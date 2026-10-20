"""Üç resmî uygulamayı aynı veri ve görevlerle çalıştıran ince uyarlama katmanı."""
import argparse
import ast
import collections
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import add_self_loops, degree, subgraph
from experiment_common import read_payload, seed_all, evaluate_matrix_row, save_result

def execute_selected(path, names, namespace):
    """Seçilen tanımları kaynak dosyadan aynen yükle; yan etkili giriş betiğini çalıştırma."""
    tree = ast.parse(Path(path).read_text())
    chosen = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in chosen} == set(names), path
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), "exec"), namespace)

def run_fedfst(cfg, payload):
    sys.path.insert(0, str(Path(cfg.sources) / "FedFST"))
    from args import parser
    from algorithm.Ours import OursClient, OursServer
    args = parser.parse_args([])
    args.input_dim, args.output_dim = payload["input_dim"], payload["classes"]
    args.clients_num, args.device_id, args.use_gpu = 3, 0, False
    args.rounds, args.local_epochs = cfg.rounds, cfg.local_epochs
    args.gen_epochs, args.kd_epochs = cfg.gen_epochs, cfg.kd_epochs
    args.gen_rounds = 2
    args.gen_num_samples_per_class = 16
    args.num_samples_per_class = 16
    args.max_iterations = 20
    args.shuffle_flag = False
    # Kaynakta getattr ile kullanılan fakat parser'da bulunmayan parametreler.
    args.gen_ce_weight, args.kd_ce_weight = 1.0, .5
    args.seed = cfg.seed
    # Erken durdurma/epoch seçimi test kümesine erişmeden sabit bütçeyle yapılır.
    args.kd_eval_start_epoch = cfg.kd_epochs + 1
    device = torch.device("cpu")
    pool = {}
    server = OursServer(args, pool, device)
    clients = [OursClient(args, i, d, pool, device, list(range(payload["classes"])))
               for i, d in enumerate(payload["clients"])]
    for i, client in enumerate(clients):
        client.tasks = copy.deepcopy(payload["tasks"][i])
    server.set_clients(clients)
    server.send_message()
    count = len(payload["tasks"][0])
    matrix = np.full((count, count), np.nan)
    history = []
    for task in range(count):
        for r in range(cfg.rounds):
            losses = []
            for c in clients:
                if int(c.tasks[task]["train_mask"].sum()) > 0:
                    losses.append(c.train(task))
                c.send_message(task)
                if int(c.tasks[task]["train_mask"].sum()) == 0:
                    pool[f"client_{c.client_id}"]["nodes_num"] = 0
            server.aggregate()
            server.send_message()
            history.append({"task": task, "round": r, "loss": float(np.mean(losses))})
        if task:
            server.feature_gen_init(task)
            for c in clients:
                c.feature_gen_init(task)
            for r in range(args.gen_rounds):
                for c in clients:
                    c.feature_gen_train(task, r != 0)
                    c.send_feature_gen(task)
                server.feature_gen_aggregate()
                server.send_feature_gen()
            server.KD_train(task)
            server.send_message()  # Son damıtılmış global ağı değerlendirmeye ve sonraki göreve gönder.
        server.update_last_global_model()
        for c in clients:
            c.last_global_model.load_state_dict(server.last_global_model.state_dict())
        def predict(i, t):
            server.global_model.eval()
            with torch.no_grad():
                return server.global_model(clients[i].tasks[t]["local_data"])[1]
        matrix[task, :task+1] = evaluate_matrix_row(payload["clients"], payload["tasks"], predict, task)
        print("FedFST task", task, matrix[task, :task+1], flush=True)
    return matrix, vars(args), history

def run_power(cfg, payload):
    sys.path.insert(0, str(Path(cfg.sources) / "POWER"))
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from config import parser
    sys.argv = original_argv
    from flcore.Ours import OursClient, OursServer
    args = parser.parse_args([])
    args.input_dim, args.output_dim = payload["input_dim"], payload["classes"]
    args.num_rounds_per_task, args.num_epochs = cfg.rounds, cfg.local_epochs
    args.num_it_recon = cfg.recon_iterations
    args.num_epoch_g = 3
    args.seed = cfg.seed
    args.task_dir = str(Path(cfg.results) / "power_cache" / f'{cfg.dataset}_{cfg.seed}')
    Path(args.task_dir).mkdir(parents=True, exist_ok=True)
    # Önbellek eski bölmelerle karışmasın.
    for path in Path(args.task_dir).glob("*.pt"):
        path.unlink()
    device = torch.device("cpu")
    pool = {}
    data = []
    for local, tasks in zip(payload["clients"], payload["tasks"]):
        mapped = []
        for task in tasks:
            mapped.append({"train_mask": task["train_mask"], "val_mask": task["valid_mask"],
                           "test_mask": task["test_mask"]})
        data.append({"data": local, "task": mapped})
    clients = [OursClient(args, i, d, pool, device) for i, d in enumerate(data)]
    server = OursServer(args, pool, device)
    count = len(data[0]["task"])
    matrix = np.full((count, count), np.nan)
    for task in range(count):
        pool["task_id"] = task
        for r in range(cfg.rounds):
            pool["round_id"] = r
            server.send_message()
            for c in clients:
                c.execute(task)
                c.send_message(task)
            server.execute()
        server.send_message()
        def predict(i, t):
            graph = clients[i].task_data(t, data[i]["data"], data[i]["task"][t])
            server.global_model.eval()
            with torch.no_grad():
                return server.global_model(graph)[0]
        matrix[task, :task+1] = evaluate_matrix_row(payload["clients"], payload["tasks"], predict, task)
        for c in clients:
            c.task_done(task)
        print("POWER task", task, matrix[task, :task+1], flush=True)
    return matrix, vars(args), []

class TorchBlock:
    """FedSPA'nın DGL sum-reducer işleminin eşdeğer PyTorch COO uygulaması."""
    def __init__(self, data):
        self.data = data
        self.srcdata = {}
        self.dstdata = {}

    def update_all(self, message_func, reduce_func):
        src, dst = self.data.edge_index
        labels = torch.full_like(self.data.y, -1)
        labels[self.data.train_mask] = self.data.y[self.data.train_mask]
        known = (labels[src] >= 0) & (labels[dst] >= 0)
        kinds = {"hete": known & (labels[src] != labels[dst]),
                 "homo": known & (labels[src] == labels[dst]), "unk": ~known}
        for frequency in ("low", "high"):
            h = self.srcdata["h_" + frequency]
            for kind, keep in kinds.items():
                result = torch.zeros_like(h)
                result.index_add_(0, dst[keep], h[src[keep]])
                self.dstdata[f"neigh_{kind}_{frequency}"] = result

def spa_operators(data):
    n = data.num_nodes
    edge, _ = add_self_loops(data.edge_index, num_nodes=n)
    values = torch.ones(edge.shape[1])
    a = torch.sparse_coo_tensor(edge, values, (n,n)).coalesce()
    rows = a.indices()[0]
    deg = torch.zeros(n).scatter_add_(0, rows, a.values())
    low = torch.sparse_coo_tensor(a.indices(), a.values() / deg[rows].clamp_min(1), (n,n)).coalesce()
    ids = torch.arange(n).repeat(2,1)
    high = torch.sparse_coo_tensor(torch.cat([ids, low.indices()], 1),
                                 torch.cat([torch.ones(n), -low.values()]), (n,n)).coalesce()
    un = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges), (n,n)).coalesce()
    return un, low, high

class SPAWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *args):
        return self.model(*args)
    def has_nan(self):
        return any(not torch.isfinite(p).all() for p in self.parameters())

def run_fedspa(cfg, payload):
    import math
    # FLGo'nun dosya üretimi ve CUDA/DGL bağımlılığı yerine aynı model ve Server.aggregate yüklenir.
    ns = dict(globals(), math=math, device=torch.device("cpu"))
    base = Path(cfg.sources) / "FedSPA/Code/benchmark001"
    execute_selected(base / "base.py", ["Seq", "MLP_spa"], ns)
    execute_selected(base / "config.py", ["freeze_parameters", "GraphConvolution", "ACM", "get_spa"], ns)
    ns["convert_to_dglgraph"] = lambda data: TorchBlock(data)
    ns["assign_pseudo_labels_to_dglgraph"] = lambda graph, ids, labels: graph
    ns["create_block_from_dglgraph"] = lambda graph: graph
    ns["mp_func_spa"] = ns["agg_func_spa"] = None
    ns["BasicServer"] = object
    definitions = ["sum_of_layer_fisher_differences", "normalize_list", "calculate_fisher_average",
                   "normalize", "calculate_weights", "fish_cal", "Server"]
    execute_selected(Path(cfg.sources) / "FedSPA/Code/flgo/algorithm/fedspa.py", definitions, ns)
    # Sabit vektör/0 toplamında bölüm belirsizliği için nötr ağırlıklar.
    ns["normalize"] = lambda x: [(float(v) - min(map(float,x))) / (max(map(float,x)) - min(map(float,x)))
                                  for v in x] if max(map(float,x)) > min(map(float,x)) else [0.0] * len(x)
    ns["normalize_list"] = lambda x: [v / sum(x) for v in x] if float(sum(x)) else [1 / len(x)] * len(x)
    # Fisher bilgisi yalnızca eğitim etiketlerinden hesaplanır.
    def fisher(net, data, un, low, high, high_list, low_list, mlp_list):
        net.zero_grad()
        out = net(data, un, low, high, high_list, low_list, mlp_list)
        F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward()
        return {name: p.grad.detach().clone().square() for name,p in net.named_parameters() if p.grad is not None}
    ns["fish_cal"] = fisher
    model = SPAWrapper(ns["get_spa"](payload["input_dim"], 64, payload["classes"], 2))
    server = ns["Server"]()
    server.model = model
    count = len(payload["tasks"][0])
    matrix = np.full((count,count), np.nan)
    history = []
    for task in range(count):
        active, operators = [], []
        for i, tasks in enumerate(payload["tasks"]):
            graph = copy.deepcopy(tasks[task]["local_data"])
            graph.train_mask = tasks[task]["train_mask"]
            if not int(graph.train_mask.sum()):
                continue
            ops = spa_operators(graph)
            operators.append(ops)
            # Kaynaktaki spektral özellik enerjisi, yoğun Laplasyen oluşturmadan hesaplanır.
            rng = np.random.RandomState(cfg.seed + i + task)
            ids = rng.choice(graph.num_features, max(1,int(.2*graph.num_features)), replace=False)
            x = graph.x[:,ids]
            src,dst = graph.edge_index
            dx = degree(src, graph.num_nodes).view(-1,1) * x
            ax = torch.zeros_like(x).index_add_(0, src, x[dst])
            den = x.square().sum(0)
            energy = float(((x*(dx-ax)).sum(0)[den>0] / den[den>0]).mean()) if (den>0).any() else 0.
            active.append(SimpleNamespace(train_data=SimpleNamespace(data=graph), datavol=int(graph.train_mask.sum()),
                                          adj_low_un=ops[0], adj_low=ops[1], adj_high=ops[2], radio_diff=energy))
        mean_energy = sum(c.radio_diff*c.datavol for c in active) / sum(c.datavol for c in active)
        for c in active:
            c.radio_diff = abs(c.radio_diff - mean_energy)
        server.clients = active
        server.received_clients = list(range(len(active)))
        if hasattr(server,"diffs"):
            del server.diffs
        for r in range(cfg.rounds):
            locals_ = []
            losses = []
            for c, ops in zip(active, operators):
                local = copy.deepcopy(server.model)
                local.train()
                c.fr_low_g_list = [layer.att_vec_low_unk for layer in local.model.gcns]
                c.fr_high_g_list = [layer.att_vec_high_unk for layer in local.model.gcns]
                c.mlp_list = [layer.att_vec_mlp for layer in local.model.gcns]
                optimizer = torch.optim.Adam(local.parameters(), lr=.005, weight_decay=5e-4)
                graph = c.train_data.data
                for _ in range(cfg.local_epochs):
                    optimizer.zero_grad()
                    out = local(graph, *ops, c.fr_high_g_list, c.fr_low_g_list, c.mlp_list)
                    loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss))
                locals_.append(local)
            server.model = server.aggregate(locals_)
            history.append({"task":task,"round":r,"loss":float(np.mean(losses))})
        def predict(i,t):
            graph = copy.deepcopy(payload["tasks"][i][t]["local_data"])
            graph.train_mask = payload["tasks"][i][t]["train_mask"]
            server.model.eval()
            with torch.no_grad():
                return server.model(graph, *spa_operators(graph))
        matrix[task,:task+1] = evaluate_matrix_row(payload["clients"], payload["tasks"], predict, task)
        print("FedSPA task",task,matrix[task,:task+1],flush=True)
    return matrix, {"rounds":cfg.rounds,"local_epochs":cfg.local_epochs,"hidden_dim":64,
                    "learning_rate":.005,"backend":"PyTorch COO equivalent of DGL sum-reducer",
                    "protocol":"class incremental adapter; original model and Server.aggregate"}, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["FedFST","POWER","FedSPA"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--sources", default="sources")
    parser.add_argument("--data", default="experiment_data")
    parser.add_argument("--results", default="results")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--gen-epochs", type=int, default=20)
    parser.add_argument("--kd-epochs", type=int, default=20)
    parser.add_argument("--recon-iterations", type=int, default=20)
    cfg = parser.parse_args()
    seed_all(cfg.seed)
    payload = read_payload(cfg.data, cfg.dataset, cfg.seed)
    start = time.perf_counter()
    matrix, settings, history = {"FedFST":run_fedfst,"POWER":run_power,"FedSPA":run_fedspa}[cfg.method](cfg,payload)
    save_result(cfg.method,payload,matrix,time.perf_counter()-start,cfg.results,settings,history)
