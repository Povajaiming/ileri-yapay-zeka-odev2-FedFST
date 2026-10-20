"""Ortak veri bölmeleri, görevler ve değerlendirme. Test verisi eğitimde kullanılmaz."""
from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, HeterophilousGraphDataset, WikipediaNetwork
from torch_geometric.utils import subgraph, to_undirected
import networkx as nx
from community import community_louvain

DATASETS = ["cora", "citeseer", "actor", "minesweeper", "squirrel"]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)

def task_graph(data, mask):
    edges, _ = subgraph(mask, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    return Data(x=data.x, y=data.y, edge_index=edges)

def make_tasks(data, classes_per_task=2, seed=24):
    rng = np.random.default_rng(seed)
    masks = [torch.zeros(data.num_nodes, dtype=torch.bool) for _ in range(3)]
    classes = int(data.y.max()) + 1
    for label in range(classes):
        ids = torch.where(data.y == label)[0].numpy()
        rng.shuffle(ids)
        n = len(ids)
        a, b = int(n * .6), int(n * .8)
        masks[0][ids[:a]] = True
        masks[1][ids[a:b]] = True
        masks[2][ids[b:]] = True
    tasks = []
    for first in range(0, classes, classes_per_task):
        handled = (data.y >= first) & (data.y < first + classes_per_task)
        tasks.append({"train_mask": masks[0] & handled,
                      "valid_mask": masks[1] & handled,
                      "test_mask": masks[2] & handled,
                      "local_data": task_graph(data, handled)})
    return tasks

def prepare_dataset(name, root, seed=24, clients=3):
    root = Path(root)
    seed_all(seed)
    raw = root / "raw" / name
    if name in ("cora", "citeseer"):
        data = Planetoid(str(raw), name)[0]
    elif name == "actor":
        data = Actor(str(raw))[0]
    elif name == "minesweeper":
        data = HeterophilousGraphDataset(str(raw), name)[0]
    else:
        data = WikipediaNetwork(str(raw), name="squirrel", geom_gcn_preprocess=True)[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.x = data.x.float()
    data.y = data.y.long().view(-1)
    classes = int(data.y.max()) + 1
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.T.tolist())
    partition = community_louvain.best_partition(graph, random_state=seed)
    communities = {}
    for node, group in partition.items():
        communities.setdefault(group, []).append(node)
    groups = [[] for _ in range(clients)]
    for nodes in sorted(communities.values(), key=len, reverse=True):
        groups[min(range(clients), key=lambda i: len(groups[i]))].extend(nodes)
    client_data, tasks = [], []
    for i, nodes in enumerate(groups):
        ids = torch.tensor(sorted(nodes))
        edge, _ = subgraph(ids, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        local = Data(x=data.x[ids].clone(), y=data.y[ids].clone(), edge_index=edge)
        local.global_ids = ids
        local_tasks = make_tasks(local, seed=seed + i)
        # Bir sınıfın bir istemcide bulunmaması tüm istemcilerde görev sayısını değiştirmez.
        while len(local_tasks) < (classes + 1) // 2:
            empty = torch.zeros(local.num_nodes, dtype=torch.bool)
            local_tasks.append({"train_mask": empty.clone(), "valid_mask": empty.clone(),
                               "test_mask": empty.clone(), "local_data": task_graph(local, empty)})
        client_data.append(local)
        tasks.append(local_tasks)
    fingerprints = []
    for local, local_tasks in zip(client_data, tasks):
        for task in local_tasks:
            train, valid, test = [task[k] for k in ("train_mask", "valid_mask", "test_mask")]
            assert not (train & valid).any() and not (train & test).any() and not (valid & test).any()
            fingerprints.append(torch.stack([train, valid, test]).numpy().tobytes())
    split_hash = hashlib.sha256(b"".join(fingerprints)).hexdigest()
    payload = {"clients": client_data, "tasks": tasks, "classes": classes,
               "input_dim": data.num_features, "dataset": name, "seed": seed,
               "split_sha256": split_hash}
    root.mkdir(parents=True, exist_ok=True)
    torch.save(payload, root / f"{name}_{seed}.pt")
    return {"Veri seti": name, "Düğüm": data.num_nodes, "Kenar yönlü": data.num_edges,
            "Özellik": data.num_features, "Sınıf": classes, "Görev": (classes + 1) // 2,
            "İstemci düğümleri": [x.num_nodes for x in client_data], "Bölme SHA256": split_hash}

def read_payload(root, name, seed):
    return torch.load(Path(root) / f"{name}_{seed}.pt", map_location="cpu", weights_only=False)

def evaluate_matrix_row(clients, tasks, predict, stage):
    row = []
    for t in range(stage + 1):
        correct = total = 0
        for i in range(len(clients)):
            mask = tasks[i][t]["test_mask"]
            if int(mask.sum()) == 0:
                continue
            pred = predict(i, t).detach().cpu().argmax(-1)
            labels = clients[i].y.cpu()
            correct += int((pred[mask] == labels[mask]).sum())
            total += int(mask.sum())
        assert total > 0, f"Task {t}: test nodes missing"
        row.append(100 * correct / total)
    return row

def save_result(method, payload, matrix, elapsed, root, settings, history=None):
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    aa = float(matrix[-1].mean())
    af = float(np.mean(np.diag(matrix)[:-1] - matrix[-1, :-1])) if n > 1 else None
    result = {"method": method, "dataset": payload["dataset"], "seed": payload["seed"],
              "AA_percent": aa, "AF_percentage_points": af, "seconds": elapsed,
              "matrix": matrix.tolist(), "split_sha256": payload["split_sha256"],
              "settings": settings, "history": history or []}
    assert np.isfinite(matrix[np.tril_indices(n)]).all()
    output = Path(root) / f'{method}_{payload["dataset"]}_{payload["seed"]}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("history", "settings")}, indent=2), flush=True)
    return result
