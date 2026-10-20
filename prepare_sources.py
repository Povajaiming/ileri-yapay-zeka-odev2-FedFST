"""Kaynak sürümlerini indirir ve belgelenmiş çalıştırılabilirlik yamalarını uygular."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import shutil

SOURCES = {
    "FedFST": ("https://github.com/YunQi572/FedFST", "d83fac0eff2ebf764d5574997a5fec0abf9cc2c7"),
    "FedSPA": ("https://github.com/OakleyTan/FedSPA", "d2b3ddb8f853a8d78c120407eba6c1a44127bc33"),
}

def prepare_sources(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (url, revision) in SOURCES.items():
        target = root / name
        if not target.exists():
            archive = root / (name + ".zip")
            urllib.request.urlretrieve(f"{url}/archive/{revision}.zip", archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(root)
                shutil.move(str(root / z.namelist()[0].split('/')[0]), target)
        manifest[name] = {"url": url, "revision": revision}
    power = root / "POWER"
    if not (power / "flcore").exists():
        archive = root / "POWER.zip"
        urllib.request.urlretrieve(
            "https://zenodo.org/api/records/15487655/files/zyl24/FCGL_POWER-official.zip/content", archive)
        assert hashlib.md5(archive.read_bytes()).hexdigest() == "567faa826ad62a3d8610682ee2e8474f"
        with zipfile.ZipFile(archive) as z:
            folder = z.namelist()[0].split('/')[0]
            z.extractall(root / "power_extract")
        if power.exists():
            shutil.rmtree(power)
        shutil.move(str(root / "power_extract" / folder), power)
    manifest["POWER"] = {"url": "https://zenodo.org/records/15487655",
                         "revision": "78e30b6", "md5": "567faa826ad62a3d8610682ee2e8474f"}

    # Kullanılmayan, depoda tanımlanmamış fonksiyonları içe aktarma hatası.
    ours = root / "FedFST/algorithm/Ours.py"
    text = ours.read_text().replace(
        ", edge_distribution_low1, edge_distribution_low2", "")
    # Eski sınıfların üreticisinde sadece eğitim verisinin özellikleri/etiketleri kullanılır.
    text = text.replace('task_data["train_mask"] | task_data["valid_mask"] | task_data["test_mask"]',
                        'task_data["train_mask"]')
    text = text.replace('task["train_mask"] | task["valid_mask"] | task["test_mask"]',
                        'task["train_mask"]')
    ours.write_text(text)
    for package in ("datasets", "models", "algorithm"):
        (root / "FedFST" / package / "__init__.py").touch()
    loader = root / "FedFST/datasets/dataset_loader.py"
    text = loader.read_text().replace("from torch_sparse import SparseTensor", "# Unused optional torch_sparse import removed")
    # Son grupta tek sınıf varsa onu da koru. Tüm veri setlerinin tüm sınıfları kullanılır.
    text = text.replace("if drop_flag:\n        tasks_num = tasks_num - 1", "if False:  # keep the final incomplete class group\n        tasks_num = tasks_num - 1")
    loader.write_text(text)

    # POWER'ın bool maskeyi düğüm indisleri gibi yineleyen yardımcı fonksiyonunu düzelt.
    utils = power / "utils.py"
    text = utils.read_text()
    start = text.index("def edge_masking(")
    end = text.index("\ndef accuracy(", start)
    text = text[:start] + '''def edge_masking(edge_index, handled, device):
    handled = handled.to(device).bool()
    keep = handled[edge_index[0]] & handled[edge_index[1]]
    return coalesce(edge_index[:, keep])

''' + text[end:]
    utils.write_text(text)
    # OpenFGL'den yalnızca bu 4 satırlık maske yardımcısı kullanılıyor.
    ours = power / "flcore/Ours.py"
    text = ours.read_text().replace("from openfgl.utils.basic_utils import idx_to_mask_tensor", '''def idx_to_mask_tensor(indices, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask''')
    ours.write_text(text)
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
