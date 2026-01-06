import logging
import os
import sys
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader

BASE_DIR = os.path.dirname(__file__)
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.datasetM import WeatherGraphNPZDataset
from inference.val_autoregressive import validate_autoregressive
from model.model_unet import HeteroGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Edit these values to control the inference window and runtime options.
INIT_TIME = "2025070312"
END_TIME = "2025070312"
NPZ_ROOT = os.path.join(BASE_DIR, "npz", "operational")
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "model",
    "UnetGNNgate7_fix_attention_autoregressive_step28_GS192x192_BS1_E1_LR1e-06_WD0.1_HD512_PRL16-16-12-8-6_ACC8_SEED3407_FFN2",
    "model_epoch_001.pth",
)
SAVE_FOLDER = None
BATCH_SIZE = 1
NUM_WORKERS = 1
ROLLOUT_STEPS = 13
GRID_SIZE = (192, 192)
BOUNDARY = 8
DEVICE = None
OUT_GRID = 116
HIDDEN_DIM = 512
HEADS = 1
PROCESSOR_LAYERS = [16, 16, 12, 8, 6]
FFN_RATIO = 2.0
USE_TRANSCONV = True
USE_CHECKPOINT = True
STRICT = True
SAVE_STEP_NPZ = True


def parse_time(value: str) -> datetime:
    formats = [
        "%Y%m%d%H",
        "%Y-%m-%d%H",
        "%Y-%m-%d %H",
        "%Y-%m-%dT%H",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Unsupported time format: {value}. "
        "Use YYYYmmddHH or YYYY-mm-dd HH."
    )


def filter_samples_by_time(dataset, start_dt: datetime, end_dt: datetime):
    filtered = []
    for sample in dataset.samples:
        base = os.path.splitext(sample)[0]
        sample_dt = datetime.strptime(base[:10], "%Y%m%d%H")
        if start_dt <= sample_dt <= end_dt:
            filtered.append(sample)
    if not filtered:
        raise ValueError(
            f"No samples found between {start_dt:%Y%m%d%H} and {end_dt:%Y%m%d%H}."
        )
    dataset.samples = filtered
    return dataset


def get_mesh_in_dim(graph) -> int:
    if hasattr(graph, "node_types"):
        mesh_nodes = [n for n in graph.node_types if str(n).startswith("mesh")]
        for node_name in sorted(mesh_nodes, reverse=True):
            if hasattr(graph[node_name], "x"):
                return int(graph[node_name].x.shape[1])
    if "mesh7" in graph:
        return int(graph["mesh7"].x.shape[1])
    raise ValueError("Unable to determine mesh feature dimension from graph.")


def load_checkpoint(model, checkpoint_path: str, device, strict: bool = True):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, torch.nn.Module):
        ckpt.to(device)
        return ckpt
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=strict)
        return model
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")


def main():
    start_dt = parse_time(INIT_TIME)
    end_dt = parse_time(END_TIME)
    if start_dt > end_dt:
        raise ValueError("init_time must be <= end_time.")

    save_folder = SAVE_FOLDER
    if save_folder is None:
        folder_name = f"{start_dt:%Y%m%d%H}_{end_dt:%Y%m%d%H}"
        save_folder = os.path.join(BASE_DIR, "inference_outputs", folder_name)

    device = torch.device(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    dataset = WeatherGraphNPZDataset(
        NPZ_ROOT,
        grid_size=GRID_SIZE,
        fourcast_steps=ROLLOUT_STEPS,
    )
    dataset = filter_samples_by_time(dataset, start_dt, end_dt)

    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    sample_graph, _, _ = dataset[0]
    in_grid = int(sample_graph["grid"].x.shape[1])
    in_mesh = get_mesh_in_dim(dataset.graph_structure)
    model = HeteroGNN(
        in_grid=in_grid,
        in_mesh=in_mesh,
        hidden_dim=HIDDEN_DIM,
        out_grid=OUT_GRID,
        heads=HEADS,
        processor_layers=PROCESSOR_LAYERS,
        use_transconv=USE_TRANSCONV,
        use_checkpoint=USE_CHECKPOINT,
        ffn_ratio=FFN_RATIO,
    )
    model = load_checkpoint(model, CHECKPOINT_PATH, device, strict=STRICT)
    model.to(device)

    validate_autoregressive(
        model=model,
        val_loader=val_loader,
        device=device,
        epoch=0,
        save_folder=save_folder,
        rollout_steps=ROLLOUT_STEPS,
        H=GRID_SIZE[0],
        W=GRID_SIZE[1],
        boundary=BOUNDARY,
        save_step_npz=SAVE_STEP_NPZ,
    )


if __name__ == "__main__":
    main()
