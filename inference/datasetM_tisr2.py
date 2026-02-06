import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from toa import integrate_toa_noaa
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _BaseWeatherGraphNPZDataset(Dataset):
    """
    Shared multi-step dataset implementation. Sub-classes toggle whether
    metadata (e.g. date) should be returned to the caller.
    """

    def __init__(self,
                 root_dir,
                 grid_size=(192, 192),
                 transform=None,
                 fourcast_steps=1,
                 *,
                 return_date: bool = False):
        super().__init__(None, transform, None)
        self.root_dir = Path(root_dir)
        self.width, self.height = grid_size
        self.graph_structure = torch.load(
            Path("model") / "hetero_graph_normalized.pt",
            weights_only=False,
        )
        self.prs_feature = self._get_feature_dirs(sub="prs")
        self.sfc_regular_feature = self._get_feature_dirs(sub="sfc/regular")
        self.sfc_average_feature = self._get_feature_dirs(sub="sfc/average")
        self.time_feature = self._get_feature_dirs(sub="time")
        self.all_samples = self._get_all_samples()
        self.fourcast_steps = fourcast_steps
        self.samples = self._get_valid_samples()
        self.stats = self._load_stats()
        self.sfc_stats = self._load_sfc_stats()
        self.lon = self._load_geo_array("lon.npy")
        self.lat = self._load_geo_array("lat.npy")
        self.return_date = return_date
        self.remove_prs = ['50cc', '100cc', '1000cc', '50w', '100w', '1000w']
        self.remove_sfc_regular = ['lsm', 'z']
        self.remove_sfc_average = ['tisr']

    def _load_geo_array(self, filename: str):
        base_dir = Path(__file__).resolve().parent.parent
        candidates = [
            base_dir / "data_collect" / filename,
            base_dir / "era5" / filename,
            Path("../../era5") / filename,
        ]
        for path in candidates:
            if path.exists():
                return np.load(path)
        logger.warning("Missing %s; set to None.", filename)
        return None

    def _get_feature_dirs(self,sub=None):
        feature_dirs = set()
        folder = self.root_dir / sub
        if folder.exists() and any(folder.iterdir()):
            first_npz = next((f for f in folder.iterdir() if f.suffix == '.npz'), None)
            if first_npz:
                data = np.load(first_npz)
                feature_dirs.update(data.keys())
        return sorted(feature_dirs)

    def _get_all_samples(self):
        reference_path = self.root_dir / "prs"
        all_samples = [f for f in os.listdir(reference_path) if f.endswith('.npz')]
        return sorted(all_samples)

    def _get_valid_samples(self):
        def consecutive_6h_steps(files, ascending=True):
            files_sorted = sorted(files, reverse=not ascending)
            times = [datetime.strptime(f[:10], "%Y%m%d%H") for f in files_sorted]
            step_list = [0]
            for i in range(1, len(times)):
                delta = times[i] - times[i - 1]
                target = timedelta(hours=6) if ascending else timedelta(hours=-6)
                if delta == target:
                    step_list.append(step_list[-1] + 1)
                else:
                    step_list.append(0)
            return step_list if ascending else step_list[::-1]

        asc_steps = consecutive_6h_steps(self.all_samples, ascending=True)
        desc_steps = consecutive_6h_steps(self.all_samples, ascending=False)

        # Create the DataFrame
        df = pd.DataFrame({
            'filename': self.all_samples,
            'asc_steps': asc_steps,
            'desc_steps': desc_steps
        })
        # ---------- 3. 濾掉「任一方向 < fourcast_steps」的樣本 ----------
        keep_mask = (df['asc_steps']  >= 1) & \
                    (df['desc_steps'] >= self.fourcast_steps)

        valid_samples = df.loc[keep_mask, 'filename'].tolist()
        # 若想保持字典序（即時間正序），最後再排序一次
        return sorted(valid_samples)

    def _load_stats(self):
        stats = {}
        df = pd.read_csv(self.root_dir.parent / "../npz/prs_variable_stats.csv")
        for _, row in df.iterrows():
            stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std'])}
        return stats

    def _load_sfc_stats(self):
        sfc_stats = {}
        df_reg = pd.read_csv(self.root_dir.parent / "../npz/sfc_regular_variable_stats.csv")
        for _, row in df_reg.iterrows():
            sfc_stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std']),'max': float(row['max']), 'min': float(row['min'])}
        df_avg = pd.read_csv(self.root_dir.parent / "../npz/sfc_average_variable_stats.csv")
        for _, row in df_avg.iterrows():
            sfc_stats[row['variable']] = {'mean': float(row['mean']), 'std': float(row['std']),'max': float(row['max']), 'min': float(row['min'])}
        return sfc_stats

    def __len__(self):
        return len(self.samples)

    def _load_features(self, sample_name, remove_prs=None,remove_sfc_regular=None,remove_sfc_average=None,time=False):
        prs_npz_path = self.root_dir / "prs" / sample_name
        prs_data = np.load(prs_npz_path)
        prs_feature = [var for var in self.prs_feature if var not in (remove_prs or [])]
        features = []
        static = []

        for var in prs_feature:
            arr = prs_data[var]
            mean= self.stats[var]['mean']
            std = self.stats[var]['std']
            arr_std = (arr - mean) / std
            features.append(arr_std.reshape(-1))
        # 你可以根據需求加入地表層特徵

        sfc_regular_npz_path = self.root_dir / "sfc/regular" / sample_name
        sfc_regular_feature = [var for var in self.sfc_regular_feature if var not in (remove_sfc_regular or [])]
        sfc_regular_data = np.load(sfc_regular_npz_path)
        for var in sfc_regular_feature:
            if var == 'z' :
                # 這個特徵不需要標準化
                arr = sfc_regular_data[var]
                max = self.sfc_stats[var]['max']
                min = self.sfc_stats[var]['min']
                arr = (arr - min) / (max - min)
                static.append(arr.reshape(-1))
                continue
            elif var == 'lsm':
                # 這個特徵不需要標準化
                arr = sfc_regular_data[var]
                static.append(arr.reshape(-1))
                continue
            arr = sfc_regular_data[var]
            mean = self.sfc_stats[var]['mean']
            std = self.sfc_stats[var]['std']
            arr_std = (arr - mean) / std
            features.append(arr_std.reshape(-1))

        sfc_average_npz_path = self.root_dir / "sfc/average" / sample_name
        sfc_average_feature = [var for var in self.sfc_average_feature if var not in (remove_sfc_average or [])]
        sfc_average_data = np.load(sfc_average_npz_path)
        for var in sfc_average_feature:
            if var == 'tisr':
                # 這個特徵不需要標準化
                arr = sfc_average_data[var]
                max = self.sfc_stats[var]['max']
                min = self.sfc_stats[var]['min']
                arr = (arr - min) / (max - min)  # Min-Max Normalization
                static.append(arr.reshape(-1))
                continue
            arr = sfc_average_data[var]
            mean = self.sfc_stats[var]['mean']
            std = self.sfc_stats[var]['std']
            arr_std = (arr - mean) / std
            features.append(arr_std.reshape(-1))

        if time:
            # 加入時間特徵
            time_feature = np.load(self.root_dir / "time" / sample_name)
            for var in self.time_feature:
                arr = time_feature[var]
                features.append(arr.reshape(-1))
        features = features + static
        return np.column_stack(features)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        logger.info(f"Loading sample: {sample_name}")
        current_idx = self.all_samples.index(sample_name)

        current_graph = self.graph_structure.clone()
        current_features = self._load_features(
            sample_name,
            remove_prs=self.remove_prs,
            time=True
        )

        prev_sample = self.all_samples[current_idx - 1]
        prev_features = self._load_features(
            prev_sample,
            remove_prs=self.remove_prs,
            remove_sfc_regular=self.remove_sfc_regular,
            time=True
        )
        print(prev_sample,)

        additional_feats = self.graph_structure['grid'].x
        curr_combined = torch.cat(
            [
                torch.tensor(current_features, dtype=torch.float32),
                torch.tensor(prev_features, dtype=torch.float32),
                additional_feats
            ],
            dim=1
        )
        zero_step = torch.zeros((curr_combined.shape[0], 1), dtype=torch.float32)
        current_graph['grid'].x = torch.cat([curr_combined, zero_step], dim=1)

        next_graphs = []
        for step in range(1, self.fourcast_steps + 1):
            future_name = self.all_samples[current_idx + step]
            print(future_name)
            future_feats = self._load_features(
                future_name,
                remove_prs=self.remove_prs,
                time=True
            )
            g = self.graph_structure.clone()
            x = torch.tensor(future_feats, dtype=torch.float32)
            step_feat = torch.full((x.shape[0], 1), float(step), dtype=torch.float32)
            g['grid'].x = torch.cat([x, step_feat], dim=1)
            next_graphs.append(g)

        if self.return_date:
            return current_graph, next_graphs, sample_name[:-4]
        return current_graph, next_graphs


class WeatherGraphNPZTrainDataset(_BaseWeatherGraphNPZDataset):
    """Multi-step dataset used for training; does not return timestamps."""

    def __init__(self, root_dir, grid_size=(192, 192), transform=None, fourcast_steps=1):
        super().__init__(root_dir, grid_size, transform, fourcast_steps, return_date=False)


class WeatherGraphNPZValDataset(_BaseWeatherGraphNPZDataset):
    """Validation variant that surfaces the sample date for bookkeeping."""

    def __init__(self, root_dir, grid_size=(192, 192), transform=None, fourcast_steps=1):
        super().__init__(root_dir, grid_size, transform, fourcast_steps, return_date=True)


# Backward compatibility: validation scripts historically import WeatherGraphNPZDataset
WeatherGraphNPZDataset = WeatherGraphNPZValDataset

@lru_cache(maxsize=1)
def _load_current_feature_table():
    feature_table_path = Path(__file__).resolve().parent / "current_feature_table.csv"
    df = pd.read_csv(feature_table_path)
    df = df.sort_values("idx")
    return df["feature"].tolist()


@lru_cache(maxsize=1)
def _get_current_feature_indices():
    features = _load_current_feature_table()
    time_indices = [i for i, f in enumerate(features) if f.startswith("time:")]
    if not time_indices:
        raise ValueError("current_feature_table missing time:* entries")
    time_start = time_indices[0]
    time_end = time_indices[-1]
    prev_indices = [i for i, f in enumerate(features) if f.startswith("prev_")]
    if not prev_indices:
        raise ValueError("current_feature_table missing prev_* entries")
    prev_start = prev_indices[0]
    prev_end = prev_indices[-1]
    try:
        tisr_idx = features.index("sfc_average:tisr")
    except ValueError as exc:
        raise ValueError("current_feature_table missing sfc_average:tisr") from exc
    try:
        graph_start = features.index("graph_const_0")
        graph_end = features.index("graph_const_6")
    except ValueError as exc:
        raise ValueError("current_feature_table missing graph_const_0..6") from exc
    return {
        "time_start": time_start,
        "time_end": time_end,
        "tisr_idx": tisr_idx,
        "graph_start": graph_start,
        "graph_end": graph_end,
        "prev_start": prev_start,
        "prev_end": prev_end,
    }


@lru_cache(maxsize=1)
def _load_grid_latlon():
    base_dir = Path(__file__).resolve().parent.parent
    lat_path_candidates = [
        base_dir / "data_collect" / "lat.npy",
        base_dir / "era5" / "lat.npy",
        Path("../../era5") / "lat.npy",
    ]
    lon_path_candidates = [
        base_dir / "data_collect" / "lon.npy",
        base_dir / "era5" / "lon.npy",
        Path("../../era5") / "lon.npy",
    ]
    lat_path = next((p for p in lat_path_candidates if p.exists()), None)
    lon_path = next((p for p in lon_path_candidates if p.exists()), None)
    if lat_path is None or lon_path is None:
        raise FileNotFoundError("Missing lat.npy/lon.npy for TOA computation.")
    return np.load(lat_path), np.load(lon_path)


@lru_cache(maxsize=1)
def _load_tisr_minmax():
    stats_path = Path(__file__).resolve().parent.parent / "npz" / "sfc_average_variable_stats.csv"
    df = pd.read_csv(stats_path)
    row = df.loc[df["variable"] == "tisr"]
    if row.empty:
        raise ValueError("sfc_average_variable_stats.csv missing tisr stats")
    return float(row["min"].iloc[0]), float(row["max"].iloc[0])


@lru_cache(maxsize=1)
def _get_grid_axes():
    lat, lon = _load_grid_latlon()
    if lat.ndim == 2 and lon.ndim == 2:
        lat2d = lat
        lon2d = lon
        if lat2d[0, 0] < lat2d[-1, 0]:
            lat2d = np.flipud(lat2d)
            lon2d = np.flipud(lon2d)
        lats = lat2d[:, 0]
        lons = lon2d[0, :]
    elif lat.ndim == 1 and lon.ndim == 1:
        lats = lat
        lons = lon
        lon2d, lat2d = np.meshgrid(lons, lats)
        if lats[0] < lats[-1]:
            lats = lats[::-1]
            lat2d = lat2d[::-1, :]
            lon2d = lon2d[::-1, :]
    else:
        raise ValueError("Unexpected lat/lon shapes for TOA computation")
    return lats, lons, lat2d, lon2d


def _parse_sample_time(sample_name: str) -> datetime:
    match = re.search(r"(\d{10})", str(sample_name))
    if not match:
        raise ValueError(f"Unable to parse sample time from {sample_name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H")


def _compute_tisr_next_from_names(sample_names, step: int, batch_vector: torch.Tensor, device: torch.device):
    if sample_names is None:
        raise ValueError("sample_names is required to compute TISR from filenames")
    times = []
    for name in sample_names:
        base_time = _parse_sample_time(name)
        times.append(base_time + timedelta(hours=step * 6))
    lats, lons, _, _ = _get_grid_axes()
    toa_noaa = integrate_toa_noaa(times, lats, lons, integration_hours=1, step_seconds=30)
    tisr_vals = toa_noaa.values.reshape(len(times), -1)
    tisr_min, tisr_max = _load_tisr_minmax()
    if tisr_max == tisr_min:
        tisr_norm = tisr_vals
    else:
        tisr_norm = (tisr_vals - tisr_min) / (tisr_max - tisr_min)

    tisr_tensor = torch.tensor(tisr_norm, dtype=torch.float32, device=device)
    counts = torch.bincount(batch_vector)
    if counts.numel() != tisr_tensor.shape[0]:
        raise ValueError("Batch size does not match sample_names length")
    expected_nodes = tisr_tensor.shape[1]
    if not torch.all(counts == expected_nodes):
        raise ValueError("Node count per sample does not match TISR grid size")
    node_idx = torch.arange(batch_vector.numel(), device=device)
    starts = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), counts[:-1]]), dim=0
    )
    within = node_idx - starts[batch_vector]
    tisr_per_node = tisr_tensor[batch_vector, within]
    return tisr_per_node


def process_data_step(
    current_data,
    next_data,
    step=0,
    first116: torch.Tensor = None,
    sample_names=None,
):
    """
    使用『已覆寫邊界後』的預報 first116 (Bn, 116) 更新 current_data['grid'].x。
    其他欄位（prev_features, tisr, graph_feature, step_feat）沿用現有設計。
    """
    assert first116 is not None, "process_data_step 需要傳入 first116（已覆寫邊界的預報）"
    prev_step_idx = step - 1
    idx = _get_current_feature_indices()
    # 依 current_feature_table 切分欄位，避免硬編號
    features = _load_current_feature_table()
    current_feature_names = features[:idx["prev_start"]]
    prev_feature_names = features[idx["prev_start"]:idx["prev_end"] + 1]
    current_name_to_idx = {name: i for i, name in enumerate(current_feature_names)}
    prev_source_indices = []
    missing_prev = []
    for name in prev_feature_names:
        base_name = name.replace("prev_", "", 1)
        if base_name in current_name_to_idx:
            prev_source_indices.append(current_name_to_idx[base_name])
        else:
            missing_prev.append(name)
    # prev_* 特徵比 current 少 (例如 sfc_regular 的 lsm/z)，因此只允許特定缺項。
    allowed_missing = {"prev_sfc_regular:lsm", "prev_sfc_regular:z"}
    unexpected_missing = [name for name in missing_prev if name not in allowed_missing]
    if unexpected_missing:
        raise ValueError(f"Missing current feature(s) for prev mapping: {unexpected_missing}")
    prev_features = current_data["grid"].x[:, prev_source_indices]
    current_features_time = next_data[prev_step_idx]['grid'].x[:, idx["time_start"]:idx["tisr_idx"]]
    tisr_next = _compute_tisr_next_from_names(
        sample_names,
        step,
        current_data["grid"].batch,
        current_data["grid"].x.device,
    )

    graph_feature = current_data['grid'].x[:, idx["graph_start"]:idx["graph_end"] + 1]
    if first116.shape[1] != idx["time_start"]:
        raise ValueError(
            f"first116 feature size {first116.shape[1]} does not match expected {idx['time_start']}"
        )
    step_feat = next_data[prev_step_idx]['grid'].x[:, -1].unsqueeze(1)
    current_features = torch.cat(
        [first116, current_features_time, tisr_next.unsqueeze(1)],
        dim=1,
    )
    current_data['grid'].x = torch.cat(
        [current_features, prev_features, graph_feature, step_feat],
        dim=1,
    )
    return current_data

def move_to_device(data, device):
    return data.to(device)
if __name__ == "__main__":
    train_dataset = WeatherGraphNPZTrainDataset("../../npz/train", grid_size=(192,192), fourcast_steps=2)
    val_dataset = WeatherGraphNPZDataset("../../npz/val", grid_size=(192,192), fourcast_steps=2)
    dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False,pin_memory=True)
    df = pd.read_csv("current_feature_table.csv")
    import time 
    start_time = time.time()
    for i, (current_data, next_data, sample_name) in enumerate(dataloader):
        for k in range(len(next_data)):
            if k != 0:
                current_data = process_data_step(
                    current_data, next_data, step=k, sample_names=sample_name
                )
            # for j in range(256):
            #     var = df['variable'][j]
            #     file_path = f"vis_channel/step{k+1}/feature_map_{j}.png"
            #     visualize_feature_map(current_data[0], feature_idx=j, save_path=file_path)
        break
    end_time = time.time()
    print(f"Data loading and processing took {end_time - start_time:.2f} seconds")
