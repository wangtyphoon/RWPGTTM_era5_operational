import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import torch
#from rmse import create_spatial_weights
import numpy as np

# ---------- 2) 空間權重 ----------
def create_spatial_weights(H=192, W=192, boundary=8, device=torch.device('cpu')):
    """
    緯度餘弦加權 + 邊界距離≤boundary 的格點置 0。
    回傳 shape = (H, W) 的 torch.Tensor
    """
    # cos(φ) 緯度加權（φ=90°–緯度；若資料自北到南，需 flip）
    lats_deg = np.linspace(0, 47.75, H)          # 依實際格網修改
    lat_w    = np.cos(np.deg2rad(lats_deg))      # shape (H,)
    w2d      = np.tile(lat_w[:, None], (1, W))   # (H, W)
    w2d = np.flipud(w2d).copy()             # ERA-I/ERA5 分段常見 N→S 需反轉

    # 距離四邊最近的格點距離
    y_idx, x_idx   = np.indices((H, W))
    dist_top       = y_idx
    dist_bottom    = H - 1 - y_idx
    dist_left      = x_idx
    dist_right     = W - 1 - x_idx
    min_dist       = np.minimum.reduce([dist_top, dist_bottom, dist_left, dist_right])

    # 將距離 ≤ boundary 的格點權重設定為 0
    w2d[min_dist <= boundary] = 0.0
    w2d = w2d / np.mean(w2d)  # 正規化權重，使總和為 1

    return torch.tensor(w2d, dtype=torch.float32, device=device)  # (H, W)

# 給定要看的日期時間字串（必須跟檔名裡的一樣，例如 '2024122718'）
TIME_TAG = "2025070500"

INPUT_DIR = Path(r"inference_outputs\2025070500_2025070500test3")
#TARGET_DIR = Path(r"UnetGNNgate7_fix_attention_target_GS192x192_BS1_E1_LR0.001_WD0.1_HD512_PRL16-16-12-8-6_ACC4_SEED3407_FFN2")
OUTPUT_DIR = INPUT_DIR / "figures"
TARGET_KEY = "target"
PROCESS_BASE_DIR = Path("inference_outputs")

# 只要檔名裡有這個時間字串 + "_step" 就會被抓到
STEP_PATTERN = f"*{TIME_TAG}*_step*.npz"
CHANNELS = [59, 76, 79, 80, 85, 86, 87]  # Channels to visualize.
SHOW_FIGURE = False
VALUE_COLORMAP = "viridis"
MSE_COLORMAP = "magma"
INCLUDE_MSE_MAP = True
COMPARE_FORECAST_PROCESS = True
WIND_SPEED_CHANNELS = (86, 87)  # (u, v) channel indices used to derive wind speed.
USE_CARTOPY_TERRAIN = True
COASTLINE_RESOLUTION = "50m"
LAND_COLOR = "0.9"
OCEAN_COLOR = "0.85"
VIS_FORECAST_ONLY = False

lat = np.load("data_collect/lat.npy")
lon = np.load("data_collect/lon.npy")

GRID_SHAPE = lat.shape
FLAT_SIZE = GRID_SHAPE[0] * GRID_SHAPE[1]
GRID_EXTENT = (
    float(np.nanmin(lon)),
    float(np.nanmax(lon)),
    float(np.nanmin(lat)),
    float(np.nanmax(lat)),
)

spatial_weights = create_spatial_weights(H=GRID_SHAPE[0], W=GRID_SHAPE[1], boundary=8)
spatial_weights = np.array(spatial_weights, dtype=np.float32)


def parse_timestamp_from_filename(path: Path, time_format: str = "%Y%m%d%H"):
    """自檔名解析時間，失敗則回傳 None。"""
    name = path.stem
    try:
        return datetime.strptime(name, time_format)
    except ValueError:
        pass

    digits = re.findall(r"(\d{10})", name)
    for cand in digits:
        try:
            return datetime.strptime(cand, "%Y%m%d%H")
        except ValueError:
            continue
    return None


def parse_step_from_filename(path: Path):
    """從檔名抓出 step 編號（_stepXX）。"""
    m = re.search(r"_step(\d+)", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else None


def load_forecast_and_target(path: Path):
    """
    讀取預報與 target；兩者皆存在同一個檔案內。
    """
    with np.load(path) as data:
        if "forecast" not in data:
            raise KeyError(f"{path.name} is missing the 'forecast' array.")
        target_key = TARGET_KEY if TARGET_KEY in data else None
        if target_key is None:
            for candidate in ("truth", "target"):
                if candidate in data:
                    target_key = candidate
                    break
        if target_key is None:
            raise KeyError(
                f"{path.name} is missing a target array (tried {TARGET_KEY}, 'truth', 'target')."
            )
        forecast = data["forecast"]
        target = data[target_key]

    return forecast, target


def load_forecast_only(path: Path):
    """讀取預報資料。"""
    with np.load(path) as data:
        if "forecast" not in data:
            raise KeyError(f"{path.name} is missing the 'forecast' array.")
        forecast = data["forecast"]
    return forecast


def load_process_only(path: Path):
    """讀取 process（邊界覆寫後）資料。"""
    with np.load(path) as data:
        if "input_data" not in data:
            raise KeyError(f"{path.name} is missing the 'input_data' array.")
        process = data["input_data"]
    return process


def find_process_path(step_path: Path):
    step_num = parse_step_from_filename(step_path)
    if step_num is None:
        return None
    match = re.match(r"^(.*)_step(\d+)\.npz$", step_path.name)
    if match:
        process_name = f"{match.group(1)}_process_step{match.group(2)}.npz"
        return step_path.with_name(process_name)
    candidates = list(step_path.parent.glob(f"*process_step{step_num:02d}.npz"))
    return candidates[0] if candidates else None


def load_forecast_and_process(path: Path):
    """讀取預報與 process（邊界覆寫後）資料。"""
    with np.load(path) as data:
        if "forecast" not in data:
            raise KeyError(f"{path.name} is missing the 'forecast' array.")
        forecast = data["forecast"]

    process_path = find_process_path(path)
    if process_path is None or not process_path.exists():
        raise FileNotFoundError(f"找不到對應的 process 檔案：{process_path}")
    process = load_process_only(process_path)
    return forecast, process


def collect_step_files():
    step_files = [
        path for path in sorted(INPUT_DIR.glob(STEP_PATTERN)) if "_process_step" not in path.name
    ]
    if not step_files:
        raise FileNotFoundError(f"No {STEP_PATTERN} files found in {INPUT_DIR}")
    return step_files


def get_layout(array: np.ndarray):
    if array.ndim == 1:
        if array.size != FLAT_SIZE:
            raise ValueError(f"Expected flattened array of size {FLAT_SIZE}, got {array.size}")
        return 1, "flat"
    if array.ndim == 2:
        if array.shape[0] == FLAT_SIZE:
            return array.shape[1], "columns"
        if array.shape[1] == FLAT_SIZE:
            return array.shape[0], "rows"
    raise ValueError(f"Array shape {array.shape} does not contain a flattened {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid.")


def normalize_index(index: int, count: int) -> int:
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    if index < 0:
        index += count
    if not 0 <= index < count:
        raise IndexError(f"Sample index {index} out of range for {count} samples.")
    return index


def flatten_sample(array: np.ndarray, index: int, layout: str) -> np.ndarray:
    if layout == "flat":
        return array
    if layout == "columns":
        return array[:, index]
    if layout == "rows":
        return array[index, :]
    raise ValueError(f"Unknown layout {layout!r}")


def reshape_grid(flat: np.ndarray) -> np.ndarray:
    if flat.size != FLAT_SIZE:
        raise ValueError(f"Cannot reshape array of size {flat.size} to {GRID_SHAPE}")
    return flat.reshape(GRID_SHAPE)


def load_channel_grid(array: np.ndarray, channel_index: int):
    sample_count, layout = get_layout(array)
    normalized_index = normalize_index(channel_index, sample_count)
    flat = flatten_sample(array, normalized_index, layout)
    return reshape_grid(flat)


def prepare_step_data(step_files, channel_index):
    prepared = []
    global_value_min = None
    global_value_max = None
    global_mse_max = None
    global_bias_min = None
    global_bias_max = None
    total_weighted_bias = 0.0
    total_weight = 0.0
    total_squared_error = 0.0
    total_squared_error_debias = 0.0
    total_count = 0

    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        forecast_raw, target_raw = load_forecast_and_target(path)

        forecast = load_channel_grid(forecast_raw, channel_index)
        target = load_channel_grid(target_raw, channel_index)

        diff = forecast - target
        squared_error = diff ** 2
        weighted = squared_error * spatial_weights       
        # 加權平均：若使用 mask，分母為有效點數；若使用連續權重，分母為權重和
        denom = spatial_weights.sum().clip(min=1e-12)       # (1,)
        mse = weighted.sum() / denom         # (C,)
        rmse = float(np.sqrt(mse))
        bias_mean = float((diff * spatial_weights).sum() / denom)
        # Debias by removing the weighted mean bias
        diff_debiased = diff - bias_mean
        squared_error_debiased = diff_debiased ** 2
        mse_debiased = (squared_error_debiased * spatial_weights).sum() / denom
        rmse_debiased = float(np.sqrt(mse_debiased))

        local_min = min(forecast.min(), target.min())
        local_max = max(forecast.max(), target.max())
        bias_min = float(diff.min())
        bias_max = float(diff.max())

        global_value_min = local_min if global_value_min is None else min(global_value_min, local_min)
        global_value_max = local_max if global_value_max is None else max(global_value_max, local_max)
        global_bias_min = bias_min if global_bias_min is None else min(global_bias_min, bias_min)
        global_bias_max = bias_max if global_bias_max is None else max(global_bias_max, bias_max)

        if INCLUDE_MSE_MAP:
            mse_max = float(squared_error.max())
            global_mse_max = mse_max if global_mse_max is None else max(global_mse_max, mse_max)

        total_squared_error += float(np.sum(squared_error))
        total_squared_error_debias += float(np.sum(squared_error_debiased))
        total_count += diff.size
        total_weighted_bias += float((diff * spatial_weights).sum())
        total_weight += float(denom)

        prepared.append(
            {
                "step_number": step_num,
                "path": path,
                "forecast": forecast,
                "target": target,
                "bias": diff,
                "mse": squared_error,
                "rmse": rmse,
                "bias_mean": bias_mean,
                "bias_debiased": diff_debiased,
                "mse_debiased": squared_error_debiased,
                "rmse_debiased": rmse_debiased,
            }
        )

    overall_rmse = float(np.sqrt(total_squared_error / total_count)) if total_count else 0.0
    overall_rmse_debias = (
        float(np.sqrt(total_squared_error_debias / total_count)) if total_count else 0.0
    )
    return (
        prepared,
        global_value_min,
        global_value_max,
        global_mse_max,
        overall_rmse,
        global_bias_min,
        global_bias_max,
        float(total_weighted_bias / total_weight) if total_weight > 0 else 0.0,
        overall_rmse_debias,
    )


def prepare_forecast_only_data(step_files, channel_index):
    prepared = []
    global_value_min = None
    global_value_max = None

    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        forecast_raw = load_forecast_only(path)

        forecast = load_channel_grid(forecast_raw, channel_index)

        local_min = float(forecast.min())
        local_max = float(forecast.max())
        global_value_min = local_min if global_value_min is None else min(global_value_min, local_min)
        global_value_max = local_max if global_value_max is None else max(global_value_max, local_max)

        prepared.append(
            {
                "step_number": step_num,
                "path": path,
                "forecast": forecast,
            }
        )

    return prepared, global_value_min, global_value_max


def prepare_forecast_process_data(step_files, channel_index):
    prepared = []
    global_value_min = None
    global_value_max = None
    global_mse_max = None
    global_bias_min = None
    global_bias_max = None
    total_squared_error = 0.0
    total_count = 0

    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        forecast_raw, process_raw = load_forecast_and_process(path)

        forecast = load_channel_grid(forecast_raw, channel_index)
        process = load_channel_grid(process_raw, channel_index)

        diff = process - forecast
        squared_error = diff ** 2
        rmse = float(np.sqrt(np.mean(squared_error)))
        bias_mean = float(np.mean(diff))

        local_min = min(forecast.min(), process.min())
        local_max = max(forecast.max(), process.max())
        bias_min = float(diff.min())
        bias_max = float(diff.max())

        global_value_min = local_min if global_value_min is None else min(global_value_min, local_min)
        global_value_max = local_max if global_value_max is None else max(global_value_max, local_max)
        global_bias_min = bias_min if global_bias_min is None else min(global_bias_min, bias_min)
        global_bias_max = bias_max if global_bias_max is None else max(global_bias_max, bias_max)

        if INCLUDE_MSE_MAP:
            mse_max = float(squared_error.max())
            global_mse_max = mse_max if global_mse_max is None else max(global_mse_max, mse_max)

        total_squared_error += float(np.sum(squared_error))
        total_count += diff.size

        prepared.append(
            {
                "step_number": step_num,
                "path": path,
                "forecast": forecast,
                "process": process,
                "bias": diff,
                "mse": squared_error,
                "rmse": rmse,
                "bias_mean": bias_mean,
            }
        )

    overall_rmse = float(np.sqrt(total_squared_error / total_count)) if total_count else 0.0
    return (
        prepared,
        global_value_min,
        global_value_max,
        global_mse_max,
        overall_rmse,
        global_bias_min,
        global_bias_max,
    )


def prepare_forecast_target_process_data(step_files, channel_index):
    prepared = []
    global_value_min = None
    global_value_max = None
    global_mse_max = None
    global_bias_min = None
    global_bias_max = None
    total_squared_error = 0.0
    total_count = 0

    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        process_path = find_process_path(path)
        if process_path is None or not process_path.exists():
            print(f"[Info] Skip step {step_num}: process file not found for {path.name}.")
            continue

        forecast_raw, target_raw = load_forecast_and_target(path)
        process_raw = load_process_only(process_path)

        forecast = load_channel_grid(forecast_raw, channel_index)
        target = load_channel_grid(target_raw, channel_index)
        process = load_channel_grid(process_raw, channel_index)

        diff = forecast - target
        squared_error = diff ** 2
        weighted = squared_error * spatial_weights
        denom = spatial_weights.sum().clip(min=1e-12)
        rmse = float(np.sqrt(weighted.sum() / denom))
        bias_mean = float((diff * spatial_weights).sum() / denom)

        process_diff = process - forecast
        process_bias_min = float(process_diff.min())
        process_bias_max = float(process_diff.max())

        local_min = min(forecast.min(), target.min(), process.min())
        local_max = max(forecast.max(), target.max(), process.max())

        global_value_min = local_min if global_value_min is None else min(global_value_min, local_min)
        global_value_max = local_max if global_value_max is None else max(global_value_max, local_max)
        global_bias_min = (
            process_bias_min if global_bias_min is None else min(global_bias_min, process_bias_min)
        )
        global_bias_max = (
            process_bias_max if global_bias_max is None else max(global_bias_max, process_bias_max)
        )

        if INCLUDE_MSE_MAP:
            mse_max = float(squared_error.max())
            global_mse_max = mse_max if global_mse_max is None else max(global_mse_max, mse_max)

        total_squared_error += float(np.sum(squared_error))
        total_count += diff.size

        prepared.append(
            {
                "step_number": step_num,
                "path": path,
                "forecast": forecast,
                "target": target,
                "process": process,
                "bias": process_diff,
                "mse": squared_error,
                "rmse": rmse,
                "bias_mean": bias_mean,
            }
        )

    if not prepared:
        raise FileNotFoundError("No steps with matching process files were found.")

    overall_rmse = float(np.sqrt(total_squared_error / total_count)) if total_count else 0.0
    return (
        prepared,
        global_value_min,
        global_value_max,
        global_mse_max,
        overall_rmse,
        global_bias_min,
        global_bias_max,
    )


def prepare_wind_speed_data(step_files, u_channel, v_channel):
    prepared = []
    global_value_min = None
    global_value_max = None
    global_mse_max = None
    global_bias_min = None
    global_bias_max = None
    total_weighted_bias = 0.0
    total_weight = 0.0
    total_squared_error = 0.0
    total_squared_error_debias = 0.0
    total_count = 0

    for step_id, path in enumerate(step_files, start=1):
        step_num = parse_step_from_filename(path) or step_id
        forecast_raw, target_raw = load_forecast_and_target(path)

        forecast_u = load_channel_grid(forecast_raw, u_channel)
        forecast_v = load_channel_grid(forecast_raw, v_channel)
        target_u = load_channel_grid(target_raw, u_channel)
        target_v = load_channel_grid(target_raw, v_channel)

        forecast = np.sqrt(forecast_u ** 2 + forecast_v ** 2)
        target = np.sqrt(target_u ** 2 + target_v ** 2)

        diff = forecast - target
        squared_error = diff ** 2
        weighted = squared_error * spatial_weights                        # (batch*node, C)
        denom = spatial_weights.sum().clip(min=1e-12)       # (1,)
        mse = weighted.sum() / denom         # (C,)
        rmse = float(np.sqrt(mse))
        bias_mean = float((diff * spatial_weights).sum() / denom)
        diff_debiased = diff - bias_mean
        squared_error_debiased = diff_debiased ** 2
        mse_debiased = (squared_error_debiased * spatial_weights).sum() / denom
        rmse_debiased = float(np.sqrt(mse_debiased))

        local_min = min(forecast.min(), target.min())
        local_max = max(forecast.max(), target.max())
        bias_min = float(diff.min())
        bias_max = float(diff.max())

        global_value_min = local_min if global_value_min is None else min(global_value_min, local_min)
        global_value_max = local_max if global_value_max is None else max(global_value_max, local_max)
        global_bias_min = bias_min if global_bias_min is None else min(global_bias_min, bias_min)
        global_bias_max = bias_max if global_bias_max is None else max(global_bias_max, bias_max)

        if INCLUDE_MSE_MAP:
            mse_max = float(squared_error.max())
            global_mse_max = mse_max if global_mse_max is None else max(global_mse_max, mse_max)

        total_squared_error += float(np.sum(squared_error))
        total_squared_error_debias += float(np.sum(squared_error_debiased))
        total_count += diff.size
        total_weighted_bias += float((diff * spatial_weights).sum())
        total_weight += float(denom)

        prepared.append(
            {
                "step_number": step_num,
                "path": path,
                "forecast": forecast,
                "target": target,
                "bias": diff,
                "mse": squared_error,
                "rmse": rmse,
                "bias_mean": bias_mean,
                "bias_debiased": diff_debiased,
                "mse_debiased": squared_error_debiased,
                "rmse_debiased": rmse_debiased,
            }
        )

    overall_rmse = float(np.sqrt(total_squared_error / total_count)) if total_count else 0.0
    overall_rmse_debias = (
        float(np.sqrt(total_squared_error_debias / total_count)) if total_count else 0.0
    )
    return (
        prepared,
        global_value_min,
        global_value_max,
        global_mse_max,
        overall_rmse,
        global_bias_min,
        global_bias_max,
        float(total_weighted_bias / total_weight) if total_weight > 0 else 0.0,
        overall_rmse_debias,
    )


def visualize_steps(
    prepared,
    value_vmin,
    value_vmax,
    mse_max,
    channel_label,
    overall_rmse,
    overall_debias_rmse=None,
    bias_vmin=None,
    bias_vmax=None,
    overall_bias=None,
    row_configs=None,
):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualization. Install it with `pip install matplotlib`."
        ) from exc
    cartopy_ctx = None
    if USE_CARTOPY_TERRAIN:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            cartopy_ctx = {
                "projection": ccrs.PlateCarree(),
                "data_crs": ccrs.PlateCarree(),
                "feature": cfeature,
            }
        except ModuleNotFoundError:
            print("[Info] cartopy not available; fallback to plain matplotlib plots.")
        except Exception as exc:
            print(f"[Info] cartopy initialization failed ({exc}); fallback to plain matplotlib plots.")

    if not SHOW_FIGURE:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if row_configs is None:
        row_configs = [
            ("forecast", "Forecast", VALUE_COLORMAP, value_vmin, value_vmax),
            ("target", "Target", VALUE_COLORMAP, value_vmin, value_vmax),
            ("bias", "Bias", "coolwarm", None, None),
        ]

    # Bias uses a symmetric range to highlight sign when present.
    uses_bias = any(key == "bias" for key, *_ in row_configs)
    if uses_bias:
        if bias_vmin is None or bias_vmax is None:
            all_bias_min = min(float(np.min(step["bias"])) for step in prepared)
            all_bias_max = max(float(np.max(step["bias"])) for step in prepared)
        else:
            all_bias_min, all_bias_max = bias_vmin, bias_vmax
        bias_limit = max(abs(all_bias_min), abs(all_bias_max))
        bias_vmin = -bias_limit
        bias_vmax = bias_limit
        row_configs = [
            (key, label, cmap, bias_vmin if key == "bias" else vmin, bias_vmax if key == "bias" else vmax)
            for key, label, cmap, vmin, vmax in row_configs
        ]

    if INCLUDE_MSE_MAP and mse_max is not None:
        mse_limit = mse_max if mse_max is not None else 0.0
        row_configs.append(("mse", "MSE", MSE_COLORMAP, 0.0, mse_limit))

    n_steps = len(prepared)
    n_rows = len(row_configs)
    subplot_kwargs = {"figsize": (4 * n_steps, 4 * n_rows)}
    if cartopy_ctx:
        subplot_kwargs["subplot_kw"] = {"projection": cartopy_ctx["projection"]}
    fig, axes = plt.subplots(n_rows, n_steps, **subplot_kwargs)
    axes = np.asarray(axes)

    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_steps == 1:
            axes = axes[:, np.newaxis]
        else:
            axes = axes.reshape(n_rows, n_steps)

    if cartopy_ctx:
        for ax in axes.ravel():
            try:
                ax.set_extent(GRID_EXTENT, crs=cartopy_ctx["data_crs"])
                ax.stock_img()
            except Exception:
                ax.add_feature(cartopy_ctx["feature"].LAND, facecolor=LAND_COLOR, zorder=0)
                ax.add_feature(cartopy_ctx["feature"].OCEAN, facecolor=OCEAN_COLOR, zorder=0)
            ax.coastlines(resolution=COASTLINE_RESOLUTION, linewidth=0.6)

    row_colorbars = [None] * n_rows

    for column, step_info in enumerate(prepared):
        for row, (key, label, cmap, vmin, vmax) in enumerate(row_configs):
            ax = axes[row, column]
            data = step_info[key]

            if key == "mse" and mse_max is not None:
                vmax = mse_max
                vmin = 0.0
            if key == "bias":
                vmax = row_configs[row][4]
                vmin = row_configs[row][3]
            imshow_kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "origin": "upper"}
            if cartopy_ctx:
                imshow_kwargs.update({"extent": GRID_EXTENT, "transform": cartopy_ctx["data_crs"]})
            im = ax.imshow(data, **imshow_kwargs)
            row_colorbars[row] = im

            title = f"Step {step_info['step_number']} {label}"
            if key == "mse":
                if "rmse_debiased" in step_info:
                    title += (
                        f" (RMSE={step_info['rmse']:.4f}, "
                        f"debias RMSE={step_info['rmse_debiased']:.4f})"
                    )
                elif "rmse" in step_info:
                    title += f" (RMSE={step_info['rmse']:.4f})"
            if key == "bias" and "bias_mean" in step_info:
                title += f" (Bias={step_info['bias_mean']:.4f})"
            ax.set_title(title)
            ax.axis("off")

    bias_text = f", overall Bias={overall_bias:.4f}" if overall_bias is not None else ""
    debias_text = (
        f", debias RMSE={overall_debias_rmse:.4f}" if overall_debias_rmse is not None else ""
    )
    title_suffix = ""
    if overall_rmse is not None:
        title_suffix = f" (overall RMSE={overall_rmse:.4f}{debias_text}{bias_text})"
    fig.suptitle(
        f"{channel_label} across steps{title_suffix}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for row, im in enumerate(row_colorbars):
        if im is None:
            continue
        color_label = "MSE" if row_configs[row][0] == "mse" else "Value"
        cbar = fig.colorbar(
            im,
            ax=axes[row, :].ravel().tolist(),
            shrink=0.75,
            pad=0.02,
        )
        cbar.ax.set_ylabel(color_label, rotation=270, labelpad=15)

    if SHOW_FIGURE:
        plt.show()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{TIME_TAG}{channel_label}_steps.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    plt.close(fig)

    if all("rmse" in step_info for step_info in prepared):
        print("\nPer-step RMSE:")
        for step_info in prepared:
            debias_text = (
                f", RMSE_debiased={step_info['rmse_debiased']:.6f}"
                if "rmse_debiased" in step_info
                else ""
            )
            print(
                f"  Step {step_info['step_number']}: "
                f"RMSE={step_info['rmse']:.6f}, "
                f"BiasMean={step_info['bias_mean']:.6f}"
                f"{debias_text}"
            )

        if overall_debias_rmse is not None:
            print(f"Overall RMSE: {overall_rmse:.6f}, debias RMSE: {overall_debias_rmse:.6f}")
        else:
            print(f"Overall RMSE: {overall_rmse:.6f}")


def main():
    step_files = collect_step_files()
    for channel_index in CHANNELS:
        if VIS_FORECAST_ONLY:
            prepared, value_vmin, value_vmax = prepare_forecast_only_data(
                step_files, channel_index
            )
            row_configs = [
                ("forecast", "Forecast", VALUE_COLORMAP, value_vmin, value_vmax),
            ]
            visualize_steps(
                prepared,
                value_vmin,
                value_vmax,
                mse_max=None,
                channel_label=f"channel_{channel_index:03d}",
                overall_rmse=None,
                row_configs=row_configs,
            )
        elif COMPARE_FORECAST_PROCESS:
            (
                prepared,
                value_vmin,
                value_vmax,
                mse_max,
                overall_rmse,
                bias_vmin,
                bias_vmax,
            ) = prepare_forecast_target_process_data(step_files, channel_index)
            row_configs = [
                ("forecast", "Forecast", VALUE_COLORMAP, value_vmin, value_vmax),
                ("target", "Target", VALUE_COLORMAP, value_vmin, value_vmax),
                ("process", "Process", VALUE_COLORMAP, value_vmin, value_vmax),
                ("bias", "Process - Forecast", "coolwarm", bias_vmin, bias_vmax),
            ]
            visualize_steps(
                prepared,
                value_vmin,
                value_vmax,
                mse_max,
                channel_label=f"channel_{channel_index:03d}",
                overall_rmse=overall_rmse,
                bias_vmin=bias_vmin,
                bias_vmax=bias_vmax,
                row_configs=row_configs,
            )
        else:
            (
                prepared,
                value_vmin,
                value_vmax,
                mse_max,
                overall_rmse,
                bias_vmin,
                bias_vmax,
                overall_bias,
                overall_rmse_debias,
            ) = prepare_step_data(step_files, channel_index)
            visualize_steps(
                prepared,
                value_vmin,
                value_vmax,
                mse_max,
                channel_label=f"channel_{channel_index:03d}",
                overall_rmse=overall_rmse,
                overall_debias_rmse=overall_rmse_debias,
                bias_vmin=bias_vmin,
                bias_vmax=bias_vmax,
                overall_bias=overall_bias,
            )

    if WIND_SPEED_CHANNELS and not VIS_FORECAST_ONLY:
        (
            prepared,
            value_vmin,
            value_vmax,
            mse_max,
            overall_rmse,
            bias_vmin,
            bias_vmax,
            overall_bias,
            overall_rmse_debias,
        ) = prepare_wind_speed_data(step_files, WIND_SPEED_CHANNELS[0], WIND_SPEED_CHANNELS[1])
        visualize_steps(
            prepared,
            value_vmin,
            value_vmax,
            mse_max,
            channel_label=f"wind_speed_{WIND_SPEED_CHANNELS[0]:03d}_{WIND_SPEED_CHANNELS[1]:03d}",
            overall_rmse=overall_rmse,
            overall_debias_rmse=overall_rmse_debias,
            bias_vmin=bias_vmin,
            bias_vmax=bias_vmax,
            overall_bias=overall_bias,
        )


if __name__ == "__main__":
    main()
