import logging
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasetM import process_data_step, move_to_device                 # 使用你剛改好的介面

BASE_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

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

@torch.no_grad()
def validate_autoregressive(model,
                            val_loader,
                            device,
                            epoch,
                            save_folder,
                            rollout_steps: int,
                            H: int = 192,
                            W: int = 192,
                            boundary: int = 8,
                            save_step_npz: bool = True):

    # 與你舊 val.py 一致的標準化資訊
    weighted_table = os.path.join(BASE_DIR, "next_feature_table_weight.csv")
    fallback_table = os.path.join(BASE_DIR, "target_mean_std.csv")
    feature_table_path = weighted_table if os.path.exists(weighted_table) else fallback_table
    feature_table = pd.read_csv(feature_table_path)
    mean = feature_table['mean'].values
    std  = feature_table['std'].values

    # 目標尺度（你原本 val.py 用 target_mean_std.csv 的 std 乘回；保持一致）
    scale_df = pd.read_csv(os.path.join(BASE_DIR, "target_mean_std.csv"))
    scale_std = torch.tensor(scale_df["std"].values, dtype=torch.float32, device=device)

    os.makedirs(save_folder, exist_ok=True)
    model.eval()

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val-AR {rollout_steps} steps]")

    # 建立單一樣本的邊界 mask（(H*W,)；True 表示要用答案覆寫）
    spatial_w = create_spatial_weights(H=H, W=W, boundary=boundary, device=device)  # (H, W)
    base_mask_flat = (spatial_w.view(-1) == 0)  # 邊界權重為 0 的格點

    # 計算邊界範圍格點數
    num_boundary_points = base_mask_flat.sum().item()
    logger.info(f"Boundary mask created with {num_boundary_points} boundary points out of {H*W} total points.")

    for batch_idx, (current_data, next_data, sample_name) in enumerate(progress_bar):
        # next_data: list[Data]，長度 = rollout_steps
        current_data = move_to_device(current_data, device)
        # next_data 每個也搬到 device
        next_data = [move_to_device(g, device) for g in next_data]

        # 取出 batch 大小與節點數，做 batch 展開
        B_idx = current_data['grid'].batch        # (batch*node,)
        batch_size = int(B_idx.max().item() + 1)
        node = H * W
        assert current_data['grid'].x.shape[0] == batch_size * node, "H,W 與資料不一致，請確認"
        # DataLoader 會把 sample_name 變成 list/tuple；統一成 list 方便索引
        if isinstance(sample_name, (list, tuple)):
            sample_names = list(sample_name)
        else:
            sample_names = [sample_name] * batch_size

        # 將單樣本的 (node,) mask 展開為 (batch*node,)
        boundary_mask_bn = base_mask_flat.repeat(batch_size)  # (batch*node,)
        # 再擴到 channel 維度：(batch*node, 116)
        boundary_mask_bnC = boundary_mask_bn.unsqueeze(1).expand(-1, 116)

        # 逐步 rollout
        for step in range(1, rollout_steps + 1):
            with torch.amp.autocast(device_type=device.type):
                x = current_data['grid'].x  # shape: (num_nodes, num_channels)

                nan_mask = torch.isnan(x)  # same shape
                nan_channels = nan_mask.any(dim=0)  # (num_channels,)

                if nan_channels.any():
                    bad_channels = torch.where(nan_channels)[0].tolist()
                    raise ValueError(f"NaN found in channels: {bad_channels}")
                outputs = model(current_data)  # 預報（標準化空間）

                # 把「模型輸出」加回基準（維持你舊 val.py 的邏輯）
                pred_step = (outputs) * scale_std + current_data['grid'].x[:, :116]  # (Bn, 116)
                if torch.isnan(pred_step).any():
                    raise ValueError("NaN found in pred data.")
            # 取出該步的答案（注意 next_data[step-1] 是未來第 step 步）
            truth_step = next_data[step - 1]['grid'].x[:, :116]  # (Bn, 116)

            # optional: 保存每一步的反標準化 npz（與舊 val.py 一致）
            if save_step_npz:
                # 反標準化到實際量級
                f_np = (pred_step.detach().float().cpu().numpy()) * std + mean
                #t_np = (truth_step.detach().float().cpu().numpy()) * std + mean
                bidx_np = B_idx.cpu().numpy()

                for i in range(batch_size):
                    mask_i = (bidx_np == i)
                    f_i = f_np[mask_i].astype(np.float32)
                    #t_i = t_np[mask_i].astype(np.float32)
                    name_i = sample_names[i] if i < len(sample_names) else sample_names[0]
                    # keep legacy list-style naming, e.g. ['2024051706']_step01.npz
                    name_i = str([name_i])
                    out_path = os.path.join(save_folder, f"{name_i}_step{step:02d}.npz")
                    if np.isnan(f_i).any():
                        # 只找出「哪些通道」出現 NaN
                        nan_channels = np.where(np.isnan(f_i).any(axis=0))[0]

                        raise ValueError(
                            f"NaN found in forecast | sample={name_i} "
                            f"| step={step:02d} | channels={nan_channels.tolist()}"
                        )
                    np.savez(out_path, forecast=f_i)

            # 在「邊界」用答案覆寫模型值
            pred_step_patched = torch.where(boundary_mask_bnC, truth_step, pred_step)

            # 若還有下一步，就把「已覆寫邊界」的 first116 餵回 dataset 的接續介面
            if step < rollout_steps:
                current_data = process_data_step(
                    current_data=current_data,
                    next_data=next_data,
                    step=step,                      # 可記錄成相對步
                    first116=pred_step_patched      # 關鍵：已覆寫邊界之後的預報
                )
    logger.info("Validation (autoregressive) completed successfully.")
