以下為**可直接貼入 Markdown (`.md`) 文件**的格式版本，已整理為標準層級與程式碼區塊結構。

---

# Repository Map

## Top-level Orchestration

* **`data_operational.py`**
  End-to-end data pipeline driver
  （download ERA5 GRIBs → process to NPZ → NaN checks/fixes → SWH resize）

* **`inference_operational.py`**
  End-to-end inference driver
  （time range filter → dataset init → model load → autoregressive rollout → save）

* **`compare_difference.py`**
  Post-processing / comparison utilities

* **`vis_muliti_step.py`**
  Multi-step visualization utilities

---

## Data Ingestion & Preprocessing (`data_collect/`)

### Downloaders

* `download_prs_operational.py`
* `download_sfc_regular_operational.py`
* `download_sfc_average_operational.py`

### GRIB → NPZ Processors

* `process_prs_operational.py`
* `process_regular_operational.py`
* `process_average_operational.py`
* `process_time_operational.py`

### Data Quality / Repair

* `check_nan_prs.py`
* `process_all_nan_filling.py`
* `resize_swh.py`

---

## Inference / Data Assembly (`inference/`)

* **`datasetM_tisr2.py`**
  Main graph dataset

  * Feature loading / normalization
  * Multi-step target construction
  * Autoregressive feature refresh (`process_data_step`)

* **`val_autoregressive.py`**

  * Rollout loop
  * Autocast
  * Boundary truth patching
  * NPZ dumps

* **`toa.py`**
  TOA radiation integration utilities
  (NOAA / PVLib / ERA5-like variants)

### Feature Schema / Statistics CSVs

* `current_feature_table.csv`
* `next_feature_table*.csv`
* `target_mean_std.csv`

---

## Model (`model/`)

### Core Architecture

* **`model_unet.py`**
  HeteroGNN encoder → pool → processor → upsample → decoder (Graph U-Net)

* **`SG_module.py`**
  SGFormer-style global attention block

### Layers (`model/layer/`)

* `gat_edge_before_ffnG_fix.py`
* `gatv2convNG.py`
* `gat_edge_ffn_upG_fix.py`
* `processor__blockG.py`

### Graph Template

* `model/hetero_graph_normalized.pt`
  (Loaded by dataset as hetero graph template)

---

## Static Statistics / Normalization Data

* `npz/*.csv`
  Variable:

  * mean
  * std
  * min
  * max

  Used for normalization and inverse scaling.

---

# Data Flow (High-Level)

```
data_operational.py
    ↓
Download ERA5 monthly GRIB files
    ↓
Processing scripts
    ↓
Time-indexed NPZ slices
(npz/operational/{prs,sfc,time})
    ↓
WeatherGraphNPZDataset
    - Load current / previous / future NPZ
    - Normalize features
    - Attach graph constants + step feature
    ↓
HeteroGNN
    - Predict grid increments / residual-like outputs
    ↓
validate_autoregressive
    - Convert outputs to next-step state
    - Patch boundaries with truth
    - Roll forward
    - Save per-step NPZ
```

---

如果你需要，我也可以幫你轉成：

* ✅ README 專業版本（含系統架構圖）
* ✅ Mermaid 架構圖（可在 GitHub 直接渲染）
* ✅ Paper appendix 風格說明版本
* ✅ 加入 training/inference separation 的 deployment 說明版本
