Top-level orchestration

data_operational.py: end-to-end data pipeline driver (download ERA5 GRIBs → process to NPZ → NaN checks/fixes → SWH resize).

inference_operational.py: end-to-end inference driver (time range filter, dataset init, model load, autoregressive rollout/save).

compare_difference.py, vis_muliti_step.py: post-processing/visualization utilities.

Data ingestion & preprocessing (data_collect/)

Downloaders:

download_prs_operational.py

download_sfc_regular_operational.py

download_sfc_average_operational.py

GRIB→NPZ processors:

process_prs_operational.py

process_regular_operational.py

process_average_operational.py

process_time_operational.py

Data quality / repair:

check_nan_prs.py

process_all_nan_filling.py

resize_swh.py

Inference/data assembly (inference/)

datasetM_tisr2.py: main graph dataset, feature loading/normalization, multi-step target construction, autoregressive feature refresh (process_data_step).

val_autoregressive.py: rollout loop, autocast, boundary truth patching, NPZ dumps.

toa.py: TOA radiation integration utilities (NOAA / PVLib / ERA5-like variants).

Feature schema/stats CSVs:

current_feature_table.csv, next_feature_table*.csv, target_mean_std.csv.

Model (model/)

model_unet.py: HeteroGNN encoder-pool-processor-upsample-decoder graph U-Net.

SG_module.py: SGFormer-style global attention block.

layer/:

gat_edge_before_ffnG_fix.py

gatv2convNG.py

gat_edge_ffn_upG_fix.py

processor__blockG.py

Uses hetero graph template file model/hetero_graph_normalized.pt (loaded from dataset).

Static stats/data

npz/*.csv: variable mean/std/min/max used for normalization and inverse scaling.

Data flow (high-level)
data_operational.py downloads ERA5 monthly GRIB files.

Processing scripts convert to time-indexed NPZ slices under npz/operational/{prs,sfc,time}.

WeatherGraphNPZDataset loads current/previous/future NPZ, normalizes features, attaches graph constants + step feature.

HeteroGNN predicts grid increments/residual-like outputs.

validate_autoregressive converts outputs to next-step state, patches boundaries with truth, rolls forward, saves per-step NPZ.
