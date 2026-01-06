import os
from datetime import datetime

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_bounds(start_date, end_date):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time()).replace(microsecond=0)
    return start_dt, end_dt


def process_time(
    start_date,
    end_date,
    input_dir=None,
    output_base=None,
    lon_file=None,
    lat_file=None,
):
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    start_dt, end_dt = build_bounds(start_date, end_date)

    if input_dir is None:
        input_dir = os.path.join(current_dir, "..", "npz", "operational", "sfc", "regular")
    if output_base is None:
        output_base = os.path.join(current_dir, "..", "npz", "operational", "time")
    if lon_file is None:
        lon_file = os.path.join(current_dir, "lon.npy")
    if lat_file is None:
        lat_file = os.path.join(current_dir, "lat.npy")

    os.makedirs(output_base, exist_ok=True)

    lon = np.load(lon_file)
    lat = np.load(lat_file)

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npz"))

    for npz_file in files:
        sample_name = npz_file[:-4]
        if len(sample_name) < 10 or not sample_name[:10].isdigit():
            print(f"Skipping unexpected file name: {npz_file}")
            continue

        dt = datetime.strptime(sample_name[:10], "%Y%m%d%H")
        if not (start_dt <= dt <= end_dt):
            continue

        print(f"Processing: {npz_file}")

        year = int(sample_name[:4])
        month = int(sample_name[4:6])
        day = int(sample_name[6:8])
        hour = int(sample_name[8:10])

        hour_ratio_base = hour / 24.0
        day_of_year = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days + 1

        output_path = os.path.join(output_base, npz_file)
        if os.path.exists(output_path):
            print(f"  -> {output_path} exists, skip.")
            continue

        hour_ratio = np.full_like(lon, hour_ratio_base, dtype=np.float32) + lon / (
            15.0 * 24
        )
        hour_ratio_sin = np.sin(hour_ratio * 2 * np.pi)
        hour_ratio_cos = np.cos(hour_ratio * 2 * np.pi)

        adjusted_hour_ratio = hour_ratio_base + lon / (15.0 * 24)
        year_ratio = np.full_like(
            lon, (day_of_year + adjusted_hour_ratio) / 365.2425, dtype=np.float32
        )
        year_ratio_sin = np.sin(year_ratio * 2 * np.pi)
        year_ratio_cos = np.cos(year_ratio * 2 * np.pi)

        time_features = {
            "hour_ratio": hour_ratio,
            "hour_ratio_sin": hour_ratio_sin,
            "hour_ratio_cos": hour_ratio_cos,
            "year_ratio": year_ratio,
            "year_ratio_sin": year_ratio_sin,
            "year_ratio_cos": year_ratio_cos,
        }

        np.savez(output_path, **time_features)
        print(f"  -> saved {len(time_features)} time features to {output_path}")


if __name__ == "__main__":
    process_time("2024-01-01", "2024-01-01")
