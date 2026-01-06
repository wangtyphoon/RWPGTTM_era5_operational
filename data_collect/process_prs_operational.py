import os
from datetime import date, datetime

import numpy as np
import xarray as xr

current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_months(start_date, end_date):
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def parse_time(value):
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f000",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported time format: {text}")


def build_bounds(start_date, end_date):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time()).replace(microsecond=0)
    return start_dt, end_dt


def process_prs(start_date, end_date, input_dir=None, output_base=None):
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    start_dt, end_dt = build_bounds(start_date, end_date)

    if input_dir is None:
        input_dir = os.path.join(current_dir, "prs")
    if output_base is None:
        output_base = os.path.join(current_dir, "..", "npz", "operational", "prs")
    os.makedirs(output_base, exist_ok=True)

    for year, month in iter_months(start_date, end_date):
        file_path = os.path.join(input_dir, f"{year:04d}-{month:02d}.grib")
        if not os.path.exists(file_path):
            print(f"{file_path} not found, skipping.")
            continue

        print(f"Processing file: {file_path}")
        ds = xr.open_dataset(file_path, engine="cfgrib")

        for valid_time in ds.time.values:
            dt = parse_time(valid_time)
            if not (start_dt <= dt <= end_dt):
                continue

            tstr = dt.strftime("%Y%m%d%H")
            npz_path = os.path.join(output_base, f"{tstr}.npz")
            if os.path.exists(npz_path):
                print(f"  -> {npz_path} exists, skip.")
                continue

            var_dict = {}
            for level in ds.isobaricInhPa.values:
                sel = ds.sel(time=valid_time, isobaricInhPa=level)
                for var in sel.data_vars:
                    key = f"{int(level)}{var}"
                    arr = sel[var].values.astype(np.float32)
                    var_dict[key] = np.squeeze(arr)

            np.savez(npz_path, **var_dict)
            print(f"  -> saved {len(var_dict)} prs-vars to {npz_path}")

        ds.close()


if __name__ == "__main__":
    process_prs("2024-01-01", "2024-01-01")
