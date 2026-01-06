import os
from datetime import datetime, timedelta

import cdsapi

current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_month_days(start_date, end_date):
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    months = {}
    current = start_date
    while current <= end_date:
        key = (current.year, current.month)
        months.setdefault(key, []).append(current.day)
        current += timedelta(days=1)
    return months


def download_sfc_regular(start_date, end_date, out_dir=None, overwrite=False):
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    month_days = build_month_days(start_date, end_date)

    if out_dir is None:
        out_dir = os.path.join(current_dir, "sfc")
    os.makedirs(out_dir, exist_ok=True)

    client = cdsapi.Client()

    for (year, month) in sorted(month_days):
        days = sorted(month_days[(year, month)])
        file_path = os.path.join(out_dir, f"{year:04d}-{month:02d}_regular.grib")

        if os.path.exists(file_path) and not overwrite:
            print(f"{file_path} already exists, set overwrite=True to replace.")
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "sea_surface_temperature",
                "forecast_logarithm_of_surface_roughness_for_heat",
                "significant_height_of_combined_wind_waves_and_swell",
                "boundary_layer_height",
                "geopotential",
                "land_sea_mask",
            ],
            "year": [year],
            "month": [month],
            "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [47.75, 102.25, 0, 150],
        }

        client.retrieve("reanalysis-era5-single-levels", request).download(file_path)
        print(f"{file_path} downloaded successfully.")


if __name__ == "__main__":
    download_sfc_regular("2024-01-01", "2024-01-01")
