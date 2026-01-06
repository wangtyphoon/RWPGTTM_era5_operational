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


def download_prs(start_date, end_date, out_dir=None, overwrite=False):
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    month_days = build_month_days(start_date, end_date)

    if out_dir is None:
        out_dir = os.path.join(current_dir, "prs")
    os.makedirs(out_dir, exist_ok=True)

    client = cdsapi.Client()

    for (year, month) in sorted(month_days):
        days = sorted(month_days[(year, month)])
        file_path = os.path.join(out_dir, f"{year:04d}-{month:02d}.grib")

        if os.path.exists(file_path) and not overwrite:
            print(f"{file_path} already exists, set overwrite=True to replace.")
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "fraction_of_cloud_cover",
                "geopotential",
                "potential_vorticity",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
            ],
            "year": [year],
            "month": [month],
            "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "pressure_level": [
                "50",
                "100",
                "150",
                "200",
                "250",
                "300",
                "400",
                "500",
                "600",
                "700",
                "850",
                "925",
                "1000",
            ],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [47.75, 102.25, 0, 150],
        }

        client.retrieve("reanalysis-era5-pressure-levels", request).download(file_path)
        print(f"{file_path} downloaded successfully.")


if __name__ == "__main__":
    download_prs("2024-01-01", "2024-01-01")
