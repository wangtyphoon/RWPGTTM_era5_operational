import os

from data_collect.check_nan_prs import check_nan_in_npz_files
from data_collect.download_prs_operational import download_prs
from data_collect.download_sfc_average_operational import download_sfc_average
from data_collect.download_sfc_regular_operational import download_sfc_regular
from data_collect.process_average_operational import process_average
from data_collect.process_prs_operational import process_prs
from data_collect.process_regular_operational import process_regular
from data_collect.process_time_operational import process_time
from data_collect.process_all_nan_filling import process_all_files as fill_nan_all
from data_collect.resize_swh import process_all_files as resize_swh_all

START_DATE = "2025-07-03"
END_DATE = "2025-07-11"


def main():
    # download_prs(START_DATE, END_DATE)
    # download_sfc_average(START_DATE, END_DATE)
    # download_sfc_regular(START_DATE, END_DATE)

    # process_prs(START_DATE, END_DATE)
    # process_average(START_DATE, END_DATE)
    # process_regular(START_DATE, END_DATE)
    # process_time(START_DATE, END_DATE)

    base_npz_dir = os.path.join(os.path.dirname(__file__), "npz", "operational")
    check_targets = [
        os.path.join(base_npz_dir, "sfc", "regular"),
        os.path.join(base_npz_dir, "sfc", "average"),
        os.path.join(base_npz_dir, "time"),
        os.path.join(base_npz_dir, "prs"),
    ]

    for npz_dir in check_targets:
        if not os.path.isdir(npz_dir):
            print(f"Skip NaN check, missing directory: {npz_dir}")
            continue
        print(npz_dir)
        check_nan_in_npz_files(npz_dir)

    # swh_dir = os.path.join(base_npz_dir, "sfc", "regular")
    # if os.path.isdir(swh_dir):
    #     resize_swh_all(swh_dir, backup=False)
    # else:
    #     print(f"Skip swh resize, missing directory: {swh_dir}")

    # if os.path.isdir(swh_dir):
    #     fill_nan_all(swh_dir)
    # else:
    #     print(f"Skip NaN filling, missing directory: {swh_dir}")


if __name__ == "__main__":
    main()
