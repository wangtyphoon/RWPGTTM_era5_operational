import csv
import os
import numpy as np


def compare_difference(tensor_a, tensor_b, atol=1e-6):
    """
    Compare two tensors and return the maximum absolute difference
    and the number of differing elements beyond a specified tolerance.

    Args:
        tensor_a (np.ndarray): First tensor to compare.
        tensor_b (np.ndarray): Second tensor to compare.
        atol (float): Absolute tolerance for comparison.

    Returns:
        max_diff (float): Maximum absolute difference between the tensors.
        num_diffs (int): Number of elements that differ beyond the tolerance.
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Tensors must have the same shape for comparison.")
    abs_diff = np.abs(tensor_a - tensor_b)
    max_diff = np.max(abs_diff)
    num_diffs = np.sum(abs_diff > atol)

    return max_diff, num_diffs


def load_channel_std(csv_path):
    std_values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "std" not in reader.fieldnames:
            raise ValueError(f"'std' column not found in {csv_path}")
        for row in reader:
            std_values.append(float(row["std"]))
    if not std_values:
        raise ValueError(f"No std values found in {csv_path}")
    return np.array(std_values, dtype=np.float32)


def apply_channel_std(arr, std, channel_dim):
    if arr.ndim <= channel_dim:
        return arr
    if arr.shape[channel_dim] != std.shape[0]:
        return arr
    shape = [1] * arr.ndim
    shape[channel_dim] = std.shape[0]
    return arr / std.reshape(shape)


def compare_npz(file_a, file_b, atol=1e-6, channel_dim=1, channel_std=None):
    data_a = np.load(file_a, allow_pickle=True)
    data_b = np.load(file_b, allow_pickle=True)

    keys_a = set(data_a.files)
    keys_b = set(data_b.files)
    shared_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    results = []
    for key in shared_keys:
        arr_a = data_a[key]
        arr_b = data_b[key]
        if channel_std is not None:
            arr_a = apply_channel_std(arr_a, channel_std, channel_dim)
            arr_b = apply_channel_std(arr_b, channel_std, channel_dim)
        abs_diff = np.abs(arr_a - arr_b)
        max_diff = float(np.max(abs_diff))
        num_diffs = int(np.sum(abs_diff > atol))
        max_index = tuple(int(i) for i in np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape))
        channel_indices = []
        if num_diffs > 0:
            if arr_a.ndim <= channel_dim:
                raise ValueError(
                    f"Key '{key}' has ndim={arr_a.ndim}, cannot use channel_dim={channel_dim}."
                )
            diff_mask = abs_diff > atol
            channel_indices = sorted(np.unique(np.where(diff_mask)[channel_dim]).tolist())
        results.append((key, max_diff, num_diffs, arr_a.shape, channel_indices, max_index))

    return results, only_a, only_b


def main():
    std_path = os.path.join("inference", "next_feature_table_weight.csv")
    channel_std = load_channel_std(std_path)
    file_a = os.path.join(
        "inference_outputs",
        "2025070500_2025070500test12",
        "['2025070500']_step01.npz",
    )
    file_b = os.path.join(
        "inference_outputs",
        "2025070500_2025070500test4",
        "['2025070500']_step01.npz",
    )
    atol = 1e-6
    channel_dim = 1

    results, only_a, only_b = compare_npz(
        file_a,
        file_b,
        atol=atol,
        channel_dim=channel_dim,
        channel_std=channel_std,
    )

    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    print(f"atol: {atol}")
    print(f"channel_dim: {channel_dim}")
    print(f"channel_std: {std_path} ({channel_std.shape[0]} channels)")

    if only_a:
        print(f"Keys only in file A ({len(only_a)}): {only_a}")
    if only_b:
        print(f"Keys only in file B ({len(only_b)}): {only_b}")

    if not results:
        print("No shared keys to compare.")
        return

    print("Shared keys comparison:")
    for key, max_diff, num_diffs, shape, channel_indices, max_index in results:
        print(
            f"- {key}: shape={shape}, max_diff={max_diff:.6e}, diffs={num_diffs}, "
            f"max_index={max_index}, channels={channel_indices}"
        )


if __name__ == "__main__":
    main()
