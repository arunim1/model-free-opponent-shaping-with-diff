import json
import math


def compare_values(val1, val2, tolerance=1e-6):
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return math.isclose(val1, val2, rel_tol=tolerance)
    elif isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        return all(compare_values(v1, v2, tolerance) for v1, v2 in zip(val1, val2))
    elif isinstance(val1, str) and isinstance(val2, str):
        return True
    else:
        return val1 == val2


def compare_json_files(file1, file2, tolerance=1e-3):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

        if isinstance(data1, dict) and isinstance(data2, dict):
            if len(data1) != len(data2):
                return False
            for key in data1:
                if key not in data2:
                    return False
                if not compare_values(data1[key], data2[key], tolerance):
                    return False
        else:
            return compare_values(data1, data2, tolerance)

    return True


# Example usage
file1 = "non_mfos_baseline_pd_lrs/out.json"
file2 = "non_mfos_baseline_pd_lrs_multi/out.json"
tolerance = 1e-6

if compare_json_files(file1, file2, tolerance):
    print("The JSON files are basically the same.")
else:
    print("The JSON files are different.")
