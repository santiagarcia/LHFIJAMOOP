"""Compare first-order predictions from one nominal run with AM Bench measurements.

usage: melt_pool_check.py RESULT_JSON CASES_CSV --exponent N

RESULT_JSON is the solver result for the nominal case, with exported derivatives.
CASES_CSV holds the measured quantities of every recipe. N is the depth exponent of the
heat source calibration. None of these ship with this public repository.
"""

import argparse
import csv
import json
from math import sqrt

from lhfijamoop.recipe import Recipe, predict, source_perturbation

QUANTITIES = [
    ("depth_um", "melt_pool_mid_1298C", "d_depth_um", "depth_um", "depth"),
    ("width_um", "melt_pool_mid_1298C", "d_width_um", "width_um", "width"),
    ("TTAM_s", "probe", "probe_sens/TTAM_s", "TTAM_s", "time above melting"),
    ("TLCR_C_s", "probe", "probe_sens/TLCR_C_s", "TLCR_C_s", "liquid cooling rate"),
    ("TTCR_C_s", "probe", "probe_sens/TTCR_C_s", "TTCR_C_s", "transition cooling rate"),
    ("TSCR_C_s", "probe", "probe_sens/TSCR_C_s", "TSCR_C_s", "solid cooling rate"),
]


def read_cases(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["case"]: row for row in csv.DictReader(handle)}


def recipe_of(row):
    return Recipe(float(row["power_W"]), float(row["speed_mm_s"]), float(row["spot_D4sigma_um"]))


def lookup(result, block, key):
    if "/" in key:
        first, second = key.split("/")
        return result[first][second]
    return result[block][key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json")
    parser.add_argument("cases_csv")
    parser.add_argument("--exponent", type=float, required=True)
    args = parser.parse_args()
    with open(args.result_json, encoding="utf-8") as handle:
        result = json.load(handle)
    cases = read_cases(args.cases_csv)
    nominal = recipe_of(cases["0"])
    print(f"{'quantity':26s}{'case':>6s}{'measured':>12s}{'predicted':>12s}{'ratio':>8s}")
    for measured_key, block, sens_key, value_key, label in QUANTITIES:
        sensitivity = lookup(result, block, sens_key)
        base_measured = float(cases["0"][measured_key])
        base_model = result[block][value_key]
        errors, flat = [], []
        for name, row in cases.items():
            if name == "0":
                continue
            change = source_perturbation(recipe_of(row), nominal, args.exponent)
            predicted = predict(0.0, sensitivity, change) / base_model
            measured = float(row[measured_key]) / base_measured - 1.0
            ratio = predicted / measured if measured else float("nan")
            print(f"{label:26s}{name:>6s}{100*measured:11.1f}%{100*predicted:11.1f}%{ratio:8.2f}")
            errors.append((predicted - measured) ** 2)
            flat.append(measured**2)
        rms = 100 * sqrt(sum(errors) / len(errors))
        rms_flat = 100 * sqrt(sum(flat) / len(flat))
        print(f"{label:26s}{'rms':>6s}{rms_flat:11.1f}%{rms:11.1f}%   (left: no change, right: first order)")


if __name__ == "__main__":
    main()
