#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rejection penalty sensitivity post-processing.

Reads all solution CSVs produced by main.py for different rejection-penalty
multipliers (files named like 'solution_rej_0p25.csv', 'solution_rej_1p0.csv', ...),
computes aggregated metrics, writes a summary CSV, and plots results.

Assumptions:
- Each solution file is produced by model.solve_and_report from model.py
  and therefore has the columns:
  ["Aircraft", "Accepted", "Width", "Length", "ETA", "Roll_In", "X", "Y",
   "ServT", "ETD", "Roll_Out", "D_Arr", "D_Dep", "Penalty_Reject",
   "Penalty_ArrivalDelay", "Penalty_DepartureDelay", "Hangar_Width",
   "Hangar_Length", "StartDate"].
- The rejection-penalty factor is encoded in the filename as '_rej_<tag>.csv'
  where <tag> is the factor with '.' replaced by 'p', e.g. 0.25 -> '0p25'.

Adjust the constants or patterns below if your naming scheme is different.
"""

import csv
import glob
import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


# Must match epsilon_p used in the model / main.py
EPSILON_P = 0.001

# Pattern to find solution files (same directory as this script by default)
DEFAULT_PATTERN = os.path.join(os.path.dirname(__file__), "solution_rej_*.csv")
# If you ran main.py into a 'results' folder, you can set e.g.:
# DEFAULT_PATTERN = "results/*_rej_*.csv"


def parse_factor_from_filename(path: str) -> float:
    """
    Extract rejection-penalty factor from a filename like
    'solution_rej_0p25.csv' or 'T3_rej_1p0.csv'.

    It looks for the substring '_rej_' in the stem, takes everything
    after it up to the extension, replaces 'p' by '.', and converts to float.
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_rej_" not in stem:
        raise ValueError(
            f"Cannot extract factor from filename {base!r}: no '_rej_' substring."
        )
    tag = stem.split("_rej_", 1)[1]
    if not tag:
        raise ValueError(
            f"Cannot extract factor from filename {base!r}: empty factor tag."
        )
    factor_str = tag.replace("p", ".")
    try:
        return float(factor_str)
    except ValueError as e:
        raise ValueError(
            f"Cannot parse factor from tag {tag!r} in filename {base!r}."
        ) from e


def summarize_solution_file(path: str) -> Dict[str, Any]:
    """
    Read one solution CSV (produced by solve_and_report) and compute:
    - #requests, #accepted, #rejected
    - total arrival/departure delays
    - objective decomposition (rejection, arrival, departure, epsilon_p term)
    - total objective (reconstructed)

    Returns a dict with these metrics.
    """
    factor = parse_factor_from_filename(path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in solution file {path!r}.")

    n_total = len(rows)
    n_current = 0
    n_requests = 0
    n_accepted_requests = 0
    n_rejected_requests = 0

    total_D_Arr = 0.0
    total_D_Dep = 0.0

    obj_rej = 0.0
    obj_arr = 0.0
    obj_dep = 0.0
    obj_eps = 0.0
    accepted_request_dep_delays: List[float] = []  # P95 departure delay over accepted requests only



    # "Future" (request) aircraft: those with positive rejection or arrival penalty.
    for row in rows:
        accepted = float(row["Accepted"])
        accepted = 1.0 if accepted >= 0.5 else 0.0  # robust 0/1

        p_rej = float(row["Penalty_Reject"])
        p_arr = float(row["Penalty_ArrivalDelay"])
        p_dep = float(row["Penalty_DepartureDelay"])
        d_arr = float(row["D_Arr"])
        d_dep = float(row["D_Dep"])

        x = float(row["X"])
        y = float(row["Y"])

        is_future = (p_rej > 0.0) or (p_arr > 0.0)
        if is_future and accepted > 0.5:
            accepted_request_dep_delays.append(d_dep)


        if is_future:
            n_requests += 1
            if accepted > 0.5:
                n_accepted_requests += 1
            else:
                n_rejected_requests += 1

            total_D_Arr += d_arr
            # Objective components for future aircraft
            obj_rej += p_rej * (1.0 - accepted)
            obj_arr += p_arr * d_arr
            obj_eps += EPSILON_P * (x + y)
        else:
            n_current += 1

        total_D_Dep += d_dep
        obj_dep += p_dep * d_dep

    acceptance_rate = (
        1.0 if n_requests == 0 else n_accepted_requests / n_requests
    )
    objective = obj_rej + obj_arr + obj_dep + obj_eps
    p95_D_Dep_req = float("nan")
    if accepted_request_dep_delays:
        p95_D_Dep_req = float(
            np.percentile(np.array(accepted_request_dep_delays, dtype=float), 95)
        )


    return {
        "factor": factor,
        "n_total": n_total,
        "n_current": n_current,
        "n_requests": n_requests,
        "n_accepted": n_accepted_requests,
        "n_rejected": n_rejected_requests,
        "acceptance_rate": acceptance_rate,
        "total_D_Arr": total_D_Arr,
        "total_D_Dep": total_D_Dep,
        "p95_D_Dep_req": p95_D_Dep_req,
        "obj_rej": obj_rej,
        "obj_arr": obj_arr,
        "obj_dep": obj_dep,
        "obj_eps": obj_eps,
        "objective": objective,
        "file": os.path.basename(path),
    }


def collect_sensitivity(pattern: str = DEFAULT_PATTERN) -> List[Dict[str, Any]]:
    """
    Scan for all solution files matching the given glob pattern,
    summarize each, and return a list of dicts sorted by factor.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files found matching pattern {pattern!r}. "
            "Make sure you've run main.py and that the naming pattern is correct."
        )

    rows = [summarize_solution_file(p) for p in paths]
    rows.sort(key=lambda r: r["factor"])
    return rows


def write_summary_csv(
    rows: List[Dict[str, Any]],
    out_path: str = "sensitivity_rej_summary.csv",
) -> None:
    """
    Write a single CSV summarizing all runs (one row per factor).
    """
    if not rows:
        raise ValueError("No rows to write in summary CSV.")

    fieldnames = [
        "factor",
        "file",
        "n_total",
        "n_current",
        "n_requests",
        "n_accepted",
        "n_rejected",
        "acceptance_rate",
        "total_D_Arr",
        "total_D_Dep",
        "p95_D_Dep_req",
        "obj_rej",
        "obj_arr",
        "obj_dep",
        "obj_eps",
        "objective",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote summary CSV to {os.path.abspath(out_path)}")


def plot_sensitivity(rows: List[Dict[str, Any]], prefix: str = "sensitivity_rej") -> None:
    """
    Create a few basic plots:
    - factor vs #rejected
    - factor vs acceptance rate
    - factor vs objective and its components
    and save them as PNG files.
    """
    factors = [r["factor"] for r in rows]
    n_rej = [r["n_rejected"] for r in rows]
    acc_rate = [r["acceptance_rate"] for r in rows]
    obj_rej = [r["obj_rej"] for r in rows]
    obj_arr = [r["obj_arr"] for r in rows]
    obj_dep = [r["obj_dep"] for r in rows]
    obj_eps = [r["obj_eps"] for r in rows]
    obj_tot = [r["objective"] for r in rows]
    p95_dep = [r.get("p95_D_Dep_req", float("nan")) for r in rows]



    # #rejected vs factor
    plt.figure()
    plt.plot(factors, n_rej, marker="o")
    plt.xlabel("Rejection penalty factor")
    plt.ylabel("Number of rejected requests")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out1 = f"{prefix}_num_rejected.png"
    plt.savefig(out1, dpi=200)
    print(f"Saved {out1}")

    # Acceptance rate vs factor
    plt.figure()
    plt.plot(factors, acc_rate, marker="o")
    plt.xlabel("Rejection penalty factor")
    plt.ylabel("Acceptance rate (requests)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out2 = f"{prefix}_acceptance_rate.png"
    plt.savefig(out2, dpi=200)
    print(f"Saved {out2}")

    # P95 departure delay vs factor (tail-delay metric)
    plt.figure()
    plt.plot(factors, p95_dep, marker="o")
    plt.xlabel("Rejection penalty factor")
    plt.ylabel("P95 departure delay (minutes)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_p95 = f"{prefix}_p95_dep_delay.png"
    plt.savefig(out_p95, dpi=200)
    print(f"Saved {out_p95}")


    # Objective decomposition vs factor
    plt.figure()
    plt.plot(factors, obj_rej, marker="o", label="Rejection cost")
    plt.plot(factors, obj_arr, marker="o", label="Arrival delay cost")
    plt.plot(factors, obj_dep, marker="o", label="Departure delay cost")
    plt.plot(factors, obj_eps, marker="o", label="Positioning (ε_p term)")
    plt.plot(factors, obj_tot, marker="o", linestyle="--", label="Total objective")
    plt.xlabel("Rejection penalty factor")
    plt.ylabel("Cost contribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out3 = f"{prefix}_objective_components.png"
    plt.savefig(out3, dpi=200)
    print(f"Saved {out3}")


def main():
    # Collect and summarize all runs
    rows = collect_sensitivity(DEFAULT_PATTERN)

    # Write table with one row per factor
    write_summary_csv(rows, out_path="Sensitivity/Rejection/sensitivity_rej_summary.csv")

    # Create a few basic plots for the report
    plot_sensitivity(rows, prefix="Sensitivity/Rejection/sensitivity_rej")

if __name__ == "__main__":
    main()
