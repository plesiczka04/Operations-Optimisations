#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Departure penalty sensitivity post-processing.

Reads all solution CSVs produced by main.py for different departure-penalty
multipliers (files named like 'solution_dep_0p25.csv', 'solution_dep_1p0.csv', ...),
computes aggregated metrics, writes a summary CSV, and plots results.

Assumptions:
- Each solution file is produced by model.solve_and_report from model.py
  and therefore has the columns:
  ["Aircraft", "Accepted", "Width", "Length", "ETA", "Roll_In", "X", "Y",
   "ServT", "ETD", "Roll_Out", "D_Arr", "D_Dep", "Penalty_Reject",
   "Penalty_ArrivalDelay", "Penalty_DepartureDelay", "Hangar_Width",
   "Hangar_Length", "StartDate"].
- The departure-penalty factor is encoded in the filename as '_dep_<tag>.csv'
  where <tag> is the factor with '.' replaced by 'p', e.g. 0.25 -> '0p25'.

Adjust the constants or patterns below if your naming scheme is different.
"""

import csv
import glob
import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt

# Must match epsilon_p used in the model / main.py
EPSILON_P = 0.001

# Pattern to find solution files
# Here we point directly to the folder where main.py writes solution_dep_*.csv
DEFAULT_PATTERN = r"C:\Users\plesiczka\Desktop\Operations3\Operations-Optimisations\Sensitivity\Przemek\departure_cost\solution_dep_*.csv"
# If you prefer relative paths instead, you could do e.g.:
# DEFAULT_PATTERN = os.path.join(os.path.dirname(__file__), "solution_dep_*.csv")


def parse_factor_from_filename(path: str) -> float:
    """
    Extract departure-penalty factor from a filename like
    'solution_dep_0p25.csv' or 'T3_dep_1p0.csv'.

    It looks for the substring '_dep_' in the stem, takes everything
    after it up to the extension, replaces 'p' by '.', and converts to float.
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_dep_" not in stem:
        raise ValueError(
            f"Cannot extract factor from filename {base!r}: no '_dep_' substring."
        )
    tag = stem.split("_dep_", 1)[1]
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
        "obj_rej": obj_rej,
        "obj_arr": obj_arr,
        "obj_dep": obj_dep,  # <-- departure cost
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
    out_path: str = r"C:\Users\plesiczka\Desktop\Operations3\Operations-Optimisations\Sensitivity\Przemek\departure_cost\sensitivity_dep_summary.csv",
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
        "obj_rej",
        "obj_arr",
        "obj_dep",
        "obj_eps",
        "objective",
    ]

    # Make sure folder exists (optional, but nice)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote summary CSV to {os.path.abspath(out_path)}")


def plot_sensitivity(
    rows: List[Dict[str, Any]],
    prefix: str = r"C:\Users\plesiczka\Desktop\Operations3\Operations-Optimisations\Sensitivity\Przemek\departure_cost\sensitivity_dep",
) -> None:
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

    BIG_FONT = 16
    BIGGER_FONT = 18
    BIG_TICKS = 14
    BIGGER_TICKS = 16

    # Ensure folder exists
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    # #rejected vs factor
    plt.figure()
    plt.plot(factors, n_rej, marker="o")
    plt.xlabel("Departure penalty factor", fontsize = BIGGER_FONT)
    plt.ylabel("Number of rejected requests", fontsize = BIGGER_FONT)
    plt.xticks(fontsize = BIGGER_TICKS)
    plt.yticks(fontsize = BIGGER_TICKS)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out1 = f"{prefix}_num_rejected.png"
    plt.savefig(out1, dpi=200)
    print(f"Saved {out1}")

    # Acceptance rate vs factor
    plt.figure()
    plt.plot(factors, acc_rate, marker="o")
    plt.xlabel("Departure penalty factor", fontsize = BIGGER_FONT)
    plt.ylabel("Acceptance rate (requests)", fontsize = BIGGER_FONT)
    plt.xticks(fontsize = BIGGER_TICKS)
    plt.yticks(fontsize = BIGGER_TICKS)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out2 = f"{prefix}_acceptance_rate.png"
    plt.savefig(out2, dpi=200)
    print(f"Saved {out2}")

    # Objective decomposition vs factor
    plt.figure()
    plt.plot(factors, obj_rej, marker="o", label="Rejection cost")
    plt.plot(factors, obj_arr, marker="o", label="Arrival delay cost")
    plt.plot(factors, obj_dep, marker="o", label="Departure delay cost")
    plt.plot(factors, obj_eps, marker="o", label="Positioning (ε_p term)")
    plt.plot(factors, obj_tot, marker="o", linestyle="--", label="Total objective")
    plt.xlabel("Departure penalty factor", fontsize = BIG_FONT)
    plt.ylabel("Cost contribution", fontsize = BIG_FONT)
    plt.xticks(fontsize = BIG_TICKS)
    plt.yticks(fontsize = BIG_TICKS)
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
    write_summary_csv(rows)

    # Create a few basic plots for the report
    plot_sensitivity(rows)

if __name__ == "__main__":
    main()
