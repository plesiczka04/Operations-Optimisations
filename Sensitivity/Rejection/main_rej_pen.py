#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hangar Scheduling Optimization Model
This module solves the hangar scheduling optimization model using Gurobi.


Authors:
Ariele Malugani
Mauro D'Errico
Przemek Lesiczka

Date: November 2025
"""

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


############ IMPORTS #############

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from csvimport import read_indexed_csv, build_sets, build_parameters
from model import build_model, solve_and_report
from datetime import datetime


############# PARAMETERS #############

HW = 150           # Hangar Width [meters]
HL = 100           # Hangar Length [meters]
Buffer = 5.0       # Buffer space between aircraft [meters]
epsilon_t = 0.1    # Minimum interval between aircraft movements [hours] (6 minutes)
epsilon_p = 0.001  # Penalty coefficient to encourage closeness to origin 


############# MAIN SCRIPT #############

def main():
    import os

    # Fixed input/output paths
    t1_path = "Sensitivity/Sensitivity_Scenario/T1.csv"
    t2_path = "Sensitivity/Sensitivity_Scenario/T2.csv"
    t3_path = "Sensitivity/Sensitivity_Scenario/T3.csv"
    out_path = "Sensitivity/Rejection/solution.csv"
    out_vars = "Sensitivity/Rejection/solution_vars.csv"
    
    # Solver options
    time_limit = None  # Time limit in seconds (optional)
    mip_gap = None     # MIP gap (e.g., 0.01 for 1%) (optional)
    threads = None     # Number of threads to use (optional)

    # Sensitivity factors for the rejection penalty
    # Each factor multiplies all P_Rej values from the input data.
    rej_factors = [0, 0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    # Read input CSVs (first column is the index)
    t1_keys, t1_map = read_indexed_csv(t1_path)
    t2_keys, t2_map = read_indexed_csv(t2_path)
    t3_keys, t3_map = read_indexed_csv(t3_path)

    # Build sets and parameters (baseline instance)
    a, c, f, M_ID = build_sets(t2_keys, t3_keys, t2_map, t3_map)
    if not a:
        raise ValueError("No aircraft found. Check T2/T3 CSVs and make sure the first column is the ID.")
    params = build_parameters(t1_map, t2_map, t3_map, a, c, f, M_ID)

    # Store a copy of the baseline rejection penalties
    base_P_Rej = params["P_Rej"].copy()

    # Produce a consistent start-date string (minutes precision)
    # Set the default start date as 1st Jan 2026 00:00
    start_date = "2026-01-01 00:00"

    # Loop over all rejection-penalty factors
    for factor in rej_factors:
        print(f"\n=== Solving model with rejection penalty factor {factor:.3g} ===")

        # Scale rejection penalties for this run
        params["P_Rej"] = {key: factor * val for key, val in base_P_Rej.items()}

        # Build model
        m, vars, bigm = build_model(params, a, c, f, HW, HL, Buffer, epsilon_t, epsilon_p)

        # Gurobi solver parameters
        m.Params.LogToConsole = 1  # Enable solver output
        if time_limit is not None:
            m.Params.TimeLimit = time_limit
        if mip_gap is not None:
            m.Params.MIPGap = mip_gap
        if threads is not None:
            m.Params.Threads = threads

        # Build factor-specific output names (to avoid overwriting files)
        tag = f"rej_{factor}".replace(".", "p")
        base_out, ext_out = os.path.splitext(out_path)
        base_vars, ext_vars = os.path.splitext(out_vars)
        out_path_factor = f"{base_out}_{tag}{ext_out}"
        out_vars_factor = f"{base_vars}_{tag}{ext_vars}"

        # Solve and write report for this factor
        solve_and_report(m, vars, a, c, f, HW, HL, params, out_path_factor, start_date, out_vars_factor)


if __name__ == "__main__":
    main()
