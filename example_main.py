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

############ IMPORTS #############

from csvimport import read_indexed_csv, build_sets, build_parameters
from model import build_model, solve_and_report


############# PARAMETERS #############

HW = 60           # Hangar Width [meters]
HL = 60         # Hangar Length [meters]
Buffer = 5.0       # Buffer space between aircraft [meters]
epsilon_t = 0.1    # Minimum interval between aircraft movements [hours] (6 minutes)
epsilon_p = 0.001  # Penalty coefficient to encourage closeness to origin 


############# MAIN SCRIPT #############

def main():
    import os

    # Fixed input/output paths
    t1_path = "Code/HangarModelResearch-main/data/T1.csv"
    t2_path = "Code/HangarModelResearch-main/data/T2.csv"
    t3_path = "Code/HangarModelResearch-main/data/random/T3-07-01.csv" 
    out_path = "solution.csv"

    # Solver options (keep None to leave unset)
    time_limit = None
    mip_gap = None
    threads = None
    start_date = None

    # Read input CSVs (first column is the index)
    t1_keys, t1_map = read_indexed_csv(t1_path)
    t2_keys, t2_map = read_indexed_csv(t2_path)
    t3_keys, t3_map = read_indexed_csv(t3_path)

    # Build sets and parameters
    a, c, f, M_ID = build_sets(t2_keys, t3_keys, t2_map, t3_map)
    if not a:
        raise ValueError("No aircraft found. Check T2/T3 CSVs and make sure the first column is the ID.")
    params = build_parameters(t1_map, t2_map, t3_map, a, c, f, M_ID)

    # Build model
    m, vars, bigm = build_model(params, a, c, f, HW, HL, Buffer, epsilon_t, epsilon_p)

    # Gurobi solver parameters
    m.Params.LogToConsole = 1 # Enable solver output
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if threads is not None:
        m.Params.Threads = threads

    # Solve and write report
    solve_and_report(m, vars, a, c, f, HW, HL, params, out_path, start_date)


if __name__ == "__main__":
    main()
