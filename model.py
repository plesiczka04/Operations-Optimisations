#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HangarScheduling — Gurobi.

This module defines functions to build and solve the hangar scheduling optimization model using Gurobi.

The function build_model constructs the optimization model.
The function solve_and_report handles solving the model and reporting the results.

Authors:
Ariele Malugani
Mauro D'Errico
Przemek Lesiczka

Date: November 2025
"""

############ IMPORTS #############

from typing import Dict, List
import gurobipy as gp
from gurobipy import GRB
import os
import csv


############# MODEL BUILDING ###############

def build_model(params: Dict, a: List[str], c: List[str], f: List[str],
                HW: float, HL: float, Buffer: float, epsilon_t: float, epsilon_p: float):
    """
    Create a gurobipy model. 
    Inputs are parameter dict and sets, as well as hangar dimensions and other constants.
    Returns the model, variables dict, and big-M values.
    """

    ### UNPACK AND INITIALIZE PARAMETERS
    W, L = params["W"], params["L"]
    ETA, ETD = params["ETA"], params["ETD"]
    ServT = params["ServT"]
    P_Rej, P_Arr, P_Dep = params["P_Rej"], params["P_Arr"], params["P_Dep"]
    X_init, Y_init = params["X_init"], params["Y_init"]

    max_eta = max(ETA.values()) if ETA else 0.0
    M_T = max_eta + sum(ServT[i] for i in a)
    M_X = HW
    M_Y = HL

    # This line produces a dictionary mapping each aircraft id to its position 
    # in the list a (1-based). Useful for constraints that require ordering.
    ord_idx = {ai: idx+1 for idx, ai in enumerate(a)} 
    

    ### MODEL INITIALIZATION
    m = gp.Model("HangarScheduling")


    ### VARIABLES
    # Intuition: 
    # X is defined as a family of variables, indexed by the iterable a (set of all aircraft).
    # Each variable is given the base name "X" and has a lower bound of 0.0.
    # Similarly for other variables.

    # Positional variables (real, non-negative)
    X = m.addVars(a, name="X", lb=0.0)
    Y = m.addVars(a, name="Y", lb=0.0)
    # Entry and exit variables (real, non-negative)
    Roll_in = m.addVars(a, name="Roll_in", lb=0.0)
    Roll_out = m.addVars(a, name="Roll_out", lb=0.0)
    # Delay variables (real, non-negative)
    D_Arr = m.addVars(f, name="D_Arr", lb=0.0)
    D_Dep = m.addVars(a, name="D_Dep", lb=0.0)
    # Acceptance variable (binary)
    Accept = m.addVars(a, vtype=GRB.BINARY, name="Accept") # 1 if aircraft is accepted, 0 otherwise
    # Relative position variables (binary)
    Right = m.addVars(a, a, vtype=GRB.BINARY, name="Right") #Aircraft ai is completely right of aircraft bi
    Above = m.addVars(a, a, vtype=GRB.BINARY, name="Above") #Aircraft ai is completely above aircraft bi
    # Relative temporal variables (binary)
    OutIn = m.addVars(a, a, vtype=GRB.BINARY, name="OutIn") #Aircraft ai rolls out before aircraft bi rolls in
    InIn = m.addVars(a, a, vtype=GRB.BINARY, name="InIn") #Aircraft ai rolls in before aircraft bi rolls in
    OutOut = m.addVars(a, a, vtype=GRB.BINARY, name="OutOut") #Aircraft ai rolls out before aircraft bi rolls out
    InOut = m.addVars(a, a, vtype=GRB.BINARY, name="InOut") #Aircraft ai rolls in before aircraft bi rolls out


    ### OBJECTIVE FUNCTION
    # In order, it features:
    # - Rejection Cost
    # - Arrival Delay Cost
    # - Departure Delay Cost
    # - Positional Penalty
    obj = gp.quicksum(P_Rej[fi] * (1 - Accept[fi]) for fi in f) \
        + gp.quicksum(P_Arr[fi] * D_Arr[fi] for fi in f) \
        + gp.quicksum(P_Dep[ai] * D_Dep[ai] for ai in a) \
        + epsilon_p * gp.quicksum(X[fi] + Y[fi] for fi in f)
    m.setObjective(obj, GRB.MINIMIZE)


    ### CONSTRAINTS
    # Please note that the naming of constraints follows the numbering of equations in the paper.
    
    # Acceptance and Scheduling Constraints 2.2-2.6
    for fi in f:
        m.addConstr(X[fi] + Y[fi] + Roll_in[fi] + Roll_out[fi] + D_Arr[fi] + D_Dep[fi]
                    <= (M_X + M_Y + 4*M_T) * Accept[fi], name=f"e02[{fi}]")
        m.addConstr(Roll_in[fi] >= ETA[fi] * Accept[fi], name=f"e03[{fi}]")
    for ai in a:
        m.addConstr(Roll_out[ai] - Roll_in[ai] >= ServT[ai] * Accept[ai], name=f"e04[{ai}]")
    for fi in f:
        m.addConstr(D_Arr[fi] >= Roll_in[fi] - ETA[fi], name=f"e05[{fi}]")
    for ai in a:
        m.addConstr(D_Dep[ai] >= Roll_out[ai] - ETD[ai], name=f"e06[{ai}]")
    
    # Hangar Physical and Non-overlapping Constraints 2.7-2.12
    for fi in f:
        m.addConstr(X[fi] >= Buffer * Accept[fi], name=f"e07[{fi}]")
        m.addConstr(X[fi] + W[fi] <= HW - Buffer + M_X * (1 - Accept[fi]), name=f"e08[{fi}]")
        m.addConstr(Y[fi] >= Buffer * Accept[fi], name=f"e09[{fi}]")
        m.addConstr(Y[fi] + L[fi] <= HL - Buffer + M_Y * (1 - Accept[fi]), name=f"e10[{fi}]")
    for ai in a:
        for bi in a:
            if ai != bi:
                m.addConstr(X[bi] + W[bi] + Buffer <= X[ai] + M_X * (1 - Right[ai, bi]), name=f"e11[{ai},{bi}]")
                m.addConstr(Y[bi] + L[bi] + Buffer <= Y[ai] + M_Y * (1 - Above[ai, bi]), name=f"e12[{ai},{bi}]")
    
    # Aircraft Physical and Temporal Sequencing Constraints 2.13-2.20
    for ai in a:
        for bi in a:
            if ord_idx[ai] < ord_idx[bi]:
                m.addConstr(Right[bi, ai] + Right[ai, bi] + Above[bi, ai] + Above[ai, bi]
                            + OutIn[ai, bi] + OutIn[bi, ai] >= Accept[ai] + Accept[bi] - 1, name=f"e13[{ai},{bi}]")
    for ai in a:
        for bi in a:
            if ai != bi:
                m.addConstr(Roll_out[ai] + epsilon_t <= Roll_in[bi] + M_T * (1 - OutIn[ai, bi]), name=f"e14[{ai},{bi}]")
    for fi in f:
        for gi in f:
            if fi != gi:
                m.addConstr(Roll_in[gi] >= Roll_in[fi] + epsilon_t - M_T * (1 - InIn[fi, gi]) - M_T * (2 - Accept[fi] - Accept[gi]), name=f"e15[{fi},{gi}]")
                m.addConstr(Roll_in[fi] >= Roll_in[gi] + epsilon_t - M_T * InIn[fi, gi] - M_T * (2 - Accept[fi] - Accept[gi]), name=f"e16[{fi},{gi}]")
    for ai in a:
        for bi in a:
            if ord_idx[ai] < ord_idx[bi]:
                m.addConstr(Roll_out[bi] >= Roll_out[ai] + epsilon_t - M_T * (1 - OutOut[ai, bi]) - M_T * (2 - Accept[ai] - Accept[bi]), name=f"e17[{ai},{bi}]")
                m.addConstr(Roll_out[ai] >= Roll_out[bi] + epsilon_t - M_T * OutOut[ai, bi] - M_T * (2 - Accept[ai] - Accept[bi]), name=f"e18[{ai},{bi}]")
    fset = set(f)
    for ai in a:
        for bi in a:
            if ai != bi:
                if (ai in fset) or (bi in fset):
                    m.addConstr(Roll_out[bi] >= Roll_in[ai] + epsilon_t - M_T * (1 - InOut[ai, bi]) - M_T * (2 - Accept[ai] - Accept[bi]), name=f"e19[{ai},{bi}]")
                    m.addConstr(Roll_in[ai] >= Roll_out[bi] + epsilon_t - M_T * InOut[ai, bi] - M_T * (2 - Accept[ai] - Accept[bi]), name=f"e20[{ai},{bi}]")
    
    # Blocking Constraints 2.21-2.22
    for ai in a:
        for bi in a:
            if ai != bi:
                m.addConstr(Roll_out[ai] >= Roll_out[bi] + epsilon_t
                            - M_T * ((1 - Above[bi, ai]) + Right[ai, bi] + Right[bi, ai] + (1 - InIn[ai, bi])),
                            name=f"e21[{ai},{bi}]")
                if (ai in fset) or (bi in fset):
                    m.addConstr(Roll_in[ai] >= Roll_out[bi] + epsilon_t
                                - M_T * ((1 - Above[bi, ai]) + Right[ai, bi] + Right[bi, ai] + InIn[ai, bi]),
                                name=f"e22[{ai},{bi}]")


    ### INITIAL CONDITIONS (2.23-2.29)
    for ci in c:
        Accept[ci].lb = 1.0; Accept[ci].ub = 1.0
        X[ci].lb = X_init[ci]; X[ci].ub = X_init[ci] # Fix X to initial value by setting lb=ub
        Y[ci].lb = Y_init[ci]; Y[ci].ub = Y_init[ci] # Fix Y to initial value by setting lb=ub 
        Roll_in[ci].lb = 0.0;  Roll_in[ci].ub = 0.0 # Fix Roll_in to 0 by setting lb=ub
    for ci in c:
        for fi in f:
            InIn[ci, fi].lb = 1.0; InIn[ci, fi].ub = 1.0 # Fix InIn to 1 by setting lb=ub
            InIn[fi, ci].lb = 0.0; InIn[fi, ci].ub = 0.0 # Fix InIn to 0 by setting lb=ub
        for dj in c:
            if ci != dj:
                InIn[ci, dj].lb = 1.0; InIn[ci, dj].ub = 1.0 # Fix InIn to 1 by setting lb=ub
    
    # Note that variable type and bound are already set at variable creation.


    ### RETURN MODEL AND VARIABLES
    return m, dict(X=X, Y=Y, Roll_in=Roll_in, Roll_out=Roll_out, D_Arr=D_Arr, D_Dep=D_Dep,
                   Accept=Accept, Right=Right, Above=Above, OutIn=OutIn, InIn=InIn, OutOut=OutOut, InOut=InOut), \
           dict(M_T=M_T, M_X=M_X, M_Y=M_Y)


############# SOLVING AND REPORTING ###############

def check_model_status(m):
    status = m.status
    if status != GRB.Status.OPTIMAL:
        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
        elif status == GRB.Status.INFEASIBLE:
            print('The model is infeasible; computing IIS') # IIS is Irreducible Inconsistent Subsystem and helps diagnose infeasibility
            m.computeIIS() # This outputs the IIS, i.e. the constraints that cannot be satisfied together
            print('The following constraint(s) cannot be satisfied:')
            for c in m.getConstrs():
                if c.IISConstr:
                    print(c.constrName)
        elif status != GRB.Status.INF_OR_UNBD:
            print('Optimization was stopped with status',status)
        exit(0)


def write_all_vars_csv(m, path, zero_tol=1e-9, only_nonzero=False):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["VarName", "Value"])
        w.writeheader()
        for v in m.getVars():
            val = float(v.X)
            if only_nonzero and abs(val) <= zero_tol:
                continue
            w.writerow({"VarName": v.VarName, "Value": val})


def solve_and_report(m: gp.Model, vars, a: List[str], c: List[str], f: List[str],
                     HW: float, HL: float,
                     params: Dict, out_csv: str, start_date: str | None = None, 
                     out_vars_csv: str | None = None):
    """
    Solve the model and write a CSV report using csv.DictWriter.
    Inputs are the model, variables dict, sets, hangar dimensions, parameters dict.
    Outputs are written to out_csv.
    """

    m.write("hangar_scheduling_model.lp")  # Write model to file for inspection
    m.optimize()
    check_model_status(m)

    # Store results
    X, Y = vars["X"], vars["Y"]
    Roll_in, Roll_out = vars["Roll_in"], vars["Roll_out"]
    D_Arr, D_Dep = vars["D_Arr"], vars["D_Dep"]
    Accept = vars["Accept"]
    W, L = params["W"], params["L"]
    ETA = params["ETA"]
    ETD = params["ETD"]
    ServT = params["ServT"]
    P_Rej, P_Arr, P_Dep = params["P_Rej"], params["P_Arr"], params["P_Dep"]

    stamp = start_date

    # Define report items
    fieldnames = ["Aircraft", "Accepted", "Width", "Length", "ETA", "Roll_In", "X", "Y",
                  "ServT", "ETD", "Roll_Out", "D_Arr", "D_Dep", "Penalty_Reject",
                  "Penalty_ArrivalDelay", "Penalty_DepartureDelay", "Hangar_Width",
                  "Hangar_Length", "StartDate"]

    # Write report to CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for ai in a:
            is_future = ai in f
            row = {
                "Aircraft": ai,
                "Accepted": int(round(Accept[ai].X)),
                "Width": W[ai],
                "Length": L[ai],
                "ETA": ETA.get(ai, 0.0),
                "Roll_In": Roll_in[ai].X,
                "X": X[ai].X,
                "Y": Y[ai].X,
                "ServT": ServT[ai],
                "ETD": ETD[ai],
                "Roll_Out": Roll_out[ai].X,
                "D_Arr": D_Arr[ai].X if is_future else 0.0,
                "D_Dep": D_Dep[ai].X,
                "Penalty_Reject": P_Rej.get(ai, 0.0),
                "Penalty_ArrivalDelay": P_Arr.get(ai, 0.0),
                "Penalty_DepartureDelay": P_Dep[ai],
                "Hangar_Width": HW,
                "Hangar_Length": HL,
                "StartDate": stamp,
            }
            writer.writerow(row)

    # Write variables CSV
    if out_vars_csv is None:
        out_vars_csv = out_csv.replace(".csv", "_vars.csv")

    write_all_vars_csv(m, out_vars_csv)

    if m.SolCount > 0:
        print("Objective:", m.ObjVal)
    else:
        print("No solution found.")
    print("Saved report to:", os.path.abspath(out_csv))
    print("Saved variables to:", os.path.abspath(out_vars_csv))
