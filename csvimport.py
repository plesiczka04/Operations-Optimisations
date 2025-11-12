#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HangarScheduling — Gurobi.

This module imports data from CSV files and constructs the necessary sets and parameters.

Inputs:
Input CSVs must have the first column as the row key (aircraft id or model id).
-   T1.csv: model parameters indexed by model id. 
    It should have the columns m (model id), W (width), L (length).
-   T2.csv: current aircraft parameters indexed by aircraft id
    It should have the columns c (aircraft id), M_ID (model id), ETD, ServT, P_Dep, Init_X, Init_Y.
-   T3.csv: future aircraft parameters indexed by aircraft id
    It should have the columns f (aircraft id), M_ID (model id), ETA, ETD, ServT, P_Rej, P_Arr, P_Dep, Is_VIP.

Outputs:
Parameter dictionaries:
- W, L: dimensions from T1
- ETA, ETD, ServT, P_Rej, P_Arr, P_Dep, Is_VIP, X_init, Y_init: aircraft parameters from T2/T3
Sets:
- a: all aircraft (current + future)
- c: current aircraft
- f: future aircraft
Mapping:
- M_ID: mapping from aircraft id to model id

Authors:
Ariele Malugani
Mauro D'Errico
Przemek Lesiczka

Date: November 2025
"""

from typing import Dict, List, Tuple
import csv
import gurobipy as gp
from gurobipy import GRB


############# DATA IMPORT HELPERS #############

def read_indexed_csv(path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Read a CSV whose first column is the index. Assumes well-formed input.
    Returns a list of keys and a mapping from key to row value in the form Dict[str, Dict[str, str]].
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [c.strip() for c in header[1:]]  # skip first column (the key)
        keys: List[str] = []
        data: Dict[str, Dict[str, str]] = {}
        for row in reader:
            key = row[0].strip()
            keys.append(key)
            # assume row has exactly len(cols) remaining cells
            row_map = {cols[i]: row[i + 1].strip() for i in range(len(cols))}
            data[key] = row_map
    return keys, data

def build_sets(t2_keys: List[str], t3_keys: List[str],
               t2_map: Dict[str, Dict[str, str]], t3_map: Dict[str, Dict[str, str]]):
    """ 
        Build the lists a, c, f and M_ID mapping.
        a: all aircraft (current + future)
        c: current aircraft
        f: future aircraft
        M_ID: mapping from aircraft id to model id
    """
    c = list(t2_keys)
    f = list(t3_keys)
    a = c + f

    M_ID: Dict[str, int] = {}
    for ci in c:
        M_ID[ci] = int(t2_map[ci]["M_ID"])
    for fi in f:
        M_ID[fi] = int(t3_map[fi]["M_ID"])
    return a, c, f, M_ID


def build_parameters(t1_map, t2_map, t3_map, a: List[str], c: List[str], f: List[str], M_ID: Dict[str, int]):
    """Create parameter dicts that include data from each aircraft."""
    
    # Dimensions from T1 and M_ID
    W: Dict[str, float] = {}
    L: Dict[str, float] = {}
    for ai in a:
        mid = str(M_ID[ai])
        if mid not in t1_map:
            raise KeyError(f"Model id {mid} for aircraft {ai} not found in T1.csv")
        W[ai] = float(t1_map[mid].get("W", 0.0))
        L[ai] = float(t1_map[mid].get("L", 0.0))

    # Initialize other parameters to be filled from T2/T3
    ETA: Dict[str, float] = {}
    ETD: Dict[str, float] = {}
    ServT: Dict[str, float] = {}
    P_Rej: Dict[str, float] = {}
    P_Arr: Dict[str, float] = {}
    P_Dep: Dict[str, float] = {}
    Is_VIP: Dict[str, int] = {}
    X_init: Dict[str, float] = {}
    Y_init: Dict[str, float] = {}

    cset = set(c)

    for ai in a:
        if ai in cset: # if current aircraft (in c)
            ETD[ai] = float(t2_map[ai].get("ETD", 0.0))
            ServT[ai] = float(t2_map[ai].get("ServT", 0.0))
            P_Dep[ai] = float(t2_map[ai].get("P_Dep", 0.0))
            X_init[ai] = float(t2_map[ai].get("Init_X", 0.0))
            Y_init[ai] = float(t2_map[ai].get("Init_Y", 0.0))
        else:  # future aircraft (in f)
            ETA[ai]   = float(t3_map[ai].get("ETA", 0.0))
            ServT[ai] = float(t3_map[ai].get("ServT", 0.0))
            ETD[ai]   = float(t3_map[ai].get("ETD", 0.0))
            P_Rej[ai] = float(t3_map[ai].get("P_Rej", 0.0))
            P_Arr[ai] = float(t3_map[ai].get("P_Arr", 0.0))
            P_Dep[ai] = float(t3_map[ai].get("P_Dep", 0.0))
            Is_VIP[ai] = int(t3_map[ai].get("Is_VIP", "0"))

    return dict(W=W, L=L,
                ETA=ETA, ETD=ETD, ServT=ServT, 
                P_Rej=P_Rej, P_Arr=P_Arr, P_Dep=P_Dep,
                Is_VIP=Is_VIP,
                X_init=X_init, Y_init=Y_init)




########### EXAMPLE USAGE #############

# Define paths to input CSV files
t1_path = "Code/HangarModelResearch-main/data/T1.csv"
t2_path = "Code/HangarModelResearch-main/data/T2.csv"
t3_path = "Code/HangarModelResearch-main/data/random/T3-07-01.csv" 

# Read input CSVs (first column is the index)
t1_keys, t1_map = read_indexed_csv(t1_path)
t2_keys, t2_map = read_indexed_csv(t2_path)
t3_keys, t3_map = read_indexed_csv(t3_path)

# Build sets and parameters
a, c, f, M_ID = build_sets(t2_keys, t3_keys, t2_map, t3_map)
if not a:
    raise ValueError("No aircraft found. Check T2/T3 CSVs and make sure the first column is the ID.")
params = build_parameters(t1_map, t2_map, t3_map, a, c, f, M_ID)


print("Parameters loaded:")
for key, value in params.items():
    print(f"  {key}: {value}")

#Print sets:
print("Sets loaded:")
print(f"  a (all aircraft): {a}")
print(f"  c (current aircraft): {c}")
print(f"  f (future aircraft): {f}")    

#Print M_ID mapping
print("M_ID mapping:")
for ai in a:
    print(f"  Aircraft {ai} -> Model ID {M_ID[ai]}")        

