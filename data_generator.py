import numpy as np
import math
import os
from collections import OrderedDict
import pandas as pd

np.random.seed(42)

def data_generator(num_initial_aircraft, num_incoming_aircraft, hangar_length, hangar_width):
    
    initial_aircraft = {}
    incoming_aircraft = {}

    for i in range(num_initial_aircraft):
        # Time of Arrival
        TOA = np.random.randint(-720, 0)  
        
        # Maintenance Duration
        MD = np.random.randint(30, 181)
                
        # Expected Departure
        ED = TOA + MD + np.random.randint(15, 61)  

        # Priority Level (0-1)
        PL = np.random.choice([0, 1], p=[0.8, 0.2])

        # Special Zone (0 or 1)
        SZ = np.random.choice([0, 1], p=[0.9, 0.1])

        # Aircraft Length (in meters)
        AL = round(np.random.uniform(20.0, 76.0), 2)

        # Aircraft Wingspan (in meters)
        AW = round(np.random.uniform(20.0, 80.0), 2)

        # Rejection Cost
        PREJ = np.random.randint(30, 181) + PL * np.random.randint(30, 181)

        # Delayed Arrival Cost
        PARR = np.random.randint(30, 181) + PL * np.random.randint(30, 181)

        # Delayed Departure Cost
        PDEP = np.random.randint(30, 181) + PL * np.random.randint(30, 181)

        # Random position with no overlap
        valid_position = False
        while not valid_position:
            POS_X = np.random.uniform(0, hangar_length)
            POS_Y = np.random.uniform(0, hangar_width)
            
            # Check no overlap with existing aircraft
            valid_position = True
            for aircraft in initial_aircraft.values():
                if (abs(POS_X - aircraft["POS_X"]) < (AL + aircraft["AL"]) / 2 and
                    abs(POS_Y - aircraft["POS_Y"]) < (AW + aircraft["AW"]) / 2):
                    valid_position = False
                    break

        initial_aircraft[i] = {
            "TOA": TOA,
            "MD": MD,
            "ED": ED,
            "PL": PL,
            "SZ": SZ,
            "AL": AL,
            "AW": AW,
            "PREJ": PREJ,
            "PARR": PARR,
            "PDEP": PDEP,
            "POS_X": POS_X,
            "POS_Y": POS_Y
        }
    
    
    
    for i in range(num_incoming_aircraft):
        # Time of Arrival
        TOA = np.random.randint(0, 1440)  
        
        # Maintenance Duration
        MD = np.random.randint(30, 181)
        
        # Expected Departure
        ED = TOA + MD + np.random.randint(15, 61)  

        # Priority Level (0-1)
        PL = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Special Zone (0 or 1)
        SZ = np.random.choice([0, 1], p=[0.9, 0.1])
        
        # Aircraft Length (in meters)
        AL = round(np.random.uniform(20.0, 76.0), 2)
        
        # Aircraft Wingspan (in meters)
        AW = round(np.random.uniform(20.0, 80.0), 2)

        # Rejection Cost
        PREJ = np.random.randint(30, 181) + PL * np.random.randint(30, 181)

        # Delayed Arrival Cost
        PARR = np.random.randint(30, 181) + PL * np.random.randint(30, 181)

        # Delayed Departure Cost
        PDEP = np.random.randint(30, 181) + PL * np.random.randint(30, 181)
        
        incoming_aircraft[i] = {
            "TOA": TOA,
            "MD": MD,
            "ED": ED,
            "PL": PL,
            "SZ": SZ,
            "AL": AL,
            "AW": AW,
            "PREJ": PREJ,
            "PARR": PARR,
            "PDEP": PDEP
        }

     
    return initial_aircraft, incoming_aircraft
    

def build_csvs(
    num_initial_aircraft: int,
    num_incoming_aircraft: int,
    hangar_length: float,
    hangar_width: float,
    out_dir: str = "."
):

    initial_aircraft, incoming_aircraft = data_generator(
        num_initial_aircraft, num_incoming_aircraft, hangar_length, hangar_width
    )

    # ---- Build T1 (model catalog) by deduping (AW, AL) across all aircraft
    unique_models = OrderedDict()
    for rec in list(initial_aircraft.values()) + list(incoming_aircraft.values()):
        key = (float(rec["AW"]), float(rec["AL"]))  # (W, L)
        if key not in unique_models:
            unique_models[key] = None

    # Stable model IDs: sort by W then L
    for idx, key in enumerate(sorted(unique_models.keys())):
        unique_models[key] = idx  # m

    def mid_for(rec) -> int:
        return unique_models[(float(rec["AW"]), float(rec["AL"]))]

    # DataFrames
    T1 = (
        pd.DataFrame(
            [{"m": m, "W": w, "L": l} for (w, l), m in unique_models.items()]
        )
        .sort_values("m")
        .reset_index(drop=True)
    )

    T2 = pd.DataFrame(
        [
            {
                "c": int(c),
                "M_ID": int(mid_for(rec)),
                "ETD": int(rec["ED"]),
                "ServT": int(rec["MD"]),
                "P_Dep": int(rec["PDEP"]),
                "Init_X": round(float(rec["POS_X"]), 3),
                "Init_Y": round(float(rec["POS_Y"]), 3),
            }
            for c, rec in initial_aircraft.items()
        ]
    ).sort_values("c").reset_index(drop=True)

    T3 = pd.DataFrame(
        [
            {
                "f": int(f),
                "M_ID": int(mid_for(rec)),
                "ETA": int(rec["TOA"]),
                "ETD": int(rec["ED"]),
                "ServT": int(rec["MD"]),
                "P_Rej": int(rec["PREJ"]),
                "P_Arr": int(rec["PARR"]),
                "P_Dep": int(rec["PDEP"]),
                "Is_VIP": int(rec["PL"]),
            }
            for f, rec in incoming_aircraft.items()
        ]
    ).sort_values("f").reset_index(drop=True)

    # Ensure exact column order per spec
    T1 = T1[["m", "W", "L"]]
    T2 = T2[["c", "M_ID", "ETD", "ServT", "P_Dep", "Init_X", "Init_Y"]]
    T3 = T3[["f", "M_ID", "ETA", "ETD", "ServT", "P_Rej", "P_Arr", "P_Dep", "Is_VIP"]]

    # Write CSVs
    os.makedirs(out_dir, exist_ok=True)
    t1_path = os.path.join(out_dir, "T1.csv")
    t2_path = os.path.join(out_dir, "T2.csv")
    t3_path = os.path.join(out_dir, "T3.csv")

    T1.to_csv(t1_path, index=False)
    T2.to_csv(t2_path, index=False)
    T3.to_csv(t3_path, index=False)

    return t1_path, t2_path, t3_path, T1, T2, T3


if __name__ == "__main__":
    paths = build_csvs(
        num_initial_aircraft=10,
        num_incoming_aircraft=15,
        hangar_length=300.0,
        hangar_width=200.0,
        out_dir="."
    )
    print("Wrote files:", paths[:3])
