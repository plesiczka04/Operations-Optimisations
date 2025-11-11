import numpy as np
import math

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
    

