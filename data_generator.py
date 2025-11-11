import numpy as np
import math

np.random.seed(42)

def data_generator():
    # Time of Arrival
    TOA = np.random.randint(0, 1440)  # in minutes from midnight
    
    # Maintenance Duration
    MD = np.random.randint(30, 181)  # in minutes
    
    # Expected Departure
    ED = TOA + MD + np.random.randint(15, 61)  # in minutes from midnight

    # Priority Level (0-1)
    PL = np.random.choice([0, 1], p=[0.8, 0.2])
    
    # Special Zone (0 or 1)
    SZ = np.random.choice([0, 1], p=[0.9, 0.1])
    
    # Aircraft Length (in meters)
    AL = round(np.random.uniform(20.0, 76.0), 2)
    
    # Aircraft Wingspan (in meters)
    AW = round(np.random.uniform(20.0, 80.0), 2)
    
    return {
        "TOA": TOA, 
        "MD": MD,
        "ED": ED,
        "PL": PL,
        "SZ": SZ,
        "AL": AL,
        "AW": AW
    }
    
if __name__ == "__main__":
    sample_data = data_generator()
    print(sample_data)
    
    

