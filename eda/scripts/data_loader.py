import pandas as pd
import numpy as np
import glob
import os
import re

def load_and_adapt_simulation_data(sim_csv_path='../data/drift/simulation.csv'):
    """
    Loads simulation CSV and adapts it to match the format of the DRIFT dataset 
    expected by the analysis functions.
    
    Parameters:
    - sim_csv_path: Path to the simulation output CSV.
    
    Returns:
    - List containing a single DataFrame (as 'sessions' usually expects a list).
    """
    
    print(f"Loading simulation data from {sim_csv_path}...")
    df = pd.read_csv(sim_csv_path)
    
    # 1. Convert Timestamp
    base_time = pd.Timestamp('2024-01-01 08:00:00')
    
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        datetimes = base_time + pd.to_timedelta(df['timestamp'], unit='s')
        df['timestamp'] = datetimes.dt.strftime('%H:%M:%S.%f').str[:-3]
        
    # 2. Ensure Geometry Columns (Width, Height, Angle)
    if 'x1' in df.columns and 'x2' in df.columns and 'y1' in df.columns and 'y2' in df.columns:
        dx1 = df['x2'] - df['x1']
        dy1 = df['y2'] - df['y1']
        len1 = np.sqrt(dx1**2 + dy1**2)
        
        dx2 = df['x3'] - df['x2']
        dy2 = df['y3'] - df['y2']
        len2 = np.sqrt(dx2**2 + dy2**2)
        
        df['dim_a'] = len1
        df['dim_b'] = len2
        
        df['width'] = df[['dim_a', 'dim_b']].max(axis=1)  # Length
        df['height'] = df[['dim_a', 'dim_b']].min(axis=1) # Width
        
        df['angle'] = 0.0
        mask_1_long = df['dim_a'] >= df['dim_b']
        
        df.loc[mask_1_long, 'angle'] = np.arctan2(dy1[mask_1_long], dx1[mask_1_long])
        df.loc[~mask_1_long, 'angle'] = np.arctan2(dy2[~mask_1_long], dx2[~mask_1_long])
        
    else:
        print("Warning: Corner columns missing. Using default car size.")
        df['width'] = 40.0
        df['height'] = 18.0
        df['angle'] = 0.0
    
    required_cols = ['track_id', 'center_x', 'center_y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in simulation data.")

    print(f"Adapted data: {len(df)} rows.")
    return [df]

