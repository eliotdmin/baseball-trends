import multiprocessing as mp
import pandas as pd
import pybaseball as pb
import os
from tqdm import tqdm

def fetch_worker(dt):
    date_str = dt.strftime('%Y-%m-%d')
    
    # Define separate paths for different data types
    sc_dir = f"data/statcast_pitches/game_date={date_str}"
    gm_dir = f"data/game_metadata/game_date={date_str}"
    
    # Check for Statcast
    if not os.path.exists(f"{sc_dir}/pitches.parquet"):
        df_sc = pb.statcast(start_dt=date_str, end_dt=date_str)
        if df_sc is not None and not df_sc.empty:
            os.makedirs(sc_dir, exist_ok=True)
            df_sc.to_parquet(f"{sc_dir}/pitches.parquet", index=False)
            
    # Add your game data fetching logic here similarly...

# 2. Protection block is MANDATORY on macOS/Windows
if __name__ == "__main__":
    # Optional: Force a specific start method if needed
    # mp.set_start_method('spawn', force=True) 
    
    dates = pd.date_range(start="2018-05-17", end="2025-12-31")
    num_processors = mp.cpu_count() - 1

    with mp.Pool(processes=num_processors) as pool:
        results = list(tqdm(pool.imap_unordered(fetch_worker, dates), total=len(dates)))

    print("Download complete.")