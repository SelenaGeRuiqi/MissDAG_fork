import os
import numpy as np
import pandas as pd
from pathlib import Path

def convert_csv_to_npy():
    """
    Convert CSV matrix files to .npy files while preserving directory structure.
    Skips evaluation_metrics.csv files.
    Each NPY file is stored in the existing subfolder with the same name as the CSV file.
    First column contains data (not labels), first row contains column headers.
    """
    # result_dir = Path('./result')
    # result_npy_dir = Path('./result_npy')
    result_dir = Path('./mvpc_dataset')
    result_npy_dir = Path('./result_npy')
    
    # Check if result directory exists
    if not result_dir.exists():
        print(f"Error: {result_dir} directory does not exist!")
        return
    
    # Check if result_npy directory exists
    if not result_npy_dir.exists():
        print(f"Error: {result_npy_dir} directory does not exist!")
        return
    
    converted_count = 0
    skipped_count = 0
    
    # Process files in result directory
    for file in os.listdir(result_dir):
        if file.endswith('.csv') and file != 'evaluation_metrics.csv':
            csv_path = result_dir / file
            # Get the base name without extension
            base_name = file.replace('.csv', '')
            # Use existing subfolder with the same name as the CSV file
            subfolder = result_npy_dir / base_name
            
            # Check if the subfolder exists
            if not subfolder.exists():
                print(f"Warning: Subfolder {subfolder} does not exist, skipping {file}")
                skipped_count += 1
                continue
            
            npy_filename = file.replace('.csv', '.npy')
            npy_path = subfolder / npy_filename
            
            try:
                # Read CSV file
                # First column contains data (not index), first row contains headers
                df = pd.read_csv(csv_path, header=0, index_col=None)
                
                # Convert to numpy array
                matrix = df.values.astype(np.float32)
                
                # Delete existing file if it exists
                if npy_path.exists():
                    npy_path.unlink()
                    print(f"Deleted existing file: {npy_path}")
                
                # Save as .npy file
                np.save(npy_path, matrix)
                
                print(f"Converted: {csv_path} -> {npy_path}")
                converted_count += 1
                
            except Exception as e:
                print(f"Error converting {csv_path}: {str(e)}")
                
        elif file == 'evaluation_metrics.csv':
            print(f"Skipped: {result_dir / file}")
            skipped_count += 1
    
    print(f"\nConversion complete!")
    print(f"Files converted: {converted_count}")
    print(f"Files skipped: {skipped_count}")

if __name__ == "__main__":
    convert_csv_to_npy()