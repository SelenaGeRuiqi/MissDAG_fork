import os
import numpy as np
import pandas as pd
from pathlib import Path

def convert_csv_to_npy():
    """
    Convert CSV matrix files to .npy files while preserving directory structure.
    Skips evaluation_metrics.csv files.
    """
    result_dir = Path('./result')
    result_npy_dir = Path('./result_npy')
    
    # Check if result directory exists
    if not result_dir.exists():
        print(f"Error: {result_dir} directory does not exist!")
        return
    
    # Create result_npy directory if it doesn't exist
    result_npy_dir.mkdir(exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    
    # Walk through all subdirectories in result
    for root, dirs, files in os.walk(result_dir):
        root_path = Path(root)
        
        # Calculate relative path from result directory
        rel_path = root_path.relative_to(result_dir)
        
        # Create corresponding directory in result_npy
        output_dir = result_npy_dir / rel_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each CSV file
        for file in files:
            if file.endswith('.csv') and file != 'evaluation_metrics.csv':
                csv_path = root_path / file
                npy_filename = file.replace('.csv', '.npy')
                npy_path = output_dir / npy_filename
                
                try:
                    # Read CSV file
                    # The first column and first row contain labels, so we'll read accordingly
                    df = pd.read_csv(csv_path, index_col=0)
                    
                    # Convert to numpy array
                    matrix = df.values.astype(np.float32)
                    
                    # Save as .npy file
                    np.save(npy_path, matrix)
                    
                    print(f"Converted: {csv_path} -> {npy_path}")
                    converted_count += 1
                    
                except Exception as e:
                    print(f"Error converting {csv_path}: {str(e)}")
                    
            elif file == 'evaluation_metrics.csv':
                print(f"Skipped: {root_path / file}")
                skipped_count += 1
    
    print(f"\nConversion complete!")
    print(f"Files converted: {converted_count}")
    print(f"Files skipped: {skipped_count}")

if __name__ == "__main__":
    convert_csv_to_npy()