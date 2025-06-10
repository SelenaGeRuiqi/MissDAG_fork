# MissDAG with MVPC Prior Integration

This repository integrates MVPC (Missing Value PC algorithm) prior knowledge into MissDAG for improved causal structure learning with missing data.

## Overview

The project combines two powerful causal discovery methods:
- **MVPC**: Handles different missingness mechanisms (MCAR, MAR, MNAR) 
- **MissDAG**: Uses EM framework for DAG structure learning with missing data

By using MVPC results as prior knowledge in MissDAG optimization, we achieve better causal discovery performance.

## Quick Start

### 1. Generate MVPC Data and Priors

Use the R scripts in your MVPC repository:

```r
# In your MVPC R repository
source("demo.R")           # Generate MVPC results
source("ExportDataset.R")  # Export datasets
```

Modify parameters in the R scripts to generate all 8 datasets listed in the codebase structure below.

### 2. Convert and Upload Data

```bash
# Convert CSV results to numpy format, you may need to change it accordingly
python convert_results.py

# Validate conversion (optional)
python check_conversion_results.py
```

Upload the `.npy` files to the `result_npy/` folder following the structure below.

### 3. Run Experiments

```python
# Modify dataset parameters in main.py
args.mvpc_subfolder = 's100_mar_v20_e5_m10_seed777'  # change dataset
args.prior_weight = 1.0                              # adjust prior strength
args.run_comparison = True                           # compare all methods

# Run experiment
python main.py
```

Results will be saved in the `output/` directory with detailed logs.

## Repository Structure

```
MissDAG_fork/
├── output/                           # Experiment outputs
│   └── 2025-06-10_11-23-07-007_9948/
│       ├── training.log              # Detailed experiment log
│       ├── config.yaml               # Experiment configuration
│       └── system_info.yaml          # System information
├── result_npy/                       # Datasets and results (numpy format)
│   ├── s100_mar_v20_e5_m10_seed777/
│   │   ├── ground_truth_cpdag_adj.npy     # Ground truth CPDAG
│   │   ├── res_mvpc_permc_adj.npy         # MVPC prior (PermC method)
│   │   ├── s100_mar_v20_e5_m10_seed777.npy # Dataset with missing values
│   │   ├── res_mvpc_drw_adj.npy           # MVPC DRW result (optional)
│   │   ├── res_com_pc_adj.npy             # Complete PC result (optional)
│   │   └── res_tw_adj.npy                 # Test-wise PC result (optional)
│   ├── s100_mnar_v20_e5_m10_seed777/      # MNAR with 100 samples
│   ├── s1000_mar_v20_e5_m10_seed777/      # MAR with 1000 samples  
│   ├── s1000_mnar_v20_e5_m10_seed777/     # MNAR with 1000 samples
│   ├── s5000_mar_v20_e5_m10_seed777/      # MAR with 5000 samples
│   ├── s5000_mnar_v20_e5_m10_seed777/     # MNAR with 5000 samples
│   ├── s10000_mar_v20_e5_m10_seed777/     # MAR with 10000 samples
│   └── s10000_mnar_v20_e5_m10_seed777/    # MNAR with 10000 samples
├── dag_methods/                      # DAG learning algorithms
├── miss_methods/                     # Missing data handling methods  
├── utils/                           # Utility functions
├── data_loader/                     # Data loading utilities
├── main.py                          # Main experiment script
├── convert_results.py               # CSV to numpy converter
├── check_conversion_results.py      # Validation script
├── requirements.txt                 # Python dependencies (MacOS)
├── environment.yml                  # Conda environment (MacOS)
└── README.md                        # This file
```

## Dataset Parameters

All datasets use the following parameters:
- **Variables**: 20 nodes
- **Extra edges**: 5
- **Missing variables**: 10  
- **Seed**: 777
- **Sample sizes**: 100, 1000, 5000, 10000
- **Missing types**: MAR (Missing At Random), MNAR (Missing Not At Random)

## Experiment Configuration

### Key Parameters in `main.py`

```python
# Dataset selection
args.mvpc_subfolder = 's100_mar_v20_e5_m10_seed777'

# Prior integration
args.prior_weight = 1.0              # Strength of MVPC prior
args.prior_conversion = 'normalize'  # Prior conversion method

# Comparison experiments  
args.run_comparison = True           # Run all three methods:
                                    # 1. MVPC prior only
                                    # 2. MissDAG without prior  
                                    # 3. MissDAG with MVPC prior

# Algorithm settings
args.dag_method_type = 'notears'        # DAG learning method
args.miss_method_type = 'miss_dag_gaussian'  # Missing data method
args.em_iter = 20                       # EM algorithm iterations
```

## Example Output

When `run_comparison = True`, you'll see performance comparison:

```
=== PERFORMANCE COMPARISON ===
F1 Score - MVPC Prior: 0.2439
F1 Score - MissDAG Only: 0.4567  
F1 Score - MissDAG + Prior: 0.5417
F1 Improvement: +0.0850
SHD Improvement: +4
```
