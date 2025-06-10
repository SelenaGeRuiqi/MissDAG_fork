## Instruction
1. Get the prior(permc), ground truth, and dataset from MVPC R repository using demo.R and ExportDataset.R
   1.1 Need to change parameters to get the 8 datasets(parameters same as the ones I listsed in result_npy/ folder in codebase structure
2. Convert .csv to .npy
3. Upload .the npy files to MissDAG_fork in result_npy/ folder following my codebase structure
4. Modify dataset parameters in main.py, and run it directly without argparse, the log files will be stored in output/ directory

## Codebase Structure

MissDAG_fork/
├── .git/
├── output/                        # Experiment output directories
│   └── 2025-06-10_11-23-07-007_9948/
│       ├── training.log
│       ├── config.yaml
│       └── system_info.yaml
├── dag_methods/
├── result_npy/                    # Results in numpy format, convert_results.py might be useful to convert the .csv files from R to .npy files
│   ├── s100_mar_v20_e5_m10_seed777/
│   │   ├── ground_truth_cpdag_adj.npy    # ground truth
│   │   ├── res_com_pc_adj.npy    # optional
│   │   ├── res_mvpc_drw_adj.npy    # optional
│   │   ├── res_mvpc_permc_adj.npy  # MVPC adjancency matrix w/ permc method
│   │   ├── res_tw_adj.npy    # optional
│   │   └── s100_mar_v20_e5_m10_seed777.npy  # dataset
│   ├── s100_mnar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
│   ├── s1000_mar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
│   ├── s1000_mnar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
│   ├── s5000_mar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
│   ├── s5000_mnar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
│   ├── s10000_mar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
│   └── s10000_mnar_v20_e5_m10_seed777/
│   │   ├── ...
│   │   ├── ...
├── utils/
├── miss_methods/
├── data_loader/
├── main.py                        # Run this, I didn't give it argpase since it might conflict with MissDAG internal usage, you can find a section for paramater settings and change there
├── convert_results.py             # Convert csv to npy, you can write your own one
├── check_conversion_results.py    # Validation script, you can write your own one
├── requirements.txt               # Python dependencies, MacOS
├── environment.yml                # Conda environment, MacOS
├── .gitignore                     # Git ignore rules, you may need to modify this
└── README.md
