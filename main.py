import logging
import os
import numpy as np

from dag_methods import Notears, Notears_ICA_MCEM, Notears_ICA, \
                            Notears_MLP_MCEM, Notears_MLP_MCEM_INIT
from data_loader import SyntheticDataset
from miss_methods import miss_dag_gaussian, miss_dag_nongaussian, miss_dag_nonlinear
from utils.config import save_yaml_config, get_args
from utils.dir import create_dir, get_datetime_str
from utils.logging import setup_logger, get_system_info
from utils.utils import set_seed, MetricsDAG, postprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MVPCDataLoader:
    """Load MVPC dataset and prior matrix"""
    
    def __init__(self, data_path, prior_path):
        self.data_path = data_path
        self.prior_path = prior_path
        
    def load_mvpc_data(self):
        data = np.load(self.data_path)
        print(f"Loaded MVPC dataset: {data.shape}")
        print(f"Missing values: {np.isnan(data).sum()}")
        print(f"Missing percentage: {np.isnan(data).sum() / data.size * 100:.2f}%")
        
        prior_matrix = np.load(self.prior_path)
        print(f"Loaded MVPC prior matrix: {prior_matrix.shape}")
        print(f"Prior matrix range: [{np.min(prior_matrix):.3f}, {np.max(prior_matrix):.3f}]")
        print(f"Prior matrix non-zero elements: {np.sum(prior_matrix != 0)}")
        print(f"Prior matrix unique values: {len(np.unique(prior_matrix))}")
        
        # Debug: show some matrix values
        print("Prior matrix sample (first 5x5):")
        print(prior_matrix[:5, :5])
        
        # Debug: check if it's binary or continuous
        unique_vals = np.unique(prior_matrix)
        print(f"Unique values in prior matrix: {unique_vals}")
        
        return data, prior_matrix
    
    def convert_prior_for_missdag(self, prior_matrix, method='normalize'):
        """Convert MVPC prior to MissDAG format"""
        if method == 'normalize':
            max_val = np.max(np.abs(prior_matrix))
            if max_val > 0:
                converted = np.abs(prior_matrix) / max_val
            else:
                converted = prior_matrix.copy()
        elif method == 'threshold':
            converted = np.where(np.abs(prior_matrix) > 0, 0.8, 0.2)
        elif method == 'sigmoid':
            converted = 1 / (1 + np.exp(-prior_matrix))
        else:
            converted = prior_matrix.copy()
            
        print(f"Prior converted using {method}: [{np.min(converted):.3f}, {np.max(converted):.3f}]")
        return converted

class MVPCSyntheticDataset:
    """Wrapper for MVPC dataset to match MissDAG interface"""
    
    def __init__(self, data_path, prior_path, ground_truth_path=None, prior_conversion='normalize'):
        self.loader = MVPCDataLoader(data_path, prior_path)
        self.data, self.prior_matrix = self.loader.load_mvpc_data()
        self.converted_prior = self.loader.convert_prior_for_missdag(
            self.prior_matrix, prior_conversion
        )
        
        self.X = self.data
        self.mask = ~np.isnan(self.data)
        
        if ground_truth_path and os.path.exists(ground_truth_path):
            self.B_bin = np.load(ground_truth_path)
            print(f"Loaded ground truth: {self.B_bin.shape}")
            print(f"Ground truth edges: {np.sum(self.B_bin != 0)}")
        else:
            n_vars = self.data.shape[1]
            self.B_bin = np.zeros((n_vars, n_vars))
            print("Warning: No ground truth provided, using zero matrix")
        
        print(f"Dataset prepared: {self.X.shape}, missing rate: {(~self.mask).sum() / self.mask.size * 100:.2f}%")

def get_mvpc_file_paths(base_dir, subfolder_name):
    """Auto-find MVPC files in subfolder"""
    import glob
    
    subfolder_path = os.path.join(base_dir, subfolder_name)
    
    if not os.path.exists(subfolder_path):
        raise ValueError(f"Subfolder not found: {subfolder_path}")
    
    # Find data file
    data_pattern = os.path.join(subfolder_path, "s*_*ar_v*_e*_m*_seed*.npy")
    data_files = glob.glob(data_pattern)
    if not data_files:
        data_pattern_csv = os.path.join(subfolder_path, "s*_*ar_v*_e*_m*_seed*.csv")
        data_files = glob.glob(data_pattern_csv)
    
    if not data_files:
        raise ValueError(f"Data file not found in {subfolder_path} with pattern s*_*ar_v*_e*_m*_seed*")
    data_path = data_files[0]
    
    # Find prior file
    prior_path = os.path.join(subfolder_path, "res_mvpc_permc_adj.npy")
    if not os.path.exists(prior_path):
        raise ValueError(f"Prior file not found: {prior_path}")
    
    # Find ground truth file
    gt_pattern = os.path.join(subfolder_path, "ground_truth*.npy")
    gt_files = glob.glob(gt_pattern)
    if not gt_files:
        print(f"Warning: No ground truth file found in {subfolder_path}")
        ground_truth_path = None
    else:
        ground_truth_path = gt_files[0]
    
    print(f"Found files in {subfolder_name}:")
    print(f"  Data: {os.path.basename(data_path)}")
    print(f"  Prior: {os.path.basename(prior_path)}")
    print(f"  Ground truth: {os.path.basename(ground_truth_path) if ground_truth_path else 'None'}")
    
    return data_path, prior_path, ground_truth_path

def get_modified_args():
    """Setup MVPC parameters"""
    args = get_args()
    
    # MVPC settings - modify these to change experiments
    args.use_mvpc_data = True
    args.mvpc_subfolder = 's10000_mnar_v20_e5_m10_seed777'  # change this for different datasets
    args.mvpc_base_dir = 'result_npy'
    args.prior_weight = 0.5  # adjust prior strength
    args.prior_weights = [0.3, 0.5, 0.7]  # multiple prior weights to test
    args.prior_conversion = 'normalize'
    args.run_comparison = True  # run all three methods for comparison
    
    # MissDAG settings
    args.dag_method_type = 'notears'
    args.miss_method_type = 'miss_dag_gaussian'
    args.em_iter = 20
    args.lambda_1_ev = 0.1
    args.graph_thres = 0.3
    
    print(f"Using DAG method: {args.dag_method_type}")
    print(f"Using miss method: {args.miss_method_type}")
    print(f"EM iterations: {args.em_iter}")
    print(f"Run comparison: {args.run_comparison}")
    
    data_path, prior_path, ground_truth_path = get_mvpc_file_paths(
        args.mvpc_base_dir, args.mvpc_subfolder
    )
    
    args.mvpc_data_path = data_path
    args.mvpc_prior_path = prior_path
    args.mvpc_ground_truth_path = ground_truth_path
    
    return args

class DAGMethodWithPrior:
    """Wrapper to add prior knowledge to DAG methods"""
    
    def __init__(self, base_method, prior_matrix, prior_weight=1.0):
        self.base_method = base_method
        self.prior_matrix = prior_matrix
        self.prior_weight = prior_weight
        print(f"DAGMethodWithPrior initialized with prior weight: {prior_weight}")
        print(f"Prior matrix shape: {prior_matrix.shape}")
        print(f"Prior matrix stats: min={np.min(prior_matrix):.3f}, max={np.max(prior_matrix):.3f}, mean={np.mean(prior_matrix):.3f}")
        
    def fit(self, X, cov_emp=None):
        if X is not None:
            print(f"X shape: {X.shape}")
        else:
            print("X is None, using covariance matrix")
        if cov_emp is not None:
            print(f"Covariance matrix shape: {cov_emp.shape}")
        
        # Check if the base method has fit_with_prior
        if hasattr(self.base_method, 'fit_with_prior'):
            print("Using native fit_with_prior method")
            return self.base_method.fit_with_prior(X, self.prior_matrix, self.prior_weight, cov_emp)
        else:
            print("Using custom prior regularization")
            return self.fit_with_prior_regularization(X, cov_emp)
    
    def fit_with_prior_regularization(self, X, cov_emp=None):
        """Add prior as regularization term"""
        
        # Handle the case where X is None (MissDAG uses covariance matrix)
        if X is None and cov_emp is not None:
            print("Working with covariance matrix instead of raw data")
            # For covariance-based methods, we need a different approach
            # Let's try to modify the covariance matrix directly with prior information
            
            # Add prior information to covariance matrix
            # This is a simple approach - you might need more sophisticated methods
            prior_influence = self.prior_weight * 0.1  # Scale down the influence
            modified_cov = cov_emp.copy()
            
            # Add small prior influence to covariance
            for i in range(modified_cov.shape[0]):
                for j in range(modified_cov.shape[1]):
                    if i != j and self.prior_matrix[i, j] > 0:
                        # Increase covariance where prior suggests connection
                        modified_cov[i, j] += prior_influence * self.prior_matrix[i, j]
                        modified_cov[j, i] += prior_influence * self.prior_matrix[i, j]
            
            print(f"Modified covariance matrix with prior influence: {prior_influence}")
            return self.base_method.fit(X=X, cov_emp=modified_cov)
        
        # Original approach for when X is not None
        # Try to access the original loss function
        original_loss = None
        if hasattr(self.base_method, '_original_loss'):
            original_loss = self.base_method._original_loss
        elif hasattr(self.base_method, 'compute_loss'):
            original_loss = self.base_method.compute_loss
        else:
            print("No loss function found, will use simple MSE")
        
        def loss_with_prior(W):
            # Original data fitting loss
            if original_loss:
                try:
                    data_loss = original_loss(W)
                except Exception as e:
                    if X is not None:
                        data_loss = np.sum((X - X @ W.T) ** 2)
                    else:
                        data_loss = 0  # Fallback
            else:
                if X is not None:
                    data_loss = np.sum((X - X @ W.T) ** 2)
                else:
                    data_loss = 0  # Fallback
            
            # Prior regularization
            prior_loss = np.sum((W - self.prior_matrix) ** 2)
            total_loss = data_loss + self.prior_weight * prior_loss
            
            return total_loss
        
        # Try to inject the modified loss function
        loss_injected = False
        if hasattr(self.base_method, 'compute_loss'):
            self.base_method._original_loss = self.base_method.compute_loss
            self.base_method.compute_loss = loss_with_prior
            loss_injected = True
        elif hasattr(self.base_method, '_loss'):
            self.base_method._original_loss = self.base_method._loss
            self.base_method._loss = loss_with_prior
            loss_injected = True
        else:
            print("Warning: Could not inject loss function, using covariance modification instead")
            # Fallback to covariance modification if available
            if cov_emp is not None:
                return self.fit_with_prior_regularization(None, cov_emp)
        
        # Call the original fit method
        try:
            result = self.base_method.fit(X, cov_emp)
        except Exception as e:
            print(f"Error in base method fit: {e}")
            raise
        
        # Restore original loss function
        if loss_injected and hasattr(self.base_method, '_original_loss'):
            if hasattr(self.base_method, 'compute_loss'):
                self.base_method.compute_loss = self.base_method._original_loss
            elif hasattr(self.base_method, '_loss'):
                self.base_method._loss = self.base_method._original_loss
            delattr(self.base_method, '_original_loss')
        
        return result

def miss_dag_with_prior(X, mask, dag_method, prior_matrix, prior_weight, em_iter, equal_variances):
    """Run MissDAG with prior knowledge"""
    dag_method_with_prior = DAGMethodWithPrior(dag_method, prior_matrix, prior_weight)
    return miss_dag_gaussian(X, mask, dag_method_with_prior, em_iter, equal_variances)

def main():
    args = get_modified_args()

    # Setup logging
    output_dir = 'output/{}'.format(get_datetime_str(add_random_str=True))
    create_dir(output_dir)
    setup_logger(log_path='{}/training.log'.format(output_dir), level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger.")

    # Log MVPC parameters
    if getattr(args, 'use_mvpc_data', False):
        _logger.info("=== MVPC PARAMETERS ===")
        _logger.info(f"MVPC Base Directory: {args.mvpc_base_dir}")
        _logger.info(f"MVPC Subfolder: {args.mvpc_subfolder}")
        _logger.info(f"MVPC Data Path: {args.mvpc_data_path}")
        _logger.info(f"MVPC Prior Path: {args.mvpc_prior_path}")
        _logger.info(f"MVPC Ground Truth Path: {getattr(args, 'mvpc_ground_truth_path', 'None')}")
        _logger.info(f"Prior Weights: {args.prior_weights}")
        _logger.info(f"Prior Conversion Method: {args.prior_conversion}")
        _logger.info("=======================")

    # Save system info and configs
    system_info = get_system_info()
    if system_info is not None:
        save_yaml_config(system_info, path='{}/system_info.yaml'.format(output_dir))
    save_yaml_config(vars(args), path='{}/config.yaml'.format(output_dir))

    set_seed(args.seed)

    # Load dataset
    if getattr(args, 'use_mvpc_data', False):
        _logger.info("Loading MVPC dataset...")
        dataset = MVPCSyntheticDataset(
            args.mvpc_data_path, 
            args.mvpc_prior_path,
            getattr(args, 'mvpc_ground_truth_path', None),
            args.prior_conversion
        )
        _logger.info("Finished loading MVPC dataset.")
    else:
        dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.noise_type,
                                   args.miss_type, args.miss_percent, args.sem_type,
                                   args.equal_variances, args.mnar_type, args.p_obs, args.mnar_quantile_q)
        _logger.info("Finished loading the original synthetic dataset.")

    # Setup DAG method
    print(f"Setting up DAG method: {args.dag_method_type}")
    
    if args.dag_method_type == 'notears':
        dag_method = Notears(args.lambda_1_ev)
    elif args.dag_method_type == 'notears_ica':
        dag_method = Notears_ICA(args.seed, args.MLEScore)
    elif args.dag_method_type == 'notears_ica_mcem':
        dag_init_method = Notears_ICA(args.seed, args.MLEScore)
        dag_method = Notears_ICA_MCEM(args.seed, args.MLEScore)
    elif args.dag_method_type == 'notears_mlp_mcem':
        dag_init_method = Notears_MLP_MCEM_INIT()
        dag_method = Notears_MLP_MCEM()
    else:
        print(f"Error: Unknown DAG method type: '{args.dag_method_type}'")
        print("Available methods: 'notears', 'notears_ica', 'notears_ica_mcem', 'notears_mlp_mcem'")
        print("Setting default to 'notears'...")
        args.dag_method_type = 'notears'
        dag_method = Notears(getattr(args, 'lambda_1_ev', 0.1))
    _logger.info("Finished setting up the structure learning method.")

    # Check for missing data
    if hasattr(dataset, 'X'):
        has_missing = np.isnan(dataset.X).any()
    else:
        has_missing = args.miss_percent > 0
    
    # Run experiments
    if not has_missing and args.miss_method_type == 'no_missing':
        if args.miss_method_type == 'no_missing':
            if hasattr(dataset, 'X_true'):
                X = dataset.X_true
            else:
                X = dataset.X
                if np.isnan(X).any():
                    _logger.warning("Data has missing values but miss_method_type is 'no_missing'. Using listwise deletion.")
                    complete_cases = ~np.isnan(X).any(axis=1)
                    X = X[complete_cases]
        else:
            raise ValueError("Add your imputation methods.")

        if args.dag_method_type not in {'notears_ica_mcem', 'notears_mlp_mcem'}:
            if getattr(args, 'use_mvpc_data', False):
                dag_method_with_prior = DAGMethodWithPrior(
                    dag_method, 
                    dataset.converted_prior, 
                    args.prior_weight
                )
                B_est = dag_method_with_prior.fit(X=X, cov_emp=None)
            else:
                B_est = dag_method.fit(X=X, cov_emp=None)
        else:
            raise ValueError("The miss_method here does not support notears_ica_mcem/notears_mlp_mcem.")

    elif args.miss_method_type == 'miss_dag_gaussian':
        results = {}
        
        if getattr(args, 'run_comparison', False) and getattr(args, 'use_mvpc_data', False):
            _logger.info("=== RUNNING COMPARISON EXPERIMENTS ===")
            
            # Experiment 1: MissDAG without prior
            _logger.info("Experiment 1: MissDAG without prior")
            B_est_no_prior, _, _ = miss_dag_gaussian(dataset.X, dataset.mask, dag_method,
                                                   args.em_iter, args.equal_variances)
            results['missdag_only'] = B_est_no_prior
            
            # Experiment 2: MissDAG with MVPC prior (multiple weights)
            for i, prior_weight in enumerate(args.prior_weights):
                _logger.info(f"Experiment {i+2}: MissDAG with MVPC prior (weight={prior_weight})")
                B_est_with_prior, _, _ = miss_dag_with_prior(
                    dataset.X, dataset.mask, dag_method,
                    dataset.converted_prior, prior_weight,
                    args.em_iter, args.equal_variances
                )
                results[f'missdag_with_prior_{prior_weight}'] = B_est_with_prior
            
            # Use the last prior weight result as the main result
            B_est = B_est_with_prior
            
        else:
            # Single experiment with first prior weight
            if getattr(args, 'use_mvpc_data', False):
                prior_weight = args.prior_weights[0]
                _logger.info(f"Running MissDAG with MVPC prior (weight: {prior_weight})")
                B_est, _, _ = miss_dag_with_prior(
                    dataset.X, dataset.mask, dag_method,
                    dataset.converted_prior, prior_weight,
                    args.em_iter, args.equal_variances
                )
            else:
                B_est, _, _ = miss_dag_gaussian(dataset.X, dataset.mask, dag_method,
                                              args.em_iter, args.equal_variances)
                                          
    elif args.miss_method_type == 'miss_dag_nongaussian':
        assert args.dag_method_type == 'notears_ica_mcem', \
                "miss_dag_nongaussian supports only notears_ica_mcem as dag_method_type"
        B_est, _, _ = miss_dag_nongaussian(dataset.X, dag_init_method,
                                         dag_method, args.em_iter, args.MLEScore, args.num_sampling, B_true=dataset.B_bin)
                                         
    elif args.miss_method_type == 'miss_dag_nonlinear':
        assert args.dag_method_type == 'notears_mlp_mcem', \
            "miss_dag_nonlinear supports only notears_mlp_mcem as dag_method_type"
        B_est, _, _ = miss_dag_nonlinear(dataset.X, dag_init_method,
                                       dag_method, args.em_iter, args.equal_variances)
    else:
        raise ValueError("Unknown method type.")
    _logger.info("Finished estimating the graph.")

    # Post-process and evaluate
    _, B_processed_bin = postprocess(B_est, args.graph_thres)
    _logger.info("Finished post-processing the estimated graph.")

    if hasattr(dataset, 'B_bin') and dataset.B_bin.size > 0 and np.any(dataset.B_bin != 0):
        
        if getattr(args, 'run_comparison', False) and 'results' in locals():
            _logger.info("=== COMPARISON RESULTS ===")
            
            # Evaluate MVPC prior with detailed debugging
            print(f"Evaluating MVPC prior with threshold {args.graph_thres}")
            print(f"Original prior matrix stats: min={np.min(dataset.prior_matrix):.3f}, max={np.max(dataset.prior_matrix):.3f}")
            print(f"Non-zero elements in prior: {np.sum(dataset.prior_matrix != 0)}")
            
            _, prior_bin = postprocess(dataset.prior_matrix, args.graph_thres)
            print(f"After postprocessing: {np.sum(prior_bin != 0)} edges")
            print(f"Ground truth edges: {np.sum(dataset.B_bin != 0)}")
            
            # Try different thresholds for MVPC prior
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            print("Trying different thresholds for MVPC prior:")
            for thresh in thresholds:
                _, prior_bin_thresh = postprocess(dataset.prior_matrix, thresh)
                result_thresh = MetricsDAG(prior_bin_thresh, dataset.B_bin).metrics
                print(f"Threshold {thresh}: edges={np.sum(prior_bin_thresh != 0)}, F1={result_thresh.get('F1', 0):.4f}")
            
            # Use the original threshold for comparison
            prior_result = MetricsDAG(prior_bin, dataset.B_bin).metrics
            _logger.info("MVPC prior result: {0}".format(prior_result))
            
            # Evaluate MissDAG without prior
            _, B_no_prior_bin = postprocess(results['missdag_only'], args.graph_thres)
            no_prior_result = MetricsDAG(B_no_prior_bin, dataset.B_bin).metrics
            _logger.info("MissDAG only result: {0}".format(no_prior_result))
            
            # Evaluate MissDAG with prior for each weight
            prior_weight_results = {}
            for prior_weight in args.prior_weights:
                key = f'missdag_with_prior_{prior_weight}'
                if key in results:
                    _, B_with_prior_bin = postprocess(results[key], args.graph_thres)
                    with_prior_result = MetricsDAG(B_with_prior_bin, dataset.B_bin).metrics
                    prior_weight_results[prior_weight] = with_prior_result
                    _logger.info(f"MissDAG with prior (weight={prior_weight}) result: {with_prior_result}")
            
            # Performance comparison
            _logger.info("=== PERFORMANCE COMPARISON ===")
            _logger.info(f"F1 Score - MVPC Prior: {prior_result.get('F1', 'N/A'):.4f}")
            _logger.info(f"F1 Score - MissDAG Only: {no_prior_result.get('F1', 'N/A'):.4f}")
            
            for prior_weight in args.prior_weights:
                if prior_weight in prior_weight_results:
                    result = prior_weight_results[prior_weight]
                    _logger.info(f"F1 Score - MissDAG + Prior (w={prior_weight}): {result.get('F1', 'N/A'):.4f}")
            
            _logger.info(f"SHD - MVPC Prior: {prior_result.get('shd', 'N/A')}")
            _logger.info(f"SHD - MissDAG Only: {no_prior_result.get('shd', 'N/A')}")
            
            for prior_weight in args.prior_weights:
                if prior_weight in prior_weight_results:
                    result = prior_weight_results[prior_weight]
                    _logger.info(f"SHD - MissDAG + Prior (w={prior_weight}): {result.get('shd', 'N/A')}")
            
            # Calculate improvements for each weight
            _logger.info("=== IMPROVEMENTS ===")
            for prior_weight in args.prior_weights:
                if prior_weight in prior_weight_results:
                    result = prior_weight_results[prior_weight]
                    f1_improvement = result.get('F1', 0) - no_prior_result.get('F1', 0)
                    shd_improvement = no_prior_result.get('shd', 0) - result.get('shd', 0)
                    _logger.info(f"Weight {prior_weight} - F1 Improvement: {f1_improvement:+.4f}, SHD Improvement: {shd_improvement:+d}")
            
            # Only save training log (no result files)
            _logger.info("Results logged to training.log (no files saved)")
            
        else:
            # Single result evaluation
            raw_result = MetricsDAG(B_processed_bin, dataset.B_bin).metrics
            _logger.info("run result:{0}".format(raw_result))
            
            if getattr(args, 'use_mvpc_data', False):
                _, prior_bin = postprocess(dataset.prior_matrix, args.graph_thres)
                prior_result = MetricsDAG(prior_bin, dataset.B_bin).metrics
                _logger.info("MVPC prior result:{0}".format(prior_result))
    else:
        _logger.warning("No valid ground truth available for evaluation.")
    
    _logger.info(f"Training log saved to {output_dir}/training.log")

if __name__ == '__main__':
    main()