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
    """load mvpc dataset and prior matrix"""
    
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
        
        return data, prior_matrix
    
    def convert_prior_for_missdag(self, prior_matrix, method='normalize'):
        """convert MVPC prior to MissDAG format"""
        if method == 'normalize':
            # normalization
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
    import glob
    
    subfolder_path = os.path.join(base_dir, subfolder_name)
    
    if not os.path.exists(subfolder_path):
        raise ValueError(f"Subfolder not found: {subfolder_path}")
    
    data_pattern = os.path.join(subfolder_path, "s*_*ar_v*_e*_m*_seed*.npy")
    data_files = glob.glob(data_pattern)
    if not data_files:
        data_pattern_csv = os.path.join(subfolder_path, "s*_*ar_v*_e*_m*_seed*.csv")
        data_files = glob.glob(data_pattern_csv)
    
    if not data_files:
        raise ValueError(f"Data file not found in {subfolder_path} with pattern s*_*ar_v*_e*_m*_seed*")
    data_path = data_files[0]
    
    prior_path = os.path.join(subfolder_path, "res_mvpc_permc_adj.npy")
    if not os.path.exists(prior_path):
        raise ValueError(f"Prior file not found: {prior_path}")
    
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
    args = get_args()
    
    # ===== modify MVPC settings here =====
    args.use_mvpc_data = True  # use MVPC dataset
    args.mvpc_subfolder = 's100_mar_v20_e5_m10_seed777'  # change dataset
    args.mvpc_base_dir = 'result_npy'
    args.prior_weight = 0.5  # prior weight
    args.prior_conversion = 'normalize'  # prior conversion method
    # =========================================
    
    args.dag_method_type = 'notears'
    args.miss_method_type = 'miss_dag_gaussian'
    args.em_iter = 20
    args.lambda_1_ev = 0.1
    args.graph_thres = 0.3
    
    print(f"Using DAG method: {args.dag_method_type}")
    print(f"Using miss method: {args.miss_method_type}")
    print(f"EM iterations: {args.em_iter}")
    
    data_path, prior_path, ground_truth_path = get_mvpc_file_paths(
        args.mvpc_base_dir, args.mvpc_subfolder
    )
    
    args.mvpc_data_path = data_path
    args.mvpc_prior_path = prior_path
    args.mvpc_ground_truth_path = ground_truth_path
    
    return args

class DAGMethodWithPrior:
    
    def __init__(self, base_method, prior_matrix, prior_weight=1.0):
        self.base_method = base_method
        self.prior_matrix = prior_matrix
        self.prior_weight = prior_weight
        
    def fit(self, X, cov_emp=None):
        # If base_method supports prior，transfer directly
        if hasattr(self.base_method, 'fit_with_prior'):
            return self.base_method.fit_with_prior(X, self.prior_matrix, self.prior_weight, cov_emp)
        else:
            return self.fit_with_prior_regularization(X, cov_emp)
    
    def fit_with_prior_regularization(self, X, cov_emp=None):
        if hasattr(self.base_method, '_original_loss'):
            original_loss = self.base_method._original_loss
        else:
            original_loss = getattr(self.base_method, 'compute_loss', None)
        
        def loss_with_prior(W):
            if original_loss:
                data_loss = original_loss(W)
            else:
                data_loss = np.sum((X - X @ W.T) ** 2)
            
            prior_loss = np.sum((W - self.prior_matrix) ** 2)
            
            total_loss = data_loss + self.prior_weight * prior_loss
            return total_loss
        
        if hasattr(self.base_method, 'compute_loss'):
            self.base_method._original_loss = self.base_method.compute_loss
            self.base_method.compute_loss = loss_with_prior
        
        result = self.base_method.fit(X, cov_emp)
        
        if hasattr(self.base_method, '_original_loss'):
            self.base_method.compute_loss = self.base_method._original_loss
            delattr(self.base_method, '_original_loss')
        
        return result

def miss_dag_with_prior(X, mask, dag_method, prior_matrix, prior_weight, em_iter, equal_variances):
    dag_method_with_prior = DAGMethodWithPrior(dag_method, prior_matrix, prior_weight)
    
    return miss_dag_gaussian(X, mask, dag_method_with_prior, em_iter, equal_variances)

def main():
    args = get_modified_args()

    # Setup for logging
    output_dir = 'output/{}'.format(get_datetime_str(add_random_str=True))
    create_dir(output_dir)  # Create directory to save log files and outputs
    setup_logger(log_path='{}/training.log'.format(output_dir), level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger.")

    # Get and save system info
    system_info = get_system_info()
    if system_info is not None:
        save_yaml_config(system_info, path='{}/system_info.yaml'.format(output_dir))

    # Save configs
    save_yaml_config(vars(args), path='{}/config.yaml'.format(output_dir))

    # Reproducibility
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
    print(f"Setting up DAG method: {args.dag_method_type}")  # 调试信息
    
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

    if hasattr(dataset, 'X'):
        has_missing = np.isnan(dataset.X).any()
    else:
        has_missing = args.miss_percent > 0
    
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

        # Estimate the DAG
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
        if getattr(args, 'use_mvpc_data', False):
            _logger.info(f"Running MissDAG with MVPC prior (weight: {args.prior_weight})")
            B_est, _, _ = miss_dag_with_prior(
                dataset.X, dataset.mask, dag_method,
                dataset.converted_prior, args.prior_weight,
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

    # Post-process estimated solution
    _, B_processed_bin = postprocess(B_est, args.graph_thres)
    _logger.info("Finished post-processing the estimated graph.")

    if hasattr(dataset, 'B_bin') and dataset.B_bin.size > 0 and np.any(dataset.B_bin != 0):
        raw_result = MetricsDAG(B_processed_bin, dataset.B_bin).metrics
        _logger.info("run result:{0}".format(raw_result))
        
        if getattr(args, 'use_mvpc_data', False):
            _, prior_bin = postprocess(dataset.prior_matrix, args.graph_thres)
            prior_result = MetricsDAG(prior_bin, dataset.B_bin).metrics
            _logger.info("MVPC prior result:{0}".format(prior_result))
    else:
        _logger.warning("No valid ground truth available for evaluation.")
        
    np.save(os.path.join(output_dir, 'B_estimated.npy'), B_est)
    np.save(os.path.join(output_dir, 'B_processed.npy'), B_processed_bin)
    if getattr(args, 'use_mvpc_data', False):
        np.save(os.path.join(output_dir, 'mvpc_prior.npy'), dataset.prior_matrix)
        np.save(os.path.join(output_dir, 'mvpc_prior_converted.npy'), dataset.converted_prior)
        if hasattr(dataset, 'B_bin'):
            np.save(os.path.join(output_dir, 'ground_truth.npy'), dataset.B_bin)
    
    _logger.info(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()