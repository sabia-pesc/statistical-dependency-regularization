# Statistical Dependency Regularization

Complete experimental source code and results for the paper "On the redlining effect and regularization approaches for fair classification", submitted to Expert Systems with Applications.

## Index Terms

- Fairness
- Fair Machine Learning
- Redlining effect
- Regularization
- Correlation Coefficients

## Abstract

Ensuring fairness in machine learning models is a fundamental challenge, particularly because of the redlining effect, in which proxy variables highly correlated with sensitive attributes unintentionally reproduce bias. Neural Network fairness-aware strategies often fail to explicitly penalize such correlations, applying dropout on hidden layers, which proves insufficient for controlling proxy discrimination.
This study introduces the Statistical Dependency Regularization (SDR), a novel in-processing regularization approach for Multi-Layer Perceptrons (MLPs) that penalizes input-layer weights proportionally to the correlation between features and sensitive attributes. 
Dependence is primarily quantified through Chatterjee’s correlation coefficient, which is robust to noise and capable of capturing complex, nonlinear, and non-monotonic relationships. 
The SDR was evaluated across four widely adopted fairness benchmarks (Adult Income, German Credit, Bank Marketing, and COMPAS Recidivism), optimizing six combined objectives involving Accuracy or the Matthews Correlation Coefficient with widely adopted fairness metrics (Statistical Parity, Equal Opportunity, and Equalized Odds). Experiments across multiple benchmark datasets and objective metrics, including hyperparameter search and resampling evaluations, indicate that SDR consistently outperforms MLPs guided by cross-entropy loss with conventional L2 regularization, and correlation alternatives (Pearson’s, Spearman’s, and Kendall’s). 
Moreover, SDR presents competitive results on fair classification problems, particularly excelling on COMPAS, the dataset with the strongest redlining bias.
Hence, by directly penalizing proxy effects, SDR provides a principled and effective contribution to fairness-aware neural networks, showing substantial potential for real-world deployment in sensitive decision-making systems.
## Installation

### Prerequisites
- Python 3.11
- Conda (Miniconda or Anaconda)

### Setup Instructions

1. **Run the setup script:**
   The setup script will automatically create the conda environment, configure PYTHONPATH, and download all required datasets:
   ```bash
   python setup.py
   ```

2. **Activate the environment:**
   ```bash
   conda activate StatisticalDependencyRegularization
   ```

3. **Install xicorrelation (required for Chatterjee's correlation coefficient):**
   ```bash
   pip install xicorrelation
   ```
   **Note:** This package is not included in `environment.yml` due to NumPy 2.0 compatibility requirements and must be installed separately after environment creation.

**What the setup script does:**
- Creates the conda environment from `environment.yml`
- Configures the project path in PYTHONPATH
- Downloads the required AIF360 datasets (Adult Income, German Credit, Bank Marketing, COMPAS)

### Key Dependencies
- **NumPy 1.26.2**: Compatible with xicorrelation
- **TensorFlow 2.15.0**: Deep learning framework
- **PyTorch 2.1.2**: Alternative deep learning framework
- **scikit-learn 1.3.2**: Machine learning utilities
- **AIF360 0.5.0**: Fairness metrics and datasets
- **Fairlearn 0.10.0**: Additional fairness tools
- **xicorrelation 0.3.0**: Chatterjee's correlation coefficient
- **Optuna 3.5.0**: Hyperparameter optimization
- **Ray 2.8.1**: Distributed computing

## Usage

### Running Experiments

The main entry point for running experiments is `hyperopt.py`, which performs hyperparameter optimization using HyperBand-TPE. The script runs experiments by iterating over combinations of datasets, fitness rules, and model initializers defined at the end of the file.

#### Configuring Experiments

To configure which experiments to run, edit the lists at the end of `hyperopt.py` (lines ~641-675):

**1. Select datasets** (uncomment the ones you want to use):
```python
datasets = [
    adult_dataset_reader,
    bank_dataset_reader,
    compas_dataset_reader,
    german_dataset_reader
]
```

**2. Select fitness rules** (combined objectives):
```python
rules = [
    mcc_parity,      # MCC + Statistical Parity Difference
    mcc_odds,        # MCC + Average Odds Difference
    mcc_opportunity, # MCC + Equal Opportunity Difference
    acc_parity,      # Accuracy + Statistical Parity Difference
    acc_odds,        # Accuracy + Average Odds Difference
    acc_opportunity  # Accuracy + Equal Opportunity Difference
]
```

**3. Select model initializers** (uncomment the methods you want to test):
```python
methods = [
    # Baseline methods
    #simple_mlp_initializer,           # MLP without regularization
    #mlp_standard_l2_initializer,      # MLP with standard L2 regularization
    
    # Correlation-based regularizers
    #mlp_preg_initializer,             # MLP with Pearson correlation penalty
    #mlp_sreg_initializer,             # MLP with Spearman correlation penalty
    #mlp_kreg_initializer,             # MLP with Kendall correlation penalty
    #mlp_xi_reg_initializer,           # MLP with Chatterjee (ξ) correlation penalty (SDR)
    
    # Fair Transition Loss variants
    #ftl_mlp_initializer,              # FTL-MLP without regularization
    #ftl_mlp_preg_initializer,         # FTL-MLP with Pearson penalty
    #ftl_mlp_sreg_initializer,         # FTL-MLP with Spearman penalty
    #ftl_mlp_xi_reg_initializer,       # FTL-MLP with Chatterjee penalty (SDR+FTL)
]
```

#### Running the Script

After configuring the desired combinations, simply run:
```bash
python hyperopt.py
```

The script will:
- Run HyperBand-TPE optimization for each combination
- Save results to `raw_results/results_TIMESTAMP.csv`
- Print best metrics for each experiment

#### Example Configuration

**To run SDR on all datasets with all fitness rules:**
```python
datasets = [
    adult_dataset_reader,
    bank_dataset_reader,
    compas_dataset_reader,
    german_dataset_reader
]

rules = [
    mcc_parity,
    mcc_odds,
    mcc_opportunity,
    acc_parity,
    acc_odds,
    acc_opportunity
]

methods = [
    mlp_xi_reg_initializer,  # SDR with Chatterjee only
]
```

**To compare SDR vs. L2 regularization on COMPAS:**
```python
datasets = [
    compas_dataset_reader
]

rules = [
    acc_parity,
]

methods = [
    mlp_standard_l2_initializer,
    mlp_xi_reg_initializer,
]
```

### Main Scripts

#### `hyperopt.py`
Main script for hyperparameter optimization and model training:
- Implements HyperBand-TPE search strategy
- Supports multiple datasets, models, and regularization approaches
- Performs cross-validation and resampling evaluations
- Saves results to CSV files in the appropriate directories
- Configurable search space for learning rate, regularization strength, network architecture, etc.

#### `models.py`
Contains model implementations:
- **MLP**: Standard Multi-Layer Perceptron with configurable architecture
- **FTL_MLP**: MLP with Fair Transition Loss integration
- **Regularizers**: Implementation of correlation-based penalties (Pearson, Spearman, Kendall, Chatterjee)
- **SDR (Statistical Dependency Regularization)**: Input-layer weight penalty proportional to feature-sensitive attribute correlation

#### `dataset_readers.py`
Dataset loading and preprocessing utilities:
- Standardized interface for AIF360 datasets
- Feature engineering and encoding
- Train/test splitting with stratification
- Sensitive attribute handling

#### `eval_grid.py`
Evaluation utilities for grid search and performance metrics:
- Fairness metrics computation (Statistical Parity, Equal Opportunity, Equalized Odds)
- Combined objective functions
- Result aggregation and statistical analysis

#### `fitness_rules.py`
Defines fitness functions and optimization objectives for multi-objective optimization.

### Customizing Hyperparameter Search

Each model initializer function defines its own hyperparameter search space using Optuna's `suggest_*` methods. To customize the search space for a specific model:

**1. Number of optimization trials** (line ~21):
```python
N_TRIALS = 100  # Increase for more thorough search
```

**2. Model-specific hyperparameters** (in each initializer function):

Example for SDR (`mlp_xi_reg_initializer`):
```python
def mlp_xi_reg_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None):
    hidden_sizes = [100, 100]
    corr_type = 'xi'
    if type(hyperparameters) is not dict:
        # Customize these ranges
        lambda_reg = hyperparameters.suggest_float('lambda_reg', 0.0, 1.0)
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
    else:
        lambda_reg = hyperparameters['lambda_reg']
        dropout = hyperparameters['dropout']
    
    model = SimpleMLP(
        sensitive_attr=sens_attr,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        batch_size=64,
        corr_type=corr_type,
        lambda_reg=lambda_reg
    )
    return model
```

**Common hyperparameters to adjust:**
- `lambda_reg`: Regularization strength (typically 0.0 to 1.0)
- `dropout`: Dropout rate (typically 0.0 to 0.5)
- `hidden_sizes`: Network architecture (e.g., [100, 100], [64, 32])
- `batch_size`: Training batch size
- Learning rate is handled internally by the model

## Project Structure

### Notebooks

The `notebooks/` directory contains Jupyter notebooks organized into three main categories:

#### `dataset_metrics/`
Contains exploratory data analysis and statistical characterization of the fairness benchmarks:
- **Dataset statistics**: Distribution analysis, feature correlations, and class imbalance metrics
- **Redlining effect quantification**: Correlation analysis between features and sensitive attributes using multiple correlation coefficients (Pearson, Spearman, Kendall, Chatterjee)
- **Bias assessment**: Measurement of proxy discrimination and fairness metrics across datasets

#### `main_results/`
Contains the primary experimental results and analysis for the paper:
- **Performance comparisons**: SDR vs. baseline methods across all datasets and fairness metrics
- **Hyperparameter optimization results**: HyperBand-TPE search outcomes and optimal configurations
- **Statistical analysis**: Significance tests, confidence intervals, and resampling evaluations
- **Visualization**: Performance plots, correlation heatmaps, and trade-off curves
- **Chatterjee correlation analysis** (`chatterjee.ipynb`): Detailed comparison of correlation coefficients and their properties

#### `tradeoff_results/`
Contains analysis of accuracy-fairness trade-offs:
- **Pareto frontier analysis**: Multi-objective optimization results showing the trade-off between accuracy/MCC and fairness metrics
- **Regularization strength impact**: Effect of different penalty weights on model performance
- **Comparative analysis**: Trade-off curves for SDR vs. alternative regularization approaches
- **Per-dataset trade-offs**: Detailed analysis for Adult Income, German Credit, Bank Marketing, and COMPAS datasets

