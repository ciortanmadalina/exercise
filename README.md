## Problem Description

The problem we are trying to solve is performing predictive maintenance on multivariate timeseries data. 
Predictive maintenance involves analyzing historical data to predict when a machine or equipment is likely to fail, allowing for proactive maintenance to be performed.

## Solution Overview
- It is valuable to predict failure as soon as possible
- Maintenance always follows failure, for simplification it is discarded 
- Goal: predict transition normal -> broken

### Two approaches 
1. Classification: n timesteps before failure= 1, normal = 0 
- XGBoost classification (see xgboost.ipynb)
- Neural networks (see lstm.ipynb)

2. Regression: time to failure (see lstm.ipynb)
- XGBoost regression (see xgboost.ipynb)


## Getting Started

To get started with our solution, follow these steps:

1. Clone the repository: `https://github.com/ciortanmadalina/exercise.git`

2. Install the required dependencies: `pip install -r requirements.txt`

3. Run the relevant noteboos: xgboost.ipynb, lstm.ipynb


### Next steps
- Perform cross validation
- Train/test splits alternatives
- Benchmark the NN models
- Use transformer models
- Unsupervised anomaly detection
- Root cause analysis