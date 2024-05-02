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

### Presentation
[Link](https://docs.google.com/presentation/d/1STqvaCWSlB4ZJVTp_rtDA35BcFQ_CCIimBSOsQoPPGU/edit#slide=id.g2d13b986bb9_0_6) to the 10 min presentation of the solution



## Getting Started

To get started with our solution, follow these steps:

1. Clone the repository: `https://github.com/ciortanmadalina/exercise.git`

2. Install the required dependencies: `pip install -r requirements.txt`

3. Save the input csv in a "data" folder at the root of this project

4. Run the relevant noteboos: 
 - preprocessing.ipynb
 - xgboost.ipynb
 - lstm.ipynb


### Next steps
- Perform cross validation
- Train/test splits alternatives
- Benchmark the NN models
- Use transformer models
- Unsupervised anomaly detection
- Root cause analysis