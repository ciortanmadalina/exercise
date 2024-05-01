
from collections import Counter
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import random
import json
from sklearn.metrics import mean_absolute_error



def preprocess(df):
    """
    Preprocess the data
    
    This function takes a DataFrame `df` as input and performs preprocessing on the data. The preprocessing steps include:
    - Clipping extreme values for each column based on the specified minimum and maximum values in the `preprocessing.json` file.
    - Scaling each column using min-max scaling.
    - Imputing missing values using forward fill and backward fill.
    
    Args:
        df (pandas.DataFrame): The input DataFrame to be preprocessed.
        
    Returns:
        tuple: A tuple containing the preprocessed DataFrame and the list of columns used for preprocessing.
    """

    df = df.copy()
    # To load the data back from the JSON file:
    with open('preprocessing.json', 'r') as f:
        meta = json.load(f)
    cols = np.array(list(meta["features"].keys()))
    # handle extreme values by clipping
    for c in cols:
        df[c] = df[c].clip(meta["features"][c]["min"], meta["features"][c]["max"])
        # min max scaling
        df[c] = (df[c] - meta["features"][c]["min"]) / (meta["features"][c]["max"] - meta["features"][c]["min"])
        # imputation
        df[c] = df[c].ffill().bfill()

    df['idx'] = np.arange(df.shape[0])
    df['y'] = df["machine_status"].map(meta["target"])
    return df, cols



def classification(X, y_classif, random_state=1):
    """Train a classification model using XGBoost.
    
    Parameters:
    X (array-like): The input features.
    y_classif (array-like): The target variable for classification.
    random_state (int, optional): The random seed for reproducibility. Defaults to 1.

    """
    idx_train, idx_test, Y_train, Y_test = train_test_split(np.arange(len(y_classif)), 
    y_classif, test_size=0.3, random_state=random_state, stratify=y_classif)

    # balance the classes by undersampling the majority class
    c0_idx_train = idx_train[np.where(Y_train == 1)[0]] # minority class
    c1_idx_train = idx_train[random.sample(list(np.where(Y_train == 0)[0]), int(len(c0_idx_train) *1.5))] # majority class
    idx_train = np.concatenate([c0_idx_train, c1_idx_train])

    class_weights = len(idx_train) / np.bincount(y_classif[idx_train])
    class_weights[0] = 2* class_weights[0] # reinforce the weight of the minority class
  
    train_weights = [class_weights[i] for i in y_classif[idx_train]]
    model = XGBClassifier()

    model.fit(X[idx_train].reshape(len(idx_train), -1), y_classif[idx_train],
    sample_weight=train_weights);

    preds = model.predict(X[idx_test].reshape(len(idx_test), -1))
    true = y_classif[idx_test]

    f1 = f1_score(true, preds)
    cm = confusion_matrix(true, preds)

    return model, f1, cm



def regression(X, y_reg, y_classif, random_state=1):
    """Train a regression model using XGBoost.

    Parameters:
    X (array-like): The input features.
    y_reg (array-like): The target variable for regression.
    y_classif (array-like): The target variable for classification.
    random_state (int, optional): The random seed for reproducibility. Defaults to 1.

    Returns:
    tuple: A tuple containing the trained model and the mean absolute error.

    """
    idx_train, idx_test, _, _ = train_test_split(np.arange(len(y_classif)), 
    y_classif, test_size=0.3, random_state=random_state, stratify=y_classif)

    dtrain = xgb.DMatrix(X[idx_train].reshape(len(idx_train), -1), label=y_reg[idx_train])
    dtest = xgb.DMatrix(X[idx_test].reshape(len(idx_test), -1), label=y_reg[idx_test])

    # Define the parameters
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'objective': 'reg:squarederror',  # error function for regression
        'eval_metric': 'rmse'  # evaluation metric for regression
    }
    num_round = 10  # the number of training iterations

    # Train the model
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    true = y_reg[idx_test]
    mae = mean_absolute_error(true, preds)
    return bst, mae



def prepare_dataset(df, cols, window_size, prediction_horizon, autocorr_window=100, incident_horizon=50000):
    """
    Prepare the dataset for training.

    Args:
        df (pandas.DataFrame): The input dataframe containing the dataset.
        cols (list): The list of column names to include in the dataset.
        window_size (int): The size of the sliding window for creating input sequences.
        prediction_horizon (int): The time horizon for the prediction task.
        autocorr_window (int, optional): The spacing between samples in the dataset. Defaults to 100.
        incident_horizon (int, optional): The maximum survival time for creating a smaller dataset. Defaults to 50000.

    Returns:
        tuple: A tuple containing the following arrays:
            - X (numpy.ndarray): The input features of shape (num_samples, num_features, window_size).
            - y_reg (numpy.ndarray): The regression target values of shape (num_samples,).
            - y_classif (numpy.ndarray): The classification target values of shape (num_samples,).
            - incident_ref (numpy.ndarray): The incident numbers corresponding to each sample of shape (num_samples,).
            - idx_ref (numpy.ndarray): The indices corresponding to each sample of shape (num_samples,).

    """
    X = []
    y_reg = []
    incident_ref = []
    idx_ref = []

    # exclude idx to ensure we have non overlapping sliding windows across failures 
    # (e.g. addind samples from failure k-1 to failure k)
    exclude_idx = []
    last_samples = df.groupby("incident_nb").agg({"idx":"min"}).values.reshape(-1)
    for _ in range(window_size - 1):
        exclude_idx.append(last_samples)
        last_samples = last_samples+1
    if len(exclude_idx)>0:
        exclude_idx = np.concatenate(exclude_idx)

    # choose ids for the samples in the dataset that can have a non overlapping size of window_size 
    # and are spaced by autocorr_window timesteps
    idx = df[(df["survival"] <= incident_horizon) & # create smaller dataset for faster training
             (df["survival"]%window_size ==0) & # avoid overlapping windows
             (df["survival"]%autocorr_window ==0) &  # spacing between samples
             (~df["idx"].isin(exclude_idx))]["idx"].values
    for window in range(window_size):
        df_= df[df["idx"].isin(idx)].sort_values(by = "idx", ascending = True)
        X.append(df_[cols].values)
        if window == 0:
            y_reg = df_["survival"].values
            incident_ref = df_["incident_nb"].values
            idx_ref = df_["idx"].values
        idx = idx -1

    # create classification target
    df["y_temp"] = (df["survival"]<prediction_horizon).astype(int)


    X = np.array(X).transpose(1,2,0)
    y_reg = np.array(y_reg)
    incident_ref = np.array(incident_ref)
    idx_ref = np.array(idx_ref)
    y_classif = df.iloc[idx_ref]["y_temp"].values
    return X, y_reg, y_classif, incident_ref, idx_ref
