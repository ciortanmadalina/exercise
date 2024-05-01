
from collections import Counter
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from umap import UMAP
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix, f1_score
import random
import json
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report


def preprocess(df):
    """
    Preprocess the data
    """
    df = df.copy()
    # To load the data back from the JSON file:
    with open('preprocessing.json', 'r') as f:
        meta = json.load(f)
    cols = np.array(list(meta["features"].keys()))
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
    idx_train, idx_test, Y_train, Y_test = train_test_split(np.arange(len(y_classif)), 
    y_classif, test_size=0.3, random_state=random_state, stratify=y_classif)


    c0_idx_train = idx_train[np.where(Y_train == 1)[0]]
    c1_idx_train = idx_train[random.sample(list(np.where(Y_train == 0)[0]), int(len(c0_idx_train) *1.5))]


    idx_train = np.concatenate([c0_idx_train, c1_idx_train])


    class_weights = len(idx_train) / np.bincount(y_classif[idx_train])

    class_weights[0] = 2* class_weights[0]
  
    train_weights = [class_weights[i] for i in y_classif[idx_train]]
    model = XGBClassifier()
    # print("input data:" , X.shape, X[idx_train].reshape(len(idx_train), -1).shape, y_classif[idx_train].shape)
    model.fit(X[idx_train].reshape(len(idx_train), -1), y_classif[idx_train],
    sample_weight=train_weights);

    preds = model.predict(X[idx_test].reshape(len(idx_test), -1))
    true = y_classif[idx_test]

    f1 = f1_score(true, preds)
    cm = confusion_matrix(true, preds)
    # print('F1: ', f1)
    # print('Confusion Matrix: \n', cm)
    return model, f1, cm



def regression(X, y_reg, y_classif, random_state=1):
    idx_train, idx_test, Y_train, Y_test = train_test_split(np.arange(len(y_classif)), 
    y_classif, test_size=0.3, random_state=random_state, stratify=y_classif)
    # print(Counter(Y_train), Counter(Y_test), len(idx_train), len(idx_test))
    # print("input data:" , X.shape, X[idx_train].reshape(len(idx_train), -1).shape, y_reg[idx_train].shape)
    # Create a DMatrix for more efficiency
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



def prepare_dataset(df, cols, window_size,  prediction_horizon,
                     autocorr_window = 100, incident_horizon = 50000):
    X = []
    y_reg = []
    incident_ref = []
    idx_ref = []

    # exclude idx to ensure we have non overlapping sliding windows across failures (e.g. addind samples from failure k-1 to failure k)
    exclude_idx = []
    last_samples = df.groupby("incident_nb").agg({"idx":"min"}).values.reshape(-1)
    for _ in range(window_size - 1):
        exclude_idx.append(last_samples)
        last_samples = last_samples+1
    if len(exclude_idx)>0:
        exclude_idx = np.concatenate(exclude_idx)


    idx = df[(df["survival"] <= incident_horizon) & 
             (df["survival"]%window_size ==0) &
             (df["survival"]%autocorr_window ==0) &  
             (~df["idx"].isin(exclude_idx))]["idx"].values
    for window in range(window_size):
        df_= df[df["idx"].isin(idx)].sort_values(by = "idx", ascending = True)
        X.append(df_[cols].values)
        if window == 0:
            y_reg = df_["survival"].values
            incident_ref = df_["incident_nb"].values
            idx_ref = df_["idx"].values
        idx = idx -1

    df["y_temp"] = (df["survival"]<prediction_horizon).astype(int)

    X = np.array(X).transpose(1,2,0)
    y_reg = np.array(y_reg)
    incident_ref = np.array(incident_ref)
    idx_ref = np.array(idx_ref)
    y_classif = df.iloc[idx_ref]["y_temp"].values
    return X, y_reg, y_classif, incident_ref, idx_ref
