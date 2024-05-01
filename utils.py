import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import json


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