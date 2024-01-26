import pandas as pd
import numpy as np
from sklearn import linear_model

"""
Author: David Awolowo
Description: calculates the moving regression over a time period
"""

def regression(dataset: pd.DataFrame, column: str) -> tuple:
    linear_regression = linear_model.LinearRegression()
    observations = len(dataset)
    dataset = dataset.loc[:, :].copy()
    dataset['x'] = range(observations)
    X = dataset['x'].values.reshape((observations,1))
    y = dataset[column].values
    linear_regression.fit(X, y)
    del dataset['x']
    return linear_regression.coef_, linear_regression.intercept_

def indicator(dataframe: pd.DataFrame, column="close", period=14) -> tuple:
    slopes = []
    intercepts = []
    start = 0
    for index in range(1, len(dataframe) + 1):
        if index > period:
            start += 1
        slices = dataframe[start:index]
        reg = regression(slices, column)
        slope = reg[0][0]   # Addition [0] to obtain the value since reg[0] is an array
        intercept = reg[1]
        slopes.append(slope)
        intercepts.append(intercept)
    return np.array(slopes), np.array(intercepts)

