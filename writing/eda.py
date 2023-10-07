# import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_logs = pd.read_csv('data/train_logs.csv')
train_scores = pd.read_csv('data/train_scores.csv')

train = train_logs.merge(train_scores, how="left", on="id")

le = LabelEncoder()
for column in train.columns:
    if train[column].dtype != int:
        train[column] = le.fit_transform(train[column])

train.to_csv('data/train.csv')
