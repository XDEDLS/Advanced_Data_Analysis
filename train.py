import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

df = pd.read_csv("data.csv")

train_x = df[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"]]
df_02 = pd.DataFrame(train_x)

# create model
k_model = KMeans(n_clusters=4, n_init=10)

# data normalization
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)

# train
k_model.fit(train_x)

predict_y = k_model.predict(train_x)

df["Classification"] = predict_y
df = pd.concat([df['Area'], df['Classification']], axis=1)
df.to_csv('result.csv')

