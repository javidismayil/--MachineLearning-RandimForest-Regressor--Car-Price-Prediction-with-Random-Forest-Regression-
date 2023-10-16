#required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
# load datset
csv_path='D:/JAVID_ISMAYILOV_PROF/MB/PYL/proje/car_price_prediction.csv'
df = pd.read_csv(csv_path)

df.head(10)
df.shape
df.columns
df.isna().sum()
filtered_values = np.where((df['Price']<100000) & (df['Price']> 1000))
re_df=df.loc[filtered_values].copy()
deleted_column = ['ID']
re_df = re_df.drop(columns=deleted_column)
re_df['Mileage'] = re_df['Mileage'].str.replace(' km', '').astype(int)
re_df.isna().sum()
re_df=re_df[re_df['Levy']!='-']
re_df['Levy']=re_df['Levy'].astype(int)
re_df.shape
df['Doors'].unique()
##dummy variable
# Create a mapping of engine volumes to ordinal categories
model_df=re_df.copy()
engine_volume=sorted(re_df['Engine volume'].unique())
engine_mapping = {value: idx for idx, value in enumerate(engine_volume)}
model_df['Engine volume'] = re_df['Engine volume'].map(engine_mapping)
inv_engine_mapping = {v: k for k, v in engine_mapping.items()}
model_df['Engine volume'].map(inv_engine_mapping)
model_df=pd.get_dummies(model_df)
