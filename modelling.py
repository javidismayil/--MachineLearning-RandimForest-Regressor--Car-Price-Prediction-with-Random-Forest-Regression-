# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:17:37 2023

@author: Javid Ismayilov
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# load datset
csv_path='D:/JAVID_ISMAYILOV_PROF/MB/PYL/proje/car_price_prediction.csv'
df = pd.read_csv(csv_path)


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

###############Train test spilt
start_time = time.time()
y=model_df['Price']
X=model_df.loc[:, model_df.columns != 'Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##################################
rf_regressor = RandomForestRegressor( n_estimators=10, criterion='squared_error',
                                     random_state=1, n_jobs=-1)
rf_regressor.fit(X_train, y_train)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
y_train_pred=rf_regressor.predict(X_train)
y_test_pred=rf_regressor.predict(X_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f"Mean Squared Error: {train_mse}")
print(f"R-squared: {train_r2}")
print(f"Mean Squared Error: {test_mse}")
print(f"R-squared: {test_r2}")

################################
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')
r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')
###################
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
ax1.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white', label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o',
            edgecolor='white', label='Training data')
ax1.set_ylabel('Residuals')
for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,
              color='black', lw=2)
plt.tight_layout()
plt.show()


###################################

