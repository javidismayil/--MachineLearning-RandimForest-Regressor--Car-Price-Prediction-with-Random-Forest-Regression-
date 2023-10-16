# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:42:54 2023

@author: Javid Ismayilov
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.metrics import mean_absolute_error
# load datset
csv_path='D:/JAVID_ISMAYILOV_PROF/MB/PYL/proje/car_price_prediction.csv'
df = pd.read_csv(csv_path)

######data preprocessing
df.head(10)
df.shape
df.columns
df.isna().sum()
filtered_values = np.where((df['Price']<100000) & (df['Price']> 1000))
re_df=df.loc[filtered_values].copy()
deleted_column = ['ID', 'Doors']
re_df = re_df.drop(columns=deleted_column)
re_df['Mileage'] = re_df['Mileage'].str.replace(' km', '').astype(int)
re_df.isna().sum()
re_df=re_df[re_df['Levy']!='-']
re_df['Levy']=re_df['Levy'].astype(int)
re_df.shape

kat_df=re_df.select_dtypes(include=['object'])
t=0
for i in np.arange(0,kat_df.shape[1]):
    a=kat_df.iloc[:,i].value_counts().count()
    b=kat_df.columns
    print(b[i], ':', a)
    t=a+t
kat_df.columns
kat_df.head

#####numeric variable
num_df=re_df.select_dtypes(exclude=['object'])
num_df.describe()
#############################vizualtion

#histogram
plt.hist(re_df['Price'], bins=50, edgecolor='black', alpha=0.8)
plt.show()
plt.hist(re_df['Cylinders'], bins=50, edgecolor='black', alpha=0.8)
plt.show()


#violin plot
plt.figure(figsize=(8, 4))
sns.violinplot(data=re_df['Price'])
plt.title('Violin Plot')
plt.xlabel('Veri')
plt.show()

#pie_chart
def pie_chart_funct(data, thereshold=0.005):
    cat = data.value_counts()
    sizes = cat / cat.sum()
    other_sum=sizes[sizes<thereshold].sum()
    sizes1=sizes[sizes>thereshold]
    if other_sum!=0:
        sizes1.loc['Others']=other_sum
    names=sizes1.index.tolist()
    return sizes1.values, names
models_chart=pie_chart_funct(re_df['Manufacturer'], thereshold=0.02)
plt.figure(figsize=(6, 6))
plt.pie(models_chart[0], labels=models_chart[1],  startangle=90 )
plt.title(' Car Manufacturer')
plt.axis('equal')
plt.show()
models_chart_gear=pie_chart_funct(re_df['Color'], thereshold=0.02)
plt.figure(figsize=(3, 3))
plt.pie(models_chart_gear[0], labels=models_chart_gear[1],  startangle=90)
plt.axis('equal')
plt.show()
#plot_bar
re_df['Leather interior'].value_counts().plot.bar()
plt.title("Leather Interior Distribution")
plt.show()
re_df['Category'].value_counts().plot.barh()
plt.title("Prices for Category")
plt.xlabel("Price")
plt.ylabel("Category")
plt.show()
re_df['Fuel type'].value_counts().plot.barh()
plt.title("Prices for Fuel type")
plt.xlabel("Price")
plt.ylabel("Fuel type")
plt.show()
re_df['Gear box type'].value_counts().plot.barh()
plt.title("Prices for Gear box type")
plt.xlabel("Price")
plt.ylabel("Gear box type")
plt.show()
#barplot
sns.barplot(x='Fuel type', y='Price', hue='Gear box type', data=re_df)
######dummy variable
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

