import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# load datset
csv_path='D:/JAVID_ISMAYILOV_PROF/MB/PYL/proje/car_price_prediction.csv'
df = pd.read_csv(csv_path)
df.head(10)
df.shape
df.columns
df.isna().sum()
filtered_values = np.where((df['Price']<100000) & (df['Price']> 1000))
re_df=df.loc[filtered_values].copy()
deleted_column = ['ID', 'Doors']
re_df = re_df.drop(columns=deleted_column)
re_df['Mileage'] = re_df['Mileage'].str.replace(' km', '').astype(int)
re_df=re_df[re_df['Levy']!='-']
re_df['Levy']=re_df['Levy'].astype(int)

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
models_chart_gear=pie_chart_funct(re_df['Gear box type'], thereshold=0.02)
plt.figure(figsize=(3, 3))
plt.pie(models_chart_gear[0], labels=models_chart_gear[1],  startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.show()
#plot_bar
re_df['Leather interior'].value_counts().plot.bar()
re_df['Category'].value_counts().plot.barh()
re_df['Fuel type'].value_counts().plot.barh()
re_df['Gear box type'].value_counts().plot.barh()
#barplot
sns.barplot(x='Fuel type', y='Price', hue='Gear box type', data=re_df)
