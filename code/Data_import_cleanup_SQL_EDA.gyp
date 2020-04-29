# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os
import sys
import pandas as pd
import numpy as np
import scipy
import datetime
import statistics as stats
import missingno as msno

import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
from sklearn.preprocessing import StandardScaler

from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
import pandas.io.sql as pd_sql

#!jt -r
get_ipython().system('jt -t monokai -f fira -fs 13 -nf ptsans -nfs 11 -N -kl -cursw 5 -cursc r -cellw 95% -T')

# %% [markdown]
# ## Creating Reference to Project Directories

# %%
project_dir = str(os.path.dirname((os.path.abspath(''))))
sys.path.append(project_dir)
print(project_dir)
figures_folder = project_dir + '/Images/'
base_path = project_dir + '/data/'
file_name = 'framingham.csv'
print(base_path + file_name)

# %% [markdown]
# ## Data Import

# %%
def data_import(file_name):
    """
    doc string for data import
    """
    file = base_path + file_name
    df = pd.read_csv(file, delimiter=',')
    df.info()
    print(df.shape)
    df.head()
    df.describe()
    return df


# %%
def feature_select(df, cols):
    df['MAP']= ((2*df['diaBP'])+df['sysBP'])/3
    df = df[cols]
    return df


# %%
file_name = 'framingham.csv'
df = data_import(file_name)

# %% [markdown]
# ## Inspective data for Null-Values

# %%
msno.matrix(df.sample(4238))
file_path = figures_folder + 'EDA_NullVal_Dist.png'
plt.savefig(file_path, format='png')
df.isnull().sum()
#df = df.dropna()


# %%
mask1 = df.glucose.isnull()
df_glu_null = df[mask1]
print(df_glu_null.shape)
mask2 = df_glu_null.diabetes == 1
df_glu_null_dia_1 = df_glu_null[mask2]
df_glu_null_dia_1
print(f'The mean BMI of people with null glucose value is ',df_glu_null.BMI.mean())
##

# %% [markdown]
# Of the 388 records, where glucose was null - only 4 had diabetes. Also the mean BMI value of the people who had null for their glucose was in the normal BMI range. It can be safely assumed that glucose tests were not done as there was no need, since they did not have diabetes. So we replaced the cases where glucose was null and diabetes was 0, with the mean glucose values.

# %%
mask3 = df.diabetes == 0
df.loc[df['glucose'].isnull() & mask3, 'glucose'] = df.glucose.mean()
mask4 = df.glucose.isnull()
df_glu_null = df[mask4]
print(f'number of records with null glucose',df_glu_null.shape[0])


# %%
mask5 = df.totChol.isnull()
df_chol_null = df[mask5]
print(f'number of records with null chol', df_chol_null.shape[0])
mask6 = df_chol_null.TenYearCHD == 0 
df_chol_null_noCHD = df_chol_null[mask6]
df_noCHD = df[df['TenYearCHD']==0]
print(f'number of records with null chol and no CHD is ',df_chol_null_noCHD.shape[0])
df.loc[df['totChol'].isnull() & mask6, 'totChol'] = df_noCHD.totChol.mean()
print(df.totChol.mean())
print(df_noCHD.totChol.mean())
#print(f'number of records with null Chol',df.totChol.shape[0])


# %%
df.loc[df['education'].isnull(), 'education'] = df.education.median()
mask7 = df.totChol > df.totChol.mean()
df.loc[df['BMI'].isnull() & mask7, 'BMI'] = df_noCHD.totChol.mean()
df = df.dropna()
df.isnull().sum()


# %%
df.head()

# %% [markdown]
# ## Sanity Checks
# %% [markdown]
# ### Exploring relations between features and explicit feature effects on target.

# %%
df.groupby(['TenYearCHD']).mean()


# %%
df.groupby(['prevalentHyp']).mean()


# %%
df.groupby(['currentSmoker']).mean()


# %%
df.groupby(['education']).mean()


# %%
df.groupby(pd.cut(df['diaBP'], 4)).mean()


# %%
df.groupby(pd.cut(df['sysBP'], 4)).mean()

# %% [markdown]
# ### Exploring relations between systolic and diastolic BP

# %%
sys_cat =pd.cut(df['sysBP'], 8)
sys_cat = sys_cat.to_frame()
sys_cat.columns = ['sys_cat']
df_new = pd.concat([df, sys_cat], axis=1)
pd.crosstab(df_new.sys_cat, df_new.TenYearCHD.astype(bool)).plot(kind='bar')
plt.title('Systolic BP vs CHD')
plt.xlabel('Systolic BP')
plt.ylabel('Frequency')


# %%
dia_cat =pd.cut(df['diaBP'], 8)
dia_cat = dia_cat.to_frame()
dia_cat.columns = ['dia_cat']
df_new = pd.concat([df, dia_cat], axis=1)
pd.crosstab(df_new.dia_cat, df_new.TenYearCHD.astype(bool)).plot(kind='bar')
plt.title('Diastolic BP vs CHD')
plt.xlabel('Diastolic BP')
plt.ylabel('Frequency')

# %% [markdown]
# ### Distribution of features

# %%
fig, ax = plt.subplots(figsize = (18, 18))
for i, col in enumerate(df.columns[1:]):
    plt.subplot(4, 4, i+1)
    sns.distplot(df.iloc[:,i],kde=False,ax=plt.gca(), color='blue')
    #plt.title(df.columns[i])
    plt.axis('on')
plt.tight_layout()


# %%
col_cont = [ 'age', 'education', 'totChol', 'sysBP','diaBP', 'BMI', 'heartRate', 'glucose', 'cigsPerDay']
col_disc = ['TenYearCHD', 'male', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

# %% [markdown]
# ### Scaling Features

# %%
df_disc = df[col_disc]
df_cont = df[col_cont]
std = StandardScaler()
std.fit_transform(df_cont.values)
df_cont = std.transform(df_cont.values)
df_cont = pd.DataFrame(df_cont)
df_cont.columns = col_cont
df_cont.head()


# %%
scaled_df = pd.concat([df_disc, df_cont], axis=1, sort=False)
scaled_df.head()
scaled_df = df.dropna()
scaled_df.shape

# %% [markdown]
# ### Correlation between Features

# %%
from yellowbrick.features import Rank2D

# Modify the figure size
fig, ax=plt.subplots(figsize=(14,7))

# Instantiate the Rank2D object with default arguments: visualizer
visualizer = Rank2D(algorithm="pearson")

# fit the visualizer
visualizer.fit_transform(df)
# Plot the visualizer with .poof() method
visualizer.poof()


# %%
# Modify the figure size
fig, ax=plt.subplots(figsize=(14,7))
# Instantiate the Rank2D object with default arguments: visualizer
visualizer = Rank2D(algorithm="pearson")
# fit the visualizer
visualizer.fit_transform(scaled_df)
# Plot the visualizer with .poof() method
visualizer.poof()


# %%
corr = scaled_df.corr()
corr.style.background_gradient()


# %%
corr_mask = corr.apply(lambda col: (abs(col)> 0.75) & (abs(col)!=1))
corr_mask.sum()


# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(df.corr(method='pearson'), cmap="seismic", annot=True, vmin=-1, vmax=1);


# %%
sns.pairplot(df, hue='TenYearCHD');

# %% [markdown]
# ## Feature Selection

# %%
df.isnull().sum()
df.columns
cols = ['age', 'totChol', 'MAP', 'BMI', 'cigsPerDay', 'heartRate',                       'glucose', 'male', 'prevalentHyp', 'TenYearCHD']
df = feature_select(df, cols)
df.head()

# %% [markdown]
# ## Upload to SQL on AWS and return as DF

# %%
# Postgres info to connect
params = {
    'host': 'localhost',  # We are connecting to our _local_ version of psql
    'port': 5432,         # port we opened on AWS
    'dbname': 'framingham',
    'user': 'samypalaniappan'
}

# sqlalchemy engine
sql_eng = create_engine(
    'postgresql+psycopg2://',
    connect_args=params
)

# connection
connection = sql_eng.connect()


# %%
def new_table(connection, df, table_name='Data'):
    # copy column headers
    pd_sql.to_sql(df, table_name, connection, if_exists='replace', index=False)


# %%
connection_string = f'postgres://samypalaniappan:{params["host"]}@{params["host"]}:{params["port"]}/framingham'
engine = create_engine(connection_string)
conn= engine.connect()


# %%
new_table(connection=conn, df=df, table_name='Data')


# %%
def psql_to_df(query, params=params, col=cols):
    connection = connect(**params)
    cursor = connection.cursor()
    cursor.execute(query)
    return pd.DataFrame(cursor.fetchall(), columns=col) 


# %%
query = """
SELECT * FROM "Data";
"""
final_df = psql_to_df(query)
final_df.head()


# %%
final_df.to_csv(base_path+"preprocessed_data.csv")


# %%


