

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os,sys
```


```python
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
#from catboost import CatBoostClassifier

#from rgf.sklearn import RGFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
#from ggplot import *
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")
```


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
```


```python
import sklearn
sklearn.__version__
```




    '0.20.1'




```python
pwd()
```




    '/mnt/data/scripts/notebooks/ashwin/test'




```python
os.chdir('flight_predictor_data')
train_df = pd.read_csv('weather_data_train.csv')
test_df = pd.read_csv('weather_data_test.csv')
```


```python
train_df.shape,test_df.shape
```




    ((2183, 1126), (100, 1126))




```python
pred = pd.read_csv('flight_data_train.csv')
```


```python
pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Day_Id</th>
      <th>Spot1 maxDistance</th>
      <th>Spot1 totalDistance</th>
      <th>Spot1 totalFlights</th>
      <th>Spot2 maxDistance</th>
      <th>Spot2 totalDistance</th>
      <th>Spot2 totalFlights</th>
      <th>Spot3 maxDistance</th>
      <th>Spot3 totalDistance</th>
      <th>Spot3 totalFlights</th>
      <th>...</th>
      <th>Spot285 totalFlights</th>
      <th>Spot286 maxDistance</th>
      <th>Spot286 totalDistance</th>
      <th>Spot286 totalFlights</th>
      <th>Spot287 maxDistance</th>
      <th>Spot287 totalDistance</th>
      <th>Spot287 totalFlights</th>
      <th>Spot288 maxDistance</th>
      <th>Spot288 totalDistance</th>
      <th>Spot288 totalFlights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>23.84</td>
      <td>41.82</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>24.40</td>
      <td>48.20</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 865 columns</p>
</div>




```python
train_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Day_Id</th>
      <th>Station1 Dew Point Height1</th>
      <th>Station1 Dew Point Height2</th>
      <th>Station1 Dew Point Height3</th>
      <th>Station1 Dew Point Height4</th>
      <th>Station1 Dew Point Height5</th>
      <th>Station1 Dew Point Height6</th>
      <th>Station1 Dew Point Height7</th>
      <th>Station1 Dew Point Height8</th>
      <th>Station1 Dew Point Height9</th>
      <th>...</th>
      <th>Station5 Wind Speed Height36</th>
      <th>Station5 Wind Speed Height37</th>
      <th>Station5 Wind Speed Height38</th>
      <th>Station5 Wind Speed Height39</th>
      <th>Station5 Wind Speed Height40</th>
      <th>Station5 Wind Speed Height41</th>
      <th>Station5 Wind Speed Height42</th>
      <th>Station5 Wind Speed Height43</th>
      <th>Station5 Wind Speed Height44</th>
      <th>Station5 Wind Speed Height45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>...</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
      <td>2183.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1161.975263</td>
      <td>5.424167</td>
      <td>4.935076</td>
      <td>4.461103</td>
      <td>4.042764</td>
      <td>3.554599</td>
      <td>3.020216</td>
      <td>2.470416</td>
      <td>1.933228</td>
      <td>1.396137</td>
      <td>...</td>
      <td>24.596934</td>
      <td>24.916352</td>
      <td>25.241739</td>
      <td>25.573145</td>
      <td>25.893053</td>
      <td>26.221007</td>
      <td>26.553247</td>
      <td>26.890890</td>
      <td>27.237081</td>
      <td>27.584012</td>
    </tr>
    <tr>
      <th>std</th>
      <td>673.965531</td>
      <td>9.471207</td>
      <td>9.538955</td>
      <td>9.693580</td>
      <td>9.822583</td>
      <td>9.861115</td>
      <td>9.916416</td>
      <td>9.997071</td>
      <td>10.070162</td>
      <td>10.135811</td>
      <td>...</td>
      <td>12.546472</td>
      <td>12.710584</td>
      <td>12.867511</td>
      <td>13.028636</td>
      <td>13.188974</td>
      <td>13.345971</td>
      <td>13.509388</td>
      <td>13.692444</td>
      <td>13.879558</td>
      <td>14.084127</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-112.889831</td>
      <td>-112.759452</td>
      <td>-112.629074</td>
      <td>-112.498696</td>
      <td>-112.368318</td>
      <td>-112.237940</td>
      <td>-112.107562</td>
      <td>-111.977184</td>
      <td>-111.846806</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.009083</td>
      <td>0.246400</td>
      <td>0.103527</td>
      <td>0.332447</td>
      <td>0.198479</td>
      <td>0.096386</td>
      <td>0.147117</td>
      <td>0.058398</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>578.500000</td>
      <td>1.232608</td>
      <td>0.752023</td>
      <td>0.237731</td>
      <td>-0.372771</td>
      <td>-0.812917</td>
      <td>-1.441226</td>
      <td>-2.268203</td>
      <td>-2.781116</td>
      <td>-3.409715</td>
      <td>...</td>
      <td>15.161252</td>
      <td>15.429079</td>
      <td>15.750631</td>
      <td>16.026380</td>
      <td>16.369179</td>
      <td>16.453653</td>
      <td>16.708970</td>
      <td>16.831048</td>
      <td>16.985326</td>
      <td>17.249704</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1162.000000</td>
      <td>6.414188</td>
      <td>5.899363</td>
      <td>5.588889</td>
      <td>5.324444</td>
      <td>4.784173</td>
      <td>4.241860</td>
      <td>3.742776</td>
      <td>3.273444</td>
      <td>2.698563</td>
      <td>...</td>
      <td>23.000000</td>
      <td>23.014218</td>
      <td>23.304348</td>
      <td>23.593723</td>
      <td>23.872340</td>
      <td>24.000000</td>
      <td>24.154587</td>
      <td>24.454918</td>
      <td>24.725689</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1746.500000</td>
      <td>11.651285</td>
      <td>11.230862</td>
      <td>10.890605</td>
      <td>10.598843</td>
      <td>10.138906</td>
      <td>9.691971</td>
      <td>9.354228</td>
      <td>8.876311</td>
      <td>8.321108</td>
      <td>...</td>
      <td>32.264178</td>
      <td>32.778230</td>
      <td>33.204379</td>
      <td>33.733717</td>
      <td>34.237816</td>
      <td>34.683905</td>
      <td>35.000000</td>
      <td>35.322170</td>
      <td>35.726893</td>
      <td>35.973528</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2332.000000</td>
      <td>20.749333</td>
      <td>20.233251</td>
      <td>20.133995</td>
      <td>20.034739</td>
      <td>19.903465</td>
      <td>19.754950</td>
      <td>19.596154</td>
      <td>19.431319</td>
      <td>19.266484</td>
      <td>...</td>
      <td>76.809359</td>
      <td>77.329289</td>
      <td>77.849220</td>
      <td>78.414720</td>
      <td>78.998832</td>
      <td>79.936937</td>
      <td>81.922156</td>
      <td>83.797688</td>
      <td>85.642643</td>
      <td>87.544545</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 1126 columns</p>
</div>



5 weather stations 
Every day around 2 AM weather data is collected for 45 heights at equal height intervals at each of these stations. Weather data collected includes pressure, temperature, dew point, wind speed and wind direction for each height. Using this weather data Laxman wants to predict whether it will be a good day or bad day for paragliding.


```python
import re
dew_point_cols = [x for x in train_df.columns if re.search(r".*Dew Point.*",x)]
pressure_cols = [x for x in train_df.columns if re.search(r".*Pressure.*",x)]
temperature_cols = [x for x in train_df.columns if re.search(r".*Temperature.*",x)]
wind_direction_cols = [x for x in train_df.columns if re.search(r".*Wind Direction.*",x)]
wind_speed_cols = [x for x in train_df.columns if re.search(r".*Wind Speed.*",x)]
```


```python
all_sensor_cols = dew_point_cols+pressure_cols+temperature_cols+wind_direction_cols+wind_speed_cols

```


```python
len(all_sensor_cols)
```




    1125



##### Checking for Outliers and treating
We have information that sometimes sensor devices at weather stations record wrong weather parameters, so we'll check for outliers and remove them using z score method. We could use robust z score method if we think the outlier values are very large and impacts the calculation of mean and standard deviation which is used in calculation z score,in our case it was checked and the normal z score method was able to identify ouliers with observations having absolute value of z score greater than 3.


```python
station1temp = [x for x in temperature_cols if re.search(r".*Station1.*",x)] 
#Station3 Dew Point Height1
```


```python
fig,ax = plt.subplots(figsize=(14,6))
g = sns.boxplot(x="variable", y="value", data=pd.melt(train_df[station1temp]))
g.set_xticklabels(g.get_xticklabels(), rotation=80)
plt.show()
```


![png](output_16_0.png)


As we can see in the boxplot of sensor measurement(temperature),there are outliers due to faulty sensors and needs to be removed before training the model,hence we calculate the z score and remove obsv with absolute values > 3 standard deviations.
def outliers_iqr(col,k=1.5):
    quartile_1, quartile_3 = np.percentile(col, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * k)
    upper_bound = quartile_3 + (iqr * k)
    #print(col.name,sum([x < lower_bound or x > upper_bound for x in col]))
    return [True if x < lower_bound or x > upper_bound else False for x in col]
from scipy import stats
import numpy as np
def remove_high_lev(df):
    for station in range(1,6):
        for col_type in (dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols):         
            station_no_col_type = [x for x in col_type if re.search(r".*Station"+str(station)+"\s+.*",x)]
            z = np.abs(stats.zscore(df[station_no_col_type]))                        
            df[station_no_col_type].drop(df[(z > 2).all(axis=1)].index,inplace=True)
    return df                           


```python
from scipy import stats
import numpy as np
def remove_high_lev(col):

    #station_no_col_type = [x for x in col_type if re.search(r".*Station"+str(station)+"\s+.*",x)]
    #z = np.abs(stats.zscore(df[station_no_col_type]))    
    z = np.abs(stats.zscore(col)) 
    return [True if x>3 else False for x in z]
    #cdf = df[col_type].drop(df[(z > 3).all(axis=1)].index).copy()
    #cdf = df[station_no_col_type].drop(df[(z > 3).all(axis=1)].index).copy()
    #return cdf                           
```


```python
outlier_mask = train_df[all_sensor_cols].apply(remove_high_lev)
```


```python
outlier_mask.shape,train_df.shape
```




    ((2183, 1125), (2183, 1126))




```python
train_df1 = train_df[~outlier_mask.any(axis=1)].copy()
```


```python
train_df1.shape
```




    (1670, 1126)



We have removed records with faulty sensor measurements and will verify the same with boxplots.


```python
fig,ax = plt.subplots(figsize=(14,6))
g = sns.boxplot(x="variable", y="value", data=pd.melt(train_df1[station1temp]))
g.set_xticklabels(g.get_xticklabels(), rotation=80)
plt.show()
```


![png](output_26_0.png)


We will join pred dataframe with train data which includes flight details for 288 paragliding spots for several days. Flight details for each spot include max distance, total distance, number of flights that took place at that spot on a particular day. The conclustion is that if the total number of flights combining all these spots is more than or equal to 15 then it is good day else it is a bad day for paragliding. 


```python
train_df_final = train_df1.merge(pred,on=['Day_Id'],how='left')
import re
total_flight_cols = [x for x in train_df_final.columns if re.search(r".*totalFlights",x)]
train_df_final['total_flights'] = train_df_final[total_flight_cols].sum(axis=1)
```


```python
train_df_final['total_flights'].unique()
```




    array([  0,   3,   2,   7,   6,  21,   1,  11,  14,   9,  29,   4,  49,
            13,  97,  53,  25,  10,  71,  74,  22, 156,  15,  16,   5,  32,
            12,  24,   8,  75,  40,  30,  19,  35,  34,  58,  18,  20,  17,
            81, 132, 102, 106,  96,  76, 111, 109,  57,  26,  91,  65,  62,
            23,  60,  37,  46,  36,  66, 105,  78,  31,  55,  72, 125,  33,
            93,  54,  27,  84,  63,  52,  28,  77, 137, 194, 110, 104, 164,
           190, 150, 161,  47,  95,  80,  90,  43, 120,  56, 243, 170,  68,
           128,  79, 158, 204,  82,  92, 142,  50,  73,  39, 126,  94, 141,
            98,  89, 145, 107, 114,  70,  61,  41, 187,  51,  99,  67,  45,
            88, 130, 100, 113, 307, 121, 222, 175, 249, 233, 172,  64,  42,
           103, 168,  59, 144, 119, 152, 116, 157, 266,  48, 198, 295, 306,
           245, 189, 140, 101,  69,  83,  44, 124, 117, 203, 394, 208, 176,
           305, 178, 200,  86, 173, 242,  38, 248, 244, 153, 148, 313, 235,
           134, 259, 177, 151, 146, 115,  85, 133, 182, 207, 171, 439, 509,
           155, 186, 210, 167, 258, 197, 206, 444, 386, 195, 339, 256, 112,
           162, 118, 224, 138, 226, 316, 344])




```python
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(15,6))
sns.distplot(train_df_final['total_flights'],ax=ax[0])
sns.boxplot(train_df_final['total_flights'],ax=ax[1],showfliers=False)##we are not showing very large values as we are only interested whether total flights was greater than 15 or not
#sns.swarmplot(train_df_final['total_flights'], zorder=0,ax=ax[1],showfliers=False)
fig.suptitle('Histogram and boxplot of total_flights')

```




    Text(0.5, 0.98, 'Histogram and boxplot of total_flights')




![png](output_30_1.png)



```python
train_df_final['label'] = np.where(train_df_final['total_flights']>=15,1,0)
```


```python
train_df_final['total_flights_bin'] = pd.qcut(train_df_final['total_flights'],q=[0, .25, .5, .75, 1.],labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q'])

train_df_final['total_flights_range'] = pd.qcut(train_df_final['total_flights'],q=[0, .25, .55, .75, 1.])

train_df_final['total_flights_range'] = train_df_final['total_flights_range'].astype('str')

sns.countplot(train_df_final['total_flights_range'])#.value_counts()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f218b2ec710>




![png](output_32_1.png)


corr_matrix = train_df_final.corr()

#corr_matrix['total'].sort_values(ascending=False).filter(regex='.*Height1$', axis=0)

corr_matrix['total_flights'].sort_values(ascending=False).filter(regex='.*Height45$', axis=0)


```python
train = train_df_final.copy()#[all_sensor_cols].copy()#.drop(columns=pred.columns,axis=1)
train_labels = train_df_final['label'].copy()
#train.drop(columns=['total_flights','label'],inplace=True)

test = test_df.copy()#[all_sensor_cols].copy()#.drop(columns=['Day_Id'],inplace=True)
```


```python
train.shape,train_labels.shape,test.shape
```




    ((1670, 1994), (1670,), (100, 1126))



## Model Building & Validation


```python
def random_grid_search_params(clf,random_grid,features):
    clf_random = RandomizedSearchCV(estimator = clf,scoring='roc_auc', param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    clf_random.fit(train_df_final[features], train_labels)
    print(">>Best_params<<\n",clf_random.best_params_)
    print(">>Best_score<<\n",clf_random.best_score_)
    return clf_random.best_estimator_
```


```python
from sklearn.model_selection import StratifiedKFold
def stratify_kfold_cv(train_data,clf,features,weights=False):
    split = StratifiedKFold(n_splits=5,random_state=43,shuffle=True)
    i=1
    X_preds = np.zeros(train_data.shape[0])
    X_preds_label = np.zeros(train_data.shape[0])
    preds = np.zeros(test_df.shape[0])
    output_dict = dict()
    for train_index,test_index in split.split(train_data,train_data['total_flights_range']):
        #print("##########")
        #print(i,'fold>')
        dict_results = dict()
        X_train , X_val = train_data.iloc[train_index],train_data.iloc[test_index]
        y_train , y_val = train_labels.iloc[train_index],train_labels.iloc[test_index]
        X_train = X_train[features]
        X_val = X_val[features]
        if weights:
            sample_weights_data = train_df_final['weights'].copy()
            clf.fit(X_train,y_train,sample_weight=sample_weights_data)
        else:
            clf.fit(X_train,y_train)
        #print(clf.feature_importances_)


        X_preds[test_index] = clf.predict_proba(X_val)[:,1]
        X_preds_label[test_index] = clf.predict(X_val)
        #preds += clf.predict_proba(test_df)[:,1]
        y_predicted_val = clf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_predicted_val)
        dict_results['accuracy'] = accuracy_score(y_val, clf.predict(X_val))
        dict_results['precision'] = precision_score(y_val, clf.predict(X_val))
        dict_results['recall'] = recall_score(y_val, clf.predict(X_val))
        dict_results['f1score'] = f1_score(y_val, clf.predict(X_val))
        dict_results['roc_auc_score'] = auc
        tn, fp, fn, tp = confusion_matrix(y_val, clf.predict(X_val)).ravel()
        dict_results['TN'] = tn
        dict_results['FP'] = fp
        dict_results['FN'] = fn
        dict_results['TP'] = tp
        
        output_dict[str(i)+"Fold"] = dict_results
        print(str(i)+'fold completed')
        i+=1
    score = roc_auc_score(train_labels, X_preds)
    mean_accuracy = accuracy_score(train_labels, X_preds_label)
    mean_precision = precision_score(train_labels, X_preds_label)
    mean_recall = recall_score(train_labels, X_preds_label)
    mean_f1_score = f1_score(train_labels, X_preds_label)
    tn, fp, fn, tp = confusion_matrix(train_labels, X_preds_label).ravel()
    mean_dict = dict()
    mean_dict['TN'] = tn
    mean_dict['FP'] = fp
    mean_dict['FN'] = fn
    mean_dict['TP'] = tp
    mean_dict['accuracy'] = mean_accuracy
    mean_dict['precision'] = mean_precision
    mean_dict['recall'] = mean_recall
    mean_dict['f1score'] = mean_f1_score
    mean_dict['roc_auc_score'] = score
    output_dict['Overall'] = mean_dict
    #print(pd.DataFrame(output_dict))
    return (pd.DataFrame(output_dict))
    
```

#### Random forest w/ RandomizedsearchCV - Original features
We will start with the original features avaiable and fit randomforest model to our data and check the performance using cross-validation mechanism


```python
random_state = np.random.RandomState(0)
rf_clf = RandomForestClassifier(random_state=random_state)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]# Number of trees in random forest
max_features = ['auto', 'sqrt']# Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]# Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]# Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]# Minimum number of samples required at each leaf node
bootstrap = [True, False]# Method of selecting samples for training each tree
# Create the random grid
rf_param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
```


```python
rf_final = random_grid_search_params(rf_clf,rf_param_grid,all_sensor_cols)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   58.6s
    [Parallel(n_jobs=-1)]: Done 130 tasks      | elapsed:  6.6min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 14.5min finished


    >>Best_params<<
     {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 70, 'bootstrap': True}
    >>Best_score<<
     0.7365894697929011



```python
rf_all_results=stratify_kfold_cv(train_df_final,rf_final,all_sensor_cols)##0.7666096465330601

```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
rf_all_results.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>55.0</td>
      <td>35.0</td>
      <td>147.0</td>
      <td>98.0</td>
      <td>0.731343</td>
      <td>0.685315</td>
      <td>0.736842</td>
      <td>0.640523</td>
      <td>0.798966</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>54.0</td>
      <td>54.0</td>
      <td>133.0</td>
      <td>94.0</td>
      <td>0.677612</td>
      <td>0.635135</td>
      <td>0.635135</td>
      <td>0.635135</td>
      <td>0.733018</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>56.0</td>
      <td>44.0</td>
      <td>138.0</td>
      <td>97.0</td>
      <td>0.701493</td>
      <td>0.659864</td>
      <td>0.687943</td>
      <td>0.633987</td>
      <td>0.752819</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>53.0</td>
      <td>51.0</td>
      <td>131.0</td>
      <td>98.0</td>
      <td>0.687688</td>
      <td>0.653333</td>
      <td>0.657718</td>
      <td>0.649007</td>
      <td>0.752201</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>61.0</td>
      <td>44.0</td>
      <td>137.0</td>
      <td>90.0</td>
      <td>0.683735</td>
      <td>0.631579</td>
      <td>0.671642</td>
      <td>0.596026</td>
      <td>0.770590</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>279.0</td>
      <td>228.0</td>
      <td>686.0</td>
      <td>477.0</td>
      <td>0.696407</td>
      <td>0.652977</td>
      <td>0.676596</td>
      <td>0.630952</td>
      <td>0.760795</td>
    </tr>
  </tbody>
</table>
</div>


As we can see,the initial model provides a reasonable performance  although in our case we would be more interested in the precision of our model because the risk in predicting it is safe for flight when it actually isn't is very high and hence would like to improve our model with better features via feature engineering and other transformations of features before training the model

```python
feat_imp = pd.DataFrame(rf_final.feature_importances_)
feat_imp.columns = ['Feature_importance']
feat_imp = feat_imp.assign(Column_name=all_sensor_cols)
feat_imp.sort_values(by=['Feature_importance'],ascending=False).set_index('Column_name').head(30).plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f21e8b79b38>




![png](output_45_1.png)

We cannot infer much from the feature importances due to the fact that our variable measurements would be highly correlated with measurements at different heights being used as features and hence the feature importance which is calcualted by mean reduction in impurity due to each feature at each split would be reduced.Although the importance of temperature and pressure is highlighted here.

Qouted from link "http://blog.datadive.net/selecting-good-features-part-iii-random-forests/"
"Random forest consists of a number of decision trees. Every node in the decision trees is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure based on which the (locally) optimal condition is chosen is called impurity. For classification, it is typically either Gini impurity or information gain/entropy and for regression trees it is variance. Thus when training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure."

"There are a few things to keep in mind when using the impurity based ranking. Firstly, feature selection based on impurity reduction is biased towards preferring variables with more categories (see Bias in random forest variable importance measures). Secondly, when the dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be used as the predictor, with no concrete preference of one over the others. But once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature. As a consequence, they will have a lower reported importance. This is not an issue when we want to use feature selection to reduce overfitting, since it makes sense to remove features that are mostly duplicated by other features. But when interpreting the data, it can lead to the incorrect conclusion that one of the variables is a strong predictor while the others in the same group are unimportant, while actually they are very close in terms of their relationship with the response variable.

The effect of this phenomenon is somewhat reduced thanks to random selection of features at each node creation, but in general the effect is not removed completely. For example, we have three correlated variables X0,X1,X2, and no noise in the data, with the output variable simply being the sum of the three features:
When we compute the feature importances, we see that X1 is computed to have over 10x higher importance than X2, while their “true” importance is very similar. This happens despite the fact that the data is noiseless, we use 20 trees, random selection of features (at each split, only two of the three features are considered) and a sufficiently large dataset.

One thing to point out though is that the difficulty of interpreting the importance/ranking of correlated variables is not random forest specific, but applies to most model based feature selection methods."
### Feature enginnering


We'll further try to bin measurements at similar heights as there wouldn't be much difference in sensor values.We could check this from boxplot of measurement values at heights closer to each other.


```python
stns=['Station1','Station2','Station3','Station4','Station5']
def bin_measured_values(df,cols,coltype):
    for stn in stns:
        col = [x for x in cols if re.search(r""+str(stn),x)]
        for i in range(0,45,5):
            
            col_height_filter = [x for x in col if x.endswith(tuple(map(lambda x: 'Height'+str(x),range(i+1,i+6))))]
            #print(col_height_filter)
            #print("Height"+str(1+i)+"-"+str(5+i))
            df[stn+" "+coltype+" "+"Height"+str(1+i)+"-"+str(5+i)] = df[col_height_filter].mean(axis=1)
    #print('completed')
    
for i,j in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
    bin_measured_values(train_df_final,j,i)
    bin_measured_values(test_df,j,i)
        #df[stn+" "+coltype+" "+"Height"+str(1+k)+"-"+str(5+k)] = df[col_height_filter].mean(axis=1)
```


```python
def overall_measured_values(df,cols,coltype):
    for stn in stns:
        col = [x for x in cols if re.search(r""+str(stn),x)]
        df[stn+" "+coltype+" "+"Height1-45"] = df[col].mean(axis=1)
for i,j in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
    overall_measured_values(train_df_final,j,i)
    overall_measured_values(test_df,j,i)
    
```


```python
binning_features = [ x for x in train_df_final.columns if re.search(r"Height\d+-\d+",x)]
```

We'll take the weighted average of each sensor measurements at each station based on their absolute values of correlation with total flights such that each feature is weighted differently, and the resulting feature tends to
be more important in feature selection process while building each individual tree


```python
for coltype in ['dew','temp','pressure','wind_dir','wind_speed']:
    test = train_df_final[['Station1 '+str(coltype)+' Height1-5','Station1 '+str(coltype)+' Height6-10','Station1 '+str(coltype)+' Height11-15',
                           'Station1 '+str(coltype)+' Height16-20','Station1 '+str(coltype)+' Height21-25','Station1 '+str(coltype)+' Height26-30',
                           'Station1 '+str(coltype)+' Height31-35','Station1 '+str(coltype)+' Height36-40','Station1 '+str(coltype)+' Height41-45',
                           'total_flights']].copy()
    print(test.corr()['total_flights'])
```

    Station1 dew Height1-5      0.043565
    Station1 dew Height6-10     0.042431
    Station1 dew Height11-15    0.043253
    Station1 dew Height16-20    0.016291
    Station1 dew Height21-25    0.027981
    Station1 dew Height26-30    0.008152
    Station1 dew Height31-35   -0.001672
    Station1 dew Height36-40   -0.006968
    Station1 dew Height41-45   -0.027207
    total_flights               1.000000
    Name: total_flights, dtype: float64
    Station1 temp Height1-5      0.193719
    Station1 temp Height6-10     0.184025
    Station1 temp Height11-15    0.172748
    Station1 temp Height16-20    0.166064
    Station1 temp Height21-25    0.147500
    Station1 temp Height26-30    0.139564
    Station1 temp Height31-35    0.133428
    Station1 temp Height36-40    0.131844
    Station1 temp Height41-45    0.136825
    total_flights                1.000000
    Name: total_flights, dtype: float64
    Station1 pressure Height1-5      0.210847
    Station1 pressure Height6-10     0.241342
    Station1 pressure Height11-15    0.253858
    Station1 pressure Height16-20    0.256485
    Station1 pressure Height21-25    0.250735
    Station1 pressure Height26-30    0.241025
    Station1 pressure Height31-35    0.232826
    Station1 pressure Height36-40    0.222451
    Station1 pressure Height41-45    0.214222
    total_flights                    1.000000
    Name: total_flights, dtype: float64
    Station1 wind_dir Height1-5     -0.010590
    Station1 wind_dir Height6-10    -0.027597
    Station1 wind_dir Height11-15   -0.008178
    Station1 wind_dir Height16-20   -0.032151
    Station1 wind_dir Height21-25   -0.028102
    Station1 wind_dir Height26-30   -0.069670
    Station1 wind_dir Height31-35   -0.082259
    Station1 wind_dir Height36-40   -0.050031
    Station1 wind_dir Height41-45   -0.038770
    total_flights                    1.000000
    Name: total_flights, dtype: float64
    Station1 wind_speed Height1-5     -0.047589
    Station1 wind_speed Height6-10    -0.117692
    Station1 wind_speed Height11-15   -0.118630
    Station1 wind_speed Height16-20   -0.138978
    Station1 wind_speed Height21-25   -0.185249
    Station1 wind_speed Height26-30   -0.197616
    Station1 wind_speed Height31-35   -0.174328
    Station1 wind_speed Height36-40   -0.160168
    Station1 wind_speed Height41-45   -0.143052
    total_flights                      1.000000
    Name: total_flights, dtype: float64



```python
fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(15,8))
axs = axs.flatten()
sns.regplot(x='Station1 Pressure Height1', y='total_flights', data=train_df_final, ax=axs[0])
#axs[0].set(xlim=(500, 970))
sns.regplot(x='Station1 Pressure Height20', y='total_flights', data=train_df_final, ax=axs[1])
#axs[1].set(xlim=(500, 970))
sns.regplot(x='Station1 Pressure Height30', y='total_flights', data=train_df_final, ax=axs[2])
#axs[2].set(xlim=(500, 970))
sns.regplot(x='Station1 Pressure Height45', y='total_flights', data=train_df_final, ax=axs[3])
#axs[3].set(xlim=(500, 970))

plt.show()
```


![png](output_54_0.png)



```python
from sklearn.preprocessing import MinMaxScaler
sclr = MinMaxScaler(feature_range=(0.01,1))
def calc_weights(stn_num,colstype):

    stn1temp = [x for x in colstype if re.search(r""+str(stn_num)+"\s+",x)]
    abs_corr  = abs(train_df_final[stn1temp+['total_flights']].corr()[['total_flights']][:-1])
    weights = sclr.fit_transform(abs_corr)
    #print(weights[0][0])
    wts = [weights[i][0] for i in range(0,45)]
    return wts
```


```python
wts_dict = dict()


wts_dict['Station1'] = {}
wts_dict['Station2'] = {}
wts_dict['Station3'] = {}
wts_dict['Station4'] = {}
wts_dict['Station5'] = {}
for stn in ['Station1','Station2','Station3','Station4','Station5']:
    wts_dict[stn]['temp'] = calc_weights(stn,temperature_cols)
    wts_dict[stn]['dew'] = calc_weights(stn,dew_point_cols)
    wts_dict[stn]['pressure'] = calc_weights(stn,pressure_cols)
    wts_dict[stn]['wind_dir'] = calc_weights(stn,wind_direction_cols)
    wts_dict[stn]['wind_speed'] = calc_weights(stn,wind_speed_cols)

```


```python

for stn in stns:
    for coltype,cols in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
        col = [x for x in cols if re.search(r""+str(stn)+"\s+",x)]
        train_df_final[str(stn)+" "+str(coltype)+"-wtavg"] = train_df_final[col].apply(lambda x:np.average(x,weights=wts_dict[stn][coltype]),axis=1)
    
```


```python
stns=['Station1','Station2','Station3','Station4','Station5']
for stn in stns:
    for coltype,cols in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
        col = [x for x in cols if re.search(r""+str(stn)+"\s+",x)]
        test_df[str(stn)+" "+str(coltype)+"-wtavg"] = test_df[col].apply(lambda x:np.average(x,weights=wts_dict[stn][coltype]),axis=1)
    
```


```python
weighted_avg_features = [x for x in train_df_final if re.search(r"wtavg",x)]
```

Create features such as max,min,variance,difference of each measurement types


```python
stns=['Station1','Station2','Station3','Station4','Station5']
def new_feat(df,stns,cols_measures,coltype):
    for stn in stns:
        cols = [x for x in cols_measures if re.search(r"^"+stn,x)]
        #print(cols)
        df[stn+" "+coltype+" "+"max"] = df[cols].max(axis=1)
        df[stn+" "+coltype+" "+"min"] = df[cols].min(axis=1)
        df[stn+" "+coltype+" "+"mean"] = df[cols].mean(axis=1)
        df[stn+" "+coltype+" "+"std"] = df[cols].std(axis=1)
        df[stn+" "+coltype+" "+"var"] = df[cols].var(axis=1)
        df[stn+" "+coltype+" "+"max-min"] = df[stn+" "+coltype+" "+"max"] - df[stn+" "+coltype+" "+"min"]
        
        
for i,j in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):                                              
    new_feat(train_df_final,stns,j,i)  
    new_feat(test_df,stns,j,i)

```


```python
aggr_features = [x for x in train_df_final.columns if x.endswith(('max','min','max-min','mean','std','var'))]
```


```python
###hist plot of binned measurement features
#num = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
numdf=pd.melt(train_df_final,value_vars=new_feat_cols1)
numgrid=sns.FacetGrid(numdf,col='variable',col_wrap=4,sharex=False,sharey=False)
numgrid=numgrid.map(sns.distplot,'value')
numgrid.savefig("All-Station-grouped-measurements-hist-output.png")
```




    <seaborn.axisgrid.FacetGrid at 0x7f964d87d320>




![png](output_63_1.png)



```python
#pd.melt(train_df_final, value_vars=new_feat_cols1,id_vars='total_flights_range')
```


```python
def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)
def gen_box_plot(station_no):
    cols = [x for x in new_feat_cols1 if re.search(r"Station"+str(station_no)+"\s",x)]
    p = pd.melt(train_df_final, value_vars=cols,id_vars='total_flights_range')
    g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(boxplot, 'total_flights_range','value')
    g.savefig("Station-"+str(station_no)+"output.png")

```


```python
for stns in range(1,6):
    gen_box_plot(stns)
```


![png](output_66_0.png)



![png](output_66_1.png)



![png](output_66_2.png)



![png](output_66_3.png)



![png](output_66_4.png)


From the plots,windspeed,temperature and pressure seems to be correlated with the no of flights on a particular day

From the PCA analysis performed in another notebook we can try using pca features in our model,especially the pressure and temperature components since the first 3 pricipal components are explaining more than 90% variance in data,moreover pressure and temperature plays an importanat role in deciding of flights on a particular day. Refer PCA analysis notebook for further understanding on explained variance for each sensor measurements across stations.


```python
scaler = StandardScaler()
pcadf = train_df_final[all_sensor_cols+['Day_Id']].copy()
pcadf[all_sensor_cols] = scaler.fit_transform(pcadf[all_sensor_cols])

```


```python
pca = PCA(n_components=3)

pcadf['PC1'+'temp'] = pca.fit_transform(pcadf[temperature_cols].values)[:,0]
pcadf['PC2'+'temp'] = pca.fit_transform(pcadf[temperature_cols].values)[:,1]
pcadf['PC3'+'temp'] = pca.fit_transform(pcadf[temperature_cols].values)[:,2]

pcadf['PC1'+'pressure'] = pca.fit_transform(pcadf[pressure_cols].values)[:,0]
pcadf['PC2'+'pressure'] = pca.fit_transform(pcadf[pressure_cols].values)[:,1]
pcadf['PC3'+'pressure'] = pca.fit_transform(pcadf[pressure_cols].values)[:,2]
```


```python
train_df_final = train_df_final.merge(pcadf[['PC1temp','PC2temp','PC3temp','PC1pressure','PC2pressure','PC3pressure','Day_Id']],on=['Day_Id'],how='left')
```


```python
test_df['PC1'+'temp'] = pca.transform(test_df[temperature_cols].values)[:,0]
test_df['PC2'+'temp'] = pca.transform(test_df[temperature_cols].values)[:,1]
test_df['PC3'+'temp'] = pca.transform(test_df[temperature_cols].values)[:,2]

test_df['PC1'+'pressure'] = pca.transform(test_df[pressure_cols].values)[:,0]
test_df['PC2'+'pressure'] = pca.transform(test_df[pressure_cols].values)[:,1]
test_df['PC3'+'pressure'] = pca.transform(test_df[pressure_cols].values)[:,2]
```


```python
pca_features = ['PC1temp','PC2temp','PC3temp','PC1pressure','PC2pressure','PC3pressure']
```


```python
engineered_features = binning_features+weighted_avg_features+aggr_features+pca_features
```

### Random forest with new features


```python
rf_new = random_grid_search_params(rf_clf,rf_param_grid,engineered_features)
rf_new_feat_results = stratify_kfold_cv(train_df_final,rf_new,engineered_features)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   34.3s
    [Parallel(n_jobs=-1)]: Done 130 tasks      | elapsed:  4.1min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  9.0min finished


    >>Best_params<<
     {'n_estimators': 1400, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': True}
    >>Best_score<<
     0.7385785041732961
    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
rf_new_feat_results.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>55.0</td>
      <td>38.0</td>
      <td>144.0</td>
      <td>98.0</td>
      <td>0.722388</td>
      <td>0.678201</td>
      <td>0.720588</td>
      <td>0.640523</td>
      <td>0.800151</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>55.0</td>
      <td>53.0</td>
      <td>134.0</td>
      <td>93.0</td>
      <td>0.677612</td>
      <td>0.632653</td>
      <td>0.636986</td>
      <td>0.628378</td>
      <td>0.737751</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>55.0</td>
      <td>42.0</td>
      <td>140.0</td>
      <td>98.0</td>
      <td>0.710448</td>
      <td>0.668942</td>
      <td>0.700000</td>
      <td>0.640523</td>
      <td>0.754435</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>51.0</td>
      <td>47.0</td>
      <td>135.0</td>
      <td>100.0</td>
      <td>0.705706</td>
      <td>0.671141</td>
      <td>0.680272</td>
      <td>0.662252</td>
      <td>0.764719</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>59.0</td>
      <td>47.0</td>
      <td>134.0</td>
      <td>92.0</td>
      <td>0.680723</td>
      <td>0.634483</td>
      <td>0.661871</td>
      <td>0.609272</td>
      <td>0.770846</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>275.0</td>
      <td>227.0</td>
      <td>687.0</td>
      <td>481.0</td>
      <td>0.699401</td>
      <td>0.657104</td>
      <td>0.679379</td>
      <td>0.636243</td>
      <td>0.765141</td>
    </tr>
  </tbody>
</table>
</div>




```python
engg_feat_imp = pd.DataFrame(rf_new.feature_importances_)
engg_feat_imp.columns = ['Feature_importance']
engg_feat_imp = engg_feat_imp.assign(Column_name=engineered_features)
engg_feat_imp.sort_values(by=['Feature_importance'],ascending=False).set_index('Column_name').head(20).plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f21a838dd30>




![png](output_78_1.png)



```python
len(engineered_features),len(all_sensor_cols)
```




    (431, 1125)



Slight improvement in performance with comparitively lesser number of features engineered from original features.We have reduced the 1125 dimesional features space to 431 dimensions through feature engineering and further improved the performance

## AdaBoost


```python
from sklearn.ensemble import AdaBoostClassifier
adabst_clf = AdaBoostClassifier(n_estimators=50,
                         learning_rate=0.5,
                         random_state=0)
adabst_result = stratify_kfold_cv(train_df_final,adabst_clf,engineered_features)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
adabst_result.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>59.0</td>
      <td>43.0</td>
      <td>139.0</td>
      <td>94.0</td>
      <td>0.695522</td>
      <td>0.648276</td>
      <td>0.686131</td>
      <td>0.614379</td>
      <td>0.769339</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>61.0</td>
      <td>53.0</td>
      <td>134.0</td>
      <td>87.0</td>
      <td>0.659701</td>
      <td>0.604167</td>
      <td>0.621429</td>
      <td>0.587838</td>
      <td>0.722250</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>65.0</td>
      <td>46.0</td>
      <td>136.0</td>
      <td>88.0</td>
      <td>0.668657</td>
      <td>0.613240</td>
      <td>0.656716</td>
      <td>0.575163</td>
      <td>0.729135</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>52.0</td>
      <td>50.0</td>
      <td>132.0</td>
      <td>99.0</td>
      <td>0.693694</td>
      <td>0.660000</td>
      <td>0.664430</td>
      <td>0.655629</td>
      <td>0.761553</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>60.0</td>
      <td>54.0</td>
      <td>127.0</td>
      <td>91.0</td>
      <td>0.656627</td>
      <td>0.614865</td>
      <td>0.627586</td>
      <td>0.602649</td>
      <td>0.725586</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>297.0</td>
      <td>246.0</td>
      <td>668.0</td>
      <td>459.0</td>
      <td>0.674850</td>
      <td>0.628337</td>
      <td>0.651064</td>
      <td>0.607143</td>
      <td>0.741207</td>
    </tr>
  </tbody>
</table>
</div>



### Light GBM


```python
lgbm_model = lgb.LGBMClassifier(lambda_l2=1.0,feature_fraction=0.6,num_boost_round=1200,num_leaves=9)
lgbm_new_feat_results = stratify_kfold_cv(train_df_final,lgbm_model,engineered_features)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
lgbm_new_feat_results.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>56.0</td>
      <td>39.0</td>
      <td>143.0</td>
      <td>97.0</td>
      <td>0.716418</td>
      <td>0.671280</td>
      <td>0.713235</td>
      <td>0.633987</td>
      <td>0.795877</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>52.0</td>
      <td>47.0</td>
      <td>140.0</td>
      <td>96.0</td>
      <td>0.704478</td>
      <td>0.659794</td>
      <td>0.671329</td>
      <td>0.648649</td>
      <td>0.750470</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>58.0</td>
      <td>50.0</td>
      <td>132.0</td>
      <td>95.0</td>
      <td>0.677612</td>
      <td>0.637584</td>
      <td>0.655172</td>
      <td>0.620915</td>
      <td>0.754148</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>44.0</td>
      <td>57.0</td>
      <td>125.0</td>
      <td>107.0</td>
      <td>0.696697</td>
      <td>0.679365</td>
      <td>0.652439</td>
      <td>0.708609</td>
      <td>0.772142</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>58.0</td>
      <td>53.0</td>
      <td>128.0</td>
      <td>93.0</td>
      <td>0.665663</td>
      <td>0.626263</td>
      <td>0.636986</td>
      <td>0.615894</td>
      <td>0.759906</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>268.0</td>
      <td>246.0</td>
      <td>668.0</td>
      <td>488.0</td>
      <td>0.692216</td>
      <td>0.655034</td>
      <td>0.664850</td>
      <td>0.645503</td>
      <td>0.766558</td>
    </tr>
  </tbody>
</table>
</div>



### XGBOOST w/ new features 


```python
xgb_model = XGBClassifier()
```


```python
#train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(3.0, 15.0]')|(train_df_final['total_flights_range']=='(15.0, 43.0]'),1,0.8)
```


```python
xgb_new_feat_results_upd = stratify_kfold_cv(train_df_final,xgb_model,engineered_features)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
xgb_new_feat_results_upd.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>54.0</td>
      <td>37.0</td>
      <td>145.0</td>
      <td>99.0</td>
      <td>0.728358</td>
      <td>0.685121</td>
      <td>0.727941</td>
      <td>0.647059</td>
      <td>0.795339</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>52.0</td>
      <td>45.0</td>
      <td>142.0</td>
      <td>96.0</td>
      <td>0.710448</td>
      <td>0.664360</td>
      <td>0.680851</td>
      <td>0.648649</td>
      <td>0.757552</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>55.0</td>
      <td>50.0</td>
      <td>132.0</td>
      <td>98.0</td>
      <td>0.686567</td>
      <td>0.651163</td>
      <td>0.662162</td>
      <td>0.640523</td>
      <td>0.754579</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>46.0</td>
      <td>49.0</td>
      <td>133.0</td>
      <td>105.0</td>
      <td>0.714715</td>
      <td>0.688525</td>
      <td>0.681818</td>
      <td>0.695364</td>
      <td>0.785787</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>56.0</td>
      <td>44.0</td>
      <td>137.0</td>
      <td>95.0</td>
      <td>0.698795</td>
      <td>0.655172</td>
      <td>0.683453</td>
      <td>0.629139</td>
      <td>0.762687</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>263.0</td>
      <td>225.0</td>
      <td>689.0</td>
      <td>493.0</td>
      <td>0.707784</td>
      <td>0.668928</td>
      <td>0.686630</td>
      <td>0.652116</td>
      <td>0.770621</td>
    </tr>
  </tbody>
</table>
</div>




```python
xgb_model.fit(train_df_final[engineered_features],train_labels)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1)




```python
test_df['Good_Bad'] = xgb_model.predict(test_df[engineered_features])
```


```python
test_df.shape
```




    (100, 1558)




```python
##Initial submission
sub = test_df[['Day_Id','Good_Bad']].copy()
sub=sub.reindex(columns=["Day_Id","Good_Bad"])
filename = 'submission.csv'
sub.to_csv(filename, index=False)
```

Very useful link to understand the hyperparameters and tune them.
https://sites.google.com/view/lauraepp/parameters

###### Hyper parameter tuning of xgboost


```python
##randomized search
param_grid = {
    'max_depth' : [4, 8, 12],
    'learning_rate' : [0.01, 0.3, 0.5],
    'n_estimators' : [20, 50, 200],              
    'objective' : ["binary:logistic"],#['multi:softprob'],
    'gamma' : [0, 0.25, 0.5],
    'min_child_weight' : [1, 3, 5],
    'subsample' : [0.1, 0.5, 1],
    'colsample_bytree' : [0.1, 0.5, 1]}


xgb_random = RandomizedSearchCV(estimator = xgb_model,scoring='roc_auc', param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# Fit the random search model
xgb_random.fit(train_df_final[engineered_features], train_labels)
print(">>Best_params<<\n",xgb_random.best_params_)
print(">>Best_score<<\n",xgb_random.best_score_)

```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1 


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   1.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    1.0s remaining:    0.0s


    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   1.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   1.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5, total=   7.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5, total=   7.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5, total=   7.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   0.8s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   1.6s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   1.7s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   1.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   2.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   2.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   2.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=  11.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=  12.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=  12.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=  10.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=  10.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=  10.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   0.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   0.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   0.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   7.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   7.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   7.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   5.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   5.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   5.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   0.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.8s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   1.7s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   1.6s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   1.6s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   1.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   1.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   1.0s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   1.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   1.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   1.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.7s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=1, total=   1.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=1, total=   1.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0.25, colsample_bytree=1, total=   1.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.5, total=   1.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.5, total=   1.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.5, total=   1.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   7.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   7.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   8.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   4.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   4.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   4.0s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5, total=   5.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5, total=   5.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=0.5, total=   5.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5, total=   1.0s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5, total=   1.0s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5, total=   0.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   2.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   2.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   2.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   2.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.5, total=   0.5s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   0.5s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   0.6s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   0.6s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.5, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.7s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   4.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   5.0s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   5.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   3.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   3.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   3.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   0.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   0.6s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   0.6s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   0.6s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   1.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   1.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   1.7s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   1.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.5s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=  11.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=  10.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=  10.8s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   9.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=  11.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=1, total=   8.9s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=8, learning_rate=0.5, gamma=0.25, colsample_bytree=0.5, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   1.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   1.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   1.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.01, gamma=0, colsample_bytree=1, total=   1.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.25, colsample_bytree=1, total=   0.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   0.5s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   0.5s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.01, gamma=0.5, colsample_bytree=1, total=   0.5s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   1.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   1.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=5, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=1, total=   1.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=1, total=   0.4s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=3, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.5, gamma=0.25, colsample_bytree=1, total=   0.3s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.01, gamma=0.5, colsample_bytree=0.1, total=   0.4s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.8s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=1, max_depth=8, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.9s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   1.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   1.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.01, gamma=0, colsample_bytree=0.1, total=   1.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   4.2s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   4.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=12, learning_rate=0.5, gamma=0.5, colsample_bytree=0.5, total=   4.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1, total=   1.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1, total=   1.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=3, max_depth=8, learning_rate=0.3, gamma=0.25, colsample_bytree=0.1, total=   1.0s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.5, colsample_bytree=0.1, total=   0.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   0.6s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   0.6s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=1, total=   0.6s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   3.8s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   3.9s
    [CV] subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5 
    [CV]  subsample=1, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=12, learning_rate=0.3, gamma=0.5, colsample_bytree=0.5, total=   3.7s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5, total=   2.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5, total=   2.3s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=200, min_child_weight=5, max_depth=4, learning_rate=0.3, gamma=0.25, colsample_bytree=0.5, total=   2.4s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=8, learning_rate=0.3, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=20, min_child_weight=1, max_depth=12, learning_rate=0.01, gamma=0.25, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1 
    [CV]  subsample=0.5, objective=binary:logistic, n_estimators=50, min_child_weight=5, max_depth=4, learning_rate=0.5, gamma=0, colsample_bytree=0.1, total=   0.2s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=0.1, total=   0.1s
    [CV] subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=0.1 
    [CV]  subsample=0.1, objective=binary:logistic, n_estimators=50, min_child_weight=1, max_depth=4, learning_rate=0.5, gamma=0.5, colsample_bytree=0.1, total=   0.1s


    [Parallel(n_jobs=1)]: Done 300 out of 300 | elapsed:  7.7min finished


    >>Best_params<<
     {'subsample': 0.5, 'objective': 'binary:logistic', 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 12, 'learning_rate': 0.01, 'gamma': 0.25, 'colsample_bytree': 1}
    >>Best_score<<
     0.7405819515699691



```python
best_xgb_params = {'subsample': 0.5, 'objective': 'binary:logistic', 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 12, 'learning_rate': 0.01, 'gamma': 0.25, 'colsample_bytree': 1}
best_xgb_model = XGBClassifier(**best_xgb_params)
```


```python
best_xgb_new_feat_results = stratify_kfold_cv(train_df_final,best_xgb_model,engineered_features)#,weights=True)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
best_xgb_new_feat_results.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>58.0</td>
      <td>35.0</td>
      <td>147.0</td>
      <td>95.0</td>
      <td>0.722388</td>
      <td>0.671378</td>
      <td>0.730769</td>
      <td>0.620915</td>
      <td>0.802772</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>50.0</td>
      <td>50.0</td>
      <td>137.0</td>
      <td>98.0</td>
      <td>0.701493</td>
      <td>0.662162</td>
      <td>0.662162</td>
      <td>0.662162</td>
      <td>0.746242</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>57.0</td>
      <td>46.0</td>
      <td>136.0</td>
      <td>96.0</td>
      <td>0.692537</td>
      <td>0.650847</td>
      <td>0.676056</td>
      <td>0.627451</td>
      <td>0.756338</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>52.0</td>
      <td>47.0</td>
      <td>135.0</td>
      <td>99.0</td>
      <td>0.702703</td>
      <td>0.666667</td>
      <td>0.678082</td>
      <td>0.655629</td>
      <td>0.779419</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>58.0</td>
      <td>49.0</td>
      <td>132.0</td>
      <td>93.0</td>
      <td>0.677711</td>
      <td>0.634812</td>
      <td>0.654930</td>
      <td>0.615894</td>
      <td>0.775347</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>275.0</td>
      <td>227.0</td>
      <td>687.0</td>
      <td>481.0</td>
      <td>0.699401</td>
      <td>0.657104</td>
      <td>0.679379</td>
      <td>0.636243</td>
      <td>0.771323</td>
    </tr>
  </tbody>
</table>
</div>



We'll provide higher weights to samples in range (3,15] as we would want a higher cost penalty for wrong predictions in this range and eventually increase our precision and decrease FP.


```python
train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(3.0, 15.0]'),1,0.8)
#train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(15.0, 49.0]'),1,np.nan)
#train_df_final['weights'] = train_df_final['weights'].fillna(0.6)
#train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(3.0, 15.0]')|(train_df_final['total_flights_range']=='(15.0, 43.0]'),1,0.8)
```


```python
best_xgb_new_feat_results1 = stratify_kfold_cv(train_df_final,best_xgb_model,engineered_features,weights=True)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
best_xgb_new_feat_results1.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>f1score</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1Fold</th>
      <td>54.0</td>
      <td>31.0</td>
      <td>151.0</td>
      <td>99.0</td>
      <td>0.746269</td>
      <td>0.699647</td>
      <td>0.761538</td>
      <td>0.647059</td>
      <td>0.802844</td>
    </tr>
    <tr>
      <th>2Fold</th>
      <td>51.0</td>
      <td>47.0</td>
      <td>140.0</td>
      <td>97.0</td>
      <td>0.707463</td>
      <td>0.664384</td>
      <td>0.673611</td>
      <td>0.655405</td>
      <td>0.742954</td>
    </tr>
    <tr>
      <th>3Fold</th>
      <td>57.0</td>
      <td>46.0</td>
      <td>136.0</td>
      <td>96.0</td>
      <td>0.692537</td>
      <td>0.650847</td>
      <td>0.676056</td>
      <td>0.627451</td>
      <td>0.755046</td>
    </tr>
    <tr>
      <th>4Fold</th>
      <td>48.0</td>
      <td>47.0</td>
      <td>135.0</td>
      <td>103.0</td>
      <td>0.714715</td>
      <td>0.684385</td>
      <td>0.686667</td>
      <td>0.682119</td>
      <td>0.778510</td>
    </tr>
    <tr>
      <th>5Fold</th>
      <td>58.0</td>
      <td>48.0</td>
      <td>133.0</td>
      <td>93.0</td>
      <td>0.680723</td>
      <td>0.636986</td>
      <td>0.659574</td>
      <td>0.615894</td>
      <td>0.779408</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>268.0</td>
      <td>219.0</td>
      <td>695.0</td>
      <td>488.0</td>
      <td>0.708383</td>
      <td>0.667122</td>
      <td>0.690240</td>
      <td>0.645503</td>
      <td>0.770798</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_xgb_model.fit(train_df_final[engineered_features],train_labels,sample_weight=train_df_final['weights'])
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0.25, learning_rate=0.01,
           max_delta_step=0, max_depth=12, min_child_weight=3, missing=None,
           n_estimators=200, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
           subsample=0.5)




```python
test_df['Good_Bad'] = best_xgb_model.predict(test_df[engineered_features])
##Second submission
sub = test_df[['Day_Id','Good_Bad']].copy()
sub=sub.reindex(columns=["Day_Id","Good_Bad"])
filename = 'submission.csv'
sub.to_csv(filename, index=False)
```

##################################################################

Due to computational limitation and time constraints we'll perfor hyper parameter tuning with Bayesian optimization instead of a comprehensive grid search of parameters


```python
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
ITERATIONS = 25 # 1000
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = make_scorer(f1_score),
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
```


```python
xgb_new_result = bayes_cv_tuner.fit(train_df_final[engineered_features].values, train_labels.values, callback=status_print)
```

    Model #1
    Best ROC-AUC: 0.5
    Best params: {'colsample_bylevel': 0.4160029192647807, 'colsample_bytree': 0.7304484857455519, 'gamma': 0.13031389926541354, 'learning_rate': 0.042815319280763466, 'max_delta_step': 13, 'max_depth': 21, 'min_child_weight': 2, 'n_estimators': 87, 'reg_alpha': 5.497557739289786e-07, 'reg_lambda': 0.05936070635912049, 'scale_pos_weight': 0.060830282487222144, 'subsample': 0.13556548021189216}
    
    Model #2
    Best ROC-AUC: 0.6225
    Best params: {'colsample_bylevel': 0.8390144719977516, 'colsample_bytree': 0.8844821246070537, 'gamma': 4.358684608480795e-07, 'learning_rate': 0.7988179462781242, 'max_delta_step': 17, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 68, 'reg_alpha': 0.0005266983003701547, 'reg_lambda': 276.5424475574225, 'scale_pos_weight': 0.3016410771843142, 'subsample': 0.9923710598637134}
    
    Model #3
    Best ROC-AUC: 0.6762
    Best params: {'colsample_bylevel': 0.4503841871781403, 'colsample_bytree': 0.9195352964526833, 'gamma': 8.168958221061441e-09, 'learning_rate': 0.07356404539935663, 'max_delta_step': 4, 'max_depth': 23, 'min_child_weight': 1, 'n_estimators': 88, 'reg_alpha': 0.00010376808625045426, 'reg_lambda': 476.96194787286544, 'scale_pos_weight': 1.3165669602830552, 'subsample': 0.387658500562527}
    
    Model #4
    Best ROC-AUC: 0.6762
    Best params: {'colsample_bylevel': 0.4503841871781403, 'colsample_bytree': 0.9195352964526833, 'gamma': 8.168958221061441e-09, 'learning_rate': 0.07356404539935663, 'max_delta_step': 4, 'max_depth': 23, 'min_child_weight': 1, 'n_estimators': 88, 'reg_alpha': 0.00010376808625045426, 'reg_lambda': 476.96194787286544, 'scale_pos_weight': 1.3165669602830552, 'subsample': 0.387658500562527}
    
    Model #5
    Best ROC-AUC: 0.6762
    Best params: {'colsample_bylevel': 0.4503841871781403, 'colsample_bytree': 0.9195352964526833, 'gamma': 8.168958221061441e-09, 'learning_rate': 0.07356404539935663, 'max_delta_step': 4, 'max_depth': 23, 'min_child_weight': 1, 'n_estimators': 88, 'reg_alpha': 0.00010376808625045426, 'reg_lambda': 476.96194787286544, 'scale_pos_weight': 1.3165669602830552, 'subsample': 0.387658500562527}
    
    Model #6
    Best ROC-AUC: 0.6762
    Best params: {'colsample_bylevel': 0.4503841871781403, 'colsample_bytree': 0.9195352964526833, 'gamma': 8.168958221061441e-09, 'learning_rate': 0.07356404539935663, 'max_delta_step': 4, 'max_depth': 23, 'min_child_weight': 1, 'n_estimators': 88, 'reg_alpha': 0.00010376808625045426, 'reg_lambda': 476.96194787286544, 'scale_pos_weight': 1.3165669602830552, 'subsample': 0.387658500562527}
    
    Model #7
    Best ROC-AUC: 0.6762
    Best params: {'colsample_bylevel': 0.4503841871781403, 'colsample_bytree': 0.9195352964526833, 'gamma': 8.168958221061441e-09, 'learning_rate': 0.07356404539935663, 'max_delta_step': 4, 'max_depth': 23, 'min_child_weight': 1, 'n_estimators': 88, 'reg_alpha': 0.00010376808625045426, 'reg_lambda': 476.96194787286544, 'scale_pos_weight': 1.3165669602830552, 'subsample': 0.387658500562527}
    
    Model #8
    Best ROC-AUC: 0.6762
    Best params: {'colsample_bylevel': 0.4503841871781403, 'colsample_bytree': 0.9195352964526833, 'gamma': 8.168958221061441e-09, 'learning_rate': 0.07356404539935663, 'max_delta_step': 4, 'max_depth': 23, 'min_child_weight': 1, 'n_estimators': 88, 'reg_alpha': 0.00010376808625045426, 'reg_lambda': 476.96194787286544, 'scale_pos_weight': 1.3165669602830552, 'subsample': 0.387658500562527}
    
    Model #9
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #10
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #11
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #12
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #13
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #14
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #15
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #16
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #17
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #18
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #19
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #20
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #21
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #22
    Best ROC-AUC: 0.6794
    Best params: {'colsample_bylevel': 0.955923206446829, 'colsample_bytree': 0.7036152301751524, 'gamma': 0.03823613443879595, 'learning_rate': 0.06786442521779147, 'max_delta_step': 8, 'max_depth': 11, 'min_child_weight': 0, 'n_estimators': 69, 'reg_alpha': 0.00022356829889037284, 'reg_lambda': 1.2908532337409298e-07, 'scale_pos_weight': 4.73588486119117, 'subsample': 0.4499578015509351}
    
    Model #23
    Best ROC-AUC: 0.6875
    Best params: {'colsample_bylevel': 0.7307968509811582, 'colsample_bytree': 0.5179152536187541, 'gamma': 0.0018406853480943382, 'learning_rate': 0.023835431281720304, 'max_delta_step': 3, 'max_depth': 23, 'min_child_weight': 0, 'n_estimators': 56, 'reg_alpha': 0.833624385244558, 'reg_lambda': 1.4112196295181561e-09, 'scale_pos_weight': 4.328907304806557, 'subsample': 0.37941367765437495}
    
    Model #24
    Best ROC-AUC: 0.6875
    Best params: {'colsample_bylevel': 0.7307968509811582, 'colsample_bytree': 0.5179152536187541, 'gamma': 0.0018406853480943382, 'learning_rate': 0.023835431281720304, 'max_delta_step': 3, 'max_depth': 23, 'min_child_weight': 0, 'n_estimators': 56, 'reg_alpha': 0.833624385244558, 'reg_lambda': 1.4112196295181561e-09, 'scale_pos_weight': 4.328907304806557, 'subsample': 0.37941367765437495}
    
    Model #25
    Best ROC-AUC: 0.6875
    Best params: {'colsample_bylevel': 0.7307968509811582, 'colsample_bytree': 0.5179152536187541, 'gamma': 0.0018406853480943382, 'learning_rate': 0.023835431281720304, 'max_delta_step': 3, 'max_depth': 23, 'min_child_weight': 0, 'n_estimators': 56, 'reg_alpha': 0.833624385244558, 'reg_lambda': 1.4112196295181561e-09, 'scale_pos_weight': 4.328907304806557, 'subsample': 0.37941367765437495}
    



```python
Best_params = {'colsample_bylevel': 1.0, 'colsample_bytree': 1.0, 'gamma': 2.92304761271297e-05, 'learning_rate': 0.23405813513519594, 'max_delta_step': 9, 'max_depth': 50, 'min_child_weight': 1, 'n_estimators': 70, 'reg_alpha': 1.0, 'reg_lambda': 1e-09, 'scale_pos_weight': 1.7055606501171967, 'subsample': 0.7217811456576027}


```


```python
upd_xgb_model = XGBClassifier(**Best_params)##(base_score=0.5, booster='gbtree', colsample_bylevel=0.1,

```


```python
upd_xgb_results = stratify_kfold_cv(train_df_final,upd_xgb_model,all_sensor_cols+new_feat_cols+new_feat_cols2+new_feat_cols1,weights=True)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
upd_xgb_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1Fold</th>
      <th>2Fold</th>
      <th>3Fold</th>
      <th>4Fold</th>
      <th>5Fold</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FN</th>
      <td>45.000000</td>
      <td>51.000000</td>
      <td>47.000000</td>
      <td>40.000000</td>
      <td>43.000000</td>
      <td>226.000000</td>
    </tr>
    <tr>
      <th>FP</th>
      <td>44.000000</td>
      <td>54.000000</td>
      <td>51.000000</td>
      <td>56.000000</td>
      <td>57.000000</td>
      <td>262.000000</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>138.000000</td>
      <td>133.000000</td>
      <td>131.000000</td>
      <td>126.000000</td>
      <td>124.000000</td>
      <td>652.000000</td>
    </tr>
    <tr>
      <th>TP</th>
      <td>108.000000</td>
      <td>97.000000</td>
      <td>106.000000</td>
      <td>111.000000</td>
      <td>108.000000</td>
      <td>530.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.734328</td>
      <td>0.686567</td>
      <td>0.707463</td>
      <td>0.711712</td>
      <td>0.698795</td>
      <td>0.707784</td>
    </tr>
    <tr>
      <th>f1score</th>
      <td>0.708197</td>
      <td>0.648829</td>
      <td>0.683871</td>
      <td>0.698113</td>
      <td>0.683544</td>
      <td>0.684755</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.710526</td>
      <td>0.642384</td>
      <td>0.675159</td>
      <td>0.664671</td>
      <td>0.654545</td>
      <td>0.669192</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.705882</td>
      <td>0.655405</td>
      <td>0.692810</td>
      <td>0.735099</td>
      <td>0.715232</td>
      <td>0.701058</td>
    </tr>
    <tr>
      <th>roc_auc_score</th>
      <td>0.792430</td>
      <td>0.754119</td>
      <td>0.744308</td>
      <td>0.766429</td>
      <td>0.766273</td>
      <td>0.764469</td>
    </tr>
  </tbody>
</table>
</div>




```python
search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    }

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        n_jobs=4,
        **params
    )
    
    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score

space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)
```


```python
upd_xgb_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1Fold</th>
      <th>2Fold</th>
      <th>3Fold</th>
      <th>4Fold</th>
      <th>5Fold</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FN</th>
      <td>45.000000</td>
      <td>52.000000</td>
      <td>48.000000</td>
      <td>44.000000</td>
      <td>43.000000</td>
      <td>232.000000</td>
    </tr>
    <tr>
      <th>FP</th>
      <td>49.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>59.000000</td>
      <td>58.000000</td>
      <td>272.000000</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>133.000000</td>
      <td>134.000000</td>
      <td>129.000000</td>
      <td>123.000000</td>
      <td>123.000000</td>
      <td>642.000000</td>
    </tr>
    <tr>
      <th>TP</th>
      <td>108.000000</td>
      <td>96.000000</td>
      <td>105.000000</td>
      <td>107.000000</td>
      <td>108.000000</td>
      <td>524.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.719403</td>
      <td>0.686567</td>
      <td>0.698507</td>
      <td>0.690691</td>
      <td>0.695783</td>
      <td>0.698204</td>
    </tr>
    <tr>
      <th>f1score</th>
      <td>0.696774</td>
      <td>0.646465</td>
      <td>0.675241</td>
      <td>0.675079</td>
      <td>0.681388</td>
      <td>0.675258</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.687898</td>
      <td>0.644295</td>
      <td>0.664557</td>
      <td>0.644578</td>
      <td>0.650602</td>
      <td>0.658291</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.705882</td>
      <td>0.648649</td>
      <td>0.686275</td>
      <td>0.708609</td>
      <td>0.715232</td>
      <td>0.693122</td>
    </tr>
    <tr>
      <th>roc_auc_score</th>
      <td>0.802306</td>
      <td>0.730055</td>
      <td>0.753178</td>
      <td>0.757914</td>
      <td>0.775383</td>
      <td>0.763625</td>
    </tr>
  </tbody>
</table>
</div>




```python
upd_xgb_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1Fold</th>
      <th>2Fold</th>
      <th>3Fold</th>
      <th>4Fold</th>
      <th>5Fold</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FN</th>
      <td>153.000000</td>
      <td>148.000000</td>
      <td>153.000000</td>
      <td>147.000000</td>
      <td>148.000000</td>
      <td>749.000000</td>
    </tr>
    <tr>
      <th>FP</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>182.000000</td>
      <td>187.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>181.000000</td>
      <td>914.000000</td>
    </tr>
    <tr>
      <th>TP</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.543284</td>
      <td>0.558209</td>
      <td>0.543284</td>
      <td>0.558559</td>
      <td>0.554217</td>
      <td>0.551497</td>
    </tr>
    <tr>
      <th>f1score</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.051613</td>
      <td>0.038961</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026490</td>
      <td>0.019868</td>
      <td>0.009259</td>
    </tr>
    <tr>
      <th>roc_auc_score</th>
      <td>0.794656</td>
      <td>0.750397</td>
      <td>0.756482</td>
      <td>0.758824</td>
      <td>0.776225</td>
      <td>0.767233</td>
    </tr>
  </tbody>
</table>
</div>




```python
upd_xgb_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1Fold</th>
      <th>2Fold</th>
      <th>3Fold</th>
      <th>4Fold</th>
      <th>5Fold</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FN</th>
      <td>145.000000</td>
      <td>147.000000</td>
      <td>151.000000</td>
      <td>145.000000</td>
      <td>142.000000</td>
      <td>730.000000</td>
    </tr>
    <tr>
      <th>FP</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>182.000000</td>
      <td>186.000000</td>
      <td>182.000000</td>
      <td>182.000000</td>
      <td>178.000000</td>
      <td>910.000000</td>
    </tr>
    <tr>
      <th>TP</th>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>9.000000</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.567164</td>
      <td>0.558209</td>
      <td>0.549254</td>
      <td>0.564565</td>
      <td>0.563253</td>
      <td>0.560479</td>
    </tr>
    <tr>
      <th>f1score</th>
      <td>0.099379</td>
      <td>0.013333</td>
      <td>0.025806</td>
      <td>0.076433</td>
      <td>0.110429</td>
      <td>0.066158</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.750000</td>
      <td>0.866667</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.052288</td>
      <td>0.006757</td>
      <td>0.013072</td>
      <td>0.039735</td>
      <td>0.059603</td>
      <td>0.034392</td>
    </tr>
    <tr>
      <th>roc_auc_score</th>
      <td>0.806795</td>
      <td>0.755998</td>
      <td>0.763305</td>
      <td>0.754712</td>
      <td>0.781311</td>
      <td>0.772210</td>
    </tr>
  </tbody>
</table>
</div>



For example if our data has 90% negative samples and 10% positive samples which is very imbalanced.
we'd try to use the parameter of scale_pos_weight and set it as 9.

It doesn't modify your data but change the weights of the positive obeservations.
The approache is called cost sensitive, the idea is to force the model to take care of the rare event by increasing the loss if it fails to correctly predict them.
It works much better than subsampling and co.


```python
#Model #20
#Best ROC-AUC: 0.7781
Best_params=  {'colsample_bylevel': 0.8659659275316633, 'colsample_bytree': 0.5502981882831188, 'gamma': 0.020737446920009037, 'learning_rate': 0.1416947638649497, 'max_delta_step': 19, 'max_depth': 38, 'min_child_weight': 3, 'n_estimators': 82, 'reg_alpha': 1.1629841239477682e-07, 'reg_lambda': 0.016531443607570532, 'scale_pos_weight': 0.059869588007783255, 'subsample': 1.0}

```


```python
best_xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8659659275316633,
       colsample_bytree=0.5502981882831188, gamma=0.020737446920009037, learning_rate=0.1416947638649497, max_delta_step=19,
       max_depth=38, min_child_weight=3, missing=None, n_estimators=82,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1.1629841239477682e-07, reg_lambda=0.016531443607570532, scale_pos_weight=0.059869588007783255, seed=None,
       silent=True, subsample=1.0
                              )
```


```python
np.linspace(0,1,20)
```




    array([0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,
           0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
           0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
           0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.        ])




```python
from sklearn.metrics import make_scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(f1_score)}
xgb_test = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8659659275316633,
       colsample_bytree=0.5502981882831188, gamma=0.020737446920009037, learning_rate=0.1416947638649497, max_delta_step=19,
       max_depth=38, min_child_weight=3, missing=None, n_estimators=82,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1.1629841239477682e-07, reg_lambda=0.016531443607570532, seed=None,
       silent=True, subsample=1.0)
grid_xgb = {'scale_pos_weight':np.linspace(0,1,20)}
clf_random = GridSearchCV(estimator = xgb_test,scoring=scoring, param_grid = grid_xgb,  cv = 3, refit='AUC', return_train_score=True)
# Fit the random search model
clf_random.fit(train_df_final[all_sensor_cols+new_feat_cols+new_feat_cols1], train_labels)
print(">>Best_params<<\n",clf_random.best_params_)
print(">>Best_score<<\n",clf_random.best_score_)
#clf_random.best_estimator_
```

    >>Best_params<<
     {'scale_pos_weight': 0.10526315789473684}
    >>Best_score<<
     0.7328617527671146



```python
final_best_xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8659659275316633,
       colsample_bytree=0.5502981882831188, gamma=0.020737446920009037, learning_rate=0.1416947638649497, max_delta_step=19,
       max_depth=38, min_child_weight=3, missing=None, n_estimators=82,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1.1629841239477682e-07, reg_lambda=0.016531443607570532, scale_pos_weight=0.04040404040404041, seed=None,
       silent=True, subsample=1.0
                              )
```


```python
final_xgb_best_results = stratify_kfold_cv(train_df_final,final_best_xgb_model,new_feat_cols1+new_feat_cols+all_sensor_cols)
```

    1fold completed
    2fold completed
    3fold completed
    4fold completed
    5fold completed



```python
final_xgb_best_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1Fold</th>
      <th>2Fold</th>
      <th>3Fold</th>
      <th>4Fold</th>
      <th>5Fold</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FN</th>
      <td>133.000000</td>
      <td>137.000000</td>
      <td>136.000000</td>
      <td>132.000000</td>
      <td>122.000000</td>
      <td>660.000000</td>
    </tr>
    <tr>
      <th>FP</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>181.000000</td>
      <td>183.000000</td>
      <td>178.000000</td>
      <td>181.000000</td>
      <td>175.000000</td>
      <td>898.000000</td>
    </tr>
    <tr>
      <th>TP</th>
      <td>20.000000</td>
      <td>11.000000</td>
      <td>17.000000</td>
      <td>19.000000</td>
      <td>29.000000</td>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.600000</td>
      <td>0.579104</td>
      <td>0.582090</td>
      <td>0.600601</td>
      <td>0.614458</td>
      <td>0.595210</td>
    </tr>
    <tr>
      <th>f1score</th>
      <td>0.229885</td>
      <td>0.134969</td>
      <td>0.195402</td>
      <td>0.222222</td>
      <td>0.311828</td>
      <td>0.221198</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.952381</td>
      <td>0.733333</td>
      <td>0.809524</td>
      <td>0.950000</td>
      <td>0.828571</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.130719</td>
      <td>0.074324</td>
      <td>0.111111</td>
      <td>0.125828</td>
      <td>0.192053</td>
      <td>0.126984</td>
    </tr>
    <tr>
      <th>roc_auc_score</th>
      <td>0.807405</td>
      <td>0.763152</td>
      <td>0.771170</td>
      <td>0.753002</td>
      <td>0.778932</td>
      <td>0.774892</td>
    </tr>
  </tbody>
</table>
</div>




```python
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([roc_auc_score(train_labels, xpred>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())
print(best_threshold)
```

    0.708771838421729
    0.41000000000000003



![png](output_128_1.png)



```python
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1Fold</th>
      <th>2Fold</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy</th>
      <td>0.712575</td>
      <td>0.706587</td>
      <td>0.709581</td>
    </tr>
    <tr>
      <th>f1score</th>
      <td>0.712575</td>
      <td>0.706587</td>
      <td>0.667124</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.712575</td>
      <td>0.706587</td>
      <td>0.693295</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.712575</td>
      <td>0.706587</td>
      <td>0.642857</td>
    </tr>
    <tr>
      <th>roc_auc_score</th>
      <td>0.712575</td>
      <td>0.706587</td>
      <td>0.770196</td>
    </tr>
  </tbody>
</table>
</div>


