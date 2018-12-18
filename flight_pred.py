
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os,sys


# In[2]:


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


# In[3]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[4]:


import sklearn
sklearn.__version__


# In[5]:


pwd()


# In[6]:


os.chdir('flight_predictor_data')
train_df = pd.read_csv('weather_data_train.csv')
test_df = pd.read_csv('weather_data_test.csv')


# In[7]:


train_df.shape,test_df.shape


# In[8]:


pred = pd.read_csv('flight_data_train.csv')


# In[9]:


pred.head()


# In[10]:


train_df.describe()


# 5 weather stations 
# Every day around 2 AM weather data is collected for 45 heights at equal height intervals at each of these stations. Weather data collected includes pressure, temperature, dew point, wind speed and wind direction for each height. Using this weather data Laxman wants to predict whether it will be a good day or bad day for paragliding.

# In[11]:


import re
dew_point_cols = [x for x in train_df.columns if re.search(r".*Dew Point.*",x)]
pressure_cols = [x for x in train_df.columns if re.search(r".*Pressure.*",x)]
temperature_cols = [x for x in train_df.columns if re.search(r".*Temperature.*",x)]
wind_direction_cols = [x for x in train_df.columns if re.search(r".*Wind Direction.*",x)]
wind_speed_cols = [x for x in train_df.columns if re.search(r".*Wind Speed.*",x)]


# In[12]:


all_sensor_cols = dew_point_cols+pressure_cols+temperature_cols+wind_direction_cols+wind_speed_cols


# In[13]:


len(all_sensor_cols)


# ##### Checking for Outliers and treating
# We have information that sometimes sensor devices at weather stations record wrong weather parameters, so we'll check for outliers and remove them using z score method. We could use robust z score method if we think the outlier values are very large and impacts the calculation of mean and standard deviation which is used in calculation z score,in our case it was checked and the normal z score method was able to identify ouliers with observations having absolute value of z score greater than 3.

# In[14]:


station1temp = [x for x in temperature_cols if re.search(r".*Station1.*",x)] 
#Station3 Dew Point Height1


# In[15]:


fig,ax = plt.subplots(figsize=(14,6))
g = sns.boxplot(x="variable", y="value", data=pd.melt(train_df[station1temp]))
g.set_xticklabels(g.get_xticklabels(), rotation=80)
plt.show()


# As we can see in the boxplot of sensor measurement(temperature),there are outliers due to faulty sensors and needs to be removed before training the model,hence we calculate the z score and remove obsv with absolute values > 3 standard deviations.
def outliers_iqr(col,k=1.5):
    quartile_1, quartile_3 = np.percentile(col, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * k)
    upper_bound = quartile_3 + (iqr * k)
    #print(col.name,sum([x < lower_bound or x > upper_bound for x in col]))
    return [True if x < lower_bound or x > upper_bound else False for x in col]
# from scipy import stats
# import numpy as np
# def remove_high_lev(df):
#     for station in range(1,6):
#         for col_type in (dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols):         
#             station_no_col_type = [x for x in col_type if re.search(r".*Station"+str(station)+"\s+.*",x)]
#             z = np.abs(stats.zscore(df[station_no_col_type]))                        
#             df[station_no_col_type].drop(df[(z > 2).all(axis=1)].index,inplace=True)
#     return df                           

# In[16]:


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


# In[17]:


outlier_mask = train_df[all_sensor_cols].apply(remove_high_lev)


# In[18]:


outlier_mask.shape,train_df.shape


# In[19]:


train_df1 = train_df[~outlier_mask.any(axis=1)].copy()


# In[20]:


train_df1.shape


# We have removed records with faulty sensor measurements and will verify the same with boxplots.

# In[21]:


fig,ax = plt.subplots(figsize=(14,6))
g = sns.boxplot(x="variable", y="value", data=pd.melt(train_df1[station1temp]))
g.set_xticklabels(g.get_xticklabels(), rotation=80)
plt.show()


# We will join pred dataframe with train data which includes flight details for 288 paragliding spots for several days. Flight details for each spot include max distance, total distance, number of flights that took place at that spot on a particular day. The conclustion is that if the total number of flights combining all these spots is more than or equal to 15 then it is good day else it is a bad day for paragliding. 

# In[22]:


train_df_final = train_df1.merge(pred,on=['Day_Id'],how='left')
import re
total_flight_cols = [x for x in train_df_final.columns if re.search(r".*totalFlights",x)]
train_df_final['total_flights'] = train_df_final[total_flight_cols].sum(axis=1)


# In[23]:


train_df_final['total_flights'].unique()


# In[24]:


fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(15,6))
sns.distplot(train_df_final['total_flights'],ax=ax[0])
sns.boxplot(train_df_final['total_flights'],ax=ax[1],showfliers=False)##we are not showing very large values as we are only interested whether total flights was greater than 15 or not
#sns.swarmplot(train_df_final['total_flights'], zorder=0,ax=ax[1],showfliers=False)
fig.suptitle('Histogram and boxplot of total_flights')


# In[25]:


train_df_final['label'] = np.where(train_df_final['total_flights']>=15,1,0)


# In[143]:


train_df_final['total_flights_bin'] = pd.qcut(train_df_final['total_flights'],q=[0, .25, .5, .75, 1.],labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q'])

train_df_final['total_flights_range'] = pd.qcut(train_df_final['total_flights'],q=[0, .25, .55, .75, 1.])

train_df_final['total_flights_range'] = train_df_final['total_flights_range'].astype('str')

sns.countplot(train_df_final['total_flights_range'])#.value_counts()


# corr_matrix = train_df_final.corr()
# 
# #corr_matrix['total'].sort_values(ascending=False).filter(regex='.*Height1$', axis=0)
# 
# corr_matrix['total_flights'].sort_values(ascending=False).filter(regex='.*Height45$', axis=0)

# In[29]:


train = train_df_final.copy()#[all_sensor_cols].copy()#.drop(columns=pred.columns,axis=1)
train_labels = train_df_final['label'].copy()
#train.drop(columns=['total_flights','label'],inplace=True)

test = test_df.copy()#[all_sensor_cols].copy()#.drop(columns=['Day_Id'],inplace=True)


# In[30]:


train.shape,train_labels.shape,test.shape


# ## Model Building & Validation

# In[31]:


def random_grid_search_params(clf,random_grid,features):
    clf_random = RandomizedSearchCV(estimator = clf,scoring='roc_auc', param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    clf_random.fit(train_df_final[features], train_labels)
    print(">>Best_params<<\n",clf_random.best_params_)
    print(">>Best_score<<\n",clf_random.best_score_)
    return clf_random.best_estimator_


# In[32]:


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
    


# #### Random forest w/ RandomizedsearchCV - Original features
# We will start with the original features avaiable and fit randomforest model to our data and check the performance using cross-validation mechanism

# In[33]:


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


# In[34]:


rf_final = random_grid_search_params(rf_clf,rf_param_grid,all_sensor_cols)


# In[35]:


rf_all_results=stratify_kfold_cv(train_df_final,rf_final,all_sensor_cols)##0.7666096465330601


# In[36]:


rf_all_results.T

As we can see,the initial model provides a reasonable performance  although in our case we would be more interested in the precision of our model because the risk in predicting it is safe for flight when it actually isn't is very high and hence would like to improve our model with better features via feature engineering and other transformations of features before training the model
# In[39]:


feat_imp = pd.DataFrame(rf_final.feature_importances_)
feat_imp.columns = ['Feature_importance']
feat_imp = feat_imp.assign(Column_name=all_sensor_cols)
feat_imp.sort_values(by=['Feature_importance'],ascending=False).set_index('Column_name').head(30).plot(kind='barh')

We cannot infer much from the feature importances due to the fact that our variable measurements would be highly correlated with measurements at different heights being used as features and hence the feature importance which is calcualted by mean reduction in impurity due to each feature at each split would be reduced.Although the importance of temperature and pressure is highlighted here.

Qouted from link "http://blog.datadive.net/selecting-good-features-part-iii-random-forests/"
"Random forest consists of a number of decision trees. Every node in the decision trees is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure based on which the (locally) optimal condition is chosen is called impurity. For classification, it is typically either Gini impurity or information gain/entropy and for regression trees it is variance. Thus when training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure."

"There are a few things to keep in mind when using the impurity based ranking. Firstly, feature selection based on impurity reduction is biased towards preferring variables with more categories (see Bias in random forest variable importance measures). Secondly, when the dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be used as the predictor, with no concrete preference of one over the others. But once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature. As a consequence, they will have a lower reported importance. This is not an issue when we want to use feature selection to reduce overfitting, since it makes sense to remove features that are mostly duplicated by other features. But when interpreting the data, it can lead to the incorrect conclusion that one of the variables is a strong predictor while the others in the same group are unimportant, while actually they are very close in terms of their relationship with the response variable.

The effect of this phenomenon is somewhat reduced thanks to random selection of features at each node creation, but in general the effect is not removed completely. For example, we have three correlated variables X0,X1,X2, and no noise in the data, with the output variable simply being the sum of the three features:
When we compute the feature importances, we see that X1 is computed to have over 10x higher importance than X2, while their “true” importance is very similar. This happens despite the fact that the data is noiseless, we use 20 trees, random selection of features (at each split, only two of the three features are considered) and a sufficiently large dataset.

One thing to point out though is that the difficulty of interpreting the importance/ranking of correlated variables is not random forest specific, but applies to most model based feature selection methods."
# ### Feature enginnering
# 

# We'll further try to bin measurements at similar heights as there wouldn't be much difference in sensor values.We could check this from boxplot of measurement values at heights closer to each other.

# In[43]:


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


# In[44]:


def overall_measured_values(df,cols,coltype):
    for stn in stns:
        col = [x for x in cols if re.search(r""+str(stn),x)]
        df[stn+" "+coltype+" "+"Height1-45"] = df[col].mean(axis=1)
for i,j in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
    overall_measured_values(train_df_final,j,i)
    overall_measured_values(test_df,j,i)
    


# In[45]:


binning_features = [ x for x in train_df_final.columns if re.search(r"Height\d+-\d+",x)]


# We'll take the weighted average of each sensor measurements at each station based on their absolute values of correlation with total flights such that each feature is weighted differently, and the resulting feature tends to
# be more important in feature selection process while building each individual tree

# In[46]:


for coltype in ['dew','temp','pressure','wind_dir','wind_speed']:
    test = train_df_final[['Station1 '+str(coltype)+' Height1-5','Station1 '+str(coltype)+' Height6-10','Station1 '+str(coltype)+' Height11-15',
                           'Station1 '+str(coltype)+' Height16-20','Station1 '+str(coltype)+' Height21-25','Station1 '+str(coltype)+' Height26-30',
                           'Station1 '+str(coltype)+' Height31-35','Station1 '+str(coltype)+' Height36-40','Station1 '+str(coltype)+' Height41-45',
                           'total_flights']].copy()
    print(test.corr()['total_flights'])


# In[47]:


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


# In[48]:


from sklearn.preprocessing import MinMaxScaler
sclr = MinMaxScaler(feature_range=(0.01,1))
def calc_weights(stn_num,colstype):

    stn1temp = [x for x in colstype if re.search(r""+str(stn_num)+"\s+",x)]
    abs_corr  = abs(train_df_final[stn1temp+['total_flights']].corr()[['total_flights']][:-1])
    weights = sclr.fit_transform(abs_corr)
    #print(weights[0][0])
    wts = [weights[i][0] for i in range(0,45)]
    return wts


# In[49]:


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


# In[50]:



for stn in stns:
    for coltype,cols in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
        col = [x for x in cols if re.search(r""+str(stn)+"\s+",x)]
        train_df_final[str(stn)+" "+str(coltype)+"-wtavg"] = train_df_final[col].apply(lambda x:np.average(x,weights=wts_dict[stn][coltype]),axis=1)
    


# In[51]:


stns=['Station1','Station2','Station3','Station4','Station5']
for stn in stns:
    for coltype,cols in zip(['dew','pressure','temp','wind_dir','wind_speed'],[dew_point_cols,pressure_cols,temperature_cols,wind_direction_cols,wind_speed_cols]):    
        col = [x for x in cols if re.search(r""+str(stn)+"\s+",x)]
        test_df[str(stn)+" "+str(coltype)+"-wtavg"] = test_df[col].apply(lambda x:np.average(x,weights=wts_dict[stn][coltype]),axis=1)
    


# In[52]:


weighted_avg_features = [x for x in train_df_final if re.search(r"wtavg",x)]


# Create features such as max,min,variance,difference of each measurement types

# In[53]:


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


# In[54]:


aggr_features = [x for x in train_df_final.columns if x.endswith(('max','min','max-min','mean','std','var'))]


# In[135]:


###hist plot of binned measurement features
#num = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
numdf=pd.melt(train_df_final,value_vars=new_feat_cols1)
numgrid=sns.FacetGrid(numdf,col='variable',col_wrap=4,sharex=False,sharey=False)
numgrid=numgrid.map(sns.distplot,'value')
numgrid.savefig("All-Station-grouped-measurements-hist-output.png")


# In[152]:


#pd.melt(train_df_final, value_vars=new_feat_cols1,id_vars='total_flights_range')


# In[150]:


def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)
def gen_box_plot(station_no):
    cols = [x for x in new_feat_cols1 if re.search(r"Station"+str(station_no)+"\s",x)]
    p = pd.melt(train_df_final, value_vars=cols,id_vars='total_flights_range')
    g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(boxplot, 'total_flights_range','value')
    g.savefig("Station-"+str(station_no)+"output.png")


# In[151]:


for stns in range(1,6):
    gen_box_plot(stns)


# From the plots,windspeed,temperature and pressure seems to be correlated with the no of flights on a particular day

# From the PCA analysis performed in another notebook we can try using pca features in our model,especially the pressure and temperature components since the first 3 pricipal components are explaining more than 90% variance in data,moreover pressure and temperature plays an importanat role in deciding of flights on a particular day. Refer PCA analysis notebook for further understanding on explained variance for each sensor measurements across stations.

# In[55]:


scaler = StandardScaler()
pcadf = train_df_final[all_sensor_cols+['Day_Id']].copy()
pcadf[all_sensor_cols] = scaler.fit_transform(pcadf[all_sensor_cols])


# In[56]:


pca = PCA(n_components=3)

pcadf['PC1'+'temp'] = pca.fit_transform(pcadf[temperature_cols].values)[:,0]
pcadf['PC2'+'temp'] = pca.fit_transform(pcadf[temperature_cols].values)[:,1]
pcadf['PC3'+'temp'] = pca.fit_transform(pcadf[temperature_cols].values)[:,2]

pcadf['PC1'+'pressure'] = pca.fit_transform(pcadf[pressure_cols].values)[:,0]
pcadf['PC2'+'pressure'] = pca.fit_transform(pcadf[pressure_cols].values)[:,1]
pcadf['PC3'+'pressure'] = pca.fit_transform(pcadf[pressure_cols].values)[:,2]


# In[57]:


train_df_final = train_df_final.merge(pcadf[['PC1temp','PC2temp','PC3temp','PC1pressure','PC2pressure','PC3pressure','Day_Id']],on=['Day_Id'],how='left')


# In[58]:


test_df['PC1'+'temp'] = pca.transform(test_df[temperature_cols].values)[:,0]
test_df['PC2'+'temp'] = pca.transform(test_df[temperature_cols].values)[:,1]
test_df['PC3'+'temp'] = pca.transform(test_df[temperature_cols].values)[:,2]

test_df['PC1'+'pressure'] = pca.transform(test_df[pressure_cols].values)[:,0]
test_df['PC2'+'pressure'] = pca.transform(test_df[pressure_cols].values)[:,1]
test_df['PC3'+'pressure'] = pca.transform(test_df[pressure_cols].values)[:,2]


# In[60]:


pca_features = ['PC1temp','PC2temp','PC3temp','PC1pressure','PC2pressure','PC3pressure']


# In[63]:


engineered_features = binning_features+weighted_avg_features+aggr_features+pca_features


# ### Random forest with new features

# In[64]:


rf_new = random_grid_search_params(rf_clf,rf_param_grid,engineered_features)
rf_new_feat_results = stratify_kfold_cv(train_df_final,rf_new,engineered_features)


# In[65]:


rf_new_feat_results.T


# In[71]:


engg_feat_imp = pd.DataFrame(rf_new.feature_importances_)
engg_feat_imp.columns = ['Feature_importance']
engg_feat_imp = engg_feat_imp.assign(Column_name=engineered_features)
engg_feat_imp.sort_values(by=['Feature_importance'],ascending=False).set_index('Column_name').head(20).plot(kind='barh')


# In[67]:


len(engineered_features),len(all_sensor_cols)


# Slight improvement in performance with comparitively lesser number of features engineered from original features.We have reduced the 1125 dimesional features space to 431 dimensions through feature engineering and further improved the performance

# ## AdaBoost

# In[72]:


from sklearn.ensemble import AdaBoostClassifier
adabst_clf = AdaBoostClassifier(n_estimators=50,
                         learning_rate=0.5,
                         random_state=0)
adabst_result = stratify_kfold_cv(train_df_final,adabst_clf,engineered_features)


# In[77]:


adabst_result.T


# ### Light GBM

# In[75]:


lgbm_model = lgb.LGBMClassifier(lambda_l2=1.0,feature_fraction=0.6,num_boost_round=1200,num_leaves=9)
lgbm_new_feat_results = stratify_kfold_cv(train_df_final,lgbm_model,engineered_features)


# In[78]:


lgbm_new_feat_results.T


# ### XGBOOST w/ new features 

# In[147]:


xgb_model = XGBClassifier()


# In[148]:


#train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(3.0, 15.0]')|(train_df_final['total_flights_range']=='(15.0, 43.0]'),1,0.8)


# In[149]:


xgb_new_feat_results_upd = stratify_kfold_cv(train_df_final,xgb_model,engineered_features)


# In[150]:


xgb_new_feat_results_upd.T


# In[160]:


xgb_model.fit(train_df_final[engineered_features],train_labels)


# In[161]:


test_df['Good_Bad'] = xgb_model.predict(test_df[engineered_features])


# In[162]:


test_df.shape


# In[163]:


##Initial submission
sub = test_df[['Day_Id','Good_Bad']].copy()
sub=sub.reindex(columns=["Day_Id","Good_Bad"])
filename = 'submission.csv'
sub.to_csv(filename, index=False)


# Very useful link to understand the hyperparameters and tune them.
# https://sites.google.com/view/lauraepp/parameters

# ###### Hyper parameter tuning of xgboost

# In[90]:


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


# In[91]:


best_xgb_params = {'subsample': 0.5, 'objective': 'binary:logistic', 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 12, 'learning_rate': 0.01, 'gamma': 0.25, 'colsample_bytree': 1}
best_xgb_model = XGBClassifier(**best_xgb_params)


# In[154]:


best_xgb_new_feat_results = stratify_kfold_cv(train_df_final,best_xgb_model,engineered_features)#,weights=True)


# In[155]:


best_xgb_new_feat_results.T


# We'll provide higher weights to samples in range (3,15] as we would want a higher cost penalty for wrong predictions in this range and eventually increase our precision and decrease FP.

# In[156]:


train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(3.0, 15.0]'),1,0.8)
#train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(15.0, 49.0]'),1,np.nan)
#train_df_final['weights'] = train_df_final['weights'].fillna(0.6)
#train_df_final['weights'] = np.where((train_df_final['total_flights_range']=='(3.0, 15.0]')|(train_df_final['total_flights_range']=='(15.0, 43.0]'),1,0.8)


# In[157]:


best_xgb_new_feat_results1 = stratify_kfold_cv(train_df_final,best_xgb_model,engineered_features,weights=True)


# In[158]:


best_xgb_new_feat_results1.T


# In[166]:


best_xgb_model.fit(train_df_final[engineered_features],train_labels,sample_weight=train_df_final['weights'])


# In[167]:


test_df['Good_Bad'] = best_xgb_model.predict(test_df[engineered_features])
##Second submission
sub = test_df[['Day_Id','Good_Bad']].copy()
sub=sub.reindex(columns=["Day_Id","Good_Bad"])
filename = 'submission.csv'
sub.to_csv(filename, index=False)


# ##################################################################

# Due to computational limitation and time constraints we'll perfor hyper parameter tuning with Bayesian optimization instead of a comprehensive grid search of parameters

# In[159]:


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


# In[87]:


xgb_new_result = bayes_cv_tuner.fit(train_df_final[engineered_features].values, train_labels.values, callback=status_print)


# In[128]:


Best_params = {'colsample_bylevel': 1.0, 'colsample_bytree': 1.0, 'gamma': 2.92304761271297e-05, 'learning_rate': 0.23405813513519594, 'max_delta_step': 9, 'max_depth': 50, 'min_child_weight': 1, 'n_estimators': 70, 'reg_alpha': 1.0, 'reg_lambda': 1e-09, 'scale_pos_weight': 1.7055606501171967, 'subsample': 0.7217811456576027}



# In[129]:


upd_xgb_model = XGBClassifier(**Best_params)##(base_score=0.5, booster='gbtree', colsample_bylevel=0.1,


# In[137]:


upd_xgb_results = stratify_kfold_cv(train_df_final,upd_xgb_model,all_sensor_cols+new_feat_cols+new_feat_cols2+new_feat_cols1,weights=True)


# In[138]:


upd_xgb_results


# In[ ]:


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


# In[132]:


upd_xgb_results


# In[124]:


upd_xgb_results


# In[113]:


upd_xgb_results


# For example if our data has 90% negative samples and 10% positive samples which is very imbalanced.
# we'd try to use the parameter of scale_pos_weight and set it as 9.
# 
# It doesn't modify your data but change the weights of the positive obeservations.
# The approache is called cost sensitive, the idea is to force the model to take care of the rare event by increasing the loss if it fails to correctly predict them.
# It works much better than subsampling and co.

# In[289]:


#Model #20
#Best ROC-AUC: 0.7781
Best_params=  {'colsample_bylevel': 0.8659659275316633, 'colsample_bytree': 0.5502981882831188, 'gamma': 0.020737446920009037, 'learning_rate': 0.1416947638649497, 'max_delta_step': 19, 'max_depth': 38, 'min_child_weight': 3, 'n_estimators': 82, 'reg_alpha': 1.1629841239477682e-07, 'reg_lambda': 0.016531443607570532, 'scale_pos_weight': 0.059869588007783255, 'subsample': 1.0}


# In[300]:


best_xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8659659275316633,
       colsample_bytree=0.5502981882831188, gamma=0.020737446920009037, learning_rate=0.1416947638649497, max_delta_step=19,
       max_depth=38, min_child_weight=3, missing=None, n_estimators=82,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1.1629841239477682e-07, reg_lambda=0.016531443607570532, scale_pos_weight=0.059869588007783255, seed=None,
       silent=True, subsample=1.0
                              )


# In[336]:


np.linspace(0,1,20)


# In[337]:


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


# In[330]:


final_best_xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8659659275316633,
       colsample_bytree=0.5502981882831188, gamma=0.020737446920009037, learning_rate=0.1416947638649497, max_delta_step=19,
       max_depth=38, min_child_weight=3, missing=None, n_estimators=82,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1.1629841239477682e-07, reg_lambda=0.016531443607570532, scale_pos_weight=0.04040404040404041, seed=None,
       silent=True, subsample=1.0
                              )


# In[331]:


final_xgb_best_results = stratify_kfold_cv(train_df_final,final_best_xgb_model,new_feat_cols1+new_feat_cols+all_sensor_cols)


# In[332]:


final_xgb_best_results


# In[179]:


thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([roc_auc_score(train_labels, xpred>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())
print(best_threshold)


# In[175]:


results

