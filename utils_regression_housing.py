#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Major Libraries for utils
import pandas as pd
import os
## sklearn -- for pipeline and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


# In[2]:


## Read the CSV file using pandas
FILE_PATH = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(FILE_PATH)


# In[3]:


## Replace the  (<1H OCEAN) to (1H OCEAN) -- will cause ane errors in Deploymnet
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')


# In[4]:


## Try to make some Feature Engineering --> Feature Extraction --> Add the new column to the main DF
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedroms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_household'] = df_housing['population'] / df_housing['households']


# In[5]:


## Split the Dataset -- Taking only train to fit (the same the model was trained on)
X = df_housing.drop(columns=['median_house_value'], axis=1)   ## Features
y = df_housing['median_house_value']   ## target

## the same Random_state (take care)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42) 


# In[6]:


## Split the Dataset -- Taking only train to fit (the same the model was trained on)
X = df_housing.drop(columns=['median_house_value'], axis=1)   ## Features
y = df_housing['median_house_value']   ## target

## the same Random_state (take care)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42) 

## Separete the columns according to type (numerical or categorical)
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float32', 'float64', 'int32', 'int64']]
categ_cols = [col for col in X_train.columns if X_train[col].dtype not in ['float32', 'float64', 'int32', 'int64']]


# In[7]:


## numerical pipeline
num_pipeline = Pipeline([
                        ('selector', DataFrameSelector(num_cols)),    ## select only these columns
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                        ])

## categorical pipeline
categ_pipeline = Pipeline(steps=[
            ('selector', DataFrameSelector(categ_cols)),    ## select only these columns
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OHE', OneHotEncoder(sparse=False))])

## concatenate both two pipelines
total_pipeline = FeatureUnion(transformer_list=[
                                            ('num_pipe', num_pipeline),
                                            ('categ_pipe', categ_pipeline)
                                               ]
                             )

X_train_final = total_pipeline.fit_transform(X_train) ## fit 


# we must use .fit_transform to make the model train on the original data set

# In[9]:


def preprocess_new(X_new):
    '''this function is used for the preprocessing for 
    a new data before prediction
    arguments
    *********
    (X_new: 2D array) --> The Features in the same order
                ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                 'population', 'households', 'median_income', 'ocean_proximity']
        All Featutes are Numerical, except the last one is Categorical.
    returns
    *******
    Preprocessed Features ready to make inference by the Model
    '''
    return total_pipeline.transform(X_new)
    


# In[ ]:




