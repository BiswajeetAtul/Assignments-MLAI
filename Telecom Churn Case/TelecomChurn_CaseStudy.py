#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn
# ## Task:
# ## 1. predict which customers are at high risk of churn.
# ## 2. identify the main indicators of churn.

# Importing all necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# Code to filter unnecessary warnings

# In[2]:


#to Suppress unnecessary warnings
warnings.filterwarnings("ignore")


# Defining the path to the Dataset

# In[3]:


teleDataFile=r'telecom_churn_data.csv'


# Reading the Dataset

# In[4]:


teleData= pd.read_csv(teleDataFile)


# Finding out the number of non-null values in the dataset

# In[5]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(teleData.info(verbose=True,null_counts =True))


# looking into the stats of all the columns of the dataset 

# In[6]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData.describe())


# Taking a peek into the dataset

# In[7]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData.head(5))


# Finding the number of unique values in each column of the dataset

# In[8]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(teleData.nunique(axis=0,dropna=False))


# we can see that there area columns with 1 or 2 unique values to as high as 82k unique values(not taking into account the mobile_number which ofcourse will have unique values) 
# 
# Printing all the unique values for columns with less then 100 unique values including null/nan:

# In[9]:


for col in list(teleData.columns):
    if(teleData[col].nunique()>100 ): continue
    else:
        print(col+":"+str(teleData[col].unique().tolist()))
        print("----------------------------------------------------------------------------------")


# In[ ]:




