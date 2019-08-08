#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn
# ## Task:
# ## 1. predict which customers are at high risk of churn.
# ## 2. identify the main indicators of churn.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[2]:


#to Suppress unnecessary warnings
warnings.filterwarnings("ignore")


# In[11]:


teleDataFile=r'telecom_churn_data.csv'


# In[12]:


teleData= pd.read_csv(teleDataFile)


# In[20]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(teleData.info(verbose=True,null_counts =True))


# In[17]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData.describe())


# In[23]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData.head(5))


# In[26]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(teleData.nunique(axis=0,dropna=False))


# In[30]:


for col in list(teleData.columns):
    if(col=='mobile_number' or teleData[col].nunique()>100 ): continue
    else:
        print(col+":"+str(teleData[col].unique().tolist()))
        print("----------------------------------------------------------------------------------")


# In[ ]:




