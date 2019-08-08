#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn
# ## Task:
# ## 1. predict which customers are at high risk of churn.
# ## 2. identify the main indicators of churn.

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[11]:


#to Suppress unnecessary warnings
warnings.filterwarnings("ignore")


# In[12]:


teleDataFile=r'telecom_churn_data.csv'


# In[13]:


teleData= pd.read_csv(teleDataFile)


# In[14]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(teleData.info(verbose=True,null_counts =True))


# In[15]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData.describe())


# In[16]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData.head(5))


# In[17]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(teleData.nunique(axis=0,dropna=False))


# In[18]:


for col in list(teleData.columns):
    if(col=='mobile_number' or teleData[col].nunique()>100 ): continue
    else:
        print(col+":"+str(teleData[col].unique().tolist()))
        print("----------------------------------------------------------------------------------")


# In[ ]:




