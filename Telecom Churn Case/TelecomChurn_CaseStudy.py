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


# # Data Preperation

# Finding the high values customers:
# 1. Creating a new columns that has average of the revenues of 6th and 7th month
# 2. Finding the 70th percentile and above.
# 3. Filtering the data to get the High Valued customers
# 

# As we can see that all the Average Revenue Columns have no null values, we can proceed forward with finding the Average
# 1. arpu_6        ---              99999 non-null float64
# 2. arpu_7        ---              99999 non-null float64
# 3. arpu_8        ---              99999 non-null float64
# 4. arpu_9        ---              99999 non-null float64

# In[10]:


teleData['avg_6_7_revenue']=(teleData["arpu_6"]+teleData["arpu_7"])/2


# In[12]:


percentile70th= teleData['avg_6_7_revenue'].quantile(0.70)
print("The 70th percentile average revenue is "+ str(percentile70th))


# In[13]:


teleData_HighValuesCustomers= teleData[teleData['avg_6_7_revenue']>=percentile70th]
teleData_HighValuesCustomers.shape


# Taking a look into the High Valued customers dataset:

# In[16]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData_HighValuesCustomers.head(10))


# Now making all Null values 0.

# In[18]:


teleData_HighValuesCustomers_Treated=teleData_HighValuesCustomers.fillna(0)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(teleData_HighValuesCustomers_Treated.info(verbose=True,null_counts =True))


# We can see that all the null values have been treated.
# Thus teleData_HighValuesCustomers_Treated is our dataset that contains the high valued customer with treated data.
# The next step is to tag the numbers as churned(1) and not churned(0)based on the following columns:
# 1. total_ic_mou_9
# 2. total_og_mou_9
# 3. vol_2g_mb_9
# 4. vol_3g_mb_9

# In[35]:


filter=((teleData_HighValuesCustomers_Treated['total_ic_mou_9']==0)&(teleData_HighValuesCustomers_Treated['vol_2g_mb_9']==0)&(teleData_HighValuesCustomers_Treated['vol_3g_mb_9']==0)&(teleData_HighValuesCustomers_Treated['total_og_mou_9']==0))
teleData_HighValuesCustomers_Treated['churned_tag']=np.where(filter, 1, 0)
teleData_HighValuesCustomers_Treated['churned_tag'].sum()


# In[26]:


#filter=((teleData_HighValuesCustomers_Treated['total_ic_mou_9']==0)&(teleData_HighValuesCustomers_Treated['vol_2g_mb_9']==0)&(teleData_HighValuesCustomers_Treated['vol_3g_mb_9']==0)&(teleData_HighValuesCustomers_Treated['total_og_mou_9']==0))
#tagged=teleData_HighValuesCustomers_Treated.ix[filter,list(teleData_HighValuesCustomers_Treated.columns)]
#display(tagged.index)


# In[33]:


teleData_HighValuesCustomers_Treated.head(5)


# Now we can see that the rows have a churned indicator against them.
# We can delete all the columns that are related to the churn phase i.e. the '_9' columns
# 
# Removing all columns that have '_9' in them.

# In[51]:


print(len([x for x in teleData_HighValuesCustomers_Treated.columns if '_9' in x]))
print(len(teleData_HighValuesCustomers_Treated.columns))
teleData_HighValuesCustomers_Treated.shape


# In[53]:


colsToDrop=[x for x in teleData_HighValuesCustomers_Treated.columns if '_9' in x]


# In[54]:


teleData_HighValuesCustomers_Tagged=teleData_HighValuesCustomers_Treated.drop(colsToDrop,axis=1)


# In[55]:


teleData_HighValuesCustomers_Tagged.shape


# Now we have removed all the columns that are related to the churn phase.
# Lets have a look at the dataset now.

# In[57]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData_HighValuesCustomers_Tagged.head(5))


# We might not need the date columns, so removing them as well.
# 
# creating a new dataset _naDates 

# In[58]:


teleData_HighValuesCustomers_naDates=teleData_HighValuesCustomers_Tagged.drop([x for x in teleData_HighValuesCustomers_Tagged.columns if 'date' in x],axis=1)
teleData_HighValuesCustomers_naDates.shape


# In[59]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(teleData_HighValuesCustomers_naDates.head(5))


# In[ ]:




