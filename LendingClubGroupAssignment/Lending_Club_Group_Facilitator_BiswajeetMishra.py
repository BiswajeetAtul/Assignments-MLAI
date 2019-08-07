
# coding: utf-8

# # Importing Necessary Libraries

# In[1]:


import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
get_ipython().magic('matplotlib inline')


# # Setting path For file

# In[2]:


try:
    pathMapping= r'loan.csv'
except Exception as ex:
    print(ex)


# # Reading the File and Data Cleaning

# In[3]:


try:
    loanDataset= pd.read_csv(pathMapping,low_memory=False)
    pd.options.display.max_columns = None
    print("Successfully Imported")
except Exception as ex:
    print(ex)


# In[4]:


loanDataset.describe()


# In[5]:


loanDataset.shape


# In[6]:


loanDataset.head(3)


# In[7]:


try:
    #Removing all coulumn and rows where all values are null (missing value) 
    loanDataset.dropna(axis=1,how='all',inplace=True)
    loanDataset.dropna(axis=0,how='all',inplace=True)

    
    #Removing % from int_rate
    loanDataset['int_rate']=loanDataset['int_rate'].str.replace('%','')
    
    #Cleaning emp_length to valid values
    loanDataset['emp_length']=loanDataset['emp_length'].apply(lambda x: '0' if '<' in str(x) else x)
    loanDataset['emp_length'].fillna(-1)
    loanDataset['emp_length']=loanDataset['emp_length'].str.extract('(\d+)')#.astype(int)
    
    #droping columns where there are more then 90% null values or only a single value.
    loanDataset=loanDataset.drop(loanDataset.loc[:,list((100*(loanDataset.isnull().sum()/len(loanDataset.index))>90))].columns, 1)
    loanDataset=loanDataset.drop(loanDataset.loc[:,list(loanDataset.nunique()==1)].columns, 1)
    
    print("After Removal of all rows and column where >90% data is NULL, Shape of dataset:")
    print(loanDataset.shape)
    
    
except Exception as ex:
    print(ex)


# # Obtaining Derived Metrics:

# In[8]:


# Obtaining the length of descriptions:
loanDataset['desc_len']=loanDataset['desc'].str.len()


# In[9]:


#Obtaining Date from issue_d
tempTable=loanDataset['issue_d'].str.split("-", expand = True)
loanDataset['issue_month']=tempTable[0]
loanDataset['issue_year']=tempTable[1]


# In[10]:


# Creating derived variables for last_payment_d
tempTable_lastpaydt=loanDataset['last_pymnt_d'].str.split("-", expand = True)
loanDataset['last_pymnt_yr']=tempTable_lastpaydt[1]
loanDataset['last_pymnt_month']=tempTable_lastpaydt[0]


# In[11]:


#Getting the zipcode fixed, removing its last two characters:
loanDataset['zip_code_clean']=loanDataset['zip_code'].apply(lambda x: x[0:-2])
loanDataset['zip_code_clean'].nunique()


# Removing text columns where there is data redundancy, and irrelevant data

# In[12]:


try:
    #dropping url cloumn as its irrelevant( 'url','desc','title') comments but we dont have means to do sentiment analysis yet.
    loanDataset.drop(['url','title','desc'],axis=1,inplace=True)
    loanDataset.drop(['member_id'],axis=1,inplace=True)
    print("After Removal of all text data columns, Shape of dataset:")
    print(loanDataset.shape)
except Exception as ex:
    print(ex)


# End of Data Cleaning

# # Starting Univariate Analysis

# In[13]:


try:
    noOfUniqueValues=loanDataset.nunique()
    print(noOfUniqueValues)
except Exception as ex:
    print(ex)


# In[14]:


try:
    uniqueValuesOfEachColumn= {}
    for x in list(loanDataset.columns):
        uniqueValuesOfEachColumn[x]=loanDataset[x].unique()
except Exception as ex:
    print(ex)


# In[15]:


try:
    #Grouping all the coulmns with loan_status
    groupedByList={}
    for x in list(loanDataset.columns):
        if(x != 'loan_status'):
            groupedByList[x]=loanDataset.groupby([x,"loan_status"])
        else:
            groupedByList['loan_status']=loanDataset.groupby(["loan_status"])
except Exception as ex:
    print(ex)


# Count of Loan status vs the loan amount

# In[16]:


unstackedPivot=groupedByList['loan_amnt'].size().unstack().fillna(0)
plt.figure(figsize=(20,18))
#plt.set_ylim()
ax1=plt.subplot(131)
plt.plot(unstackedPivot['Fully Paid'], marker='', color='green', label="Fully Paid")
plt.ylabel('Count')
plt.xlabel('Loan Amount')
plt.legend()
ax2=plt.subplot(132, sharex=ax1, sharey=ax1)
plt.plot(unstackedPivot['Current'], marker='', color='blue', label="Current")
plt.ylabel('Count')
plt.xlabel('Loan Amount')
plt.legend()
ax3=plt.subplot(133, sharex=ax1, sharey=ax1)
plt.plot(unstackedPivot['Charged Off'], marker='', color='red',label="Charged Off")
plt.ylabel('Count')
plt.xlabel('Loan Amount')

plt.legend()


# It can be observed that more people take loans of small amounts, and thus this range has higher denstiy of being charged off. However the plots dont provide any insight other then this.

# In[17]:


unstackedPivot=groupedByList['pub_rec_bankruptcies'].size().unstack().fillna(0)
plt.figure(figsize=(12,11))
plt.plot(unstackedPivot['Fully Paid'], marker='', color='green', label="Fully Paid")
plt.plot(unstackedPivot['Current'], marker='', color='blue', label="Current")
plt.plot(unstackedPivot['Charged Off'], marker='', color='red',label="Charged Off")
plt.ylabel('Count')
plt.xlabel('Recorded Bankruptcies')
plt.grid()
plt.legend()


# In[18]:


unstackedPivot=groupedByList['emp_length'].size().unstack().fillna(0)
plt.figure(figsize=(12,11))
plt.plot(unstackedPivot['Fully Paid'], marker='', color='green', label="Fully Paid")
plt.plot(unstackedPivot['Current'], marker='', color='blue', label="Current")
plt.plot(unstackedPivot['Charged Off'], marker='', color='red',label="Charged Off")
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.xlabel('Year of Exp')
plt.grid()


# In[19]:


unstackedPivot=groupedByList['mths_since_last_delinq'].size().unstack().fillna(0)
plt.figure(figsize=(12,8))
#plt.set_ylim()
#ax1=plt.subplot(131)
plt.plot(unstackedPivot['Fully Paid'], marker='', color='green', label="Fully Paid")
plt.plot(unstackedPivot['Current'], marker='', color='blue', label="Current")
plt.plot(unstackedPivot['Charged Off'], marker='', color='red',label="Charged Off")
plt.ylabel('Count of Loans')
plt.xlabel('Months Since last delinquency')
plt.grid()
plt.legend()




# In[20]:


unstackedPivot=groupedByList['home_ownership'].size().unstack().fillna(0)
plt.figure(figsize=(12,11))
plt.plot(unstackedPivot['Fully Paid'], marker='', color='green', label="Fully Paid")
plt.plot(unstackedPivot['Current'], marker='', color='blue', label="Current")
plt.plot(unstackedPivot['Charged Off'], marker='', color='red',label="Charged Off")
plt.ylabel('Count of Loans')
plt.legend()


# In[21]:


# Checking the ratio of counts in terms of percent of 
ownership_check = groupedByList['home_ownership'].agg({'home_ownership': 'count'})
ownership_check1 = ownership_check.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack().fillna("--")
ownership_check1


# The loan count and the ratio is almost same for all categories of "Home Ownership". 
# So, it is NOT a driving factor.

# In[22]:


groupedByList['verification_status']['id'].size().unstack().plot(kind='bar',stacked=False,figsize=(15,10))
plt.ylabel("Count")
plt.show()


# In[23]:


verification_check = groupedByList['verification_status'].agg({'verification_status': 'count'})
verification_check1 = verification_check.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack()
verification_check1


# **verification_status** is a driving factor. Most of the verified statused loans get charged off.

# In[24]:


groupedByList['delinq_2yrs'].size().unstack().plot(figsize=(12,11))
plt.grid()


# In[25]:


groupedByList['purpose']['id'].size().unstack().plot(kind='bar',stacked=False,figsize=(15,10))
plt.ylabel("Count")
plt.show()


# In[26]:


# Check in Percentage of row total
purpose_check = groupedByList['purpose'].agg({'purpose': 'count'})
purpose_check1 = purpose_check.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack()
purpose_check1


# **purpose** is a driving factor behind loan default.
# Result: "Small Bussinesses" have high ratio of Defaulters. It is risky to invest in them.
#         Whereas "Debt Consolidation" has the highest number of loans and has comparitively less Defaulters ratio.

# In[27]:


'''
The below graph suggests most of the Charged Off loans happen to lie within 25000.
'''
charged_off_dist = loanDataset.loc[(loanDataset['loan_status']=='Charged Off'),['funded_amnt']]
count_charged_off = loanDataset.loc[(loanDataset['loan_status']=='Charged Off'),['funded_amnt']].count()
plt.figure(figsize=(12,11))
plt.hist(charged_off_dist['funded_amnt'],bins=50)


# In[28]:


#Let us also see 95% of the charged off loans is what amount
charged_off_dist.quantile(0.95)


# In[29]:


fully_paid_dist = loanDataset.loc[(loanDataset['loan_status']=='Fully Paid'),['funded_amnt']]
count_fully_paid = loanDataset.loc[(loanDataset['loan_status']=='Fully Paid'),['funded_amnt']].count()
plt.figure(figsize=(12,11))
plt.hist(fully_paid_dist['funded_amnt'], bins=50)


# In[30]:


fully_paid_dist.quantile(0.95)


# Based on this understand, let us also create bins for funded_amnt that can be sed later for finer analysis of data.

# In[31]:


def funded_amnt_bin(x):
    bin =0
    if x<= 5000:
        bin = 5000
    elif x<=10000:
        bin = 10000
    elif x<=15000:
        bin = 15000
    elif x<=20000:
        bin =20000
    elif x<=25000:
        bin= 25000
    elif bin<=30000:
        bin=30000
    else:
        bin=35000
    return bin

loanDataset['funded_amnt_bin']=loanDataset['funded_amnt'].apply(funded_amnt_bin)
loanDataset['funded_amnt_bin'].unique()


# **Conclusion: Both Charged Off loans and Fully Paid loans show similar trends for funded_amnts. More loans are given in multiple of 5K i.e. 5K, 10K, 15K upto 35K.*However since there is no difference between these two plots, funded_amnt does not give any clear indication of default*.**

#  **total_rec_prncp** : we created a derived variable to calculate proportion of principal paid to the overall principal. It will be 1.0 for all Fully Paid loans. For Charged off loans, it will be less than 1.0 since full principal payment happens at the very end of the scheduled tenure.

# In[32]:


loanDataset['proportion_rec_prncp'] = round(loanDataset['total_rec_prncp']/loanDataset['funded_amnt'],2)
#check for Fully Paid loans. This must come to 1.0 for all Fully Paid loans.
loanDataset.loc[(loanDataset['loan_status']=='Fully Paid'),['proportion_rec_prncp']].mean()
'''
Now for Charged Off loans the summary metrics of this variable are given below. 

Here, we can see the mean (0.35) and median (0.31) are not very spread from each other. This means the values are uniformly 
distributed and there is potentially no skew. 

However, values vary from 0.0 to 0.99 - so no definite conclusion can be made just by using these statistics. So, let us 
analyze distribution plots to draw more insights
'''
(loanDataset.groupby('loan_status'))['proportion_rec_prncp'].describe()


# In[33]:


'''
Let us also make plots to see the distribution of this variable to derive more insights. By plotting the scatter plot across
different funded amounts to see whether amounts had an impact on the proportion of principal paid. 
'''
proportion_rec_prncp_dist =loanDataset.loc[(loanDataset['loan_status']=='Charged Off'),['proportion_rec_prncp','funded_amnt']]
plt.figure(figsize=(20,10))
plt.grid()
plt.scatter(proportion_rec_prncp_dist['funded_amnt'],proportion_rec_prncp_dist['proportion_rec_prncp'])
(loanDataset.loc[(loanDataset['loan_status']=='Charged Off')]).groupby('funded_amnt_bin')['proportion_rec_prncp'].describe() 


# In[34]:


'''
Running a correlation between proportion_rec_prncp & funded_amnt also shows only a small negative. But it does not
seem suggestive enough to make a conclusive decision on its impact on default loan. 
This corroborates the scatter graph shown above.
'''
proportion_rec_prncp_dist.corr()


# **Conclusion: As one can see, from the above plot, across different funded amounts, the proportion of principal paid is uniformly distributed from 0 to 1. There is no typical value that can be used to signal a default. Hence it does not tell us much. Though the proportion seems to slightly reduce with the increase in funded_amnt, the correlation is not strong enough (only -0.11). Hence this ratio does not signal us much.**

# In[35]:


groupedByList['term'].size().unstack().plot(kind='bar',stacked=False,rot=90,figsize=(18,12))
plt.grid()


# We can see that those with 36 months as the **term** have low ratio of charged off to fully paid loans, where as those with 60 months as the **term** the ratio is obviously higher.

# In[36]:


groupedByList['int_rate'].size().unstack().plot(kind='kde',stacked=False,rot=90,figsize=(18,12))
plt.grid()


# In[37]:


groupedByList['emp_length'].size().unstack().plot(kind='bar',stacked=False,rot=90,figsize=(18,12))
plt.grid()


# In[38]:


empexp_check = groupedByList['emp_length'].size()
empexp_check = empexp_check.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack()
empexp_check.transpose().fillna('--')
print(empexp_check)
empexp_check.plot(kind='kde',rot=90,figsize=(15,10))


# We can see that those with experience of 10 or more years have taken more loans, and have charged off the maximum times. However the distribution of their percentages are quite different for charged off and fully paid loans. We can conclude that the **emp_length** has a certain effect on loan status. From those with less and more experience the ratio of charged off to fully paid is low, where as for those in intermediate  phases the ratios are high.

# In[39]:


# Check in Percentage of row total
zipcode_check = groupedByList['zip_code_clean'].agg({'zip_code_clean': 'count'})
zipcode_check = zipcode_check.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).unstack()
zipcode_check.transpose().fillna('--')


# There is no significant change in ratios of defaulter and fully paid for zip_code column.
# So, **zip_code** is not a  driving factor.

# In[40]:


groupedByList['addr_state'].size().unstack().plot(kind='bar',stacked=False,figsize=(18,12))
plt.ylabel("Count")
plt.show()


# In[41]:


groupedByList['dti'].size().unstack().plot(kind='kde',stacked=False,figsize=(10,8),subplots=True)
groupedByList['dti'].size().unstack().plot(kind='line',stacked=False,figsize=(10,8),subplots=True)
plt.show()


# **dti** column shows an increasing trend for the percentage of defaulters from 5 to 25. After which it stars decreasing. Therefore the higher dti range i.e. from 15 to 25 are more likely to default.

# In[42]:


#let us first find the total interest to be paid by each consumer.
#For this, we first find the total interest payable i.e. term * installment - funded_amnt.

#For this we first calculate the numerical value of term:
loanDataset['term_value'] = loanDataset['term'].apply(lambda x:int(x.lstrip()[0:2]))

#Now find total interest payable
loanDataset['total_payable_int'] = loanDataset['term_value']*loanDataset['installment'] - loanDataset['funded_amnt']
loanDataset['proportion_rec_int'] = round(loanDataset['total_rec_int']/loanDataset['total_payable_int'],2)
print((loanDataset.groupby('loan_status'))['proportion_rec_int'].describe())


# In[43]:


'''
The below graph has interesting insights. Above 5000, the total_rec_int by various defaulters is a uniform distribution
i.e. there is no specific pattern based inference. 
However, below a ticket size of 5000, we see defaulters are more startified across levels. Defaulters seem to pay the 
interest cover in discrete steps i.e. 0.01, 0.02 & so on till they have paid 0.5 of the total payable interest. 
It maybe interesting to use this to pro-actively engage with users of such ticket sizes.
'''

proportion_rec_int_dist =loanDataset.loc[(loanDataset['loan_status']=='Charged Off'),['proportion_rec_int','total_payable_int']]
plt.figure(figsize=(10,20))
plt.grid()
plt.scatter(proportion_rec_int_dist['total_payable_int'],proportion_rec_int_dist['proportion_rec_int'], c='red')


# In[44]:


#Let us also compare how proportion of received interest is for Fully Paid loans
proportion_rec_int_dist =loanDataset.loc[(loanDataset['loan_status']=='Fully Paid'),['proportion_rec_int','total_payable_int']]
plt.figure(figsize=(10,20))
plt.grid()
plt.scatter(proportion_rec_int_dist['total_payable_int'],proportion_rec_int_dist['proportion_rec_int'])


# In[45]:


proportion_rec_int_dist.corr()


# **Conclusion: specific pattern cant be inferred from the proportion of interest paid. The scatter plot is quite similar for both Fully Paid and Charged Off loans. 
# However, for ticket size of 5000 and below, we see defaulters are more layered across levels. Defaulters seem to pay the 
# interest cover in discrete steps i.e. 5%, 10% & so on till they have paid 50% of the total payable interest.**

# In[46]:


groupedByList['last_pymnt_yr']['last_pymnt_yr'].describe().unstack().fillna('--')


# In[47]:


groupedByList['last_pymnt_month']['last_pymnt_yr'].describe().unstack().fillna('--')


# **Conclusion: It can be clearly seen that in cases of Charged Off loans, most last_pymnt_d fell between 10th-13th of the month. In case of Fully Paid loans, most last_pymnt_d fell between 13th-15th of the month. This is a very strong indicator of possible defaults / Charge Offs. However, month does not seem a big indicator since it follows the same trend for both Fully Paid and Charged off loans.**

# In[48]:


groupedByList['grade'].size().unstack().plot(kind='bar',figsize=(15,12))


# It can be seen that loans with **grade** as A have very low ratio of charged off to fully paid loans. Whereas G, F, E, D,C have an increasing ratio,indicating that loans in these grades tend to be defaulted more. 

# In[49]:


groupedByList['sub_grade'].size().unstack().plot(kind='bar',figsize=(15,12))


# It can be seen that loans with **sub_grade** in A have very low ratio of charged off to fully paid loans. The ratio then varies for other sub grades, For G sub groups the ratio are very high. And the ratio increases starting from D2. 
# Thus Grades and Sub Grades can be a driving factor towards deciding the loan to be charged off.

# In[50]:


for_fully_paid = round(loanDataset.loc[(loanDataset['loan_status']=='Fully Paid')&(loanDataset['last_pymnt_amnt']<loanDataset['installment']),['funded_amnt']].count()/count_fully_paid,3)
for_charged_off = round(loanDataset.loc[(loanDataset['loan_status']=='Charged Off')&(loanDataset['last_pymnt_amnt']<loanDataset['installment']),['funded_amnt']].count()/count_charged_off,3)


# In[51]:


'''
We see that for Fully Paid cases, ~10% of the instances last_pymnt_amnt < installement.
However, for Charged Off cases, ~18% of the instances last_pymnt_amnt < installement.
This is a significant difference between the two. So we can analyze this attribute further for more insights.
'''
for_fully_paid, for_charged_off


# In[52]:


'''
Extending the above analysis, we will find the mean/median of such cases for both Fully Paid and Charged Off. 
If we run the below, we see such cases are more prominent for higher ticket sizes
'''
loanDataset.loc[(loanDataset['loan_status']=='Fully Paid')&(loanDataset['last_pymnt_amnt']<loanDataset['installment']),['funded_amnt']].describe()


# In[53]:


loanDataset.loc[(loanDataset['loan_status']=='Charged Off')&(loanDataset['last_pymnt_amnt']<loanDataset['installment']),['funded_amnt']].describe()


# **Conclusion: From the above, it seems if the ticket size is high (~15000) and the last_pymnt_amnt < installment, there is a 18% probability that it may default. The company can take appropriate measures accordingly. Hence this attribute is important for the early detection of default.**

# # Bivariate Analysis

# In[54]:


try:
    CorrelationMatrix=loanDataset.corr()   
    CorrelationMatrix.dropna(axis=1,how='all',inplace=True)
    print(loanDataset.shape)
    CorrelationMatrix.dropna(axis=0,how='all',inplace=True)
    plt.figure(figsize=(25,25))
    sns.heatmap(CorrelationMatrix,annot=True,linewidth=0.5)
except Exception as ex:
    print(ex)


# Inferences from the correlation:
# 
# 1. **total_pymnt_inv** has a very strong corrletaion to **total_pymnt (0.97)**. Hence, we must ignore this variable and consider only total_pymnt. 
# 2. **pub_rec_bankruptcies** correlates fairly strongly to **public_rec (0.86)**. Hence, we can take one among these
# 3. Our derived variable **proportion_rec_prncp** correlates very strongly to another derived variable **proportion_rec_int (0.93)**. So between principal and interest payment, we can choose only one.
# 4. Two derived variables **proportion_rec_prncp & proportion_late_fee** show slight negative correlation to funded_amnt. 

# # Conclusions

# 1. **Verification_status** shows a trend that percentage of defaulters from total for Verified status is more compared to Non-Verified.
# 2. **Purpose** shows that the percntage of defaulter for 'Small_bussniness' is relatively are than all others puposes. Therefore this category of laoan are more likely to default.
# 3. **dti** column shows an increasing trend for the percentage of defaulters from 5 to 25. After which it stars decreasing. Therefore the higher dti range i.e. from 15 to 25 are more likely to default.
# 4. Most **last_pymnt_d** fall between year 2010-2013. In case of Fully Paid loans, most last_pymnt_d fall between 2013-2015. There might be some economic crisis during these periods. This is a very strong indicator of possible defaults / Charge Offs during economic crisis.
# 5. **emp_length** has a certain effect on loan status. From those with less and more experience the ratio of charged off to fully paid is low, where as for those in intermediate  phases the ratios are high
# 6. **grade** and **sub_grade** affect the loan status too.
# 7.  If the ticket size is high (~15000) and the **last_pymnt_amnt** < **installment**, there is a 18% probability that it may default. The company can take appropriate measures accordingly. Hence this attribute is important for the early detection of default
# 
