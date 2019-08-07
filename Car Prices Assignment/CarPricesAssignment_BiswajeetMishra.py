
# coding: utf-8

# # Car Prices Assignment- Linear Regression
# 
# #### Problem Statement:
# 
# Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# 
# Which variables are significant in predicting the price of a car?
# How well those variables describe the price of a car?
# Model the price of cars with the available independent variables.
# 
# - Determine All driving factors for the prices of cars in US market for chinese company
# - Create a Linear Regression Model to predict prices
# 

# #### Importing Necessary Libraries
# 

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


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


try:
    pathMapping= r'CarPrice_Assignment.csv'
    carPricesDataset=pd.read_csv(pathMapping)
    pd.options.display.max_columns = None
except Exception as ex:
    print(ex)
carPrices=carPricesDataset


# In[4]:


carPrices.head()


# In[5]:


carPrices.set_index('car_ID',inplace=True)


# In[6]:


carPrices[['Car','Name']]=carPrices['CarName'].str.split(" ", n = 1, expand = True) 
#Dropping carName and Name as they are not needed and not relevant anymore

carPrices.drop(['CarName','Name'],axis=1,inplace=True)


# In[7]:


carPrices.head()


# In[8]:


carPrices.nunique()


# In[9]:


carPrices.describe()


# In[10]:


carPrices.info()


# In[11]:


try:
    uniqueValuesOfEachColumn= {}
    for x in list(carPrices.columns):
        if(len(carPrices[x].unique())<=10):
            uniqueValuesOfEachColumn[x]=carPrices[x].unique()
except Exception as ex:
    print(ex)

lablesForCarMakers=carPrices['Car'].unique()


# In[12]:


uniqueValuesOfEachColumn


# In[13]:


lablesForCarMakers


# In[14]:


#There are misspelled names in the dataset. so converting all to same correct names
def NormalizeMakerName(x):    
    x=x.lower()
    if(x in['maxda','mazda']):
        return'mazda'
    elif(x in['vokswagen','volkswagen','vw']):
        return'volkswagen'
    elif(x in['toyota','toyouta']):
        return'toyota'
    elif(x in['porsche','porcshce']):
        return'porsche'
    else:
        return x

carPrices['Car']=carPrices['Car'].apply(NormalizeMakerName)
lablesForCarMakers=carPrices['Car'].unique()

print(lablesForCarMakers)


# I will convert all the car makers to two categories: Luxury and Non-Luxury. Why? Because the names directly wont effect **Geely**. And **Geely** will always remain **Geely**. It wont ever become Audi or BMW. But its higly likly possible that **Geely** may launch its car in either luxury or non luxury category, and as such this data can help it price its car accordingly to gain market.

# In[15]:



luxuryBrands=['alfa-romero', 'audi', 'bmw', 'jaguar', 'mercury', 'porsche', 'volvo']
#nonLuxuryBrands=['chevrolet', 'dodge', 'honda', 'isuzu', 'mazda', 'buick', 'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen']
carPrices['Car']=carPrices['Car'].apply(lambda x: 'luxury' if x in luxuryBrands else 'non-luxury')


# In[16]:


carPrices['Car'].unique()


# We can see that there are 9 categorical variables viz.
# **{CarName,symboling,fueltype,aspiration,doornumber,carbody,drivewheel,enginelocation,enginetype,cylindernumber,fuelsystem}**
# 
# Out of these **doornumber** and **cylindernumber** are just numbers in text format, we can convert them directly into their corresponding digit representatives.
# 
# The **car_ID** column had unique values, so i made it the index of the dataset, it has no effect on the prices anyways(as given in the data dictionary)
# 
# The **CarName** column has the company name and the car model concatenated together. We can separate them into two different columns containing the company name **Car** and the car model names **Name**. 
# We might not need the car model names 'Name', because the prices of a model are determined by its brand, and specifications, not model name.
# 
# Moreover **Car** is has a lot of company names, and they are either a luxury brand or a non luxury brand. so i have converted them into two categories viz. luxury and non-luxury.
# 
# The **symboling** column has ordered categorical values, but as they are integers I can use them directly without changing them.
# 
# The **fueltype** , **aspiration** , **enginelocation** have 2 categories respectively. we can use 0 and 1 to represent the values.
# 
# All other categorical columns can be either dummy enoded or hot-encoded.
# 
# There are no null values as evident from the carPrice.info() functions' value, except in derived **Name** column where 2 values are missing for the **CarName** 'Subaru'. As I wont be using **Name** anyways, its not a problem.

# # 2. Visualization of Data

# Creating pariplots in scatter mode to see if there are any correlations and visualize patterns.

# In[17]:


pairplt=sns.pairplot(carPrices)


# The scatter plots show patterns for some combinations and are random for others.
# price with all variables except carheight, peakrpm shows patterns.  carheight and  stroke are much dense in the middle bottom of the plots. Compression ratio absolutely divided into two different sections. peakratio seems to be random. citympg, highwaympg seem to have a negative slope.

# Creating a Heat Map to visualize correlations properly to determine which variables are highly correlated to each other.

# In[18]:


plt.figure(figsize=(25,25))
sns.heatmap(carPrices.corr(),annot=True)


# There looks to be high correlation between price and curbweight, enignesize, horsepower, peakrpm. Enginesize has high correlation with horsepower and curbweight. carlenght and wheelbase have high corelation too.

# Creating boxplots to visualize the categorical variables.

# In[19]:


plt.figure(figsize=(20, 12))
#plt.subplot(321)
sns.boxplot(x = 'fueltype', y = 'price', data = carPrices)
plt.show()
plt.figure(figsize=(20, 12))
#plt.subplot(322)
sns.boxplot(x = 'aspiration', y = 'price', data = carPrices)
plt.show()
plt.figure(figsize=(20, 12))
#plt.subplot(323)
sns.boxplot(x = 'carbody', y = 'price', data = carPrices)
plt.show()
plt.figure(figsize=(20, 12))
#plt.subplot(324)
sns.boxplot(x = 'drivewheel', y = 'price', data = carPrices)
plt.show()
plt.figure(figsize=(20, 12))
#plt.subplot(325)
sns.boxplot(x = 'enginelocation', y = 'price', data = carPrices)
plt.show()
plt.figure(figsize=(20, 12))
#plt.subplot(326)
sns.boxplot(x = 'enginetype', y = 'price', data = carPrices)
plt.show()
plt.figure(figsize=(20, 12))
sns.boxplot(x = 'fuelsystem', y = 'price', data = carPrices)
plt.show()


# 1. We can see that with fuelsystems, idi and mpfi have more price spread then others, they can be significant factors for price.
# 2. for enginetype ohcv has highest spead for price.
# 3. There are less cars with engine at the rear end, but they cost very high.
# 4. Rear wheel drive car are more in number and so are the prices for them.
# 5. convertibles  and hardtops cost more then other body type categories.
# 6. diesel cars have higher median value, but petrol cars have more outliers.
# 7. cars with std aspirations have more outliers but median for turbo is higher. and they are oppositely skewed. 

# # 3. Preparing data for model creation

# Converting all categorical variables into numerical data so that it can be used in creation of models.

# In[20]:


wordToDigtsMapper={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
                    'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
                    'nineteen':19,'twenty':20}
wordToDigtsMapper


# In[21]:


def binary_map(x):
    return x.map(wordToDigtsMapper)

carPrices[['doornumber','cylindernumber']]=carPrices[['doornumber','cylindernumber']].apply(binary_map)
carPrices[['doornumber','cylindernumber']]


# In[22]:


carPrices.describe()


# In[23]:


plt.figure(figsize=(8,5))
sns.scatterplot(x="doornumber", y="price", data=carPrices)
plt.show()
sns.boxplot(x="doornumber", y="price", data=carPrices)
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(x="cylindernumber", y="price", data=carPrices)
plt.show()
sns.boxplot(x="cylindernumber", y="price", data=carPrices)
plt.show()


# We can see that cars with more number of cylinders have more price. at least the median values increases with increase in cylinder numbers. except for 2 and 3 cylinders. 
# For the doornumbers the boxes overlap. and thus there might not be much influence of these on price. 

# Converting other categorical variables into numerical values.

# In[24]:


categoricalColumns=['fueltype','aspiration','enginelocation']
conversionFactors={}
for x in categoricalColumns:
    uniquesList=carPrices[x].unique()
    tempDictToStoreMapping={}
    for y in range(len(uniquesList)):
        tempDictToStoreMapping.update({uniquesList[y]:y})
    conversionFactors[x]=tempDictToStoreMapping

conversionFactors


# In[25]:


columnstoDummify=['Car','carbody','drivewheel','enginelocation','enginetype','fuelsystem']


# In[26]:


#mapping all categorical values to their corresponding number assignments

for x in conversionFactors:
    carPrices[x]=carPrices[x].apply(lambda y: conversionFactors[x][y])

carPrices


# In[27]:


carPrices.describe()


# In[28]:


carPrices.shape


# In[29]:


#Dummifying the car makers column
# Let's drop the first column from status df using 'drop_first = True'
carMakers = pd.get_dummies(carPrices[columnstoDummify], drop_first = True)
# Add the results to the original housing dataframe
carPrices = pd.concat([carPrices, carMakers], axis = 1)
#Dropping the original columns, we alredy have all the unique values for each columns in uniqueValuesOfEachColumn variable

carPrices.drop(columnstoDummify,axis=1,inplace=True)


# In[30]:


carPrices.shape


# In[31]:


carPrices.describe()


# In[32]:


plt.figure(figsize=(30,30))
sns.heatmap(carPrices.corr(),annot=True)


# # 4. Building Model

# In[33]:


#Splitting for test and train
from sklearn.model_selection import train_test_split
nm.random.seed(0)
carPrices_Train,carPrices_Test = train_test_split(carPrices,train_size = 0.7, test_size = 0.3, random_state = 100)


# In[34]:


carPrices_Train.describe()


# In[35]:


carPrices_Test.describe()


# In[36]:


#Normalizing values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns 
listCols=list(carPrices.columns)
# for x in lablesForCarMakers:
#     if(x in listCols):
#         listCols.remove(x)

carPrices_Train[listCols] = scaler.fit_transform(carPrices_Train[listCols])
carPrices_Test[listCols] = scaler.fit_transform(carPrices_Test[listCols])


# In[37]:


carPrices_Train.head(2)


# In[38]:


carPrices_Test.head(2)


# In[39]:


carPrices_Train.describe()


# In[40]:


carPrices_Test.describe()


# In[41]:


carPrices_Train.shape


# In[42]:


carPrices_Test.shape


# In[43]:


#dividng into X and Y
y_train = carPrices_Train.pop('price') #contains the price column as dependent variable
X_train = carPrices_Train #contains the predictor variables
y_test = carPrices_Test.pop('price') #contains the price column as dependent variable
X_test = carPrices_Test #contains the predictor variables


# ## Building model using SciKit Learn

# In[44]:


# Importing RFE and LinearRegression and other necessary modules.
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[45]:


# Running RFE with the output number of the variable equal to 20
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe_20 = RFE(lm, 20)             # running RFE
rfe_20 = rfe_20.fit(X_train, y_train)


# In[46]:


list(zip(X_train.columns,rfe_20.support_,rfe_20.ranking_,lm.coef_))


# In[47]:


rfe_25 = RFE(lm, 25)             # running RFE
rfe_25 = rfe_25.fit(X_train, y_train)


# In[48]:


list(zip(X_train.columns,rfe_25.support_,rfe_25.ranking_,lm.coef_))


# In[49]:


rfe_15 = RFE(lm, 15)             # running RFE
rfe_15 = rfe_15.fit(X_train, y_train)


# In[50]:


list(zip(X_train.columns,rfe_15.support_,rfe_15.ranking_,lm.coef_))


# In[51]:


X_test.head(2)


# In[52]:


y_test_pred = lm.predict(X_test)
y_train_pred=lm.predict(X_train)

# RMSE
print("RMSE TEST: "+str(nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("RMSE TRAIN: "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("R2-Score TEST: "+str(r2_score(y_test, y_test_pred)))
print("R2-Score TRAIN: "+str(r2_score(y_train, y_train_pred)))


# The next part is a bit complex, I m going to store the supported and not supported columns obtained from the RFE for 25, 20, 15 variables in lists and then i will use that list to create 3 LM models using OLS to see which is better. I will continue with the best model in hand.

# In[53]:


#taking the columns which have support as true
supportedCols_25_20_15 = [X_train.columns[rfe_25.support_],X_train.columns[rfe_20.support_],X_train.columns[rfe_15.support_]]
supportedCols_25_20_15


# In[54]:


notSupportedCols_25_20_15=[X_train.columns[~rfe_25.support_],X_train.columns[~rfe_20.support_],X_train.columns[~rfe_15.support_]]
notSupportedCols_25_20_15


# #### Creating LM model using statsmodel for all the metrics and stats:

# In[55]:


#test set with RFE selected variables
X_train_rfe_25 = X_train[supportedCols_25_20_15[0]] #index 0 in this list has columns for rfe_25
import statsmodels.api as sm  
#adding constant 
X_train_rfe_25 = sm.add_constant(X_train_rfe_25)
lm_25 = sm.OLS(y_train,X_train_rfe_25).fit()
print(lm_25.summary())


# The model indicates that there are multicollinearity problems. So they need to be dealt with. Moreover **curbweight**, **carbody_wagon** have high p values, and are insignificant. we will remove them one after the other. Dealing with Multicollinearity first:

# In[56]:


#test set with RFE selected variables
X_train_rfe_20 = X_train[supportedCols_25_20_15[1]] #index 1 inthis list has columns for rfe_20
 
#adding constant
X_train_rfe_20 = sm.add_constant(X_train_rfe_20)
lm_20 = sm.OLS(y_train,X_train_rfe_20).fit()
print(lm_20.summary())


# The model indicates that there are multicollinearity problems. So they need to be dealt with. Moreover **curbweight**, **carbody_wagon** have high p values, and are insignificant. we will remove them one after the other. Dealing with Multicollinearity first:

# In[57]:


#test set with RFE selected variables
X_train_rfe_15 = X_train[supportedCols_25_20_15[2]] #index 2 in this list has columns for rfe_15
X_test_rfe_15=X_test[supportedCols_25_20_15[2]] #index 2 in this list has columns for rfe_15
import statsmodels.api as sm  
#adding constant
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
lm_15 = sm.OLS(y_train,X_train_rfe_15).fit()
print(lm_15.summary())


# All the models indicates that there are multicollinearity problems. So they need to be dealt with. Moreover **curbweight**, **carbody_wagon** have high p values, and are insignificant. we will remove them one after the other.
# However for model with 15 variables the f-stat is high and prob(f-stat) is nearest to 0.
# 
# So we will consider only this model as our base model.
# Now
# Dealing with Multicollinearity first:

# In[58]:


X_train_rfe = X_train_rfe_15
X_test_rfe=X_test_rfe_15


# In[59]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# The above values show that VIF is very high for most of the variables. We need to drop most of them. We will look into the significance of the variables too. but from here we will look only into rfe_15

# **compressionratio** has a p value of 0.402. Removing it:

# In[60]:


X_train_rfe_1st=X_train_rfe.drop('compressionratio',axis=1)
X_test_rfe_1st=X_test_rfe.drop('compressionratio',axis=1)
lm_1st=sm.OLS(y_train,X_train_rfe_1st).fit()
print(lm_1st.summary())


# Now all columns have significance, and their p values are good too, <0.05 so lets look into VIF for these columns.

# In[61]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_1st.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Removing **fueltype** as it has high VIF:
# 

# In[62]:


X_train_rfe_2nd=X_train_rfe_1st.drop('fueltype',axis=1)
X_test_rfe_2nd=X_test_rfe_1st.drop('fueltype',axis=1)
lm_2nd=sm.OLS(y_train,X_train_rfe_2nd).fit()
print(lm_2nd.summary())


# we can see that the multicollinearity warning is no longer displayed.

# In[63]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_2nd.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Removing **enginesize** as it has high VIF:

# In[64]:


X_train_rfe_enginesizeDropped=X_train_rfe_2nd.drop('enginesize',axis=1)
X_test_rfe_enginesizeDropped=X_test_rfe_2nd.drop('enginesize',axis=1)
lm_enginesizeDropped=sm.OLS(y_train,X_train_rfe_enginesizeDropped).fit()
print(lm_enginesizeDropped.summary())


# In[65]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_enginesizeDropped.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping enginesize decreases all statistics, and increases most p values. so lets not drop it and drop the next largest VIF valued cloumn: i.e. **cylindernumber**

# In[66]:


X_train_rfe_cylindernumberDropped=X_train_rfe_2nd.drop('cylindernumber',axis=1) #deleting 3rd column
X_test_rfe_cylindernumberDropped=X_test_rfe_2nd.drop('cylindernumber',axis=1)
lm_cylindernumberDropped=sm.OLS(y_train,X_train_rfe_cylindernumberDropped).fit()
print(lm_cylindernumberDropped.summary())
X_train_rfe_3rd=X_train_rfe_cylindernumberDropped
X_test_rfe_3rd=X_test_rfe_cylindernumberDropped


# The stats obtained for this LM model is better then the one obtained upon removing **enginesize** so lets go with this. Again we have some variables with p values > -.05. lets drop them and see what happens: Dropping **enginetype_ohcf**

# In[67]:


#kept the name as 4th as we are removing the 4th column now
X_train_rfe_4th=X_train_rfe_3rd.drop('enginetype_ohcf',axis=1) 
X_test_rfe_4th=X_test_rfe_3rd.drop('enginetype_ohcf',axis=1)
lm_4th=sm.OLS(y_train,X_train_rfe_4th).fit()
print(lm_4th.summary())


# We can see that the stat values have increased: lets drop the next insignificant variable: **boreratio**

# In[68]:


#kept the name as 5th as we are removing the 4th column now
X_train_rfe_5th=X_train_rfe_4th.drop('boreratio',axis=1) 
X_test_rfe_5th=X_test_rfe_4th.drop('boreratio',axis=1) 
lm_5th=sm.OLS(y_train,X_train_rfe_5th).fit()
print(lm_5th.summary())


# In[69]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_5th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# **enginetype_dohcv** has p value 0.361 we will remove that.
# Also we can notice from VIF,
# **carwidth** can be deleted here as, by domain knowledge, larger engine size has higher power.

# In[70]:


#6th model
X_train_rfe_6th=X_train_rfe_5th.drop('enginetype_dohcv',axis=1) 
X_test_rfe_6th=X_test_rfe_5th.drop('enginetype_dohcv',axis=1) 

lm_6th=sm.OLS(y_train,X_train_rfe_6th).fit()
print(lm_6th.summary())


# In[71]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_6th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# **peakrpm** has p-value 0.06, it needs to be deleted.

# In[72]:


#7th model
X_train_rfe_7th=X_train_rfe_6th.drop('peakrpm',axis=1) 
X_test_rfe_7th=X_test_rfe_6th.drop('peakrpm',axis=1)
lm_7th=sm.OLS(y_train,X_train_rfe_7th).fit()
print(lm_7th.summary())


# In[73]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_7th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Now, again all variables have good p-values. So we look into VIF, and delete **carwidth**

# In[74]:


#8th model
X_train_rfe_8th=X_train_rfe_7th.drop('carwidth',axis=1) 
X_test_rfe_8th=X_test_rfe_7th.drop('carwidth',axis=1) 

lm_8th=sm.OLS(y_train,X_train_rfe_8th).fit()
print(lm_8th.summary())


# In[75]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_8th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Lets see at this point how much good the model is.

# In[76]:


X_train_rfe_final=X_train_rfe_8th
X_test_rfe_final=sm.add_constant(X_test_rfe_8th)

lm_final = LinearRegression()
lm_final.fit(X_train_rfe_final, y_train)

y_train_pred=lm_final.predict(X_train_rfe_final)
y_test_pred = lm_final.predict(X_test_rfe_final)

# RMSE
print("RMSE TEST: "+str(nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("RMSE TRAIN: "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("RMSE difference(Train-Test): "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))-nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("R2-Score TEST: "+str(r2_score(y_test, y_test_pred)))
print("R2-Score TRAIN: "+str(r2_score(y_train, y_train_pred)))
print("R2- Score Difference(Train-Test): "+str(r2_score(y_train, y_train_pred)-r2_score(y_test, y_test_pred)))


# The R Squared score decreases by almost 0.15 from Test to Train set. the RMSE increases too

# I would like to add the varibal of mcar makes to the model, as i feel that luxury and non luxury car types do influence price.

# In[77]:


#9th model
X_train_rfe_9th=X_train_rfe_8th
X_test_rfe_9th=X_test_rfe_8th
X_train_rfe_9th["non_luxury"]=X_train["Car_non-luxury"]
X_test_rfe_9th["non_luxury"]=X_test["Car_non-luxury"]
lm_9th=sm.OLS(y_train,X_train_rfe_9th).fit()
print(lm_9th.summary())


# As we can see the stats increased upon addition of luxury category. lets check its VIF

# In[78]:


X_train_rfe_final=X_train_rfe_9th
X_test_rfe_final=sm.add_constant(X_test_rfe_9th)

lm_final = LinearRegression()
lm_final.fit(X_train_rfe_final, y_train)

y_train_pred=lm_final.predict(X_train_rfe_final)
y_test_pred = lm_final.predict(X_test_rfe_final)

# RMSE
# RMSE
print("RMSE TEST: "+str(nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("RMSE TRAIN: "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("RMSE difference(Train-Test): "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))-nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("R2-Score TEST: "+str(r2_score(y_test, y_test_pred)))
print("R2-Score TRAIN: "+str(r2_score(y_train, y_train_pred)))
print("R2- Score Difference(Train-Test): "+str(r2_score(y_train, y_train_pred)-r2_score(y_test, y_test_pred)))


# - The RMSE values increases from train to test by almost.02
# - The R squared value decreases by 0.1 from train to test.
# 
# #### Lets see if we can further decrease this difference by adding or deleting any other variables.

# In[79]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_9th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Let's try removing the **stroke** variabe and see how the model changes:
# 

# In[80]:


#10th model
X_train_rfe_10th=X_train_rfe_9th.drop('stroke',axis=1) 
X_test_rfe_10th=X_test_rfe_9th.drop('stroke',axis=1) 

lm_10th=sm.OLS(y_train,X_train_rfe_10th).fit()
print(lm_10th.summary())


# The values here are pretty goo too, except **enginetype_ohc** which has p value of 0.114. we can try dropping it. lets see into the VIF values too:  

# In[81]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_10th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# I will try deleting both parallely and look into the changes in the models.
# 

# In[82]:


#11.a th model
X_train_rfe_11_ath=X_train_rfe_10th.drop('enginesize',axis=1) 
X_test_rfe_11_ath=X_test_rfe_10th.drop('enginesize',axis=1) 

lm_11_ath=sm.OLS(y_train,X_train_rfe_11_ath).fit()
print(lm_11_ath.summary())


# In[83]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_11_ath.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[84]:


X_train_rfe_final=X_train_rfe_11_ath
X_test_rfe_final=sm.add_constant(X_test_rfe_11_ath)

lm_final = LinearRegression()
lm_final.fit(X_train_rfe_final, y_train)

y_train_pred=lm_final.predict(X_train_rfe_final)
y_test_pred = lm_final.predict(X_test_rfe_final)

# RMSE
# RMSE
print("RMSE TEST: "+str(nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("RMSE TRAIN: "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("RMSE difference(Train-Test): "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))-nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("R2-Score TEST: "+str(r2_score(y_test, y_test_pred)))
print("R2-Score TRAIN: "+str(r2_score(y_train, y_train_pred)))
print("R2- Score Difference(Train-Test): "+str(r2_score(y_train, y_train_pred)-r2_score(y_test, y_test_pred)))


# In[85]:


#11.b th model
X_train_rfe_11_bth=X_train_rfe_10th.drop('enginetype_ohc',axis=1) 
X_test_rfe_11_bth=X_test_rfe_10th.drop('enginetype_ohc',axis=1) 

lm_11_bth=sm.OLS(y_train,X_train_rfe_11_bth).fit()
print(lm_11_bth.summary())


# In[86]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_11_bth.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[87]:


X_train_rfe_final=X_train_rfe_11_bth
X_test_rfe_final=sm.add_constant(X_test_rfe_11_bth)

lm_final = LinearRegression()
lm_final.fit(X_train_rfe_final, y_train)

y_train_pred=lm_final.predict(X_train_rfe_final)
y_test_pred = lm_final.predict(X_test_rfe_final)

# RMSE
print("RMSE TEST: "+str(nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("RMSE TRAIN: "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("RMSE difference(Train-Test): "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))-nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("R2-Score TEST: "+str(r2_score(y_test, y_test_pred)))
print("R2-Score TRAIN: "+str(r2_score(y_train, y_train_pred)))
print("R2- Score Difference(Train-Test): "+str(r2_score(y_train, y_train_pred)-r2_score(y_test, y_test_pred)))


# As we can see the stats are not good, when prediction is being done after adding removinh **enginesize**. Moreover everytime i test by dropping engine size, the stats reduce drastically. So it can plausibly be said that **enginesize** is  an important driver in car prices.
# 
# The values for model 11 b look good though. So lets continue on it. and delete **horsepower** to see how the model is affected.
# 

# In[88]:


#12th model
X_train_rfe_12th=X_train_rfe_11_bth.drop('horsepower',axis=1) 
X_test_rfe_12th=X_test_rfe_11_bth.drop('horsepower',axis=1) 

lm_12th=sm.OLS(y_train,X_train_rfe_12th).fit()
print(lm_12th.summary())


# In[89]:


X_train_rfe_final=X_train_rfe_12th
X_test_rfe_final=sm.add_constant(X_test_rfe_12th)

lm_final = LinearRegression()
lm_final.fit(X_train_rfe_final, y_train)

y_train_pred=lm_final.predict(X_train_rfe_final)
y_test_pred = lm_final.predict(X_test_rfe_final)

# RMSE
print("RMSE TEST: "+str(nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("RMSE TRAIN: "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))))
print("RMSE difference(Train-Test): "+str(nm.sqrt(metrics.mean_squared_error(y_train, y_train_pred))-nm.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
print("R2-Score TEST: "+str(r2_score(y_test, y_test_pred)))
print("R2-Score TRAIN: "+str(r2_score(y_train, y_train_pred)))
print("R2- Score Difference(Train-Test): "+str(r2_score(y_train, y_train_pred)-r2_score(y_test, y_test_pred)))


# In[90]:


vif = pd.DataFrame()
X_train_withoutConstant=X_train_rfe_12th.drop('const',axis=1)
X = X_train_withoutConstant
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# This model looks good. the VIF are all below 5 and one is just 0.3 more then 5. The p values for all the variables are good. The overall model fit is good, with a higher F-stat and almost 0 prob(F-stat). Lets consider this model for Residual analysis.

# # 5. Residual Analysis

# We have taken model **#12** for residual analysis as its overall values were the best optimized version.

# In[91]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms Train', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)     


# We can  see that the error terms are normally distributed with mean at 0. so our assumptions hold good.

# In[92]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)


# We can certaily see a pattern in the distribution of predicited and test data

# In[93]:


y_original=y_test.reset_index(drop=True)
test=pd.DataFrame([y_original,y_test_pred])
test=test.transpose()
test.head()


# In[94]:


fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(test['price'], 'b-o', label="Data") # the test original data
ax.plot(test['Unnamed 0'], 'r-^', label="predicted") #the perdicted data
ax.legend(loc="best");


# We can see that the predicted and orignal prices are very near to each other. 

# # Conclusion

# We have the following conclusion from the models and the plots:
# 
# The following are the factors that drive the market price of US cars:
# - carheight            
# - enginesize           
# - enginetype_rotor     
# - fuelsystem_idi       
# - non_luxury
# 
# The prepared model is good enough with RMSE and R square values as
# - RMSE TEST: 0.08911508483587784
# - RMSE TRAIN: 0.08138226244635197
# - RMSE difference(Train-Test): -0.00773282238952587
# - R2-Score TEST: 0.8185463853690147
# - R2-Score TRAIN: 0.856622862765285
# - R2- Score Difference(Train-Test): 0.03807647739627029
# 
# The residual analysis shows that the error terms are normally distributed, thus our assumptions hold true.
