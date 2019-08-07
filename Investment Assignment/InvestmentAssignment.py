# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:29:31 2019

@author: Biswajeet
"""
import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
warnings.filterwarnings("ignore")
pathCompanies="D:/upgradMLAI/InvestmentAnalysisAssignment/companies.txt"
pathRounds="D:/upgradMLAI/InvestmentAnalysisAssignment/rounds2.csv"
pathCountries="D:/upgradMLAI/InvestmentAnalysisAssignment/OfficialEngCountries.csv"
pathMappings="D:/upgradMLAI/InvestmentAnalysisAssignment/mapping.csv"


#encoding 'cp437' 'raw_unicode_escape' 'ISO-8859-1'
try:
    print("----------------Reading Data---------------------------------")
    companies= pd.read_csv(pathCompanies, sep='\t',encoding='latin_1')
#    )
#    print(companies)
    rounds=pd.read_csv(pathRounds, sep=',',encoding='latin_1')
    countries=pd.read_csv(pathCountries)
    mappings=pd.read_csv(pathMappings)
#    
#    print(rounds)
#    print(companies_indexed)
#    print(rounds_indexed)
#    t1=rounds.iloc[-1,0]
#    t2=companies.iloc[-1,0]
#    print(t1.lower() +"---" + t2.lower())
#    print(t1.lower() == t2.lower())
    convertTolower=lambda x:x.lower()
    print("-----------------------CONVERTING INTO LOWER----------------")
#    companies_indexed[0]=companies_indexed[0].applymap(convertTolower(companies_indexed.iloc[0]) ,axis=1)
#    rounds_indexed[0]=rounds_indexed[0].applymap(convertTolower(rounds_indexed.iloc[0]),axis=1)
    companies['indexed_permalink']=companies['permalink'].str.lower()
    rounds['indexed_company_permalink']=rounds['company_permalink'].str.lower()
    companies_indexed=companies.set_index("permalink",verify_integrity=True)
    rounds_indexed=rounds.set_index("funding_round_permalink")
#    print(companies_indexed)
#    print(rounds_indexed)
    print("------------------MERGING------------------------")
    master_frame=pd.merge(rounds_indexed,companies_indexed,how="inner",left_on="indexed_company_permalink",right_on="indexed_permalink")
    ########################How many unique companies are present in rounds2?#####################################
    print("No of unique Companies in Rounds2= "+ str(rounds_indexed['indexed_company_permalink'].nunique()))
    ########################How many unique companies are present in companies?#####################################
    print("No of unique Companies in companies= "+ str(companies_indexed['indexed_permalink'].nunique()))
#    print(master_frame.head())
    print("----------------CHECKPOINT 2----------------------")
    cleanedData=master_frame[['funding_round_type','raised_amount_usd']]
    cleanedData.fillna(0,inplace=True)
    averageTable=cleanedData.groupby('funding_round_type')['raised_amount_usd'].mean()
    print(averageTable.idxmax())
    dataForVenture=master_frame[master_frame['funding_round_type']=='venture'].fillna(0)
    dataForAngel=master_frame[master_frame['funding_round_type']=='angel'].fillna(0)
    dataForSeed=master_frame[master_frame['funding_round_type']=='seed'].fillna(0)
    dataForPEquity=master_frame[master_frame['funding_round_type']=='private_equity'].fillna(0)
    print("VENTURE, mean="+str(dataForVenture['raised_amount_usd'].mean()))
    print("Angel, mean="+str(dataForAngel['raised_amount_usd'].mean()))
    print("seed, mean="+str(dataForSeed['raised_amount_usd'].mean()))
    print("equity, mean="+str(dataForPEquity['raised_amount_usd'].mean()))
    
    #################CHECKPOINT 3 ######################
    print("----------------CHECKPOINT 3----------------------")

    top_9=master_frame[["funding_round_type","raised_amount_usd","country_code"]]
    top_9=top_9[top_9["funding_round_type"]=="venture"]
    top_9=top_9.groupby("country_code")["raised_amount_usd"].sum().reset_index()
    top_9=top_9.sort_values(by="raised_amount_usd", ascending=False)[0:10]
    print(top_9)
    top_3=pd.merge(top_9,countries,how="inner",left_on="country_code",right_on="Code")[0:3]
    top_3=top_3.drop(["Code"], axis=1)
    print(top_3.head())
    print("--------------------------CHECKPOINT 4----------")
    print("Starting Sector Analysis 1")

    mappings['category_list']=mappings['category_list'].str.replace('0','na')
    mappings_reduced=pd.melt(mappings,id_vars=["category_list"])
    mappings_reduced=mappings_reduced.where(mappings_reduced["value"]==1)
    mappings_reduced=mappings_reduced.dropna()
    mappings_reduced=mappings_reduced.rename(columns={"variable": "main_sector"})
    mappings_reduced=mappings_reduced.drop('value',axis=1)
    primary_sectors=master_frame['category_list'].str.split('|',n=1,expand=True)
    master_frame["primary_sector"]=primary_sectors[0]
    master_frame=pd.merge(master_frame,mappings_reduced, how='left',left_on='primary_sector',right_on='category_list')
    
    master_frame.drop(["category_list_y","category_list_x"],axis=1,inplace=True);
    print("xxxxxx")
    print("End Sector Analysis 1")
    print("--------------------------CHECKPOINT 5----------")
    print("Starting Sector analysis 2")

    FrameFundingTypeVenture=master_frame[(master_frame["funding_round_type"]=='venture') & (master_frame['raised_amount_usd'] >= 5000000) & (master_frame['raised_amount_usd'] <= 15000000)]
    print("Creating D1 for USA")
    D1=FrameFundingTypeVenture[FrameFundingTypeVenture["country_code"]==top_3.loc[0][0]]
    D1=D1.where((D1['raised_amount_usd'] >= 5000000) & (D1['raised_amount_usd'] <= 15000000))
    D1.dropna(subset=['raised_amount_usd'])
    print("Creating D2 for GRB")
    D2=FrameFundingTypeVenture[FrameFundingTypeVenture["country_code"]==top_3.loc[1][0]]
    D2=D2.where((D2['raised_amount_usd'] >= 5000000) & (D2['raised_amount_usd'] <= 15000000))
    D2.dropna(subset=['raised_amount_usd'])
    print("Creating D3 for IND")
    D3=FrameFundingTypeVenture[FrameFundingTypeVenture["country_code"]==top_3.loc[2][0]]
    D3=D3.where((D3['raised_amount_usd'] >= 5000000) & (D3['raised_amount_usd'] <= 15000000))
    D3.dropna(subset=['raised_amount_usd'])
    
    TotalCountInvestmentsD1=D1.groupby("main_sector")["raised_amount_usd"].count().reset_index(drop=False).sort_values(by="raised_amount_usd",ascending=False)
    TotalCountInvestmentsD2=D2.groupby("main_sector")["raised_amount_usd"].count().reset_index(drop=False).sort_values(by="raised_amount_usd",ascending=False)
    TotalCountInvestmentsD3=D3.groupby("main_sector")["raised_amount_usd"].count().reset_index(drop=False).sort_values(by="raised_amount_usd",ascending=False)
    TotalAmountInvestmentsD1=D1.groupby("main_sector")["raised_amount_usd"].sum().reset_index(drop=False).sort_values(by="raised_amount_usd",ascending=False)
    TotalAmountInvestmentsD2=D2.groupby("main_sector")["raised_amount_usd"].sum().reset_index(drop=False).sort_values(by="raised_amount_usd",ascending=False)
    TotalAmountInvestmentsD3=D3.groupby("main_sector")["raised_amount_usd"].sum().reset_index(drop=False).sort_values(by="raised_amount_usd",ascending=False)
    TotalCountInvestmentsD1.rename(inplace=True,  columns={"raised_amount_usd": "count_investments"})
    TotalCountInvestmentsD2.rename(inplace=True,  columns={"raised_amount_usd": "count_investments"})
    TotalCountInvestmentsD3.rename(inplace=True,  columns={"raised_amount_usd": "count_investments"})
    TotalAmountInvestmentsD1.rename(inplace=True,  columns={"raised_amount_usd": "amount_investments"})
    TotalAmountInvestmentsD2.rename(inplace=True, columns={"raised_amount_usd": "amount_investments"})
    TotalAmountInvestmentsD3.rename(inplace=True, columns={"raised_amount_usd": "amount_investments"})
    TotalInvestmentsD1=pd.merge(TotalCountInvestmentsD1,TotalAmountInvestmentsD1,left_on="main_sector",right_on="main_sector",how="inner")
    TotalInvestmentsD2=pd.merge(TotalCountInvestmentsD2,TotalAmountInvestmentsD2,left_on="main_sector",right_on="main_sector",how="inner")
    TotalInvestmentsD3=pd.merge(TotalCountInvestmentsD3,TotalAmountInvestmentsD3,left_on="main_sector",right_on="main_sector",how="inner")
    D1=pd.merge(D1,TotalInvestmentsD1,left_on="main_sector",right_on="main_sector",how="inner")
    D2=pd.merge(D2,TotalInvestmentsD1,left_on="main_sector",right_on="main_sector",how="inner")
    D3=pd.merge(D3,TotalInvestmentsD1,left_on="main_sector",right_on="main_sector",how="inner")
    ##############Total number of Investments (count)
    print("Total number Of investments in D1="+str(D1["raised_amount_usd"].count()))
    print("Total number Of investments in D2="+str(D2["raised_amount_usd"].count()))
    print("Total number Of investments in D3="+str(D3["raised_amount_usd"].count()))

    
    ##################Total amount of investment (USD)
    print("Total amount Of investments in D1="+str(D1["raised_amount_usd"].sum()))
    print("Total amount Of investments in D2="+str(D2["raised_amount_usd"].sum()))
    print("Total amount Of investments in D3="+str(D3["raised_amount_usd"].sum()))

#    TopSectorsD1D2D3=list()
#    TopSectorsD1D2D3.append(list(TotalInvestmentsD1.values[0]),list(TotalInvestmentsD2.values[0]),list(TotalInvestmentsD3.values[0]))
#    TopSectorsD1D2D3.append(list(TotalInvestmentsD1.values[1]),list(TotalInvestmentsD2.values[1]),list(TotalInvestmentsD3.values[1]))
#    TopSectorsD1D2D3.append(list(TotalInvestmentsD1.values[2]),list(TotalInvestmentsD2.values[2]),list(TotalInvestmentsD3.values[2]))
    
#    print(TopSectorsD1D2D3)
    ################Top Sector name (no. of investment-wise)
    print("Top Sector in D1="+str(TotalInvestmentsD1.values[0]))
    print("Top Sector in D2="+str(TotalInvestmentsD2.values[0]))
    print("Top Sector in D3="+str(TotalInvestmentsD3.values[0]))
    ###################Second Sector name (no. of investment-wise)
    print("2nd Top Sector in D1="+str(TotalInvestmentsD1.values[1]))
    print("2nd Top Sector in D2="+str(TotalInvestmentsD2.values[1]))
    print("2nd Top Sector in D3="+str(TotalInvestmentsD3.values[1]))
    ####################3Third Sector name (no. of investment-wise)
    print("3rd Top Sector in D1="+str(TotalInvestmentsD1.values[2]))
    print("3rd Top Sector in D2="+str(TotalInvestmentsD2.values[2]))
    print("3rd Top Sector in D3="+str(TotalInvestmentsD3.values[2]))
    ##################Number of investments in top sector (3)
    print("No. of Investments in Top Sector - \'"+TotalInvestmentsD1.values[0][0]+ "\' For D1="+str(TotalInvestmentsD1.values[0][1] ))
    print("No. of Investments in Top Sector - \'"+TotalInvestmentsD2.values[0][0]+ "\' For D2="+str(TotalInvestmentsD2.values[0][1] ))
    print("No. of Investments in Top Sector - \'"+TotalInvestmentsD3.values[0][0]+ "\' For D3="+str(TotalInvestmentsD3.values[0][1] ))
    ####################Number of investments in second sector (4)
    print("No. of Investments in 2nd Top Sector - \'"+TotalInvestmentsD1.values[1][0]+ "\' For D1="+str(TotalInvestmentsD1.values[1][1] ))
    print("No. of Investments in 2nd Top Sector - \'"+TotalInvestmentsD2.values[1][0]+ "\' For D2="+str(TotalInvestmentsD2.values[1][1] ))
    print("No. of Investments in 2nd Top Sector - \'"+TotalInvestmentsD3.values[1][0]+ "\' For D3="+str(TotalInvestmentsD3.values[1][1] ))
    ##################Number of investments in third sector (5)
    print("No. of Investments in 3rd Top Sector - \'"+TotalInvestmentsD1.values[2][0]+ "\' For D1="+str(TotalInvestmentsD1.values[2][1] ))
    print("No. of Investments in 3rd Top Sector - \'"+TotalInvestmentsD2.values[2][0]+ "\' For D2="+str(TotalInvestmentsD2.values[2][1] ))
    print("No. of Investments in 3rd Top Sector - \'"+TotalInvestmentsD3.values[2][0]+ "\' For D3="+str(TotalInvestmentsD3.values[2][1] ))
    ################For point 3 (top sector count-wise), which company received the highest investment?
    ############For point 4 (second best sector count-wise), which company received the highest investment?
    D1["lower_company_name"]=D1['name'].str.lower()
    D2["lower_company_name"]=D2['name'].str.lower()
    D3["lower_company_name"]=D3['name'].str.lower()
    TopCompanyD1=D1.groupby(["indexed_company_permalink","lower_company_name"])["raised_amount_usd"].sum().reset_index(drop=False).sort_values(by='raised_amount_usd',ascending=False)
    TopCompanyD2=D2.groupby(["indexed_company_permalink","lower_company_name"])["raised_amount_usd"].sum().reset_index(drop=False).sort_values(by='raised_amount_usd',ascending=False)
    TopCompanyD3=D3.groupby(["indexed_company_permalink","lower_company_name"])["raised_amount_usd"].sum().reset_index(drop=False).sort_values(by='raised_amount_usd',ascending=False)
    print("For D1, Company with highest investment for Top Sector \'"+TotalInvestmentsD1.values[0][0]+"\' is \'"+TopCompanyD1.values[0][1]+"\' with total investment of, $"+str(TopCompanyD1.values[0][2]))
    print("For D1, Company with highest investment for 2nd Top Sector \'"+TotalInvestmentsD1.values[1][0]+"\' is \'"+TopCompanyD1.values[1][1]+"\' with total investment of, $"+str(TopCompanyD1.values[1][2]))
    print("For D2, Company with highest investment for Top Sector \'"+TotalInvestmentsD2.values[0][0]+"\' is \'"+TopCompanyD2.values[0][1]+"\' with total investment of, $"+str(TopCompanyD2.values[0][2]))
    print("For D2, Company with highest investment for 2nd Top Sector \'"+TotalInvestmentsD2.values[1][0]+"\' is \'"+TopCompanyD2.values[1][1]+"\' with total investment of, $"+str(TopCompanyD2.values[1][2]))
    print("For D3, Company with highest investment for Top Sector \'"+TotalInvestmentsD3.values[0][0]+"\' is \'"+TopCompanyD3.values[0][1]+"\' with total investment of, $"+str(TopCompanyD3.values[0][2]))
    print("For D3, Company with highest investment for 2nd Top Sector \'"+TotalInvestmentsD3.values[1][0]+"\' is \'"+TopCompanyD3.values[1][1]+"\' with total investment of, $"+str(TopCompanyD3.values[1][2]))
    
    print("End Sector Analysis 2")
    
    
    print("CheckPoint 6- Plots")
    #for Total investments global:
    print("plot 1")
    plt.figure(figsize=(15,8))
    plt.title("Total Investments Globally- All Sectors")
    TotalInvestmentsGlobal=cleanedData.groupby('funding_round_type')['raised_amount_usd'].sum().reset_index(drop=False)
    averageTable=averageTable.reset_index(drop=False)
    plt.yscale("log")
    plt.xlabel("Funding Types")
    plt.ylabel("Amount")
    plt.bar(TotalInvestmentsGlobal["funding_round_type"],TotalInvestmentsGlobal["raised_amount_usd"],label='Total Investment')
    plt.xticks(rotation=90)
    plt.bar(averageTable.index,averageTable["raised_amount_usd"],label='Mean Investment')
    plt.legend(loc='best')
    plt.show()
    
    print("Plot 2")
    plt.figure(figsize=(15,12))
    plt.title("Top 9 Countries")
    plt.yscale("log")
    plt.xlabel("Countries")
    plt.ylabel("raised_amount_usd")
    plt.bar(top_3["country_code"],top_3["raised_amount_usd"])
    plt.bar(top_9["country_code"],top_9["raised_amount_usd"])
#    plt.legend(loc='best')
    plt.show()
    
    print("Plot 3")
    plt.figure(figsize=(15,9))
    plt.yscale("log")
    plt.xlabel("Top 3 Sectors")
    plt.ylabel("Total Number of Investments in Main Sectors")
    
    #1st subplot
    subplotAlabel="Total Investments in D1," +top_3.values[0][0]+" for each sector"
    plt.subplot(131)
    plt.title(subplotAlabel)
    plt.xticks(rotation=90)
    plt.bar(TotalInvestmentsD1["main_sector"],TotalInvestmentsD1["count_investments"])
    #2nd subplot
    subplotAlabel="Total Investments in D2," +top_3.values[1][0]+" for each sector"
    plt.subplot(132)
    plt.title(subplotAlabel)
    plt.xticks(rotation=90)
    plt.bar(TotalInvestmentsD2["main_sector"],TotalInvestmentsD2["count_investments"])
    #3rd subplot
    subplotAlabel="Total Investments in D3," +top_3.values[2][0]+" for each sector"
    plt.subplot(133)
    plt.title(subplotAlabel)
    plt.xticks(rotation=90)
    plt.bar(TotalInvestmentsD3["main_sector"],TotalInvestmentsD3["count_investments"])
except Exception as e:
    print("exception= "+str(e))