import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


df = pd.read_csv('/Users/ladi./Developer/data_analyst/PERSONAL PROJECTS/SPOTIFY/Spotify Quarterly.csv')

df.shape

df.head(20)

df.columns

df.dtypes

df.describe()

#Data Prep

df.head()


#Found missing values for almost all colums
df.isna().sum() 

#No duplicated values 
df.duplicated()

#To find duplicated location (none found )
df.loc[df.duplicated]

#df.loc[df.duplicated(subset=['Premium Revenue'])]

missing_values = df.isnull().sum()

print("Missing Values:",missing_values)

#---------------------------
df_cleaned = df.dropna()

df_cleaned.isna().sum() 

df.dropna(inplace=True)

print(df.isna().sum())

#-------------------------------------------------------------------------

# What is the distribution of costs across different 
# categories such as sales and marketing, research and development, 
# and general administrative costs? 
# How have these costs changed over time, 
# and what impact do they have on overall profitability?



# The business landscape is rather competitive in today's market
# it's paramount to orginizational success. Cost analysis offers
# a foundational pillar in the descision-making process, which in turn
# provides invaluable insight into operational effeciency, resource allocation,
# and overall financial literacy.
# Checking cost structures over time we can get a better view of potential inefficiencies,
# and maximise profitability
# The objectives of this 'Cost Analysis' is to understand the relationship
# between cost dynamics / overall profits, which should help in the understandings
# of the fluctuations in costs. Also to show cost expenditure that can be furhter optimized
# By understanding costs there can be financial organization which as stated in the beginning
# is needed for today's competitve landscapce



# Data Description/Overview of Dataset:

print('Dataset Overview: ')
print(f'\nNumber of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
print('\nData Field:')
print(df.columns)


# First 10 rows:

print('\nFirst few rows:')
print(df.head())

# Dataset Stats:

print('\nDataset Stats: ')
print(df.describe())


#Dataset info, datatype, nullCount:

print('\nInformation about Dataset: ')
print(df.info())

#Cost categories in Dataset:

cost_columns = ['Sales and Marketing Cost', 
                'Research and Development Cost',
                'Genreal and Adminstraive Cost'
                ]
cost_data = df[cost_columns]

print("\nColumns related to cost categories:")
print(cost_data.head())


#Visualization Interpretation: