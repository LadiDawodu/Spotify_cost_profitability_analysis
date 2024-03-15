#Imports and Read

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


df = pd.read_csv('/Users/ladi./Developer/data_analyst/PERSONAL PROJECTS/SPOTIFY/Spotify Quarterly.csv')

# --------------------------- EDA -------------------------------------
#Data Understanding

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

#---------------------------------------------------


#Feature Understanding
 
df
df['Total Revenue'].plot(kind='line')


def histo_Total_Revenue():
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Total Revenue'])
        plt.title('Total Revenue')
        plt.xlabel('Total Revenue')
        plt.ylabel('Frequency')
        plt.show()

histo_Total_Revenue()


#Gross Profit / Cost Breakdwon


#Gross for each revenue stream
"""
def gross_Rev():
        
        df['Gross Profit'] =  df['Total Revenue'] - df['Cost of Revenue']

        cost_categories = ['Sales and Marketing Cost', 'Research and Development Cost', 'Genreal and Adminstraive Cost']

        df['Total Cost'] = df[cost_categories].sum(axis=1)

        for category in cost_categories:
                df[f'{category} (%)'] = (df[category] / df['Total Revenue']) * 100
                
        plt.plot(df['Total Cost'])  # Assuming index represents meaningful x-axis values
        plt.xlabel('Total cost')  # Replace 'X-axis Label' with an appropriate label for your x-axis
        plt.ylabel('Total Cost')  # Set y-axis label as 'Total Cost'
        plt.title('Total Cost Over Time')  # Set plot title
        plt.show()
        

gross_Rev()
"""

#Descriptive

revenue_columns = ["Total Revenue", "Premium Revenue","Ad Revenue"]
cost_columns = ['Sales and Marketing Cost', 'Research and Development Cost', 'Genreal and Adminstraive Cost']

def rev_stats(): 
        print("Revenue Stats:")       
        for col in revenue_columns:
                print(df[col].describe())

rev_stats()


def cost_stats(): 
        print("Cost Stats:")       
        for cost in cost_columns:
                print(df[cost].describe())

cost_stats()


def skew():
        for cols in revenue_columns + cost_columns:
                print(f"{cols}: Skewness - {df[cols].skew()}")
        
skew()

#----------------------------------------------------------------

#FEATURE RELATIONSHIPS

#Revenue vs Costs

def rev_costs():
        df.plot(kind="scatter",
                x="Total Revenue",
                y="Sales and Marketing Cost",
                title= 'Total Revenue vs. Cost of Revenue')
        plt.show()
rev_costs()


#Change date
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Year']
decades_map = {
        2017: '2010s',
    2018: '2010s',
    2019: '2010s',
    2020: '2020s',
    2021: '2020s',
    2022: '2020s',
    2023: '2020s'
}

df['decades'] = df['Year'].map(decades_map)

df.dtypes()

sns.reset_defaults()
# def ad_rev_ad_cost_rev():
        #df['Ad Revenue'] = df['Ad Revenue'].astype(int)
ad_rev_ad_cost_rev = sns.scatterplot(data=df,
                x='Ad Revenue',
                y='Ad Cost of revenue',
                hue="decades")
ad_rev_ad_cost_rev.set_title('Ad Revenue vs. Ad Cost of revenue')
plt.show()
#ad_rev_ad_cost_rev() 

#print(df.head())


#----------------------------------------------------------
# Analysis:

# How has total revenue, premium revenue, 
# and ad revenue evolved over the years? 
# Are there any noticeable trends or seasonal patterns in revenue generation?

#Total Rev
def revenue_over_years():
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=df, x='Year', y='Total Revenue', label= 'Total Revenue')
        ax.set_title('Total Revenue over Years')
        plt.show()
revenue_over_years()


#Premium Rev
def premium_rev():
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=df, x='Year', y='Premium Revenue', label = 'Premium Revenue')
        ax.set_title('Premium Revenue over Years')
        plt.show()
premium_rev()


#Ad Revenue
def ad_rev():
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=df, x='Year', y='Ad Revenue', label = 'Ad Revenue')
        ax.set_title('Ad Revenue over Years')
        plt.show()
ad_rev()
        




# Trend of total revenue, premium revenue, 
# and ad revenue over the years
     
def trend_rev_over_years():
           plt.figure(figsize=(10, 6))
           # Total Revenue
           sns.lineplot(data=df, x='Year', y='Total Revenue', label='Total Revenue')
           # Premium Revenue
           sns.lineplot(data=df, x='Year', y='Premium Revenue', label='Premium Revenue')
           # Ad Revenue
           sns.lineplot(data=df, x='Year', y='Ad Revenue', label='Ad Revenue')
           plt.title('Trend of Revenue Over Years')
           plt.xlabel('Year')
           plt.ylabel('Revenue')
           plt.legend()
           plt.show()
trend_rev_over_years()

#----------------------------------------------------------

# What is the distribution of costs across different categories such as sales and marketing, 
# research and development, and general administrative costs? 
# How have these costs changed over time. 
# What impact do they have on overall profitability?

# Calculate Distribution of Cost over Time:

cost_columns = [
        'Sales and Marketing Cost',
        'Research and Development Cost', 
        'Genreal and Adminstraive Cost'
        ]

df['Total Cost'] = df[cost_columns].sum(axis=1)

print(df[['Date', 'Total Cost']])


# Visualized Calculate Total Cost
def cost_over_time():
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=df, x='Date', 
                        y='Total Cost', label = 'Cost Over Time')
        ax.set_title('Cost Prof')
        plt.show()

cost_over_time()

#------------------------------------------------------------------------

# Is there a correlation between advertising spending (ad revenue)
# and its effectiveness (e.g., increase in total revenue or growth in monthly active users)? 
# How does the return on investment (ROI) vary across different advertising channels or campaigns?


# Calculate Advertising ROI:

df['ROI'] = (df['Total Revenue'] - df['Ad Cost of revenue']) / df['Ad Cost of revenue']
print(df[['Date', 'Total Revenue', 'Ad Cost of revenue', 'ROI']])


scaled_roi = df['ROI'] * 100
def advertising_roi():
           plt.figure(figsize=(10, 6))
           
           #plt.subplots(ylim=(min(df['ROI']), max(df['ROI'])))  # Adjust limits as needed
           
           # Total Revenue
           sns.lineplot(data=df, x='Year', y='Total Revenue', label='Total Revenue')
           # Premium Revenue
           sns.lineplot(data=df, x='Year', y='Ad Cost of revenue', label='Ad Cost of revenue')
           # Ad Revenue
           sns.lineplot(data=df, x='Year', y=scaled_roi, label='ROI', linewidth=2)
           plt.title('Advertising ROI')
           plt.xlabel('Year')
           plt.ylabel('Revenue')
           plt.legend()
           plt.show()
advertising_roi()

df['ROI']


# Correlation:
# quantify the relationship between ad spending and various effectiveness metrics, 
# such as total revenue or growth in monthly active users

ad_corr = df['Total Revenue'].corr(df['Ad Cost of revenue'])

ad_mau_corr = df['Total Revenue'].corr(df['MAUs'])

print(ad_mau_corr)
print(ad_corr)

# Visualize

def revenue_relationship():
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=df,  
        x='Total Revenue', y='Ad Cost of revenue', label = 'Ad Cost of revenue relationship')
        
        ax2 = sns.lineplot(data=df,  
        x='Total Revenue', y='MAUs', label = 'Ad MAUs')
        
        plt.title('Ad Cost of revenue & MAU corr')
        plt.xlabel('Total Revenue')
        plt.ylabel('Ad Cost of revenue & MAU ')
        plt.legend()
        plt.show()

revenue_relationship()


def muas_relationship():
        plt.figure(figsize=(10,6))
        ax = sns.scatterplot(data=df,  
        x='Total Revenue', y='MAUs', label = 'Ad Cost of revenue relationship')
        ax.set_title('Ad Cost of revenue + Total Revenue Relationship ')
        plt.show()

muas_relationship()


# Growth rates of premium MAUs and ad MAUs over time

def growth_mau():
        df['Premium MAUs Growth'] = df['Premium MAUs'].pct_change()
        df['Ad MAUs Growth'] = df ['Ad MAUs'].pct_change()

        plt.figure(figsize=(10,6))
        sns.lineplot(data=df, y='Premium MAUs', x='Date', 
                label='Premium MAUs')
        sns.lineplot(data=df, y='Ad MAUs', x='Date', label='Ad MAUs Growth')
        plt.xlabel('Date')
        plt.ylabel('Growth Rate %')
        plt.title('Growth rates of premium MAUs and ad MAUs over time')
        plt.show()
growth_mau()



#df.describe()



premium_arpu = df['Premium Revenue'] / df['Premium MAUs']
ad_arpu = df['Ad Revenue'] / df['Ad MAUs']

# Visualize ARPU Comparison
plt.figure(figsize=(10, 6))
plt.bar(['Premium', 'Ad'], [premium_arpu.mean(), ad_arpu.mean()])
plt.xlabel('User Type')
plt.ylabel('Average Revenue per User (ARPU)')
plt.title('Comparison of ARPU between Premium and Ad Users')
plt.show()


# Gross Profit:

df['Gross Profit'] = df['Total Revenue'] - df['Ad Cost of revenue']

# Gross Profit Margin:

df['Gross Profit Margin'] = (df['Gross Profit'] / df['Total Revenue'])

print(df['Gross Profit'])
print(df['Gross Profit Margin']) 

fig,(ax1, ax2) = plt.subplots(2,1, figsize=(10,8))

sns.lineplot(data=df, x='Date', y='Gross Profit', 
        label='Gross Profit', ax=ax1)
ax1.set_ylabel('Gross Profit')

sns.lineplot(data=df, x='Date', y='Gross Profit Margin', 
        label='Gross Profit', ax=ax2)
ax1.set_ylabel('Gross Profit Margin %')

fig.suptitle('Gross Profit and Margin over Time')
plt.show()







