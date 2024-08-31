PREDICTING CUSTOMER CHURN FOR SYRIATEL
Name: Amos Kipngetich Rotich

Student Pace: Part Time

Scheduled Project Review Date/Time: September 1, 2024

Instructor Name: Winnie Anyoso

1. Introduction
Background:
Customer churn/turnover is a significant concern for telecommunications companies like SyriaTel. Churn occurs when customers discontinue their service, leading to revenue loss and increased costs for acquiring new customers. Understanding and predicting churn enables companies like SyriaTel to take proactive measures to retain customers at risk of leaving, thereby improving customer satisfaction and reducing financial losses.

Objective:
The primary objective of this project is to develop a predictive model that identifies SyriaTel customers likely to end their relationship with the company. By accurately predicting the turnover, SyriaTel can focus its retention efforts on at-risk customers, ultimately reducing churn rates and enhancing the company's profitability.

Dataset Overview:
The dataset used in this project contains records of 3,333 SyriaTel customers. The dataset includes various attributes related to customer demographics, account details, service usage, and customer service interactions. The target variable is churn, a binary indicator of whether a customer has stopped using SyriaTel’s services.

Business Problem:
For SyriaTel, retaining customers is crucial to maintaining a stable revenue stream. By analyzing data on customer behavior and service usage, the company can identify patterns that suggest a higher likelihood of turnover. The predictive model developed in this project will assist SyriaTel in implementing targeted retention strategies, such as personalized offers or improved customer support, to reduce churn.

2. Data Processing
We start by handling any necessary preprocessing steps, including converting categorical variables, handling missing values (if any), and scaling numerical features.

Converting Categorical Variables:
We'll convert categorical variables such as state, international plan, and voice mail plan into numerical values using one-hot encoding.

Scaling Numerical Features:
For models sensitive to feature scales, such as logistic regression, we will standardize the numerical features.

Let's start with the preprocessing:

1. Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
2. Import Dataset
data = 'bigml_59c28831336c6604c800002a.csv'
df = pd.read_csv(data)
3. Exploratory Data Analysis
df.shape
(3333, 21)
The dataset consists of 3,333 customer records, each containing 21 attributes.

df.head()
state	account length	area code	phone number	international plan	voice mail plan	number vmail messages	total day minutes	total day calls	total day charge	...	total eve calls	total eve charge	total night minutes	total night calls	total night charge	total intl minutes	total intl calls	total intl charge	customer service calls	churn
0	KS	128	415	382-4657	no	yes	25	265.1	110	45.07	...	99	16.78	244.7	91	11.01	10.0	3	2.70	1	False
1	OH	107	415	371-7191	no	yes	26	161.6	123	27.47	...	103	16.62	254.4	103	11.45	13.7	3	3.70	1	False
2	NJ	137	415	358-1921	no	no	0	243.4	114	41.38	...	110	10.30	162.6	104	7.32	12.2	5	3.29	0	False
3	OH	84	408	375-9999	yes	no	0	299.4	71	50.90	...	88	5.26	196.9	89	8.86	6.6	7	1.78	2	False
4	OK	75	415	330-6626	yes	no	0	166.7	113	28.34	...	122	12.61	186.9	121	8.41	10.1	3	2.73	3	False
5 rows × 21 columns

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3333 entries, 0 to 3332
Data columns (total 21 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   state                   3333 non-null   object 
 1   account length          3333 non-null   int64  
 2   area code               3333 non-null   int64  
 3   phone number            3333 non-null   object 
 4   international plan      3333 non-null   object 
 5   voice mail plan         3333 non-null   object 
 6   number vmail messages   3333 non-null   int64  
 7   total day minutes       3333 non-null   float64
 8   total day calls         3333 non-null   int64  
 9   total day charge        3333 non-null   float64
 10  total eve minutes       3333 non-null   float64
 11  total eve calls         3333 non-null   int64  
 12  total eve charge        3333 non-null   float64
 13  total night minutes     3333 non-null   float64
 14  total night calls       3333 non-null   int64  
 15  total night charge      3333 non-null   float64
 16  total intl minutes      3333 non-null   float64
 17  total intl calls        3333 non-null   int64  
 18  total intl charge       3333 non-null   float64
 19  customer service calls  3333 non-null   int64  
 20  churn                   3333 non-null   bool   
dtypes: bool(1), float64(8), int64(8), object(4)
memory usage: 524.2+ KB
​
