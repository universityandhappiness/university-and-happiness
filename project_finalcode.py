
# coding: utf-8

# In[649]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
from sklearn import neighbors, datasets, linear_model
from scipy.stats import linregress


# In[650]:

data2 = pd.read_csv('C:/Users/Apilium/Desktop/2015.csv',index_col=['HappinessRank'])
data2.head(10)


# In[651]:

data = pd.read_csv('C:/Users/Apilium/Desktop/cwurData.csv',index_col=['world_rank'])
data.head(10)


# In[652]:

np.nanmean(data[data.country=='Switzerland'].score)


# In[653]:

len(data)


# In[654]:

data.dtypes


# In[655]:

data2.dtypes


# In[656]:

data.describe()


# In[657]:

data2.describe()


# normalizing the number of occurences of each country

data.country.value_counts(normalize=True)


# In[659]:

data["country"].value_counts()


# In[660]:

data.groupby("country").score.mean().sort_values(ascending=False)


# In[661]:

data.country.value_counts().plot(kind="bar")


# get histogram for switzerland

d = data.dropna()
plt.hist(d[d.country=='Switzerland'].score)
plt.title('Switzerland')
plt.xlabel('Score')
plt.ylabel('Count')


# In[663]:

data2.groupby("Country").HappinessScore.agg(["mean"])


# get data in matrix

data_matrix=[]
data_matrix=data.as_matrix()
print(data_matrix)


# In[665]:

mean_array=[]
country_array=[]

for country in data.country:
    if not country in country_array:
        country_array.append(country)
        mean_array.append(np.nanmean(data[data.country==country].score))
                                
print(mean_array)


# get scores and country lists 

score_array=[]
for HappinessScore in data2.HappinessScore:
    score_array.append(HappinessScore)

happycountry_array=[]
for Country in data2.Country:
    happycountry_array.append(Country)


# creating clean arrays for countries, scores and means to plot graphs 

cleancountry_array=[]
cleanscore_array=[]

for i in range (0,59):
    for j in range (0,158):
        if (country_array[i]==happycountry_array[j]):
            cleanscore_array.append(score_array[j])
            cleancountry_array.append(happycountry_array[j])
            
cleanmean_array=[]
for i in range(0,59):
    for j in range(0,57):
        if(country_array[i]==cleancountry_array[j]):
            cleanmean_array.append(mean_array[i])

print(cleancountry_array)


# In[668]:

print(cleanmean_array)


# Relationship between university score and happiness score

get_ipython().magic('matplotlib inline')
plt.scatter(cleanmean_array, cleanscore_array)
plt.xlabel('Mean of University Score')
plt.ylabel('Happiness Score')
plt.title("Relationship between university score and happiness score")


# In[671]:

len(cleancountry_array)


# In[672]:

dict = {}
for data in data_matrix:
    if data[1] not in dict and data[1] in cleancountry_array:
        dict[data[1]] = data
        
for d in dict:
    print(dict[d])
len(dict)    


# In[673]:

newscore_array=[]
for d in dict:
    newscore_array.append(dict[d][11])


# In[674]:

print(cleancountry_array)
print(newscore_array)


# university score and happiness score

get_ipython().magic('matplotlib inline')
plt.scatter(newscore_array, cleanscore_array)
plt.xlabel('University Score')
plt.ylabel('Happiness Score')


# economy and happiness score

get_ipython().magic('matplotlib inline')
plt.scatter(data2.Economy, data2.HappinessScore)
plt.xlabel('University Score')
plt.ylabel('Happiness Score')


# In[677]:

linregress(data2.Economy, data2.HappinessScore)


# In[678]:

fit = np.polyfit(data2.Economy,data2.HappinessScore,1)
fit_fn = np.poly1d(fit) 

# economy and happiness score

plt.plot(data2.Economy,data2.HappinessScore, 'yo', data2.Economy, fit_fn(data2.Economy), '--k')


# highest scored uni of each country and happiness score

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(newscore_array[:10], cleanscore_array[:10], s=10, c='b', marker="s", label='Top 10')
ax1.scatter(newscore_array[-10:], cleanscore_array[-10:], s=10, c='r', marker="o", label='Bottom 10')
ax1.set_xlabel('University Score')
ax1.set_ylabel('Happiness Score')
plt.legend(loc='lower right');
plt.title("Highest Scored University of a Country and Happiness Score")
plt.show()


# In[680]:

linregress(cleanmean_array,cleanscore_array)


# In[681]:

fit = np.polyfit(cleanmean_array,cleanscore_array,1)
fit_fn = np.poly1d(fit) 

# regression line of means and happiness score

plt.plot(cleanmean_array,cleanscore_array, 'yo', cleanmean_array, fit_fn(cleanmean_array), '--k')
plt.xlim(43, 55)
plt.ylim(3, 8)
plt.xlabel('University Score Mean')
plt.ylabel('Happiness Score')
plt.title('Regression line of means and happiness score')


# In[682]:

fit = np.polyfit(cleanmean_array,cleanscore_array,1)
fit_fn = np.poly1d(fit) 

# freedom and happiness score 

plt.plot(data2.Freedom,data2.HappinessScore, 'yo', data2.Freedom, fit_fn(data2.Freedom), '--k')
plt.xlim(0, 0.7)
plt.ylim(2.5, 8)
plt.xlabel('Freedom Score')
plt.ylabel('Happiness Score')
plt.title('Freedom and happiness score')


# In[683]:

fit = np.polyfit(cleanmean_array,cleanscore_array,1)
fit_fn = np.poly1d(fit) 

# regression line of means and happiness score

plt.plot(data2.HappinessScore,data2.Economy, 'yo', data2.HappinessScore, fit_fn(data2.HappinessScore), '--k')
plt.xlim(0, 0.7)
plt.ylim(2.5, 8)
plt.xlabel('University Score Mean')
plt.ylabel('Happiness Score')
plt.title('Regression line of means and happiness score')


# polynomial line of means and happiness score

fit = np.polyfit(cleanmean_array,cleanscore_array,3)
fit_fn = np.poly1d(fit) 

plt.plot(cleanmean_array,cleanscore_array, 'yo', cleanmean_array, fit_fn(cleanmean_array), '--k')
plt.xlim(43, 55)
plt.ylim(3, 8)
plt.xlabel('University Score Mean')
plt.ylabel('Happiness Score')
plt.title('Polynomial line of means and happiness score')


# In[685]:

data2f = data2.corr()


# In[686]:

data2f


# heatmap of happiness value

import seaborn as sns
fig = plt.figure(figsize=(8,8))
sns.heatmap(data2f);


# number of occurences of each region

data2.Region.value_counts()


# In[689]:

data2.head(10)


# In[690]:

data2 = data2.reset_index(drop=True)


# In[691]:

data2.head(10) # index is ordered as well as group


# In[692]:

dict2 = {}
for data3 in data3_matrix:
    if data3[2] not in dict2 and data3[2] in cleancountry_array:
        dict2[data3[2]] = data3
tempcountry_array=[]
for d in dict2:
    tempcountry_array.append(dict2[d][2])
    
    
print(tempcountry_array)
len(tempcountry_array)


# In[693]:

len(dict2)


# fitting linear regression plot

m = regr.coef_[0]
b = regr.intercept_
print(' y = {0} * x + {1}'.format(m, b))
plt.scatter(scores, cleanmean_array, color='blue')
plt.plot([scores[0], scores[-1]], [m*scores[0] + b, m*scores[-1] + b], 'r')
plt.title('Fitting linear regression', fontsize=14)
plt.xlabel('Happiness Score', fontsize=13)
plt.ylabel('University Score Mean', fontsize=13)


# estimation of new happiness value in graph

estimationOfX = m * 6 + b
plt.scatter(scores, cleanmean_array, color='blue')
plt.scatter([6], [estimationOfX], color='red')
plt.plot([scores[0], scores[-1]], [m*scores[0] + b, m*scores[-1] + b], 'r')
plt.title('Estimation of A Happiness Score of 6', fontsize=14)
plt.xlabel('Happiness Score', fontsize=13)
plt.ylabel('University Score Mean', fontsize=13)


# error of estimation

estimation_error = estimationOfX - cleanmean_array[6]
print(estimation_error)


# estimation of new happiness value

estimationOfX = m * 6 + b
print(estimationOfX)


# real vs predictive model

predictions = [m * cleanscore_array[score] + b for score in range(0,57)]
plt.scatter(years, cleanmean_array, color='blue') #real
plt.scatter(years, predictions, color='red') #model
plt.title('Model Distribution versus Real Distribution', fontsize=14)
plt.xlabel('Happiness Score', fontsize=13)
plt.ylabel('University Score Mean', fontsize=13)

