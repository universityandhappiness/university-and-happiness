'''
Importing pandas for showing the datasets in a better interface, numpy is for calculating for mean and matplotlib for plots.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Top 10 countries in university rankings according to CWUR Data
'''
data = pd.read_csv('d:/Users/SUUSER/Desktop/university data/cwurData.csv',index_col=['world_rank'])
data.head(10)

'''
Top 10 countries in World Happiness Index 2015
'''
data2 = pd.read_csv('d:/Users/SUUSER/Desktop/happiness/2015.csv',index_col=['Country'])
data2.head(10)

'''
Calculating the mean of scores for Switzerland
'''
np.nanmean(data[data.country=='Switzerland'].score)
d = data.dropna()
plt.hist(d[d.country=='Switzerland'].score)
plt.title('Switzerland')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Iceland
'''
np.nanmean(data[data.country=='Iceland'].score)
d = data.dropna()
plt.hist(d[d.country=='Iceland'].score)
plt.title('Iceland')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Denmark
'''
np.nanmean(data[data.country=='Denmark'].score)
d = data.dropna()
plt.hist(d[d.country=='Denmark'].score)
plt.title('Denmark')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Norway
'''
np.nanmean(data[data.country=='Norway'].score)
d = data.dropna()
plt.hist(d[d.country=='Norway'].score)
plt.title('Norway')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Canada
'''
np.nanmean(data[data.country=='Canada'].score)
d = data.dropna()
plt.hist(d[d.country=='Canada'].score)
plt.title('Canada')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Finland
'''
np.nanmean(data[data.country=='Finland'].score)
d = data.dropna()
plt.hist(d[d.country=='Finland'].score)
plt.title('Finland')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Netherlands
'''
np.nanmean(data[data.country=='Netherlands'].score)
d = data.dropna()
plt.hist(d[d.country=='Netherlands'].score)
plt.title('Netherlands')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Sweden
'''
np.nanmean(data[data.country=='Sweden'].score)
d = data.dropna()
plt.hist(d[d.country=='Sweden'].score)
plt.title('Sweden')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for New Zealand
'''
np.nanmean(data[data.country=='New Zealand'].score)
d = data.dropna()
plt.hist(d[d.country=='New Zealand'].score)
plt.title('New Zealand')
plt.xlabel('Score')
plt.ylabel('Count')

'''
Calculating the mean of scores for Australia
'''
np.nanmean(data[data.country=='Australia'].score)
d = data.dropna()
plt.hist(d[d.country=='Australia'].score)
plt.title('Australia')
plt.xlabel('Score')
plt.ylabel('Count')