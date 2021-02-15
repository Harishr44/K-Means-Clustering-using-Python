# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:38:39 2020

@author: Harish
"""

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
crime= pd.read_csv("crime_data.csv")
def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
df_norm=norm_func(crime.iloc[:,1:])

k = list(range(2,15))
k
TWSS = []  
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
    plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
# optimal k value=4

model=KMeans(n_clusters=4).fit(df_norm)
model.labels_ 
md=pd.Series(model.labels_) 
md.value_counts() 
crime['clust']=md
crime = crime.iloc[:,[5,0,1,2,3,4]]

crime.groupby(crime.clust).mean()
# Infernce:-
# second cluster cities have lower average values for murder assault and rape than other cluster cities
# second cluster cities are safer to live than other cities.