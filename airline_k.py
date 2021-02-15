# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:11:13 2020

@author: Harish
"""

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

airline_k=pd.read_excel("EastWestAirlines.xlsx")
def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
df_norm=norm_func(airline_k.iloc[:,1:])

k = list(range(2,20))
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
# optimal k value= 5
model=KMeans(n_clusters=5).fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object

md.value_counts() 
airline_k['clust']=md # creating a  new column and assigning it to new column 
a1_mean=pd.DataFrame(airline_k.groupby(airline_k.clust).mean())
