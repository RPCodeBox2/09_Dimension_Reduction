# In[1] - Documentation
"""
Script - 01_DimRed_PCA_Breast_Cancer.py
Decription - 
Author - Rana Pratap
Date - 2020
Version - 1.0
"""
print(__doc__)


# In[2] - Import Libraries

import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
import seaborn as sns
#%matplotlib inline

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()

# In[3] - PCA visualization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='rainbow')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# In[4] - Interpreting the components
pca.components_

map= pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(map,cmap='twilight')

# In[5] - 
del(cancer,df,map,scaled_date,x_pca)

