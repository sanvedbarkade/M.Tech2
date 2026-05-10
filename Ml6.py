# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("/content/Mall_Customers - Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

# To see the distribution of data
data.boxplot(figsize=(8,4))

data.describe()

# To see frequency in the data
data.hist(figsize=(10,6))
plt.show()

# Extracting Data without labels using iloc

X_val = data.iloc[:,3:]

from sklearn.cluster import KMeans

val=[]

for i in range(1,15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=3)
    kmeans.fit(X_val)
    val.append(kmeans.inertia_)
print(val)


X_val.head()

# Finding the optimum number of clusters using Elbow method
plt.plot(range(1,15),val)

plt.xlabel('The No of clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('The Elbow Method')
plt.show()

# We can see the elbow on arm relies on 5 so will take cluster count as 5
# Kmeans Clustering algorithm
kmeans = KMeans(n_clusters = 5, init='k-means++')
kmeans.fit(X_val)

#To show centroids of clusters
kmeans.cluster_centers_

#Prediction of K-Means clustering
y_kmeans = kmeans.fit_predict(X_val)
y_kmeans

# Convert X_val to numpy array if it's a pandas DataFrame
if isinstance(X_val, pd.DataFrame):
    X_val = X_val.values

# Now create the scatter plot to visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_val[y_kmeans == 0, 0], X_val[y_kmeans == 0, 1], c='red', s=100, label='Cluster 1')
plt.scatter(X_val[y_kmeans == 1, 0], X_val[y_kmeans == 1, 1], c='green', s=100, label='Cluster 2')
plt.scatter(X_val[y_kmeans == 2, 0], X_val[y_kmeans == 2, 1], c='orange', s=100, label='Cluster 3')
plt.scatter(X_val[y_kmeans == 3, 0], X_val[y_kmeans == 3, 1], c='blue', s=100, label='Cluster 4')
plt.scatter(X_val[y_kmeans == 4, 0], X_val[y_kmeans == 4, 1], c='yellow', s=100, label='Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           s=300, c='brown', marker='*', label='Centroids')

plt.title('K-Means Clustering')
plt.legend()
plt.show()