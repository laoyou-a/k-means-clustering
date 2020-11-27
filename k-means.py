# Step (0): Include libraries
#----------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Step (1): Preparations
#----------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('k-means Clustering')


# Step (2): Generate some sample data
#----------------------------------------------
n_clusters = 6
n_dimensions = 2 #for n=3 its harder to plot; you need mplot3d for example
X, y = make_blobs(n_samples=350,
                       centers=n_clusters,
                       n_features=n_dimensions,
                       #random_state=0,
                       cluster_std=1.5)


# Step (3): Visualize this data
#----------------------------------------------
print(X.shape)
ax1.scatter(X[:, 0], X[:, 1]) #2D data


# Step (4): Perform k-means algorithm
#----------------------------------------------
kmeans = KMeans(n_clusters=n_clusters) #here its possible to give more oir less than n_clusters clusters
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# Step (5): Visualize clustered data
#----------------------------------------------
#if(n_dimensions == 2):
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='black', alpha=0.5);


# Step (6): Plot
#----------------------------------------------
plt.show()
#plt.savefig('img.png')