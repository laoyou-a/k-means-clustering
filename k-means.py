#**************************************************************************************************
#k-means Clustering Algorithm
#Fabian Luettel
#inspired by VanderPlan, Jake (2016): Python Data Science Handbook, O'Reilly Media, Inc., https://bit.ly/39nA4Ke (Zugriff: 27.11.2020)
#**************************************************************************************************

# Step (0): Include libraries
#----------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs #Gaussian data
from sklearn.datasets import make_moons #crescent shape data with two crescents
from sklearn.datasets import make_circles #circular data in 2D
from sklearn.cluster import KMeans
import random
import sys
import tkinter as tk #for dialogue
from tkinter import messagebox as mb #for dialogue
import tkinter.font as font

# Step (1): GUI settings
#----------------------------------------------
gui = tk.Tk(className="k-means Algorithm")
gui.geometry("300x300")
gui.configure(bg="#3f888f")

def answ_gauss():
    global data_distribution_type
    data_distribution_type = "type_gauss"
    gui.destroy() #also possible:  #mb.showinfo("Mode chosen. Calculation will start.")
def answ_crescent():
    global data_distribution_type
    data_distribution_type = "type_crescent"
    gui.destroy()
def answer_circular():
    global data_distribution_type
    data_distribution_type = "type_circular"
    gui.destroy()

myFont = font.Font(size=30)
Button_A = tk.Button(text='Gauss Data',
                     bg='#0052cc', fg='#ffffff',
                     height = 3, width = 5,
                     command=answ_gauss).pack(fill=tk.X)
Button_B = tk.Button(text='Crescent Data',
                     bg='#0052cc', fg='#ffffff',
                     height = 3, width = 5,
                     command=answ_crescent).pack(fill=tk.X)
Button_C = tk.Button(text='Circular Data',
                     bg='#0052cc', fg='#ffffff',
                     height = 3, width = 5,
                     command=answer_circular).pack(fill=tk.X)

#see more types here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
gui.mainloop()

# Step (2): Preparations
#----------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('k-means Clustering')

# Step (3): settings
#----------------------------------------------
n_clusters_algorithm = 2 #how many clusters should the algorithm detect?
n_samples = 350

if(data_distribution_type == "type_gauss"):
    cluster_std = 1.5
    n_dimensions = 2  # for n=3 its harder to plot; you need mplot3d for example
    n_clusters_real = 6

elif(data_distribution_type == "type_crescent"):
    value_noise = 0.05

elif(data_distribution_type == "type_circular"):
    value_noise = 0.05

#else:
    #throw error...


# Step (4): Generate some sample data
#----------------------------------------------
if(data_distribution_type == "type_gauss"):
    X, y = make_blobs(n_samples=n_samples,
                           centers=n_clusters_real,
                           n_features=n_dimensions,
                           #random_state=0,
                           cluster_std=cluster_std)

elif(data_distribution_type == "type_crescent"):
    X, y = make_moons(n_samples=n_samples,
                           #random_state=0,
                           noise=value_noise)

elif(data_distribution_type == "type_circular"):
    X, y = make_circles(n_samples=n_samples,
                           #random_state=0,
                           noise=value_noise)
#else:
    #throw error...


# Step (5): Visualize this data
#----------------------------------------------
print(X.shape)
ax1.scatter(X[:, 0], X[:, 1]) #2D data


# Step (6): Perform k-means algorithm
#----------------------------------------------
kmeans = KMeans(n_clusters=n_clusters_algorithm)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# Step (7): Visualize clustered data
#----------------------------------------------
#if(n_dimensions == 2):
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='black', alpha=0.5);


# Step (8): Plot
#----------------------------------------------
plt.show()
#plt.savefig('img.png')