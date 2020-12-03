#**************************************************************************************************
#k-means Clustering Algorithm
#Fabian Luettel
#The k-means examples are inspired by VanderPlan, Jake (2016): Python Data Science Handbook, O'Reilly Media, Inc., https://bit.ly/39nA4Ke (Zugriff: 27.11.2020)
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

from skimage.segmentation import slic #skimage for k-means for image
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# Step (1): GUI settings
#----------------------------------------------
gui = tk.Tk(className="k-means Algorithm")
gui.geometry("400x400")
gui.configure(bg="azure")
myFont = font.Font(size=10)
global mode_on
mode_on = True

def closeGUI(): #close GUI
    if mb.askokcancel("Quit", "Do you want to quit?"):
        gui.destroy() #also possible:  #mb.showinfo("Mode chosen. Calculation will start.")
        global mode_on
        mode_on = False
        plt.close('all') #close also all figure windows

def answ_gauss(): #start k-means algorithm for gauss distribution
    global data_distribution_type
    data_distribution_type = "type_gauss"
    perform_calculations_method() #this is performing the k-means calculations defined below

def answ_crescent(): #start k-means algorithm for crescent distribution
    global data_distribution_type
    data_distribution_type = "type_crescent"
    perform_calculations_method()

def answer_circular(): #start k-means algorithm for circular distribution
    global data_distribution_type
    data_distribution_type = "type_circular"
    perform_calculations_method()

def answer_image():  # start k-means algorithm for an image
    global data_distribution_type
    data_distribution_type = "type_image"
    perform_image_calculation_method()

# Step (2): Defining the function for using the k-means algorithm
# ----------------------------------------------
def perform_calculations_method():

    # Step (2.1): Preparations
    #----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Step (2.2): settings
    #----------------------------------------------
    n_samples = 350

    if(data_distribution_type == "type_gauss"):
        cluster_std = 1.2
        n_dimensions = 2  # for n=3 its harder to plot; you need mplot3d for example
        n_clusters_real = 5 # real amount of clusters
        n_clusters_algorithm = 5  # how many clusters should the algorithm detect?
        data_type = "Gauß data"

    elif(data_distribution_type == "type_crescent"):
        value_noise = 0.05
        n_clusters_algorithm = 2  # how many clusters should the algorithm detect?
        data_type = "crescent-shaped data"

    elif(data_distribution_type == "type_circular"):
        value_noise = 0.05
        n_clusters_algorithm = 2  # how many clusters should the algorithm detect?
        data_type = "circle-shaped data"

    #see more types here: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
    #for circular, crescent etc better use DBSCAN etc.


    elif(data_distribution_type == "type_image"):
        value_noise = 0.05
        #n_clusters_algorithm = 2  # how many clusters should the algorithm detect?
        #data_type = "circle-shaped data"

    #else:
        #throw error...

    # Step (2.3): Generate some sample data
    #----------------------------------------------
    if(data_distribution_type == "type_gauss"):
        X, y = make_blobs(n_samples=n_samples,
                               centers=n_clusters_real,
                               n_features=n_dimensions,
                               random_state=4, #seed no4 is a good example with this setting
                               cluster_std=cluster_std)

    elif(data_distribution_type == "type_crescent"):
        X, y = make_moons(n_samples=n_samples,
                               #random_state=0,
                               noise=value_noise)

    elif(data_distribution_type == "type_circular"):
        X, y = make_circles(n_samples=n_samples,
                               #random_state=0,
                               noise=value_noise,
                               factor=0.5)
    #else:
        #throw error...

    # Step (2.4): Visualize this data
    #----------------------------------------------
    print(X.shape)
    ax1.scatter(X[:, 0], X[:, 1]) #2D data
    ax1.set(xlabel='Attribute 1', ylabel='Attribute 2')

    # Step (2.5): Perform k-means algorithm
    #----------------------------------------------
    kmeans = KMeans(n_clusters=n_clusters_algorithm, algorithm="auto") #according to Python doc: The k-means problem is solved using either Lloyd’s or Elkan’s algorithm.
    kmeans.fit(X) #compute k-means clustering
    y_kmeans = kmeans.predict(X) #compute cluster centers and predict cluster index for each sample

    # Step (2.6): Visualize clustered data
    #----------------------------------------------
    #if(n_dimensions == 2):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], c='black', alpha=0.5);
    ax2.set(xlabel='Attribute 1', ylabel='Attribute 2')

    # Step (2.7): Plot
    #----------------------------------------------
    for ax in (ax1, ax2): # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.label_outer()
    fig.suptitle('k-means clustering with '+data_type)
    plt.savefig('clustering_' + data_distribution_type + '.png', dpi=300)
    plt.show()



# Step (3): Defining the function for using the k-means algorithm for images
# ----------------------------------------------
def perform_image_calculation_method():

    # Step (3.1): Preparations
    #----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Step (3.2): settings
    #----------------------------------------------
    n_clusters_algorithm = 2  # how many clusters should the algorithm detect?; here 2: remove background and keep the foreground (temple)

    # Step (3.3): Read sample image
    #----------------------------------------------
    image = img_as_float(io.imread('raohe_temple.jpg'))

    # Step (3.4): Visualize original picture
    #----------------------------------------------
    ax1.imshow(image)
    ax1.axis('off')

    # Step (3.5): Perform k-means algorithm
    #----------------------------------------------
    # slic is k means clustering for images, see https://www.kite.com/python/docs/skimage.segmentation.slic
    # sigma is for smoothing
    # compactness: high value: focus on spatial distance; low value: focus on color proximity
    segments = slic(image,
                    n_segments=n_clusters_algorithm,
                    sigma=1,
                    compactness=0.05)

    # Step (3.6): Visualize clustered image
    #----------------------------------------------
    ax2.imshow(mark_boundaries(image = image,
                               label_img = segments,
                               color = (0,255,127),
                               outline_color = (240, 248, 255)
                               ))
    ax2.axis('off')

    # Step (3.7): Plot
    #----------------------------------------------
    fig.suptitle('k-means clustering for image data with '+ str(n_clusters_algorithm) + ' clusters')
    plt.savefig('clustering_image.png', dpi=300)
    plt.show()

# Step (4): Loop all the stuff using GUI buttons
# ----------------------------------------------
while(mode_on == True):

    Button_Start_Gauss = tk.Button(text='Gauß Data',
                         bg='lawn green', fg='black',
                         height = 3, width = 5,
                         command=answ_gauss)
    Button_Start_Crescent = tk.Button(text='Crescent Data',
                         bg='lawn green', fg='black',
                         height = 3, width = 5,
                         command=answ_crescent)
    Button_Start_Circular = tk.Button(text='Circular Data',
                         bg='lawn green', fg='black',
                         height = 3, width = 5,
                         command=answer_circular)
    Button_Start_Image = tk.Button(text='Image Data',
                         bg='gold', fg='black',
                         height = 3, width = 5,
                         command=answer_image)
    Button_Close = tk.Button(text='Terminate',
                         bg='firebrick1', fg='white',
                         height = 3, width = 5,
                         command=closeGUI)

    gui.protocol("WM_DELETE_WINDOW", closeGUI)  # close window via "X"

    Button_Start_Gauss['font'] = myFont
    Button_Start_Crescent['font'] = myFont
    Button_Start_Circular['font'] = myFont
    Button_Start_Image['font'] = myFont
    Button_Close['font'] = myFont

    Button_Start_Gauss.pack(fill=tk.X)
    Button_Start_Crescent.pack(fill=tk.X)
    Button_Start_Circular.pack(fill=tk.X)
    Button_Start_Image.pack(fill=tk.X)
    Button_Close.pack(fill=tk.X)

    gui.mainloop()

print("Program terminated successfully.")