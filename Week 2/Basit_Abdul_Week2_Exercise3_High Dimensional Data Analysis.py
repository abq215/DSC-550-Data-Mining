
# coding: utf-8

# # Basit_Abdul_Week2_Exercise3_High Dimensional Data Analysis
# 
# File: Basit_Abdul_Week2_Exercise3_High Dimensional Data Analysis.ipynb
# Name: Abdul Basit
# Date: 06/21/2020
# Course: DSC 550 Data Mining
# Instructor: Professor Brant Abeln
# Exercise: 2.1
# Assignment: High Dimensional Data Analysis

# Write a script to do the following:
# 
# a) Hypersphere Volume: Plot the volume of a unit hypersphere as a function of dimension. Plot for d=1,⋯,50.
# 
# b) Hypersphere Radius: What value of radius would one need to maintain a hypersphere volume of 1 with increasing d. Plot this value for d=1,⋯,100.
# 
# c) Nearest Neighbors: Assume we have a unit hypercube centered at (0.5,⋯,0.5). Generate n=10000 uniformly random points in d dimensions, in the range (0,1) in each dimension. Find the ratio of the nearest and farthest point from the center of the space. Also store the actual distance of the nearest dn and farthest df points from the center. Plot these value for d=1,⋯,100.
# 
# d) Fraction of Volume: Assume we have a hypercube of edge length l=2 centered at the origin (0,0,⋯,0). Generate n=10,000 points uniformly at random for increasing dimensionality d=1,⋯,100. Now answer the following questions:
# Plot the fraction of points that lie inside the largest hypersphere that can be inscribed inside the hypercube with increasing d. After how many dimensions does the fraction go to essentially zero?
# Plot the fraction of points in the thin shell of width ϵ=0.01 inside the hypercube (i.e., the difference between the outer hypercube and inner hypercube, or the thin shell along the boundaries). What is the trend that you see? After how many dimensions does the fraction of volume in the thin shell go to 100% (use binary search or increase the dimensionality in steps of 10 to answer this. You may use maximum dimensions of up to 2000, and you may use a threshold of 0.0001 to count the volume as essentially being 1 in the shell, i.e., a volume of 0.9999 can be taken to be equal to 1 for finding the smallest dimension at which this happens).
# 
# e) Diagonals in High Dimensions
# 
# Your goal is the compute the empirical probability mass function (EPMF) for the random variable X that represents the angle (in degrees) between any two diagonals in high dimensions.
# 
# Assume that there are d primary dimensions (the standard axes in cartesian coordinates), with each of them ranging from -1 to 1. There are 2d additional half-diagonals in this space, one for each corner of the d-dimensional hypercube.
# 
# Write a script that randomly generates n=100000 pairs of half-diagonals in the d-dimensional hypercube, and computes the angle between them (in degrees).
# 
# Plot the EPMF for three different values of d, as follows d=10,100,1000. What is the min, max, value range, mean and variance of X for each value of d?
# 
# What would you have expected to have happened analytically? In other words, derive formulas for what should happen to angle between half-diagonals as d→∞. Does the EPMF conform to this trend? Explain why? or why not?
# 
# What is the expected number of occurrences of a given angle θ between two half-diagonals, as a function of d (the dimensionality) and n (the sample size)?

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.special import gamma


# ## Hypersphere Volume

# In[12]:


def Vol1_s(d):
    return np.pi ** (d/ 2.0)/ gamma(d/ 2.0 + 1)

d = np.linspace(1,50+1,25)
plt.figure()
plt.plot(d,Vol1_s(d))
plt.yscale("log")
plt.xlabel("Dimension d")
plt.title("Volume of the d-Dimensional Unit Hypersphere") 
plt.ylabel("Volume")
plt.grid(True)
plt.show()


# ## Hypersphere Radius

# In[13]:


import math
from math import pi

xdata = []
ydata = []
radius = pd.DataFrame(columns=['Dimension', 'Radius'])
for dimension in range(1,101):
    n = float(dimension)
    volume = 1
    radius = math.sqrt(1 / (pi * n))
    xdata.append(n)
    ydata.append(radius)
for i in range(1,len(x)):
    radius_df = pd.DataFrame({'Hypersphere Dimension': xdata, 'Hypersphere Radius': ydata})
print(radius_df)
plt.xlabel('Dimension')
plt.plot(xdata, ydata, label='Radius')
plt.title('Hypersphere Radius')
plt.legend()
plt.show()


# ## Nearest Neighbors

# In[2]:


import random


# In[9]:


# Define a function to produce 100 x 10000 arbitrary patterns

def search_neighbors(d):
    
    cubehype= []
    for x in range(len(d)):
        uniform = []
        for n in range(10000):
            uniform.append(random.uniform(0, 1))
        cubehype.append(uniform)

    return cubehype


# In[10]:


# Define a Function to get number close enough to 0.5
def closeenough(list, k=0.5):
    
    return list[min(range(len(list)), key=lambda i: abs(list[i]-k))]


# In[11]:


# Similarly define a Function to get number far enough from 0.5
def farenough(list, k=0.5):
    
    return list[max(range(len(list)), key=lambda i: abs(list[i]-k))]


# In[12]:


# Create a function to produce a list of distance between close and far point for each value of Dimension d
# In order to calculate Nearest Neighbor k-nn
def obtain_knn(cubehype):
    closer = []
    farther = []
    for i in range(len(cubehype)):
        
        # Increment after creating absolute value difference from the close enough point
         closer.append(abs(0.5-closeenough(cubehype[i]))) 
            
        # Increment after creating absolute value difference from the far enough point
         farther.append(abs(0.5-farenough(cubehype[i])))   
    return closer, farther


# In[13]:


# creating function for plot of nearest and farthest 
def create_plot(d, closer, farther):
    # Plot close enough distances for each d
    plt.plot(d, closer)
    plt.xlabel('Dimension - $d$')
    plt.ylabel('Distance of close enough dnear')
    plt.title('Distance from closer point vs d')
    plt.show()

    # Plot fart enough distances for each d
    plt.plot(d, farther)
    plt.xlabel('Dimension - $d$')
    plt.ylabel('Distance of far enough dfar')
    plt.title('Distance from Farther point vs d')
    plt.show()

    return


# In[14]:


if __name__ == '__main__':

    # Create a list of dimensions (d) within range 1 - 100
    d = [x for x in range(1, 100)]
    # Produce 100 x 10000 arbitrary patterns
    cubehype = search_neighbors(d)
    # Obtain both close enough and far enough distances for each d
    closer, farther = obtain_knn(cubehype)        
    # Create plot
    create_plot(d, closer, farther)           


# ## Fraction of Volume

# In[2]:


import numpy as np 
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Define a function to simulate fraction of points that lie inside the hypersphere
def fraction_of_points(N=int(10e4), l=2, d=1):
    # Generate n=10000 uniformly random points in d dimensions
    unif = np.random.uniform(0, 1, size=(N, d))

    count = 0
    for i in range(N):
        if euclidean(np.zeros((1,d)), unif[i]) < l:
            count += 1

    # Volume of hypercube of edge length l=2
    hypercube_volume = (2*l)**d 

    return count / float(N)


# D for range between 1 and 100
D = range(1, 100)

lst = []
for d in D:
    frac = fraction_of_points(d=d)
    lst.append(frac)

# Plot the fraction of points that lie inside the hypersphere
plt.plot(lst)
plt.xlabel("Dimension $d$")
plt.ylabel("Fraction of points in hypersphere")
plt.show()


# Question: After how many dimensions does the fraction go to essentially zero?
#     
# Answer: Fraction goes to zero after approximately 17 dimensions.    

# In[3]:


fraction_of_shellpoints=[]
d=[x for x in range(1,400)] 
def calculate_volume_of_thin_shell(d): 
    for i in d:   
        l=2
        e=0.01
        r=l/2
        # with the higher dimention it goes to 1
        volume=1-(1-e/r)**i
        fraction_of_shellpoints.append(volume)  
        
        
 # creating function to plot       
def plot(d,fraction_of_shellpoints):
    plt.plot(d, fraction_of_shellpoints)
    plt.xlabel("d")
    plt.ylabel("Fraction Of point of thin shell")
    plt.show()
print(fraction_of_shellpoints)


# In[4]:


if __name__=='__main__':
    # calling function inside try blck to catch the eror
    try:
        calculate_volume_of_thin_shell(d)
        plot(d,fraction_of_shellpoints)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args));
    finally:
        print("Code is executed irrespective of exceptions")


# Question: What is the trend that you see? After how many dimensions does the fraction of volume in the thin shell go to 100% (use binary search or increase the dimensionality in steps of 10?                                                                                                                  
# Answer: Fraction of points of thin shell increases with increase in dimensions. Considering the dimensionality with steps of 10, fraction goes to 100% after 40 dimensions.                                         

# ## Diagonals in High Dimensions

# In[3]:


import itertools as it
from collections import Counter


# In[4]:


def compute_angle(point1,point2):
    #since the unit length must be sqrt(D)
    return np.dot(point1,point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))


# ### D = 10

# In[5]:


N = 100000 # randomly select 10000 pairs
D = 10
# generate the pairs and calcuate the angle btw each pair
results = np.zeros(N)
i=0
while(i < N):
    points_pre = np.random.rand(2,int(D)) # generate 2*D array
    points_pre[points_pre<=0.5] = -1
    points_pre[points_pre>0.5] = 1
    results[i] = compute_angle(points_pre[0],points_pre[1])
    i = i+1


# In[6]:


# Compute min, max, value range, mean and variance of X for each value of d
print('min is ')
print(min(results))
print('max is ')
print(max(results))
print('range is ')
print(max(results)-min(results))
print('mean is ')
print(np.mean(results))
print('variance is ')
print(np.var(results))


# In[7]:


plt.hist(results, bins=50, normed=True)
plt.xlabel("Primary Dimensions")
plt.ylabel("PMF at $d$=10")
plt.show()


# ### D = 100

# In[8]:


N = 100000 # randomly select 10000 pairs
D = 100
# generate the pairs and calcuate the angle btw each pair
results = np.zeros(N)
i=0
while(i < N):
    points_pre = np.random.rand(2,int(D)) # generate 2*D array
    points_pre[points_pre<=0.5] = -1
    points_pre[points_pre>0.5] = 1
    results[i] = compute_angle(points_pre[0],points_pre[1])
    i = i+1


# In[9]:


# Compute min, max, value range, mean and variance of X for each value of d
print('min is ')
print(min(results))
print('max is ')
print(max(results))
print('range is ')
print(max(results)-min(results))
print('mean is ')
print(np.mean(results))
print('variance is ')
print(np.var(results))


# In[10]:


plt.hist(results, bins=50, normed=True)
plt.xlabel("Primary Dimensions")
plt.ylabel("PMF at $d$=100")
plt.show()


# ### D = 1000

# In[11]:


N = 100000 # randomly select 10000 pairs
D = 1000
# generate the pairs and calcuate the angle btw each pair
results = np.zeros(N)
i=0
while(i < N):
    points_pre = np.random.rand(2,int(D)) # generate 2*D array
    points_pre[points_pre<=0.5] = -1
    points_pre[points_pre>0.5] = 1
    results[i] = compute_angle(points_pre[0],points_pre[1])
    i = i+1


# In[12]:


# Compute min, max, value range, mean and variance of X for each value of d
print('min is ')
print(min(results))
print('max is ')
print(max(results))
print('range is ')
print(max(results)-min(results))
print('mean is ')
print(np.mean(results))
print('variance is ')
print(np.var(results))


# In[13]:


plt.hist(results, bins=50, normed=True)
plt.xlabel("Primary Dimensions")
plt.ylabel("PMF at $d$=1000")
plt.show()

