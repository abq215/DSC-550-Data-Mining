
# coding: utf-8

# # Basit_Abdul_Week2_Exercise1_Closed_Itemset_Mining
# 
# File: Basit_Abdul_Week2_Exercise1_Closed_Itemset_Mining.ipynb
# Name: Abdul Basit
# Date: 06/21/2020
# Course: DSC 550 Data Mining
# Instructor: Professor Brant Abeln
# Exercise: 2.1
# Assignment: Closed Itemset Mining
# Reference: https://www.youtube.com/watch?v=2oVMmMdeCOQ
#            https://github.com/weeklyDataScience/apriori/blob/master/apriori.py

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import pandas as pd


""" This program creates the brute force algorithm for itemset mining. This algorithm is detailed on page 223 in Data Mining and Machine Learning
"""

def create_dict_from_file(filename):
    """ Read in a file of itemsets each row is considered the transaction id and each line contains the items associated with it. This function returns a dictionary that has a key set as the tid and has values of the list of items (strings)
    """
    f = open(filename, 'r')
    d = {}
    for tids, line_items in enumerate(f):
           d[tids] = [j for j in line_items.split(' ')
                           if j != '\n']
    return d

def create_database(itemset):
    "Uses dummy indexing to create the binary database"
    return pd.Series(itemset).str.join('|').str.get_dummies()

def compute_support(df, column):
    "Exploits the binary nature of the database"
    return df[column].sum()


# ## Run code with minsup = 3000

# In[2]:


if __name__ == '__main__':
    # Check if the command line arguments are given
    minsup=3000
    filename = 'mushroom.txt'
    dict_itemset = create_dict_from_file(filename)
    database = create_database(dict_itemset)
    
    # Executes the brute force algorithm
    # NOTE: a list comprehension is faster
    freq_items = []
    for col in database.columns:
        sup = compute_support(database, col)
        if sup >= minsup:
            freq_items.append(int(col))
        else:
            pass

    print('There are %d items with frequency'          ' greater than or equal to minsup 3000' % len(freq_items))
    print(sorted(freq_items))


# ## Run code with minsup = 5000

# In[3]:


if __name__ == '__main__':
    # Check if the command line arguments are given
    minsup=5000
    filename = 'mushroom.txt'
    dict_itemset = create_dict_from_file(filename)
    database = create_database(dict_itemset)
    
    # Executes the brute force algorithm
    # NOTE: a list comprehension is faster
    freq_items = []
    for col in database.columns:
        sup = compute_support(database, col)
        if sup >= minsup:
            freq_items.append(int(col))
        else:
            pass

    print('There are %d items with frequency'          ' greater than or equal to minsup 5000' % len(freq_items))
    print(sorted(freq_items))

