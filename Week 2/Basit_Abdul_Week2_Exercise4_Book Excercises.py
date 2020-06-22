
# coding: utf-8

# # Basit_Abdul_Week2_Exercise4_Book Excercises
# 
# File:        Basit_Abdul_Week2_Exercise4_Book Excercises.ipynb
# Name:        Abdul Basit
# Date:        06/21/2020
# Course:      DSC 550 Data Mining_Summer Term T302
# Instructor:  Professor Brant Abeln
# Exercise:    2.1
# Assignment:  Book Excercises 3.1.1, 3.2.1, 3.3.3 & 3.4.1
# Book:        Mining of Massive Datasets by Anand Rajaraman and Jeffrey D. Ullman

# ## Exercise 3.1.1

# Compute the Jaccard similarities of each pair of the following three sets: {1, 2, 3, 4}, {2, 3, 5, 7}, and {2, 4, 6}.

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.special import gamma
from itertools import combinations
from pandas import DataFrame


# In[2]:


# Three sets provided in the question
lst1 = {1, 2, 3, 4}
lst2 = {2, 3, 5, 7}
lst3 = {2, 4, 6}


# In[3]:


# Define function to compute Jaccard Similarity
def jacc_similarity(s1, s2):
    return len(s1.intersection(s2))/len(s1.union(s2))


# In[4]:


if __name__ == '__main__':
    print('Jaccard Similarity between set1 & set2 - ', round(jacc_similarity(lst1, lst2), 3))
    print('Jaccard Similarity between set1 & set3 - ', round(jacc_similarity(lst1, lst3), 3))
    print('Jaccard Similarity between set2 & set3 - ', round(jacc_similarity(lst2, lst3), 3))


# ## Exercise 3.2.1

# What are the first ten 3-shingles in the first sentence of Section 3.2?

# In[5]:


# Book text in question
book_text = 'The most effective way to represent documents as sets, for the purpose of identifying lexically similar '        'documents is to construct from the document the set of short strings that appear within it'

# Create 3 constant shingles
k = 3

if __name__ == '__main__':
    # Print given text
    print('Book Text in question ->>', book_text, '\n')              
    # Remove commas from text to make it plain, split each word
    tokens = book_text.replace(',', '').split()
    
    print("First ten 3-shingles in the text are ->> " + str([tokens[x:x + k] for x in range(0, 10)]))


# ## Exercise 3.3.3

# In Fig. 3.5 is a matrix with six rows.
# 
# Element S1 S2 S3 S4
#    0     0  1  0  1
#    1     0  1  0  0
#    2     1  0  0  1
#    3     0  0  1  0
#    4     0  0  1  1
#    5     1  0  0  0
# 
# (a) Compute the minhash signature for each column if we use the following three hash functions: h1(x) = 2x + 1 mod 6; h2(x) = 3x + 2 mod 6; h3(x) = 5x + 2 mod 6.
# (b) Which of these hash functions are true permutations?
# (c) How close are the estimated Jaccard similarities for the six pairs of columns to the true Jaccard similarities?

# In[6]:


import sys


def initialstep():
    # Given hash matrix
    h_dict = {'Element': [0, 1, 2, 3, 4, 5], 'S1': [0, 0, 1, 0, 0, 1], 'S2': [1, 1, 0, 0, 0, 0], 'S3': [0, 0, 0, 1, 1, 0],
          'S4': [1, 0, 1, 0, 1, 0]}
    # Transform into a Dataframe
    matrix = pd.DataFrame(data=h_dict)     
    return matrix


# In[7]:


# Define a function to calculate 3 hash functions for each of the columns
def evaluate_hash(h_matrix):
    
    # Convert values of Element column to a list
    rows = list(h_matrix['Element']) 
    # Copy initial matrix
    h_matrix1 = h_matrix             

    # Create iterations for all elements in the matrix
    for r in h_matrix.itertuples():   
        i = r.Element
        # First hash function
        h_matrix1.loc[r.Index, 'h1'] = ((2 * i) + 1) % 6
        # Second hash function
        h_matrix1.loc[r.Index, 'h2'] = ((3 * i) + 2) % 6
        # 3rd hash function
        h_matrix1.loc[r.Index, 'h3'] = ((5 * i) + 2) % 6

    # Convert the result into integers
    h_matrix1["h1"] = h_matrix1["h1"].astype(int)
    h_matrix1["h2"] = h_matrix1["h2"].astype(int)
    h_matrix1["h3"] = h_matrix1["h3"].astype(int)

    return h_matrix1


# In[8]:


# Define a function to calculate Hash Signature
def evaluate_hashsign(df):
    
    # Define a list for signature matrix
    signmatrix = []
    # 3x4 List of lists for storing Hash Signatures
    for i in range(3):                               
        # Set each elements of signature matrix with high value
        signmatrix.append([sys.maxsize] * 4)

    h_list = []
    # 6x3 List of lists for Hash Functions
    h_list = df[['h1', 'h2', 'h3']].values.tolist()
    
    # Logic to calculate hash Signature
    # Iterate over rows of hash Matrix
    for i, row in df.iterrows():                     
        col_count = 0
        # Iterate over columns S1, S2, S3, and S4
        for col in df.columns[1:5]:                  
            col_count += 1
            # If row-col has value '0' do nothing
            if row[col] == 0:                        
                continue
            # Iterate 3 times for 3 hash functions    
            for s in range(3):                       
                # Check if sign matrix value is greater than hash value
                if signmatrix[s][col_count-1] > h_list[i][s]:   
                    # Replace signature value with hash value
                    signmatrix[s][col_count-1] = h_list[i][s]   

    # Convert Hash signature matrix into DataFrame
    signmatrix_df = pd.DataFrame(signmatrix, columns=['S1', 'S2', 'S3', 'S4'], index=['h1', 'h2', 'h3'])

    return signmatrix_df


# In[9]:


def permut_hash(h_matrix):
    """
    Function to check the true permutations
    """
    # Generate lists
    Element_l = sorted(h_matrix['Element'].tolist())
    h1_l = sorted(h_matrix['h1'].tolist())
    h2_l = sorted(h_matrix['h2'].tolist())
    h3_l = sorted(h_matrix['h3'].tolist())

    # Compare each hash function list with Element list to check for true permutations.
    if Element_l == h1_l:
        print('\n->> First Hash function is true permutations of Element list.', '\n')
    elif Element_l == h2_l:
        print('\n->> Second Hash function is true permutations of Element list.', '\n')
    elif Element_l == h3_l:
        print('\n->> Third Hash function is true permutations of Element list.', '\n')
    return


# In[10]:


# Create a function to compute Jaccard similarity
def jacc_similarity(a, b):
    intr = len(list(set(a).intersection(b)))            
    uni = (len(a) + len(b)) - intr                       
    return intr/uni


# In[11]:


# Define a function to compare Jaccard similarities between 6 pairs of columns
def compre_jacc(h_matrix, h_sign):
    # Initialize list for True Jaccard similarities
    TruJ = []  
    # Initialize list for Estimate Jaccard similarities
    EstJ = []        

    # Compute True Jaccard similarities between column pairs
    TruJ.append(jacc_similarity(h_matrix['S1'], h_matrix['S2']))
    TruJ.append(jacc_similarity(h_matrix['S1'], h_matrix['S3']))
    TruJ.append(jacc_similarity(h_matrix['S1'], h_matrix['S4']))
    TruJ.append(jacc_similarity(h_matrix['S2'], h_matrix['S3']))
    TruJ.append(jacc_similarity(h_matrix['S2'], h_matrix['S4']))
    TruJ.append(jacc_similarity(h_matrix['S3'], h_matrix['S4']))

    # Compute Estimate Jaccard similarities between column pairs
    EstJ.append(jacc_similarity(h_sign['S1'], h_sign['S2']))
    EstJ.append(jacc_similarity(h_sign['S1'], h_sign['S3']))
    EstJ.append(jacc_similarity(h_sign['S1'], h_sign['S4']))
    EstJ.append(jacc_similarity(h_sign['S2'], h_sign['S3']))
    EstJ.append(jacc_similarity(h_sign['S2'], h_sign['S4']))
    EstJ.append(jacc_similarity(h_sign['S3'], h_sign['S4']))

    # Add lists to dictionary
    J_dict = {'TruJ': TruJ, 'EstJ': EstJ}     
    J_matrix = pd.DataFrame(J_dict, index=['S1-S2', 'S1-S3', 'S1-S4', 'S2-S3', 'S2-S4', 'S3-S4'])

    print('Comparision of Jaccard Similarities:\nTruJ- True Jaccard Similarity | EstJ- Estimate Jaccard Similarity\n',
          J_matrix)
    if TruJ == EstJ:
        print('\n->> The Estimated Jaccard similarity is close to True Jaccard similarity')
    else:
        print('\n->> The Estimated Jaccard similarity is NOT close to True Jaccard similarity')

    return


# In[12]:


if __name__ == '__main__':

    # Calculate the minhash signature for each column if we use 3 hash functions
    matrix = initialstep()
    print('Given Matrix - \n', matrix, '\n')

    h_matrix = evaluate_hash(matrix)
    print('Matrix with hash function values -\n', h_matrix, '\n')
    h_sign = evaluate_hashsign(h_matrix)
    print('==> Minhash Signature Matrix -\n', h_sign)

    # Determine which of the hash functions are actual permitations
    permut_hash(h_matrix)

    # Determine how similar are the estimated Jaccard similarities with the actual true Jaccard similarities
    compre_jacc(h_matrix, h_sign)


# ## Exercise 3.4.1

# Evaluate the S-curve 1−(1−sr)b for s = 0.1, 0.2, . . . , 0.9, for the following values of r and b:
# • r = 3 and b = 10.
# • r = 6 and b = 20.
# • r = 5 and b = 50.

# In[13]:


# Values of s provided in the question
s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# In[14]:


# Define a function to evaluate S-Curve
def scurve_eval(r, b):
    s_curve_val = []
    for i in s:
        s_curve_val.append(1-(1-i**r)**b)
    return s_curve_val


# In[15]:


# # Define a function to plot graph of S-Curve
def scurve_graph(r, b):
    # Call the function scurve_eval to determine S-Curve values
    s_curve_val = scurve_eval(r, b) 
    
    # Plot a graph between s-values and S-Curve values
    plt.plot(s, s_curve_val)          
    plt.xlabel('S Values')
    plt.ylabel('S-Curve')
    plt.title('S-Value vs S-Curve : r={} b={}'.format(r, b))
    plt.show()


# In[16]:


if __name__ == '__main__':

    # Plot for r=3 and b=10
    scurve_graph(r=3, b=10) 
    # Plot for r=6 and b=20
    scurve_graph(r=6, b=20)  
    # Plot for r=5 and b=50
    scurve_graph(r=5, b=50)  

