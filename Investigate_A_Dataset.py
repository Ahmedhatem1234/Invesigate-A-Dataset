#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate a Dataset - [tmdb-movies]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# > I choose the TMDb movie data set for data analysis. This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
# 
# ### Questions
# > 1) What is the year that has the highest release of movies? /
# > 2) what is the movie with the highest And lowest budget?
#  

# In[2]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > After Observing the dataset and the questions related to this dataset for the analysis we will be keeping only relevent data and deleting the unused data.
# 
# 
# ### General Properties
# 

# In[5]:


# Load your data. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
# Read the CSV File Using Pandas read_csv function
z = pd.read_csv("tmdb-movies.csv")
z.head()


# In[6]:


# print the shape of the dataset 
z.shape


# In[7]:


# print concise summery of the dataset
z.describe()


# In[8]:


# print the concise summery of the dataset
z.info()


# In[9]:


z.head()


# In[10]:


z.hist(figsize = (12,10))


# In[11]:


z.fillna(z.mean(),inplace=True)
z.info()


# In[18]:


z.dropna(inplace=True)
z.info()


# 
# ### Data Cleaning
# > Information That We Need To Delete Or Modify:
# We need to remove duplicate rows from the dataset
#  

# In[20]:


#'duplicated()' function in pandas return the duplicate row as True and othter as False
#for counting the duplicate elements we sum all the rows
sum(z.duplicated())


# In[21]:


#After calculating the duplicate row we can drop these row using 'drop_duplicates()' function
z.drop_duplicates(inplace=True)
#after removing duplicate value from the dataset
sum(z.duplicated())


# In[22]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section
z.drop(['id', 'imdb_id', 'cast', 'director', 'keywords', 'overview', 'production_companies', 'release_date', 'budget_adj', 'revenue_adj' ,'homepage', 'tagline'], axis=1, inplace=True)


# In[23]:


print("Rows With Zero Values In The Budget Column:",z[(z['budget']==0)].shape[0])
print("Rows With Zero Values In The Revenue Column:",z[(z['revenue']==0)].shape[0])

z = z[z.budget != 0]


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# 
# ### Research Question 1 (What is the year that has the highest release of movies?)

# In[24]:


import pandas as pd
z = pd.read_csv("tmdb-movies.csv")
# make group for each year and count the number of movies in each year 
a = z.groupby('release_year').count()['original_title']
print(a.tail())

#make group of the data according to their release year and count the total number of movies in each year and pot.
z.groupby('release_year').count()['original_title'].plot(xticks = np.arange(1960,2016,5))

#set the figure size and labels
sn.set(rc={'figure.figsize':(12,10)})
plt.title("the number of movies over the years",fontsize = 16)
plt.xlabel('release year',fontsize = 12)
plt.ylabel('number Of movies',fontsize = 12)
sn.set_style("whitegrid")


# ### Research Question 2  (what is the movie with the highest And lowest budget?)

# In[25]:


#use the function 'idmin' to find the index of lowest budget movie.
#use the function 'idmax' to find the index of Highest budget movie.
#print the row related to this index.
def find_minmax(x):
    min_index = z[x].idxmin()
    high_index = z[x].idxmax()
    high = pd.DataFrame(z.loc[high_index,:])
    low = pd.DataFrame(z.loc[min_index,:])
    
    
    print("Movie Which Has Highest "+ x + " : ",z['original_title'][high_index])
    print("Movie Which Has Lowest "+ x + "  : ",z['original_title'][min_index])
    return pd.concat([high,low],axis = 1)


find_minmax('budget')

import pandas as pd
z = pd.read_csv("tmdb-movies.csv")
#make a plot which contain top 10 highest budget movies.
#sort the 'budget' column in decending order and store it in the new dataframe.
a = pd.DataFrame(z['budget'].sort_values(ascending = False))
a['original_title'] = z['original_title']
b = list(map(str,(a['original_title'])))

#extract the top 10 budget movies data from the list and dataframe.
x = list(b[:10])
y = list(a['budget'][:10])

#plot the figure and setup the title and labels.
ax = sn.pointplot(x=y,y=x)
sn.set(rc={'figure.figsize':(12,10)})
ax.set_title("movies from the lowest budget to the highest",fontsize = 16)
ax.set_xlabel("Budget",fontsize = 12)
sn.set_style("whitegrid")


# ### Research Question 3  (what is the relation  between Revenue And Budget?)

# In[26]:


#how does revenue change according to their budget.
#make a scatter plot using 'regplot' between ''revenue' and 'budget'.
z = pd.read_csv("tmdb-movies.csv")
a = sn.regplot(x=z['revenue'], y=z['budget'],color='c')

#set the title and labels of the figure
a.set_title("Revenue Vs Budget",fontsize=16)
a.set_xlabel("Revenue",fontsize=10)
a.set_ylabel("Budget",fontsize=10)
#set the figure size
sn.set(rc={'figure.figsize':(6,4)})
sn.set_style("whitegrid")

#find the correlation between them
#change zero into NAN in budget and revenue column for the exact calculation.
z['budget'] = z['budget'].replace(0,np.NAN)
z['revenue'] = z['revenue'].replace(0,np.NAN)


#find the correlation using 'corr()' function.
#it returns a dataframe which contain the correlation between all the numeric columns.
data_corr = z.corr()
print("Correlation Between Revenue And Budget : ",data_corr.loc['revenue','budget'])


# <a id='conclusions'></a>
# ## Conclusions
# > question (1)
# 
# > 2015 has the highest release of movies.
# 
# > question (2)
# 
# > 1) The Warrior's Way is the movie with the highest budget.
# 
# > 2) The Hobbit: An Unexpected Journey is the movie with the lowest budget.
# 
# > question (3)
# 
# > Revenue is directly connected to the budget.
# 
# 
# 
# ### Limitations
# > It's not 100 percent guaranteed solution that this formula is gonna work, But it shows us that we have high probability of making high profits if we had similar characteristics as such. If we release a movie with these characteristics, it gives people high expectations from this movie. This was just one example of an influantial factor that would lead to different results, there are many that have to be taken care of.
# 
# ## Submitting your Project 
# 
# 

# In[5]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




