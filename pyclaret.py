#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pycaret')


# In[2]:


# check pycaret version
import pycaret
print('PyCaret: %s' % pycaret.__version__)


# In[3]:


# load the sonar dataset
from pandas import read_csv
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# summarize the shape of the dataset
print(df.shape)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# summarize the first few rows of data
print(df.head())


# In[4]:


# compare machine learning algorithms on the sonar classification dataset
from pandas import read_csv
from pycaret.classification import setup
from pycaret.classification import compare_models
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# evaluate models and compare models
best = compare_models()
# report the best model
print(best)


# In[7]:


# tune model hyperparameters on the sonar classification dataset
import sklearn
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from pycaret.classification import setup
from pycaret.classification import tune_model
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# tune model hyperparameters
best = tune_model(ExtraTreesClassifier(), n_iter=200, choose_better=True)
# report the best model
print(best)


# In[8]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier 

X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


# In[9]:


clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() > 0.999


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




