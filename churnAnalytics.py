#!/usr/bin/env python
# coding: utf-8

# # Churn Analytics

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# ### Import Dependencies

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# #### Importing Algorithms

# In[4]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ### Import Dataset

# In[5]:


df = pd.read_csv(r"C:\Users\sgaks\Downloads\SNU_AI_DS\4th_sem\ML_LAB\Ex4\Telco-Customer-Churn.csv")
df1 = df.copy()


# In[6]:


df1


# ### Pre-processing data:

# In[7]:


df1 = df1.drop('customerID',axis=1).copy()


# In[8]:


df1['TotalCharges']=pd.to_numeric(df1['TotalCharges'], errors='coerce')


# ### Data Analysis

# In[9]:


print(f"Total number of rows in dataset : {df1.shape[0]}\nTotal number of columns in dataset : {df1.shape[1]}")


# In[10]:


df1.info()


# ### Label Encoding

# In[11]:


categorical_features = df1.select_dtypes(include='object')
categorical_features_col = categorical_features.columns


# In[12]:


Label_Encoder = LabelEncoder()
for i in df1.select_dtypes(include='object').columns:
    df1[i] = Label_Encoder.fit_transform(df1[i].astype(str))


# In[13]:


df1.isna().sum() 


# - Total Charges column have 11 null values which needs to be filled

# In[14]:


df1['TotalCharges'].value_counts()


# In[15]:


df1['TotalCharges']=df1['TotalCharges'].fillna(df1['TotalCharges'].mean())


# ### Data Visualisation

# In[16]:


df1.describe()


# In[17]:


plt.figure(figsize=(25,25))
sns.heatmap(df1.corr(),annot=True,cmap='cool')


# ### Splitting Data

# In[18]:


x = df1.iloc[:,:-1].values
y = df1.iloc[:,-1].values


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# ## Logistic Regression

# ##### GridSearchCV

# In[20]:


LR = LogisticRegression()
grid = {'C':np.logspace(-3,3,7),'penalty':['l1','l2']}
LR_cv = GridSearchCV(LR,grid,cv=10)
LR_cv.fit(x_train,y_train)##### GridSearchCV


# In[21]:


print(f"Best Accuracy Score:{LR_cv.best_score_}")


# In[22]:


print("Best params",LR_cv.best_params_)


# ##### Randomized Search

# In[23]:


from sklearn.model_selection import RandomizedSearchCV
params = {'C':np.logspace(-5,5,7),
         'penalty':['l1','l2']}
rnd_search = RandomizedSearchCV(LR,params,cv=9)
rnd_search.fit(x_train,y_train)


# In[24]:


print(f"Best Accuracy Score:{rnd_search.best_score_}")


# In[25]:


print(f"Best Params:{rnd_search.best_params_}")


# ## K-Nearest Neighbour [KNN]

# In[26]:


param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}
knn = KNeighborsClassifier()


# ##### GridSearch CV

# In[27]:


knn_grid = GridSearchCV(knn, param_grid,scoring='accuracy', cv=10)
knn_grid.fit(x_train, y_train)


# In[28]:


print("Best parameter (k):", knn_grid.best_params_)


# In[29]:


print("Accuracy score of Best parameter (k):", knn_grid.best_score_)


# - From the all the parameter we kind that the knn algorithm works best with 14 nearest neighbour
# - Now, implementing KNN algorithm with [n] as 14 

# In[30]:


knn = KNeighborsClassifier(knn_grid.best_params_['n_neighbors'])
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# ##### Randomized Search

# In[31]:


grid = RandomizedSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(x_train,y_train)


# In[32]:


print(f"Accuracy Score : {grid.best_score_}")


# In[33]:


print(f"Best Parameters : {grid.best_params_['n_neighbors']}")


# ## Naive Bayes

# In[34]:


a = np.random.dirichlet(np.ones(2),size=20)
params_bayes = {'priors': [i for i in a]}


# ##### GridSearch CV

# In[35]:


clf = GaussianNB()
baiyes_grid =  GridSearchCV(clf, param_grid=params_bayes, scoring='accuracy', cv=10)
baiyes_grid.fit(x_train,y_train)


# In[36]:


a = baiyes_grid.best_params_['priors']
print("Accuracy score of Best parameter (k):", baiyes_grid.best_params_)


# In[37]:


print("Best parameter (k):", baiyes_grid.best_score_)


# - From the all the parameter, we find that the Naive Baiyes algorithm works best with array ([0.93593143, 0.06406857]
# - Now, implementing Naive Baiyes algorithm with array ([0.93593143, 0.06406857] 

# In[38]:


clf = GaussianNB(priors=a)


# In[39]:


clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[40]:


accuracy_score(y_test, y_pred)


# ##### Randomized Search CV

# In[41]:


grid = RandomizedSearchCV(estimator=GaussianNB(),param_distributions=params_bayes,cv=10)
grid.fit(x_train,y_train)


# In[42]:


print(f"Best Parameters : {grid.best_params_}")


# In[43]:


print(f"Best Accuracy : {grid.best_score_}")


# In[44]:


accuracy_score(y_test, y_pred)


# ## Decision Tree:

# In[45]:


params_dt = {'max_features': [i for i in range(21)]}


# ##### Grid Search CV

# In[46]:


dt = DecisionTreeClassifier()
dt_grid =  GridSearchCV(dt, param_grid=params_dt, scoring='accuracy', cv=10)
dt_grid.fit(x_train,y_train)


# In[47]:


print("Best parameter (k):", dt_grid.best_params_)


# In[48]:


a = baiyes_grid.best_score_
print("Accuracy score of Best parameter (k):", dt_grid.best_score_)


# In[49]:


accuracy_score(y_test, y_pred)
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)


# In[50]:


accuracy_score(y_test, y_pred)


# In[51]:


r_grid = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=params_dt, cv=10)
r_grid.fit(x_train,y_train)


# In[52]:


print(f"Best Parameters : {grid.best_params_}")


# In[53]:


print(f"Best Accuracy : {grid.best_score_}")


# In[54]:


y_pred = dt.predict(x_test)


# In[55]:


accuracy_score(y_test, y_pred)


# ## SVM

# In[56]:


param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','polynomial','gausian','linear']}


# ##### Grid Search CV

# In[ ]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(x_train, y_train)


# In[ ]:


print(f"Best Parameters : {grid.best_params_}")


# In[ ]:


print(f"Best Accuracy : {grid.best_score_}")


# ##### Randomized Search CV

# In[ ]:


grid = RandomizedSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(x_train, y_train)


# In[ ]:


print(f"Best Parameters : {grid.best_params_}")


# In[ ]:


print(f"Best Accuracy : {grid.best_score_}")

