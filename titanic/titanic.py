#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[101]:


cwd = os.getcwd()
dirname = os.path.dirname(cwd)
titanic_train_path = os.path.join(dirname, r'titanic\titanic-dataset\train.csv')
titanic_test_path = os.path.join(dirname, r'titanic\titanic-dataset\test.csv')

training = pd.read_csv(titanic_train_path)
test = pd.read_csv(titanic_test_path)

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training, test])

get_ipython().run_line_magic('matplotlib', 'inline')
all_data.columns


# In[17]:


training.info()


# In[18]:


training.describe()


# In[19]:


#columns
training.describe().columns


# In[20]:


#look at numeric and categorical values
df_num = training[['Age', 'SibSp','Parch', 'Fare']]
df_cat = training[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]


# In[21]:


#distributions for all numeric variables
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[22]:


print(df_num.corr())
sns.heatmap(df_num.corr())


# In[23]:


# compare survival rate across Age, SibSp, Parch, and Fare
pd.pivot_table(training, index = 'Survived', values=['Age', 'SibSp', 'Parch', 'Fare'])


# In[25]:


for i in df_cat.columns:
    sns.barplot(x = df_cat[i].value_counts().index, y = df_cat[i].value_counts()).set_title(i)
    plt.show()


# In[28]:


#Comparing survival and each of these categorical variables
print(pd.pivot_table(training, index = 'Survived', columns ='Pclass', values='Ticket',
                    aggfunc = 'count'))
print()
print(pd.pivot_table(training, index = 'Survived', columns ='Sex', values='Ticket',
                    aggfunc = 'count'))
print()
print(pd.pivot_table(training, index = 'Survived', columns ='Embarked', values='Ticket',
                    aggfunc = 'count'))


# In[33]:


df_cat.Cabin
training['cabin_multiple'] = training.Cabin.apply(lambda x:0 if pd.isna(x) else len(x.split(' ')))
training['cabin_multiple'].value_counts()


# In[34]:


pd.pivot_table(training, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc = 'count')


# In[37]:


#creates categories based on the cabin letter (n stands for null)
#in this case we will treat null values like it's own category

training['cabin_adv'] = training.Cabin.apply(lambda x : str(x)[0])


# In[40]:


#comparing survival rate by cabin
print(training.cabin_adv.value_counts())
pd.pivot_table(training, index='Survived', columns='cabin_adv', values='Name', aggfunc = 'count')


# In[43]:


#understand ticket values better
#numeric vs non numeric
training['numeric_ticket'] = training.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
training['ticket_letters'] = training.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').
                                                   lower() if len(x.split(' ')[:-1]) >0 else 0)


# In[45]:


training['numeric_ticket'].value_counts()


# In[47]:


#let us view all rows in dataframe through scrolling. This is for convenience
pd.set_option('max_rows', None)
training['ticket_letters'].value_counts()


# In[52]:


#difference in numeric vs non-numberic tickets in survival rate
pd.pivot_table(training, index='Survived', columns='numeric_ticket', values='Ticket', aggfunc = 'count')


# In[53]:


#survival rate across different ticket types
pd.pivot_table(training, index='Survived', columns='ticket_letters', values='Ticket', aggfunc = 'count')


# In[61]:


#feature engineering on person's title
training.Name.head(50)
training['name_title'] = training.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[62]:


training['name_title'].value_counts()


# In[79]:


#create alll categorical variables that we did above for both training and test sets
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x:0 if pd.isna(x) else len(
x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x:str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x:1 if x.isnumeric() 
                                                   else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(
    x.split(' ')[:-1]).replace(',','.').replace('/','.').lower() 
                                                   if len(x.split(' ')[:-1]) >0 
                                                   else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data
all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Fare = all_data.Fare.fillna(training.Fare.mean())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test
all_data.dropna(subset=['Embarked'], inplace=True)

#tried log norm of sibsp (not used)
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()

#log norm of far (used)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

#converted fare to category for pd.get_dummies()
all_data.Pclass = all_data.Pclass.astype(str)

#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','norm_fare',
                                      'Embarked', 'cabin_adv', 'cabin_multiple','numeric_ticket',
                                      'name_title','train_test']])

#Split to train test again
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis = 1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis = 1)

y_train = all_data[all_data.train_test == 1].Survived
y_train.shape
all_dummies


# In[76]:


#Scale data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']] = scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis = 1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis = 1)

y_train = all_data[all_data.train_test==1].Survived


# In[86]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[82]:


gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[83]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[84]:


#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: '+ str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))


# In[85]:


lr = LogisticRegression()
param_grid = {'max_iter'  : [2000],
             'penalty'    : ['l1', 'l2'],
             'C'          : np.logspace(-4,4,20),
             'solver'     : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train_scaled,y_train)
clf_performance(best_clf_lr,'Logistic Regression')


# In[94]:


knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3,5,7,9],
             'weights'     : ['uniform', 'distance'],
             'algorithm'   : ['auto', 'ball_tree','kd_tree'],
             'p'           : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train_scaled, y_train)
clf_performance(best_clf_knn, 'KNN')


# In[ ]:




