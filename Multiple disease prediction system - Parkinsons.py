
# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection & Analysis

# In[2]:


# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('parkinsons.csv')


# In[3]:


# printing the first 5 rows of the dataframe
parkinsons_data.head()


# In[4]:


# number of rows and columns in the dataframe
parkinsons_data.shape


# In[5]:


# getting more information about the dataset
parkinsons_data.info()


# In[6]:


# checking for missing values in each column
parkinsons_data.isnull().sum()


# In[7]:


# getting some statistical measures about the data
parkinsons_data.describe()


# In[8]:


# distribution of target Variable
parkinsons_data['status'].value_counts()


# 1  --> Parkinson's Positive
# 
# 0 --> Healthy
# 

# In[9]:


# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()


# Data Pre-Processing

# Separating the features & Target

# In[10]:


X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']


# In[11]:


print(X)


# In[12]:


print(Y)


# Splitting the data to training data & Test data

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Support Vector Machine Model

# In[15]:


model = svm.SVC(kernel='linear')


# In[16]:


# training the SVM model with training data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[17]:


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[18]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[19]:


# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[20]:


print('Accuracy score of test data : ', test_data_accuracy)


# Building a Predictive System

# In[21]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")


# Saving the trained model

# In[22]:


import pickle


# In[23]:


filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[24]:


# loading the saved model
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# In[25]:


for column in X.columns:
  print(column)


# In[ ]:




