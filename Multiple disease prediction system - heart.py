

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Processing

# In[2]:


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart.csv')


# In[3]:


# print first 5 rows of the dataset
heart_data.head()


# In[4]:


# print last 5 rows of the dataset
heart_data.tail()


# In[5]:


# number of rows and columns in the dataset
heart_data.shape


# In[6]:


# getting some info about the data
heart_data.info()


# In[7]:


# checking for missing values
heart_data.isnull().sum()


# In[8]:


# statistical measures about the data
heart_data.describe()


# In[9]:


# checking the distribution of Target Variable
heart_data['target'].value_counts()


# 1 --> Defective Heart
# 
# 0 --> Healthy Heart

# Splitting the Features and Target

# In[10]:


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


# In[11]:


print(X)


# In[12]:


print(Y)


# Splitting the Data into Training data & Test Data

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[15]:


model = LogisticRegression()


# In[16]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[17]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[18]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[19]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[20]:


print('Accuracy on Test data : ', test_data_accuracy)


# Building a Predictive System

# In[21]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# Saving the trained model

# In[22]:


import pickle


# In[23]:


filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[24]:


# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))


# In[25]:


for column in X.columns:
  print(column)


# In[ ]:




