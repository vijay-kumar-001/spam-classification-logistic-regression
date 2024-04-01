#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer #It convert text document in


# In[2]:


df = pd.read_csv(r"C:\Users\vijay\Desktop\spam.csv", encoding='latin1')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


cols = df.columns.difference(['v1','v2'])


# In[7]:


df = df.drop(columns =cols )


# In[8]:


df.head()


# In[9]:


df.rename(columns = {'v1':'category','v2':'message'},inplace = True)


# In[10]:


df.duplicated().sum()


# In[11]:


df =df.drop_duplicates(keep = "first")


# In[12]:


df.duplicated().sum()


# In[13]:


df.shape


# In[14]:


df.groupby('category').describe()


# In[15]:


#data visualisation
plt.pie(df['category'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[16]:


df['category'] =df['category'].apply(lambda x: 0 if x == 'spam' else 1)


# In[17]:


df


# In[18]:


#dependent and independent variable 
x = df['message']
y = df['category']


# In[19]:


x.head()


# In[20]:


y


# In[21]:


#testing the data 
x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state =3)


# In[22]:


x_train.head()


# In[23]:


y_train.head()




# In[24]:


print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[25]:


print(y.shape)
print(y_train.shape)
print(y_test.shape)


# In[26]:


feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_train_features=feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.transform(x_test)


# In[27]:


y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[28]:


y_train


# In[29]:


df.iloc[4443]


# In[30]:


y_test


# In[31]:


x_train


# In[32]:


x_test


# In[33]:


print(x_test_features)


# In[ ]:





# In[34]:


#model  
model = LogisticRegression()
model.fit(x_train_features,y_train)


# In[35]:


prediction_on_train_data =model.predict(x_train_features)
accuracy  =accuracy_score(y_train,prediction_on_train_data)
accuracy


# In[36]:


y_prediction_on_test_data =model.predict(x_test_features)
accuracy  =accuracy_score(y_test,y_prediction_on_test_data)
accuracy


# In[37]:


input_mail = ["XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

print("Our Mail is: ", prediction)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")


# In[38]:


#checking how many false postives


# In[39]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_prediction_on_test_data)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)


# Displaying the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[40]:


precision_score(y_test,y_prediction_on_test_data)


# In[ ]:




