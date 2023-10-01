#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyforest
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE


# In[2]:


# Creating SQLAlchemy engine using the psycopg2 connection
engine = create_engine('postgresql+psycopg2://postgres:Dishit@127.0.0.1:5432/Project_1')

# Loading sample of the data
query = "SELECT * FROM your_table;"
df = pd.read_sql_query(query, engine)

# Displaying basic statistics
print(df.describe())

# Close the database connection
engine.dispose()


# In[3]:


# Distribution of Amount
plt.figure(figsize=(10, 6))
sns.histplot(df['amount'], bins=50, kde=True)
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Amount')
plt.show()

# Distribution of Time
plt.figure(figsize=(10, 6))
sns.histplot(df['time'], bins=50, kde=True)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Time')
plt.show()


# In[4]:


# Class distribution
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# In[6]:


# Separating features (X) and target (y)
X = df.drop(columns=['class'])
y = df['class']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Applying SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Checking the class distribution after resampling
print("Class distribution after SMOTE:")
print(y_train_resampled.value_counts())


# In[8]:


# Creating a Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Training the model on the resampled training data
clf.fit(X_train_resampled, y_train_resampled)


# In[9]:


from sklearn.metrics import classification_report, precision_recall_curve, auc

# Predicting on the test data
y_pred = clf.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred))

# Calculating AUPRC
precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
auprc = auc(recall, precision)
print(f'AUPRC: {auprc:.2f}')

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

