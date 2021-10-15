### Human Activity Recognation ###
"""
Description:
		Human activity recognition is the problem of classifying sequences of data
		recorded by specialized harnesses or smart phones into known well-defined Human activities,
		The problem will be solved using K-Nearest-Neighbor (KNN) algorithm where we will train the model on dataset of collection of 
		human activities in order to classifying them based on categories.
"""

## Importing modules ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score

## 1 | Data Preprocessing ##
"""Prepare and analyse the dataset before training"""

# 1.1 Reading the data
train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

# 1.2 Combine both data frames in train and test sets
train['Data'] = 'Train'
test['Data']  = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
both['subject'] = '#' + both['subject'].astype(str)
print(both.columns)

# 1.3 Showing combined data
print(f"Dataset head :\n{both.head()}\n")

# 1.4 Data statistics
stats = pd.DataFrame()
stats['Missing value']  = both.isnull().sum()
stats['N unique value'] = both.nunique()
stats['dtype'] = both.dtypes
print(f"Data statistics :\n{stats}\n")

# 1.5 Human activites
activity = both['Activity']
label_counts = activity.value_counts()
plt.figure(figsize= (12, 8))
plt.bar(label_counts.index, label_counts)
plt.title("Activites", fontsize=25)
plt.show()

# 1.6 Prepare train set
Data = both['Data']
Subject = both['subject']
train = both.copy()
train = train.drop(['Data','subject','Activity'], axis =1)

# 1.7 Standardize train set :
# Rescaling the distribution of value so that the mean of observed values is 0 
# and the standard deviation is 1
slc = StandardScaler()
train = slc.fit_transform(train)

# 1.8 Dimensionality reduction :
# Transformation of data from a high-dimensional space into a low-dimensional space 
pca = PCA(n_components=0.9, random_state=0)
train = pca.fit_transform(train)

# 1.9 Splitting the  data into training and testing
X_train, X_test, y_train, y_test = train_test_split(train, activity, test_size = 0.2, random_state = 0)

## 2 | Model Creation ##
"""Create model to fit it to the data"""

# 2.1 Model creation
model = KNeighborsClassifier(algorithm='auto', n_neighbors=8, p=1, weights='distance')

# 2.2 Fit model to the data
model.fit(X_train, y_train) 

# 2.3 Predictions on the test set
y_pred = model.predict(X_test)

## 3 | Model Evaluation ##
"""Evaluate model performance"""

# 3.1 Cross validation score
_ = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
results = (_.mean(), _.std())

# 3.2 Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {round(accuracy*100, 2)} %")

# 3.3 Classification report
print(f"\nClassification report : \n{classification_report(y_test, y_pred)}")

# 3.4 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix : \n")
sns.heatmap(cm, annot=True)
plt.title("Confusion matrix")
plt.show()
