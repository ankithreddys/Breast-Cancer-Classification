#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
#%%
dataframe = pd.DataFrame(np.c_[dataset['data'], dataset['target']], columns = np.append(dataset['feature_names'], ['target']))
print(dataframe.shape)
#%%
x = dataframe.drop(["target"],axis=1)
y = dataframe["target"]
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=(5))
#%%
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
#%%
y_pred = model.predict(x_test)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
#%%
sns.heatmap(cm,annot=True)
#%%
print(classification_report(y_test, y_pred))
#%%
min_train = x_train.min()
range_train = (x_train - min_train).max()
x_train_scaled = (x_train - min_train)/range_train
#%%
min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_scaled = (x_test - min_test)/range_test
#%%
from sklearn.svm import SVC
model = SVC()
model.fit(x_train_scaled, y_train)
#%%
y_pred_scaled = model.predict(x_test_scaled)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred_scaled)
print(cm)
print(accuracy_score(y_test, y_pred_scaled))
#%%
sns.heatmap(cm,annot=True)
#%%
print(classification_report(y_test, y_pred_scaled))
#%%







