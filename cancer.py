#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
#%%
dataset.keys()
#%%
print(dataset["data"])
#%%
print(dataset["target"])
#%%
print(dataset["frame"])
#%%
print(dataset["target_names"])
#%%
print(dataset["DESCR"])
#%%
print(dataset["feature_names"])
#%%
print(dataset["filename"])
#%%
print(dataset["data_module"])
#%%
print(dataset["data"].shape)
#%%
dataframe = pd.DataFrame(np.c_[dataset['data'], dataset['target']], columns = np.append(dataset['feature_names'], ['target']))
print(dataframe.shape)
#%%
dataframe.head()
#%%
dataframe.tail()
#%%
sns.pairplot(dataframe,hue = "target" ,vars=["mean radius","mean symmetry","concavity error","worst perimeter","worst fractal dimension"])
#%%
sns.countplot(dataframe,x="target")
#%%
plt.figure(figsize=(20,20))
sns.heatmap(dataframe.corr(),annot=True)
plt.show()
#%%
x = dataset["data"]
y = dataset["target"]
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






