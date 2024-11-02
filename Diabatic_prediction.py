import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
a=pd.read_csv("/content/drive/MyDrive/diabetes.csv")
a.head()
x=a.drop(columns="Outcome",axis=1)
y=a["Outcome"]

from sklearn.model_selection import train_test_split
xt,xte,yt,yte=train_test_split(x,y,test_size=.1,stratify=y,random_state=3)
# from sklearn.preprocessing import StandardScaler
# z=StandardScaler()
# xt=z.fit_transform(xt)
# xte=z.fit_transform(xte)
##model on which you want to work
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
k=LogisticRegression(max_iter=1000)
#train model
k.fit(xt,yt)

# prediction
n=k.predict(xte)
# import pickle
# pickle.dump(k,open("mod.sav","wb"))
print(accuracy_score(n,yte))
print(k.coef_,k.intercept_)
data=confusion_matrix(yte,n)
plt.figure(figsize=(10,10))
sns.heatmap(data,annot=True,fmt=" ", annot_kws={'size': 124, 'weight': 'bold', 'color': 'red'})
plt.xlabel("orginal")
plt.ylabel('prediction')
plt.show()
 