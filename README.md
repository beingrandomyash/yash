#iris
#iris.cvs   for data science project

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from google.colab import files
uploaded = files.upload()
import pandas as pd
dataset = pd.read_csv('Iris.csv')  
print(dataset)
description = dataset.describe()
print(description)
data = dataset.values
x = data[: , 1:4]
y = data[: , 5]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train
from sklearn.svm import SVC
svn = SVC()
svn.fit(x_train, y_train)
predictions = svn.predict(x_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)*100
print('%.2f' %acc)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
