# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:HARISHA S 
RegisterNumber: 212223040063
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/fe4f32b3-b657-43fc-ab14-7f7ef670bbf4)
```
data.info()
```
![image](https://github.com/user-attachments/assets/deeff6fe-e15d-49c8-a768-ce9982d4cabb)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/536157e6-069a-4418-a922-240e4854edc2)
```
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/888db987-02b3-4b53-8b7f-cfdf468e1d66)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/61b66830-ecd4-4435-9782-15021a49b974)
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
![image](https://github.com/user-attachments/assets/a95c4f79-0d01-4265-a910-dbd08d8d70ec)

```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/62c04fee-356e-4827-aa51-7e50d21c6c8e)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![download](https://github.com/user-attachments/assets/282511a6-31b1-4e80-88c6-2432b1def4b6)

## Output:
![decision tree classifier model](sam.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
