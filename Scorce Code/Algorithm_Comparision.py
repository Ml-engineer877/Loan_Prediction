import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#LOADING DATASET

dataset=pd.read_csv(r"C:\Users\TBM\Desktop\Git Project\Loan_Prediction\Data\Loan_Dataset.csv")
print(dataset.head(5))

#CHECKING FOR MISSING VALUES

print(dataset.isnull().sum())

#FILLING THE MISSING VALUES

dataset["Gender"]=dataset["Gender"].fillna("1")
dataset["Married"]=dataset["Married"].fillna("1")
dataset["Dependents"]=dataset["Dependents"].fillna("1")
dataset["Self_Employed"]=dataset["Self_Employed"].fillna("0")
dataset["LoanAmount"]=dataset["LoanAmount"].fillna(dataset["LoanAmount"].mean())
dataset["Loan_Amount_Term"]=dataset["Loan_Amount_Term"].fillna(dataset["Loan_Amount_Term"].mean())
dataset["Credit_History"]=dataset["Credit_History"].fillna("1")
print(dataset.isnull().sum())

#SPLITTING THE DATASET FOR INPUT(X) AND OUTPUT(Y)

x=dataset.iloc[:,:-1].values
y=dataset["Loan_Status"]
##print(x)
##print(y)

#SEGGREGATE THE DATASET FOR TRAINING AND TESTING

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
##print(x_train.shape)
##print(x_test.shape)


#ALGORITHM COMPARISION

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
models=[]
models.append(("RFC",RandomForestClassifier()))
models.append(("GNB",GaussianNB()))
models.append(("LR",LogisticRegression()))
models.append(("SVC",SVC()))
models.append(("DTC",DecisionTreeClassifier()))
models.append(("KNC",KNeighborsClassifier()))

names=[]
result=[]
res=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10)
    score=cross_val_score(model,x_train,y_train,cv=kfold,scoring="accuracy")
    result.append(score)
    names.append(name)
    res.append(score.mean())
    print('%s:%f'%(name,score.mean()))
plt.ylim(.990,.999)
plt.bar(names,res,color="b",width=0.5)
plt.show()
