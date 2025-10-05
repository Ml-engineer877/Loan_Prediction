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

dataset["Gender"]=dataset["Gender"].fillna("1.0")
dataset["Married"]=dataset["Married"].fillna("1")
dataset["Dependents"]=dataset["Dependents"].fillna("1")
dataset["Self_Employed"]=dataset["Self_Employed"].fillna("0")
dataset["LoanAmount"]=dataset["LoanAmount"].fillna(dataset["LoanAmount"].mean())
dataset["Loan_Amount_Term"]=dataset["Loan_Amount_Term"].fillna(dataset["Loan_Amount_Term"].mean())
dataset["Credit_History"]=dataset["Credit_History"].fillna("1")
print(dataset.isnull().sum())

#VISUALIZING THE CORRELATION OF DATA

from matplotlib import pyplot as plt
import seaborn as sns
sns.heatmap(dataset[["ApplicantIncome","CoapplicantIncome","Credit_History"]].corr(),annot=True,cmap="coolwarm")
plt.show()

#CHECKING FOR OUTLIERS DATA
outlier=pd.DataFrame()
num_col=dataset[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
for col in num_col:
    q1=dataset[col].quantile(0.25)
    q3=dataset[col].quantile(0.75)
    iqr=q3-q1
    low=q1-1.5*iqr
    up=q3+1.5*iqr
    outlier=dataset[(dataset[col]<(q1-1.5*iqr)) | (dataset[col]>q3+1.5*iqr)]
    print(f"{col} -> outlier:{outlier.shape[0]}")

#VISUALIZING THE OUTLIERS

plt.figure(figsize=(10,6))
sns.boxplot(dataset[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]])
plt.title("Boxplot for Numeric Columns")
plt.show()

#REPLACING THE OUTLIERS
for col in num_col:
    dataset[col]=dataset[col].clip(lower=low,upper=up)
    q1=dataset[col].quantile(0.25)
    q3=dataset[col].quantile(0.75)
    iqr=q3-q1
    low=q1-1.5*iqr
    up=q3+1.5*iqr
    outlier=dataset[(dataset[col]<(q1-1.5*iqr)) | (dataset[col]>q3+1.5*iqr)]
    print(f"{col} -> outlier:{outlier.shape[0]}")

plt.figure(figsize=(10,6))
sns.boxplot(dataset[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]])
plt.title("Boxplot for Numeric Columns")
plt.show()
            
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

#MODEL TRAINING

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#MODEL TESTING
predicted=model.predict(x_test)
print("Predicted Output:",predicted)

#MODEL ACCURACY AND CONFUSION MATRIX

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Of The Model Is :",accuracy_score(predicted,y_test)*100)
print("Confusion Matrix Is :",confusion_matrix(predicted,y_test))


###CONVERTING THE PYTHON FILE TO PICKLE FILE
##
##import pickle
##with open('Model.pkl','wb') as f:
##    pickle.dump(model,f)
