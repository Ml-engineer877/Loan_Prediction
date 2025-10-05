#IMPORTING THE PICKLE FILE

import pickle
def load_model():
    with open('Model.pkl','rb') as f:
        return pickle.load(f)
model=load_model()

#ASKING INPUT FROM USERS

columns=["Gender","Married","Dependent","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"]
Gender=int(input("Enter Your Gender (1-Male,0-Female):"))
Married=int(input("You Are Married Or Not (1-Yes,0-No):"))
Dependent=int(input("Enter Number Of Dependents:"))
Education=int(input("Enter Your Education Level (1-Graduate,0-Not Graduate):"))
Self_Employed=int(input("You Are Self Employee Or Not(1-Yes,0-No):"))
ApplicantIncome=int(input("Enter Your Income:"))
CoapplicantIncome=int(input("Enter Your CoapplicantIncome:"))
LoanAmount=int(input("Enter Your Requested Loan Amount:"))
Loan_Amount_Term=int(input("Enter Your Loan Amount Term:"))
Credit_History=int(input("Enter Your Credit History (1 Or 0):"))
Property_Area=int(input("Enter Your Area (1-Rural,0-Urban,2-SemiUrban):"))

user_input=[[Gender,Married,Dependent,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]]
pred=model.predict(user_input)
if pred==1:
    print("You Are Eligible To Get Loan")
else:
    print("You Are Not Eligible To Get Loan")
