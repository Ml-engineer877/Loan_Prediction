🏦 Loan Prediction using Logistic Regression
📘 Overview

The Loan Prediction System is a Machine Learning project that predicts whether a loan applicant is likely to get loan approval or not.
It uses Logistic Regression, a supervised classification algorithm, to learn patterns from historical loan data and predict loan approval outcomes based on applicant information.

🎯 Objective

To develop a predictive model using Logistic Regression that automates the loan approval process and helps financial institutions make data-driven decisions quickly and accurately.

📊 Dataset

The dataset contains details of past loan applicants. Each record includes personal, financial, and credit-related information used to determine loan eligibility.

Feature	Description
Loan_ID	Unique Loan Identification Number
Gender	Male / Female
Married	Applicant’s marital status
Dependents	Number of dependents
Education	Graduate / Not Graduate
Self_Employed	Yes / No
ApplicantIncome	Applicant’s income
CoapplicantIncome	Co-applicant’s income
LoanAmount	Loan amount (in thousands)
Loan_Amount_Term	Loan term (in months)
Credit_History	Credit history (1 = good, 0 = bad)
Property_Area	Urban / Rural / Semiurban
Loan_Status	Target variable (Y = Approved, N = Not Approved)
⚙️ Technologies Used

Python 3.9

NumPy

Pandas

Matplotlib / Seaborn (Visualization)

Scikit-learn (Machine Learning)


🧠 Machine Learning Workflow
1️⃣ Data Preprocessing

Handle missing values using mean/median/mode imputation.

Encode categorical variables using LabelEncoder.

Normalize continuous features if necessary.

2️⃣ Exploratory Data Analysis (EDA)

Study correlations between features and target variable.

<img src="images/Correlation.png" alt="Correlation" width="500">

Check for outliers and feature importance.

Box Plot Of Outliers:

<img src="images/Outliers.png" alt="Boxplot" width="500">

Box Plot Of After Replacing Outliers:

<img src="images/Without Outliers.png" alt="Boxplot" width="500">

3️⃣ Model Building – Logistic Regression

Logistic Regression is a binary classification algorithm used to predict two outcomes: loan approved (Y) or not approved (N).

The sigmoid function is used to estimate probabilities.

4️⃣ Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

5️⃣ Model Evaluation

Evaluate the model using:

Accuracy Score

Confusion Matrix


Example:

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

📈 Results
Metric	Score
Accuracy	~81%

✅ The Logistic Regression model performed consistently well and is interpretable, making it ideal for financial use cases.


📜 Future Enhancements

Apply other ML algorithms (Random Forest, XGBoost) for comparison.

Build a simple Streamlit web app for user interaction.

Deploy model using Flask API or Heroku/AWS.

🤝 Contributors

Gobi M — Machine Learning Engineer

Open to contributions! Feel free to fork, suggest improvements, or open issues.