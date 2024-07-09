import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_path = r'F:\loan_fix\Dataset\train.csv'
train = pd.read_csv(train_path)

train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})

print(train.isnull().sum())

Loan_status = train['Loan_Status']
train.drop('Loan_Status', axis=1, inplace=True)

test_path = r'F:\loan_fix\Dataset\test.csv'
test = pd.read_csv(test_path)

Loan_ID = test['Loan_ID']
data = train.append(test)

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})

data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
data['Dependents'].fillna(data['Dependents'].median(), inplace=True)

data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Self_Employed'].fillna(np.random.randint(0, 2), inplace=True)

data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Rural': 0, 'Semiurban': 1})

data['Credit_History'].fillna(np.random.randint(0, 2), inplace=True)
data['Married'].fillna(np.random.randint(0, 2), inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
data['Gender'].fillna(np.random.randint(0, 2), inplace=True)

data.drop('Loan_ID', inplace=True, axis=1)

print(data.isnull().sum())

train_X = data.iloc[:len(train)]
X_test = data.iloc[len(train):]

train_X, test_X, train_y, test_y = train_test_split(train_X, Loan_status, random_state=7)

models = []
models.append(('logreg', LogisticRegression()))
models.append(('tree', DecisionTreeClassifier()))
models.append(('lda', LinearDiscriminantAnalysis()))
models.append(('svc', SVC()))
models.append(('knn', KNeighborsClassifier()))
models.append(('nb', GaussianNB()))

results = []
names = []
scoring = 'accuracy'
seed = 7

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed) 
    cv_results = cross_val_score(model, train_X, train_y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))


logreg = LogisticRegression()
logreg.fit(train_X, train_y)
pred = logreg.predict(test_X)

print("Accuracy Score:", accuracy_score(test_y, pred))
print("Confusion Matrix:\n", confusion_matrix(test_y, pred))
print("Classification Report:\n", classification_report(test_y, pred))

submission_predictions = logreg.predict(X_test).astype(int)
df_output = pd.DataFrame({'Loan_ID': Loan_ID, 'Loan_Status': submission_predictions})

output_path = r'F:\loan_fix\Dataset\output.csv'
df_output.to_csv(output_path, index=False)

print("Predictions saved successfully to", output_path)
