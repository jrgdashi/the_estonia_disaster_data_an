import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.formula as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from treelib import Tree
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r'C:\Users\jrgda\OneDrive\Desktop\new2.txt')

print("Df:")
print(df)

print("I don't care about the PassengerId, Firstname, Lastname therefore I will be droping those columns")
df = df.drop('PassengerId', 1)
df = df.drop('Firstname', 1)
df = df.drop('Lastname', 1)
df = df.drop('Country', 1)

print("New df:")
print(df)
print()

print("Describe df:")
print(df.describe())
print()

print("Man and Women survival:")
sns.countplot(x='Sex', hue='Survived', alpha=0.9, data=df)
sns.despine()
plt.title("Man Women Survival")
plt.show()

print("Passenger and Crew survival:")
sns.countplot(x='Category', hue='Survived', alpha=0.9, data=df)
sns.despine()
plt.title("Passenger Crew Survival")
plt.show()

print("Age survival:")
plt.figure(figsize=[12,4])
sns.countplot(x='Age', hue='Survived', alpha=0.9, data=df)
sns.despine()
plt.title("Age by survived")
plt.show()

'''
I'll be changing the categorical data to numbers so that I can use them, this data has either M or F and P or C so 
it's very convenient as-well. 
'''
Category = {'C': 1, 'P': 2}
df.Category = [Category[item] for item in df.Category]

Sex = {'M': 1, 'F': 2}
df.Sex = [Sex[item] for item in df.Sex]
print(df)

sns.boxplot(x="Survived",y="Age",data=df)
plt.show()



# This is not what I expected since I would have thought there was going to be more in common with surviving and age, sex.
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# MODELS


Y = df['Survived']
X = df.drop(columns=['Survived'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)

# KNN
knncla = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
knncla.fit(X_train, Y_train)
Y_predict6 = knncla.predict(X_test)
knncla_cm = confusion_matrix(Y_test, Y_predict6)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(knncla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
plt.title('KNN Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()
test_acc_knncla = round(knncla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)
train_acc_knncla = round(knncla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
model1 = pd.DataFrame({
    'Model': ['KNN'],
    'Train Score': [train_acc_knncla],
    'Test Score': [test_acc_knncla]
})
print(model1)

# REGRESSION MODEL

t_data = pd.concat([X_train, Y_train], axis=1)

print(t_data.head())

model2= smf.ols('Survived ~ Age + Sex', data= t_data).fit()

print(model2.summary())

coef = model2.params
rs = pd.Series([model2.rsquared], index=["R_squared"])
r1= coef.append(rs)
r1= pd.DataFrame(data=r1, columns=["Value"])
print(r1)

# K-fold

model2_score = cross_val_score(LinearRegression(), X_train[['Age','Sex']], Y_train, cv=10)
print(model2_score)
print(np.mean(model2_score))

# RMSE
y_prediction= model2.predict(X_test[['Age','Sex']])
print(np.sqrt(metrics.mean_squared_error(Y_test.T.squeeze(), y_prediction)))

#   MODEL 3

#  x and y together created the train database
t_data = pd.concat([X_train, Y_train], axis=1)

print(t_data.head())

model3= smf.ols('Survived ~ Category + Sex', data= t_data).fit()

print(model3.summary())

coef = model3.params
rs = pd.Series([model3.rsquared], index=["R_squared"])
r3= coef.append(rs)
r3= pd.DataFrame(data=r3, columns=["Value"])
print(r3)

# K-fold

model3_score = cross_val_score(LinearRegression(), X_train[['Category', 'Sex']], Y_train, cv=10)
print(model3_score)
print(np.mean(model3_score))

# RMSE
y_prediction= model3.predict(X_test[['Category', 'Sex']])
print(np.sqrt(metrics.mean_squared_error(Y_test.T.squeeze(), y_prediction)))

"""
Both regression models have really low accuracy, where as the KNN seems to have high accuracy, therefore so far
I would rather keep the first MODEL (KNN)
"""
log_regression = LogisticRegression()
log_regression.fit(X_train,Y_train)
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()





"""
References
https://www.kaggle.com/code/khotijahs1/the-estonia-disaster-model-random-forest-vs-k-nn
"""