# importing the tools 
import pandas as pd
import numpy as np

#importing the ml tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
# importing the evaluation tools
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,roc_auc_score)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score,recall_score,f1_score
# additional tool
from IPython.display import display


# --- Step 1: Generate the Dataset ---
n = 600
rng = np.random.default_rng(seed=42)

monthly_income = rng.uniform(10, 100, size=n)
loan_amount = rng.uniform(50, 500, size=n)
credit_score = rng.uniform(300, 850, size=n)
num_existing_loans = rng.integers(0, 6, size=n)

# Compute risk score and binary target
risk_score = (
    -0.05 * monthly_income 
    + 0.008 * loan_amount 
    - 0.012 * credit_score 
    + 1.5 * num_existing_loans 
    + rng.normal(0, 2, size=n)
)
will_default = (risk_score > 0).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'monthly_income': monthly_income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'num_existing_loans': num_existing_loans,
    'target': will_default
})

X = df.drop('target', axis=1)
y = df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#training model with desicion tree
pipe=make_pipeline(
    MinMaxScaler(),
    DecisionTreeClassifier(max_depth=3,random_state=42)
)
pipe.fit(X_train,y_train)
# training the model with RandomForestClassifier
pipe2=make_pipeline(
    MinMaxScaler(),
    RandomForestClassifier(n_estimators=100,max_depth=3,random_state=42)
)
pipe2.fit(X_train,y_train)


#Evaluation for the Desicion tree
print("-------- Desicion tree evaluation --------")
print()
prediction=pipe.predict(X_test)
print()
print("------------- Accuracy Scores -------------")
print("The training Score : ",pipe.score(X_train,y_train)*100,"%")
print("The testing score : ",pipe.score(X_test,y_test)*100,"%")
print()
print("------------- Confusion matrix -------------")
matrix=confusion_matrix(prediction,y_test)
print(matrix)
tn=matrix[0][0]
fp=matrix[0][1]
fn=matrix[1][0]
tp=matrix[1][1]
print()
print(f"tn : {tn} | fp : {fp}\nfn : {fn}  | tp : {tp}")
print("---------- Classification report ----------")
print(classification_report(prediction,y_test))
print()
print("----------------- ROC-AUC -----------------")
# Get probabilities for the positive class
probab = pipe.predict_proba(X_test)[:, 1]

# Correct order: (y_true, y_score)
roc_score = roc_auc_score(y_test, probab)

print(f"ROC AUC Score: {roc_score:.4f}")

print()
print()
print("<>"*50)
# Evlation for Random Forest
print("-------- Random Forest evaluation --------")

print()
print()
prediction_rf=pipe2.predict(X_test)
print("------------- Accuracy Scores -------------")
print("The training Score : ",pipe2.score(X_train,y_train)*100,"%")
print("The testing score : ",pipe2.score(X_test,y_test)*100,"%")
print()
print("------------- Confusion matrix -------------")
matrix=confusion_matrix(prediction_rf,y_test)
print(matrix)
tn=matrix[0][0]
fp=matrix[0][1]
fn=matrix[1][0]
tp=matrix[1][1]
print()
print(f"tn : {tn} | fp : {fp}\nfn : {fn}  | tp : {tp}")
print()
print("---------- Classification report ----------")
print(classification_report(prediction_rf,y_test))
print()
print("----------------- ROC-AUC -----------------")
# Get probabilities for the positive class
probab_rf = pipe2.predict_proba(X_test)[:, 1]

# Correct order: (y_true, y_score)
roc_score = roc_auc_score(y_test, probab_rf)

print(f"ROC AUC Score: {roc_score:.4f}")

print("_"*100)
precisions,recalls,thresholds=precision_recall_curve(y_test,probab_rf)

# claculating the f1 score for all the thresholds and the one with the highest thresholds will be the best threshold
# adding the 1e-9 prevents it dividing by 0
f1_scores=(
    (2*precisions[:-1]*recalls[:-1])/
    (precisions[:-1]+recalls[:-1]+1e-9)
)

best_idx=f1_scores.argmax()
best_threshold=thresholds[best_idx]
y_pred_defoult=(probab_rf>=0.5).astype(int)
y_pred_costume=(probab_rf>=best_threshold).astype(int)

print("The Best Threshold : ",best_threshold)
print("The best precisions : ",precisions[best_idx])
print("The best recalls : ",recalls[best_idx])
print("The best f1 score : ",f1_scores[best_idx])
print("_"*100)
# Step 5 
prec_defoult=precision_score(y_pred_defoult,y_test)
recall_defoult=recall_score(y_pred_defoult,y_test)
f1_defoult=f1_score(y_test,y_pred_defoult)

prec_custom=precision_score(y_pred_costume,y_test)
recall_custom=recall_score(y_pred_costume,y_test)
f1_custom=f1_score(y_test,y_pred_costume)

comparision={
    "Metrics":["Precision","recall score","F1 Score"],
    "Defoult":[prec_defoult,recall_defoult,f1_defoult],
    "Costumed":[prec_custom,recall_custom,f1_custom]

}
comp=pd.DataFrame(comparision)
print("The Comparision table of metrics with diffrent thresholds : ")
print()
display(comp)

