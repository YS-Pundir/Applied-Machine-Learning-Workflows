# importing  the tools
import pandas as pd 
import numpy as np
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.pipeline import make_pipeline



from numpy.random import default_rng

rng = default_rng(seed=99)
n = 500

study_hours        = rng.uniform(1, 10, size=n)       # daily study hours
attendance_percent = rng.uniform(40, 100, size=n)     # class attendance %
assignments_done   = rng.uniform(0, 10, size=n)       # assignments submitted (out of 10)

scores = (
    20
    + 5.5  * study_hours
    + 0.4  * attendance_percent
    + 3.0  * assignments_done
    + rng.normal(0, 8, size=n)
)

# Label: 1 = Pass, 0 = Fail
y = (scores >= 70).astype(int)
X = np.column_stack([study_hours, attendance_percent, assignments_done])
# Converting the database into the pandas dataframe 
x=pd.DataFrame(X,columns=["study_hours", "attendance_percent", "assignments_done"])
y=pd.DataFrame(y,columns=["Pass(1) or Fail(0)"])


# Spliting the data into the training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=99)
# training the model
pipeline=make_pipeline(
    MinMaxScaler(),
    LogisticRegression()
)

pipeline.fit(x_train,y_train)
print("-"*20,"Class Distribution","-"*20)
print(y.value_counts())

prediction=pipeline.predict(x_test)
actual=y_test
probab_pass=pipeline.predict_proba(x_test)

df_display=x_test.copy()
df_display["Actual"]=actual
df_display["Predicted"]=prediction
df_display["P(pass)"]=probab_pass[:,1]
df_display["Correct"]=df_display["Actual"]==df_display["Predicted"]
print("-"*60)
print()
print("----TABLE FOR 10 STUDENTS----")
print()
display(df_display.head(10))

matrix=confusion_matrix(y_test,prediction)
tn=matrix[0][0]
fp=matrix[0][1]
fn=matrix[1][0]
tp=matrix[1][1]
print("-"*60)
print()
print("-"*20,"  Break-Down   ","-"*20)
print()
print(f"                   Pred MALIGNANT(0) | Pred BENIGN(1)")
print(f"Actual MALIGNANT : TN = {tn:<12} | FP = {fp} (Danger!)")
print(f"Actual BENIGN    : FN = {fn:<12} | TP = {tp}")

print()
print("-"*20,"  Accuracy  ","-"*20)
print("Accuracy : ",(tn+tp)/(tn+tp+fn+fp)*100,"%")



print("-"*20,"Appling Thresholds","-"*20)
threshold=0.6
pred_custom=(df_display["P(pass)"]>=threshold).astype(int)
print("-"*60)

print("THRESHOLD=0.6")
print()
print("No of Pass and Fail-->")
print(pred_custom.value_counts())
print("Accuracy Acore : ",accuracy_score(y_test,pred_custom)*100,"%")
print()
print()
print("THRESHOLD=0.5")
threshold=0.5
pred_custom=(df_display["P(pass)"]>=threshold).astype(int)
print("-"*60)

print()
print("No of Pass and Fail-->")
print(pred_custom.value_counts())
print("Accuracy Acore : ",accuracy_score(y_test,pred_custom)*100,"%")
