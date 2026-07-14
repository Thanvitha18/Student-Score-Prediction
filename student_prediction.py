import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("student_scores.csv")
print(df.head(5))
df.info()
print(df.describe())
print(df.shape)
print(df.isnull().sum())
X=df[["Hours"]]
y=df["Scores"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
print(X_train.shape)
print(X_test.shape)
model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
print(y_predict)
comparison=pd.DataFrame({"Actual":y_test,"Predicted":y_predict})
print(comparison)
print("MAE:",mean_absolute_error(y_test,y_predict))
print("MSE:",mean_squared_error(y_test,y_predict))
print("R^2 Score:",r2_score(y_test,y_predict))
new_prediction=model.predict([[9.5]])
print("Predicted Score:",new_prediction[0])
plt.scatter(X,y)
plt.plot(X,model.predict(X),color='yellow')
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Student Score Prediction using Linear Regression")
plt.savefig("graph_output.png", dpi=300, bbox_inches="tight") 
plt.show()





