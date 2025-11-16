import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
df = pd.read_csv("train_energy_data.csv")
print(df.dtypes)

# preprocess
def preprocessDataflow(df):
    mapBuildingTypes = {
        "Residential": 0 , "Commercial": 0.5, "Industrial": 1
    }
    df = df.drop(["DayOfWeek", "AverageTemperature"], axis=1)
    df["BuildingType"] = df["BuildingType"].map(mapBuildingTypes)
    return df

data = preprocessDataflow(df)
print("-----PreProcessed-----")
print(data.dtypes)
# features
X = data.drop("EnergyConsumption", axis=1)
y = data["EnergyConsumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
# normalize & scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model tune
def tuneModel(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = tuneModel(X_train, y_train)



with open("EnergyConsumptionLinearModel.pkl", "wb") as f:
    pickle.dump(model, f) 
with open("EnergyConsumptionScaler.pkl", "wb") as f: 
    pickle.dump(scaler, f)

# evaluation
y_pred = model.predict(X_test)


# metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
sns.lineplot(x=[y_test.min(), y_test.max()],
             y=[y_test.min(), y_test.max()],
             color="red")
plt.title("Actual vs Predicted (Linear Regression)")
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.show()

# Residual plot
plt.figure(figsize=(8, 5))
sns.residplot(x=y_pred, y=y_test - y_pred)
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()
# save

