# packages
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# housing data
X = pd.read_csv("data/housing_prices/train.csv", index_col ="Id")
X_test_full = pd.read_csv("data/housing_prices/test.csv", index_col ="Id")

# remove rows with missing target, separate target from predictors
X.dropna(axis = 0, subset = ["SalePrice"], inplace = True)
y = X.SalePrice
X.drop(["SalePrice"], axis = 1, inplace = True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,
                                                                train_size=0.8,
                                                                test_size=0.2,
                                                                random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]
# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in
                ['int64', 'float64']]
# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Define and fit the model
xgb = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, random_state=0)
xgb.fit(X_train, y_train)

# predict
pred1 = xgb.predict(X_valid)

# calculate MAE
mae = mean_absolute_error(pred1, y_valid)
print(f"Mean absolute error: {mae}")
