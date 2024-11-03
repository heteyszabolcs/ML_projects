# data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Import relevant libraries for Stacking Ensembles
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
# Import metrics
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

# fetch data
boston = fetch_openml('boston')
boston.frame.info()

boston.data['RAD'] = boston.data['RAD'].astype(int).apply(lambda x: 0 if x==24 else x)
boston.data['CHAS'] = boston.data['CHAS'].astype(int)
boston.data.info()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2,
                                                    random_state=42)
# Function for evaluating the models' performance
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    print('----------------------------------------------\n')
    print('Train R2 Score: ', round(r2_score(y_true=y_train, y_pred=train_pred), 6))
    print('Train Root Mean Squared Error: ', sqrt(mean_squared_error(y_true=y_train, y_pred=train_pred)))
    print('----------------------------------------------\n')
    test_pred = model.predict(X_test)
    print('Test R2 Score: ', round(r2_score(y_true=y_test, y_pred=test_pred), 6))
    print('Test Root Mean Squared Error: ', sqrt(mean_squared_error(y_true=y_test, y_pred=test_pred)))

# xgboost
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
evaluate_model(xgb, X_train, y_train, X_test, y_test)
# light gbm
light = LGBMRegressor(random_state=42, verbosity=-1)
light.fit(X_train, y_train)
evaluate_model(light, X_train, y_train, X_test, y_test)
# cat gbm
cat = CatBoostRegressor(random_state=42, verbose=0, cat_features=['CHAS', 'RAD'])
cat.fit(X_train, y_train)
evaluate_model(cat, X_train, y_train, X_test, y_test)

# stacking
estimators = [
    ('xgb', xgb),
    ('lgb', light),
    ('cat', cat)
]

# creating a stacked model with the base models and Elastic Net as the meta-model
stack = StackingRegressor(estimators=estimators, final_estimator=ElasticNetCV())
stack.fit(X_train, y_train)
evaluate_model(stack, X_train, y_train, X_test, y_test)

