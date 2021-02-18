

# # Importing Essentials
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


housing = pd.read_csv(r"C:\MyPythonProjects\practice_folder\ml-model-house-data.py")
housing.drop(['Id'], axis=1, inplace=True)
# I can test it by comparing column_count before and after


# # 1. Dealing with missing values
housing.isnull().sum().sort_values(ascending=False)


# # 2. Fixing missing values explicitly

# Replacing categorical columns with None
'''
cat_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']
'''
df = housing
cat_columns = df.select_dtypes(include=['object']).columns

for col in cat_columns:
    df[col] = df[col].fillna("None")

#Changing LotFrontage to mean LotFrontage in the same Neighborhood
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#Replacing numerical column null values with 0
num_columns = df.select_dtypes(exclude=['object']).columns
for col in num_columns:
    if col != 'Electrical':
        df[col] = df[col].fillna(int(0))

#Replacing 'Electrical' with mode
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

#Dropping Utilities
df = df.drop(['Utilities'],axis=1)

#Checking the count of null values again
df.isnull().apply(sum).max()

df.info()


# # 3. Dealing with Outliers

# Removing noisy data which is above 0.999 quantile
num_attributes = df[num_columns]

high_quant = df.quantile(.999)

for col in num_columns:
    df = df.drop(df[col][df[col]>high_quant[col]].index)

df.info()


# # 4. Dealing with correlated attributes

# Removing highly correlated features calculated in the EDA Notebook while viewing scatter plot and corr values

attributes_drop = ['MiscVal', 'MoSold', 'YrSold', 'BsmtFinSF2', 'BsmtHalfBath', 'MSSubClass', 'GarageArea',
                  'GarageYrBlt', '3SsnPorch']
df.drop(attributes_drop, axis=1, inplace=True)

# Removing columns with lots of missing values - PoolQC: 1453, MiscFeature: 1406, Alley: 1369, Fence: 1179
attributes_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
df.drop(attributes_drop, axis=1, inplace=True)


# # 5. Handling Text and Categorical Values

df.select_dtypes(include=['object']).columns


# # 5. Handling Text and Categorical Values

# Transforming Categorial variables using OneHotEncoder
cat_encoder = OneHotEncoder()
df_cat_processed = cat_encoder.fit_transform(df)
df_cat_processed


# # Data Transformation

#Separate features and target variables
housing_X = df.drop('SalePrice', axis=1)
housing_y = df['SalePrice']

# Getting list of numerical and categorical values separately
num_attributes = housing_X.select_dtypes(exclude=['object'])
cat_attributes = housing_X.select_dtypes(include=['object'])

num_attribs = list(num_attributes)
cat_attribs = list(cat_attributes)

# Numerical pipeline to impute any missing values with the median and scale attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])


#Full pipeline that handles both numerical and categorical column's transformation
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

# Description before applying transforms
print("housing_y:\n",housing_y.describe())

# Applying log transformation to sales price - remember right-skewed data
housing_y_prepared = np.log(housing_y)

# Running transformation pipeline on all other attributes
housing_X_prepared = full_pipeline.fit_transform(housing_X)

# Description before applying transform
print("\nhousing_y_prepared:\n",housing_y_prepared)

housing_X_prepared


# # 6. Creating and Assessing ML Models

# # a) Trial 1 with Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Splitting train and test set
X_train, X_test, y_train, y_test = train_test_split(housing_X_prepared, housing_y_prepared, test_size=0.2, random_state=7)


# Training the model on training data

#Training the model on Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model

print("Accuracy%:", model.score(X_test, y_test)*100)


# # b) Training on multiple ML models to see which fits best

# RMSE (Root mean sqaure error) will be used and since we took a log of the target variable, we need to inverse it before calculating error

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import xgboost

# Function to invert target variable array from log scale
def inv_y(transformed_y):
    return np.exp(transformed_y)

# Series to collect RMSE for the different algorithms: "algortihm name + RMSE"
rmse_compare = pd.Series()
rmse_compare.index.name = "Model"

# Series to collect the accuracy for the different algorithms: "algorithms name + score"
scores_compare = pd.Series()
scores_compare.index.name = "Model"

# Model 1: Linear Regression =======================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_val_predictions = linear_model.predict(X_test)
linear_val_rmse = mean_squared_error(inv_y(linear_val_predictions), inv_y(y_test))
linear_val_rmse = np.sqrt(linear_val_rmse)
rmse_compare['LinearRegression'] = linear_val_rmse

lr_score = linear_model.score(X_test, y_test)*100
scores_compare['LinearRegression'] = lr_score

#Model 2: Decision Tress ===========================
dtree_model = DecisionTreeRegressor(random_state=5)
dtree_model.fit(X_train, y_train)

dtree_val_predictions = dtree_model.predict(X_test)
dtree_val_rmse = mean_squared_error(inv_y(dtree_val_predictions), inv_y(y_test))
dtree_val_rmse = np.sqrt(dtree_val_rmse)
rmse_compare['DecisionTreeRegressor'] = dtree_val_rmse

dtree_score = dtree_model.score(X_test, y_test)*100
scores_compare['DecisionTreeRegressor'] = dtree_score

# Model 3: Random Forest ==========================
rf_model = RandomForestRegressor(random_state=5)
rf_model.fit(X_train, y_train)

rf_val_predictions = rf_model.predict(X_test)
rf_val_rmse = mean_squared_error(inv_y(rf_val_predictions), inv_y(y_test))
rf_val_rmse = np.sqrt(rf_val_rmse)
rmse_compare['RandomForest'] = rf_val_rmse

rf_score = rf_model.score(X_test, y_test)*100
scores_compare['RandomForest'] = rf_score


# Model 4: Gradient Boostinf Regression ===========
gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)
gbr_model.fit(X_train, y_train)

gbr_val_predictions = gbr_model.predict(X_test)
gbr_val_rmse = mean_squared_error(inv_y(gbr_val_predictions), inv_y(y_test))
gbr_val_rmse = np.sqrt(gbr_val_rmse)
rmse_compare['GradientBoostingRegression'] = gbr_val_rmse

gbr_score = gbr_model.score(X_test, y_test)*100
scores_compare['GradientBoostingRegression'] = gbr_score


print("RMSE values for different algorithms:")
rmse_compare.sort_values(ascending=True).round()


print("Accuracy scores for different algorithms")
scores_compare.sort_values(ascending=False).round(3)


# # 6.1 Conclusion 1

# Conclusion from above 4 models:
# * LinearRegression and Random Forest have better accuracy than the rest but still have high RMSE. This means that either we need to improve the features or the model is underfitting.
# * Decision Tree should be able to form complex non-linear relationships but it seems that this model is overfitting the training set.
# * Random Forest works by training many decision trees on random subsets of features and then averaging the predictions. This is why the accuracy of Random Forest is higher than Decision Tree.

# # c) Evaluation using Cross-validation

from sklearn.model_selection import cross_val_score

# Performing K fold cross-validation, with K=10 on Linear model
scores = cross_val_score(linear_model, X_train, y_train,
                        scoring="neg_mean_squared_error", cv=10)
linear_rmse_scores = np.sqrt(-scores)

# Printing results
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation", scores.std())
    
display_scores(linear_rmse_scores)


from sklearn.model_selection import cross_val_score

# Performing K fold cross-validation, with K=10 on Randon Forest
scores = cross_val_score(rf_model, X_train, y_train,
                        scoring="neg_mean_squared_error", cv=10)
rf_rmse_scores = np.sqrt(-scores)

# Printing results
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation", scores.std())
    
display_scores(rf_rmse_scores)

