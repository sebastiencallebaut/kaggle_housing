import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Load the files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Get number of observations for test and train
print([len(x) for x in [train_df, test_df]])

# Combine it into one large file for data exploration and cleaning
combined_df = pd.concat([train_df, test_df])

# Get a first view
print(combined_df)

# Classify int variables into category if needed
combined_df["MSSubClass"] = combined_df["MSSubClass"].astype("category")

# Quick look at potential missing values
print(combined_df.info())

# List of cols with missing values
print([col for col in combined_df.columns if combined_df[col].isnull().any()])

# Handling missing values correctly might greatly help us, we spend some time here
missing  = ['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
            'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
            'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
            'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
            'GarageArea', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SalePrice']

# List of cols with missing values in test set only
missing_test = [col for col in test_df.columns if test_df[col].isnull().any()]

print([x for x in missing if x not in missing_test]) # Only "Electrical" is only missing in train set, so we do nothing

# Categorical data impute with mode
missing_vals = ["MSZoning", "Alley", "Utilities", 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2',
                "Electrical",'KitchenQual', 'Functional','GarageType',"SaleType", 'GarageFinish','GarageQual','GarageCond',
                'Exterior1st', 'Exterior2nd','FireplaceQu', "PoolQC", "Fence", "MiscFeature"]

for missing_val in missing_vals:
    combined_df[missing_val].fillna((combined_df[missing_val].mode()[0]), inplace=True)

# Add "other" category as few elements are missing
combined_df["PoolQC"] = combined_df["PoolQC"].fillna("Other")

# Continuous data
missing_vals = ["LotFrontage", 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF1','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                'GarageCars', 'GarageArea',]
impute_vals = ["LotConfig" ,"Neighborhood",'BsmtFinType1', 'BsmtFinType2','BsmtQual', 'BsmtQual', 'BsmtQual',
               'GarageType', 'GarageType']

for missing_val, impute_val in zip(missing_vals, impute_vals):
    combined_df[missing_val] = combined_df[missing_val].fillna(combined_df.groupby(impute_val)[missing_val].transform('mean'))

# Continuous impute data based on other continuous data
missing_vals = ['GarageYrBlt']
impute_vals = ['YearBuilt']

for missing_val, impute_val in zip(missing_vals, impute_vals):
    combined_df[missing_val] = combined_df[missing_val].fillna(combined_df[impute_val])

# Fill all leftovers with mean
for missing_val in combined_df.columns.values.tolist():

    if missing_val == "SalePrice":
        pass

    else:
        try:

            combined_df[missing_val] = combined_df[missing_val].mean()
        except:
            pass

# List of cols with missing values
print([col for col in combined_df.columns if combined_df[col].isnull().any()])

# Quick look at potential missing values
print(combined_df.info())

# Get a sense of the data
print(combined_df.describe())

"""
# Check the sale price distribution by different types of variables
for element in ["MSSubClass", "MSZoning", "Utilities", "HouseStyle", "CentralAir", "PoolQC", "SaleType"]:
    cat_plot = sns.catplot(y="SalePrice", x= element, kind="swarm", legend="full", data=combined_df, height=4.5, aspect=3 / 3,);
    cat_plot.set_xticklabels(rotation=90)

for element in ["1stFlrSF", "LotArea", "OverallQual", "OverallCond", "YearBuilt","ExterQual", "YrSold"]:
    re_plot = sns.relplot(y="SalePrice", x= element, legend="full", data=combined_df, height=4.5, aspect=3 / 3,);
    re_plot.set_xticklabels(rotation=90)


# Correlation matrix for imputing missing values
corr_mat = combined_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_mat, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
"""

# Get dummies for our data set
combined_df = pd.get_dummies(combined_df)

# Split the data set so to build our model
train_df = combined_df[combined_df["SalePrice"] > 0 ]
test_df = combined_df[combined_df["SalePrice"].isna() ]

# Create the X and y sets
X = train_df.drop(["SalePrice"], axis = 1)
y = train_df["SalePrice"]

# Scale the data
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02

# Drop ID cols

# Add features (e.g. total SF)

# Feature selection (only keep variables with some variance)

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)

gbr_model_full_data = gbr.fit(X, y)



