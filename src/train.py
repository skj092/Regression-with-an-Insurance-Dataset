import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

path = Path('data')
train_df = pd.read_csv(path / 'train.csv')
test_df = pd.read_csv(path / 'test.csv')


# Handle datetime column
def handle_datetime(df, col):
    df[col] = pd.to_datetime(df[col])
    df['year'] = df[col].dt.year.astype('float32')
    df['month'] = df[col].dt.month.astype('float32')
    df['day'] = df[col].dt.day.astype('float32')
    df['dow'] = df[col].dt.dayofweek.astype('float32')
    df['seconds'] = (df[col].astype('int64') // 10**9).astype('float32')
    df.drop(columns=[col], axis=1, inplace=True)
    return df


# Handle datetime columns
train_df = handle_datetime(train_df, 'Policy Start Date')
test_df = handle_datetime(test_df, 'Policy Start Date')

# Handle target column
train_df['Premium Amount'] = np.log1p(train_df['Premium Amount'])

# List categorical columns
cat_cols = train_df.select_dtypes(include='object').columns.tolist()


# Fill missing values
def fill_missing(df, cat_cols):
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].fillna(df[col].mode().values[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


train_df = fill_missing(train_df, cat_cols)
test_df = fill_missing(test_df, cat_cols)


# Cross-Fold Target Encoding
def cross_fold_target_encode(train_df, test_df, cat_cols, target_col, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = train_df[target_col].mean()

    for col in cat_cols:
        train_df[f'{col}_te'] = 0
        test_df[f'{col}_te'] = 0

        # Create folds and encode
        for train_idx, valid_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_valid = train_df.iloc[valid_idx]

            # Calculate mean excluding current fold
            fold_mean = fold_train.groupby(col)[target_col].mean()

            # Map mean to validation fold
            train_df.loc[valid_idx, f'{col}_te'] = fold_valid[col].map(
                fold_mean)

        # Replace NaNs with global mean
        train_df[f'{col}_te'].fillna(global_mean, inplace=True)

        # Apply encoding to test data
        test_mean = train_df.groupby(col)[target_col].mean()
        test_df[f'{col}_te'] = test_df[col].map(test_mean).fillna(global_mean)

    # Drop original categorical columns
    train_df.drop(columns=cat_cols, inplace=True)
    test_df.drop(columns=cat_cols, inplace=True)

    return train_df, test_df


train_df, test_df = cross_fold_target_encode(
    train_df, test_df, cat_cols, 'Premium Amount')

# Cross-validation
X = train_df.drop(['Premium Amount'], axis=1)
y = train_df['Premium Amount']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': -1,
    'num_leaves': 64,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'verbosity': -1,
    'random_state': 42
}

tik = time.time()
model = LGBMRegressor(**lgb_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
loss = np.sqrt(mean_squared_error(y_valid, y_pred))
tok = time.time()

print(f"RMSE: {loss:.3f} | Time Taken: {tok-tik:.2f}s")

# Inference
y_test = model.predict(test_df)
submission = pd.DataFrame(
    {'id': test_df['id'], 'Premium Amount': np.expm1(y_test)}
)
submission.to_csv('submission.csv', index=False)
