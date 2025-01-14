import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from lightgbm import LGBMRegressor

path = Path('data')
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')

# For simplicity drop the column 'Policy Start Date'
train_df.drop(columns=['Policy Start Date'], axis=1, inplace=True)
test_df.drop(columns=['Policy Start Date'], axis=1, inplace=True)

# list categorical and numerical columns
cat_cols = train_df.select_dtypes(include='object').columns.tolist()
num_cols = train_df.select_dtypes(include=np.number).drop(
    columns='Premium Amount').columns.tolist()


# Fill Missing
def fill_missing(df, cat_cols=cat_cols):
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].fillna(df[col].mode().values[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


train_df = fill_missing(train_df, cat_cols)
test_df = fill_missing(test_df, cat_cols)


# Handle categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])


# cross validation
X = train_df.drop(['Premium Amount'], axis=1)
y = train_df['Premium Amount']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)


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
loss = mean_squared_log_error(y_pred, y_valid)
tok = time.time()
print(f"loss : {math.sqrt(loss):.3f} | Time Taken {tok-tik:.2f}s")


# Inference
y_test = model.predict(test_df)
submission = pd.DataFrame({'id': test_df['id'], 'Premium Amount': y_test})
submission.to_csv('submission.csv', index=False)
