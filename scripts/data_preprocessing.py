from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
 
from config import RANDOM_STATE, TEST_SIZE, VALID_SIZE, TARGET_COL
 
 
def prepare_data(df: pd.DataFrame):
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
 
    # First carve out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
 
    # Then split remaining into train / validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=VALID_SIZE, stratify=y_temp,
        random_state=RANDOM_STATE
    )
 
    # Scale Time and Amount using stats from the TRAIN set only (no leakage)
    scaler = StandardScaler()
    for split_X in (X_train, X_valid, X_test):
        split_X[["Time", "Amount"]] = split_X[["Time", "Amount"]].astype(float)
 
    scaler.fit(X_train[["Time", "Amount"]])
    for split_X in (X_train, X_valid, X_test):
        split_X[["Time", "Amount"]] = scaler.transform(split_X[["Time", "Amount"]])
 
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
 
    valid_df = X_valid.copy()
    valid_df[TARGET_COL] = y_valid.values
 
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
 
    print(
        f"Split sizes -> train: {len(train_df):,} "
        f"({train_df[TARGET_COL].sum()} fraud), "
        f"valid: {len(valid_df):,} ({valid_df[TARGET_COL].sum()} fraud), "
        f"test: {len(test_df):,} ({test_df[TARGET_COL].sum()} fraud)"
    )
 
    return train_df, valid_df, test_df