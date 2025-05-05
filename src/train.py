import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

def read(file_name):
    file_path = os.path.join(parent_dir, "data", file_name)

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        return df
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

def preprocess(df, target_column, scalar_type):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(numerical_features) > 0:
        num_impute = SimpleImputer(strategy='mean')
        X_train[numerical_features] = num_impute.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = num_impute.transform(X_test[numerical_features])

        if scalar_type == 'StandardScaler':
            scaler = StandardScaler()
        elif scalar_type == 'MinMaxScaler':
            scaler = MinMaxScaler()

        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    if len(categorical_features) > 0:
        cat_impute = SimpleImputer(strategy='most_frequent')
        X_train[categorical_features] = cat_impute.fit_transform(X_train[categorical_features])
        X_test[categorical_features] = cat_impute.transform(X_test[categorical_features])

        X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

        X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

        return X_train_encoded, X_test_encoded, y_train, y_test

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model, saved_name):
    model.fit(X_train, y_train)

    model_path = os.path.join(parent_dir, 'models', saved_name + '.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy