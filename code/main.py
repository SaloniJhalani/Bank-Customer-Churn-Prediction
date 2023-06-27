from pathlib import Path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


def one_hot_encoding(df):
    categorical_columns = df.select_dtypes(include='object').columns
    one_hot_encoders = {}

    # Iterate over each categorical column and fit one hot encoder
    for column in categorical_columns:
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(df[[column]])
        encoded_features = encoder.transform(df[[column]])
        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df.drop(column, axis=1), encoded_data], axis=1)
        one_hot_encoders[column] = encoder
    return df, one_hot_encoders


def data_split(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def normalization(X):
    continuous_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = MinMaxScaler()

    # Fit the scaler on the training data
    scaler.fit(X[continuous_columns])

    # Perform normalization
    X[continuous_columns] = scaler.transform(X[continuous_columns])
    return X, scaler


def perform_oversampling(X, y):
    # Instantiate the SMOTE object
    smote = SMOTE()
    # Perform oversampling
    X_oversampled, y_oversampled = smote.fit_resample(X, y)
    return X_oversampled, y_oversampled


if __name__ == "__main__":
    # Load Data
    df = pd.read_csv(str(Path(__file__).parents[1] / 'data/churn_data.csv'))

    #Drop fields not used for training
    drop_columns = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(drop_columns, axis=1, inplace=True)

    # Split features & label
    X = df.drop('Exited', axis=1)  # Features
    y = df['Exited']  # Target variable

    X, one_hot_encoder = one_hot_encoding(X)  # One Hot Encoding

    X, scaler = normalization(X)  # Normalization
    X_oversampled, y_oversampled = perform_oversampling(X, y) #Oversampling
    X_train, X_test, y_train, y_test = data_split(X_oversampled, y_oversampled)  # Test Train Split

    # Build Model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test))

    # Save Model
    with open(str(Path(__file__).parents[1] / 'model/model.pickle'), 'wb') as f:
        pickle.dump((model, one_hot_encoder, scaler), f)
