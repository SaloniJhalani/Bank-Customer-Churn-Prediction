from pathlib import Path
import pandas as pd
import pickle

def predict(X):
    # Load the model from the saved file
    with open(str(Path(__file__).parents[1] / 'model/model.pickle'), 'rb') as f:
        model, one_hot_encoder, scaler = pickle.load(f)

    # One Hot Encoding
    for column, encoder in one_hot_encoder.items():
        encoded_features = encoder.transform(X[[column]])
        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([column]))
        X = pd.concat([X.drop(column, axis=1), encoded_data], axis=1)

    # Normalization
    continuous_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    X[continuous_columns] = scaler.transform(X[continuous_columns])

    # Prediction
    pred = model.predict(X)
    return pred
