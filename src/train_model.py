import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(reference_path, new_data_path):
    try:
        reference_data = pd.read_csv(reference_path)
        new_data = pd.read_csv(new_data_path)
        return pd.concat([reference_data, new_data], ignore_index=True) # new_data is appended to reference_data without Outcome
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def prepare_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def build_pipeline(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def main():
    # Configuration
    reference_data_path = 'data/processed/reference_from_train_data.csv'
    new_data_path = 'data/processed/new_test_data.csv'
    model_path = 'models/model_pipeline.pkl'
    
    df = load_data(reference_data_path, new_data_path)
    if df.empty:
        return
    
    X, y = prepare_data(df)
    numeric_features = X.columns.tolist()
    
    pipeline = build_pipeline(numeric_features)
    pipeline.fit(X, y)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()