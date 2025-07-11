import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class TransformationConfig:
    preprocessor_path: str = os.path.join("Artifacts", "preprocessor.pkl")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataTransformation:
    def __init__(self):
        self.config = TransformationConfig()
        
    def transformation(self):
        try:
            logging.info("Loading train and test data...")
            train_data = pd.read_csv("Artifacts/train.csv")
            test_data = pd.read_csv("Artifacts/test.csv")

            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]

            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            logging.info("Splitting successful")

            # -------------------------
            # Label encode the target
            # -------------------------
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train)
            y_test_encoded = encoder.transform(y_test)

            # -------------------------
            # Get columns
            # -------------------------
            categorical_features = X_train.select_dtypes(include='object').columns.tolist()
            numeric_features = X_train.select_dtypes(exclude='object').columns.tolist()

            # -------------------------
            # Create transformers
            # -------------------------
            cat_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            num_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", cat_transformer, categorical_features),
                    ("num", num_transformer, numeric_features)
                ]
            )

            logging.info("Fitting preprocessor on train data")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(self.config.preprocessor_path, preprocessor)

            logging.info("Preprocessor saved successfully")

            train_array = np.c_[X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed, y_train_encoded]
            test_array = np.c_[X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed, y_test_encoded]

            return (train_array, test_array, self.config.preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys)
