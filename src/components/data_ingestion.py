import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Modelling

@dataclass
class DataConfig:
    raw_data_path = os.path.join("Artifacts",'data.csv')
    train_data_path = os.path.join("Artifacts",'train.csv')
    test_data_path = os.path.join("Artifacts",'test.csv')

def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)
    return df

class DataIngestion:
    def __init__(self):
        self.data_path = DataConfig()

    def initiate_ingestion(self):
        try:
            logging.info("Data Loading started")

            train = pd.read_csv("src/notebook/Training.csv")
            test = pd.read_csv("src/notebook/Testing.csv")

            # ✅ CLEAN COLUMN NAMES
            train = clean_columns(train)
            test = clean_columns(test)

            os.makedirs("Artifacts", exist_ok=True)

            logging.info("Data Converted into Dataframe successfully")
            logging.info("train-test split started")

            train, test1 = train_test_split(train, test_size=0.2, random_state=42)
            test = pd.concat([test1, test]).reset_index(drop=True)

            train.to_csv(self.data_path.train_data_path, index=False)
            test.to_csv(self.data_path.test_data_path, index=False)

            full_data = pd.concat([train, test]).reset_index(drop=True)
            full_data.to_csv(self.data_path.raw_data_path, index=False)

            logging.info("Data saved into Artifacts directory")
            return (self.data_path.train_data_path, self.data_path.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

        


if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_ingestion()

transformation = DataTransformation()
train_array, test_array, preprocessor_path, encoder_path = transformation.transformation()

obj2 = Modelling()
best_model, best_acc, model_path = obj2.ModelTrainer(train_array, test_array)

