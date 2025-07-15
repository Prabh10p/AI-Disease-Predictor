import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from src.utils import save_object,evaluate_model
from dataclasses import dataclass
from sklearn.multiclass import OneVsRestClassifier


@dataclass
class ModelConfig:
      model_path = os.path.join("Artifacts","model.pkl")

class Modelling:
    def __init__(self):
        self.config = ModelConfig()


    def ModelTrainer(self, train_array,test_array):
        try:
            logging.info("Splitting train and test arrays...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            logging.info('Splitting completed')


            logging.info("Model training started")

           



            models = {
    "AdaBoost": OneVsRestClassifier(AdaBoostClassifier()),
    #"GradientBoost": OneVsRestClassifier(GradientBoostingClassifier()),
    #"RandomForest": OneVsRestClassifier(RandomForestClassifier(class_weight="balanced")),
    "Logistic": OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000)),
    #"CatBoost": OneVsRestClassifier(CatBoostClassifier(verbose=0, class_weights='Balanced')),
    #"DecisionTree": OneVsRestClassifier(DecisionTreeClassifier(class_weight="balanced")),
    #"SVC": OneVsRestClassifier(SVC(probability=True, class_weight="balanced"))
}


            

            params = {
    "AdaBoost": {
        "estimator__n_estimators": [10, 50, 100],
        "estimator__learning_rate": [0.01, 0.1, 1.0]
    },
    "GradientBoost": {
        "estimator__n_estimators": [50, 100],
        "estimator__learning_rate": [0.05, 0.1],
        "estimator__max_depth": [3, 5]
    },
    "RandomForest": {
        "estimator__n_estimators": [50, 100],
        "estimator__max_depth": [None, 10],
        "estimator__max_features": ["sqrt"]
    },
    "Logistic": {
        "estimator__C": [0.01, 0.1, 1.0, 10],
        "estimator__solver": ['lbfgs', 'liblinear']
    },
    "CatBoost": {
        "estimator__depth": [4, 6],
        "estimator__learning_rate": [0.03, 0.1],
        "estimator__iterations": [100, 200]
    },
    "DecisionTree": {
        "estimator__max_depth": [None, 10, 20],
        "estimator__criterion": ["gini", "entropy"]
    },
    "SVC": {
        "estimator__C": [0.1, 1, 10],
        "estimator__kernel": ["linear", "rbf"]
    }
}


            logging.info("model training done")



            report = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_name, (best_model,best_model_train_accuracy, best_model_accuracy) = max(
                report.items(), key=lambda x: x[1][2]  # Use test accuracy
            )


            save_object(
                  file_path=self.config.model_path,
                  obj=best_model

            )
            logging.info(f"Best Model: {best_model_name} with training accuracy :{best_model_train_accuracy:.4f} and testing Accuracy: {best_model_accuracy:.4f}")
            return best_model,best_model_accuracy,self.config.model_path




        except Exception as e:
              raise CustomException(e,sys)
             





