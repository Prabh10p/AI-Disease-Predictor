import os
import sys
import numpy as np
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold



def save_object(file_path,obj):
    try:
      path = os.path.dirname(file_path)
      os.makedirs(path,exist_ok=True)

      with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = {}  
        for name, model in models.items():
            param = params.get(name, {})
            print(f"Training {name}...")

            v_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduce folds to 3
            grid = GridSearchCV(model, param_grid=param, cv=v_strategy, n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # ✅ Optional: Remove cross_val_score for speed
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            report[name] = (best_model, train_accuracy, test_accuracy)
            print(f"{name} done ✅ | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)




def load_object(file_path):
     try:
          with open(file_path,"rb") as file_obj:
               return pickle.load(file_obj)
     except Exception as e:
          raise CustomException(e,sys)
          
             

