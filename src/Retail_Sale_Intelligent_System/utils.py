import os
import sys
from src.Retail_Sale_Intelligent_System.exception import CustomException
from src.Retail_Sale_Intelligent_System.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score  # CHANGED: r2_score -> accuracy_score
from dataclasses import dataclass
from dotenv import load_dotenv
import pymysql
import pickle


load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL database")
    try:
        mydb = pymysql.connect(
            host = host,
            user = user,
            password = password,
            db = db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query('SELECT * FROM superstore_sales', mydb)
        print(df.head())
        

        return df
        
    except Exception as ex:
        raise CustomException(ex)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # Using GridSearchCV to find best parameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Updating model with best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # CHANGED: Using accuracy_score instead of r2_score for Classification
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)