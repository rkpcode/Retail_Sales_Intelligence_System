import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from src.Retail_Sale_Intelligent_System.exception import CustomException
from src.Retail_Sale_Intelligent_System.logger import logging
from src.Retail_Sale_Intelligent_System.utils import save_object, evaluate_models
import dagshub

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        """
        Calculate metrics for classification:
        - Accuracy: Overall correctness
        - Precision/Recall/F1: Weighted average to handle class imbalance
        """
        accuracy = accuracy_score(actual, pred)
        # 'weighted' calculates metrics for each label, and finds their average weighted by support
        precision = precision_score(actual, pred, average='weighted', zero_division=1)
        recall = recall_score(actual, pred, average='weighted', zero_division=1)
        f1 = f1_score(actual, pred, average='weighted', zero_division=1)
        return accuracy, precision, recall, f1

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # --- MODELS DICTIONARY ---
            # Using Classifiers. 
            # class_weight='balanced' is crucial here to penalize mistakes on the minority class (Loss).
            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
                "XGBClassifier": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False, auto_class_weights='Balanced'),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42)
            }

            # --- HYPERPARAMETERS ---
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    # scale_pos_weight can be added here if imbalance is severe
                },
                "CatBoosting Classifier": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200],
                    'subsample': [0.8, 1.0]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10, 20, None]
                }
            }

            logging.info("Starting Model Training with Hyperparameter Tuning")
            
            # evaluate_models function in utils.py MUST be compatible with Classifiers 
            # (it should calculate accuracy/score, not R2)
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get best model score
            best_model_score = max(sorted(model_report.values()))

            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f"\n==============================")
            print(f"Best Model Found: {best_model_name}")
            print(f"Accuracy Score: {best_model_score * 100:.2f}%")
            print(f"==============================\n")

            # --- MLflow / DagsHub Logging ---
            dagshub.init(repo_owner='rkpcode', repo_name='Retail_Sales_Intelligence_System', mlflow=True)
            mlflow.set_registry_uri("https://dagshub.com/rkpcode/Retail_Sales_Intelligence_System.mlflow")

            with mlflow.start_run():
                predicted_classes = best_model.predict(X_test)
                
                (accuracy, precision, recall, f1) = self.eval_metrics(y_test, predicted_classes)

                # Log Metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                
                # Log Confusion Matrix (Visible in Console)
                # Format: [[True Neg, False Pos], [False Neg, True Pos]]
                cm = confusion_matrix(y_test, predicted_classes)
                print("Confusion Matrix (Reality Check):")
                print(cm)
                print("Legend: [Top-Left: True Loss, Bottom-Right: True Profit]")

                # Save Model
                import pickle
                with open("artifacts/best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                    
                # Optional: Log model to remote registry
                # mlflow.sklearn.log_model(best_model, "model")

            # Threshold Check
            # Lowered to 0.55 because real world prediction without data leakage is hard.
            if best_model_score < 0.55:
                raise CustomException(f"Model performance is too poor ({best_model_score:.2f}). Check data quality or add more features.", sys)
            
            logging.info(f"Best found model saved: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)