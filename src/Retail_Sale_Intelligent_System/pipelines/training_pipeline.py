import os
import sys
from src.Retail_Sale_Intelligent_System.exception import CustomException
from src.Retail_Sale_Intelligent_System.logger import logging
from src.Retail_Sale_Intelligent_System.components.data_ingestion import DataIngestion
from src.Retail_Sale_Intelligent_System.components.data_transformation import DataTransformation
from src.Retail_Sale_Intelligent_System.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info(">>>>> Training Pipeline Started <<<<<")
        
        # 1. Data Ingestion
        logging.info("Step 1: Data Ingestion")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion("raw_data_path_argument_is_ignored_internally")
        print(f"Data Ingestion Completed. Train path: {train_data_path}, Test path: {test_data_path}")

        # 2. Data Transformation
        logging.info("Step 2: Data Transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transormation(train_data_path, test_data_path)
        print("Data Transformation Completed.")

        # 3. Model Training
        logging.info("Step 3: Model Training")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model Training Completed. Best Model Accuracy: {accuracy*100:.2f}%")
        
        logging.info(">>>>> Training Pipeline Completed Successfully <<<<<")

    except Exception as e:
        logging.error("Error in Training Pipeline")
        raise CustomException(e, sys)