import os
import sys
from src.Retail_Sale_Intelligent_System.exception import CustomException
from src.Retail_Sale_Intelligent_System.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Retail_Sale_Intelligent_System.utils import read_sql_data

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path: str):
        logging.info("Starting data ingestion process")
        try:
            # Read the dataset from SQL
            df = read_sql_data()
            logging.info("Dataset read successfully")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # --- Feature Engineering ---
            
            # 1. Create Target Variable (This is the ONLY place Profit is used)
            df['Profitable'] = (df['Profit'] > 0).astype(int)
            
            # 2. Date Handling 
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
            df['Ship_Date']  = pd.to_datetime(df['Ship_Date'])
            
            # 3. Create Time Features
            df['Year'] = df['Order_Date'].dt.year
            df['Month'] = df['Order_Date'].dt.month_name()
            df['Weekday'] = df['Order_Date'].dt.day_name()

            # 4. Calculate Missing Features
            # Calculate Ship_Duration
            df['Ship_Duration'] = (df['Ship_Date'] - df['Order_Date']).dt.days
            
            # REMOVED: Profit Margin calculation. It causes Data Leakage.

            # Split Data based on Year (Temporal Split)
            train_df = df[df['Year'] < 2017].copy()
            test_df  = df[df['Year'] == 2017].copy()

            # Define columns to keep
            # REMOVED: 'Profit', 'Profit Margin (%)' -> We want to PREDICT these, not use them.
            feature_cols = [
               'Sales', 'Quantity', 'Discount', 'Ship_Duration',
               'Region', 'Segment', 'Category', 'Sub_Category', 'Ship_Mode', 
               'Month', 'Weekday'
             ]
            
            # Check for missing columns before proceeding
            missing_cols = [col for col in feature_cols if col not in train_df.columns]
            if missing_cols:
                raise CustomException(f"The following required columns are missing from the dataframe: {missing_cols}", sys)

            X_train = train_df[feature_cols].copy()
            y_train = train_df['Profitable']
            X_test  = test_df[feature_cols].copy()
            y_test  = test_df['Profitable']
            
            # Derived Features Loop
            for df_ in [X_train, X_test]:
                df_['Log_Sales'] = np.log1p(df_['Sales'])
                df_['High_Discount'] = (df_['Discount'] >= 0.3).astype(int)
                
                # Map Month and Weekday
                month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
                df_['Month_Num'] = df_['Month'].map(month_map)
                
                dow_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
                df_['Weekday_Num'] = df_['Weekday'].map(dow_map)
                
                # Target Encoding for Sub_Category
                # Note: We map using Training means ONLY to avoid leakage
                means_sub = X_train.join(y_train).groupby('Sub_Category')['Profitable'].mean()
                df_['SubCat_ProfRate'] = df_['Sub_Category'].map(means_sub)
                
                # Fill NaN values (for categories present in Test but not Train)
                if 'SubCat_ProfRate' in df_.columns and df_['SubCat_ProfRate'].isnull().any():
                     df_['SubCat_ProfRate'] = df_['SubCat_ProfRate'].fillna(means_sub.mean())

            train_set = pd.concat([X_train, y_train], axis=1)
            test_set = pd.concat([X_test, y_test], axis=1)

            logging.info("Data split into training and testing sets")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)