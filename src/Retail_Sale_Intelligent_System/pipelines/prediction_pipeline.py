import sys
import os
import pandas as pd
import numpy as np
from src.Retail_Sale_Intelligent_System.exception import CustomException
from src.Retail_Sale_Intelligent_System.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths for artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Loading Model and Preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("Transforming Data...")
            data_scaled = preprocessor.transform(features)
            
            print("Predicting...")
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class is responsible for mapping the raw input data from the UI 
    to the format required by the model (Data Frame).
    """
    def __init__(self,
        sales: float,
        quantity: int,
        discount: float,
        order_date: str,
        ship_date: str,
        region: str,
        segment: str,
        category: str,
        sub_category: str,
        ship_mode: str):

        self.sales = sales
        self.quantity = quantity
        self.discount = discount
        self.order_date = order_date
        self.ship_date = ship_date
        self.region = region
        self.segment = segment
        self.category = category
        self.sub_category = sub_category
        self.ship_mode = ship_mode

    def get_data_as_data_frame(self):
        try:
            # 1. Handle Date logic (Same as Data Ingestion)
            order_date_dt = pd.to_datetime(self.order_date)
            ship_date_dt = pd.to_datetime(self.ship_date)

            # 2. Calculate Derived Features
            # Ship Duration
            ship_duration = (ship_date_dt - order_date_dt).days
            
            # Log Sales
            log_sales = np.log1p(self.sales)
            
            # Month Number (1-12)
            month_num = order_date_dt.month
            
            # Weekday Number (Mon=1, Sun=7)
            # Python's weekday() is Mon=0, so we add 1
            weekday_num = order_date_dt.weekday() + 1

            # 3. Create Dictionary with EXACT column names expected by Preprocessor
            custom_data_input_dict = {
                "Sales": [self.sales],
                "Quantity": [self.quantity],
                "Discount": [self.discount],
                "Ship_Duration": [ship_duration],
                "Log_Sales": [log_sales],
                "Month_Num": [month_num],
                "Weekday_Num": [weekday_num],
                "Region": [self.region],
                "Segment": [self.segment],
                "Category": [self.category],
                "Sub_Category": [self.sub_category],
                "Ship_Mode": [self.ship_mode],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)