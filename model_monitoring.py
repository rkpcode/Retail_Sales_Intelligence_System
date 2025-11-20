import os
import sys
import pandas as pd
from src.Retail_Sale_Intelligent_System.logger import logging
from src.Retail_Sale_Intelligent_System.exception import CustomException

# --- MAINE TRY/EXCEPT HATA DIYA HAI ---
# Ab agar error aayega, toh Python poora chittha khol dega
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
# --------------------------------------

class ModelMonitoring:
    def __init__(self):
        # Paths - Adjusted for root execution
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.log_data_path = os.path.join("artifacts", "prediction_logs.csv")
        self.report_path = os.path.join("artifacts", "monitoring_report.html")

    def initiate_monitoring(self):
        logging.info("Starting Model Monitoring...")
        try:
            if not os.path.exists(self.log_data_path):
                raise CustomException("No prediction logs found. Run the app and make predictions first.", sys)

            print("Loading Training Data...")
            train_df = pd.read_csv(self.train_data_path)
            
            print("Loading Live Logs...")
            current_df = pd.read_csv(self.log_data_path)
            
            # Columns Mapping
            numerical_features = ['Sales','Quantity','Discount','Ship_Duration','Log_Sales','Month_Num','Weekday_Num']
            categorical_features = ['Region','Segment','Category','Sub_Category','Ship_Mode']
            
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = numerical_features
            column_mapping.categorical_features = categorical_features
            
            logging.info("Generating Data Drift Report...")
            print("Calculating Drift... Please wait.")
            
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=train_df, current_data=current_df, column_mapping=column_mapping)

            report.save_html(self.report_path)
            print(f"\nSUCCESS: Report Generated! Check: {self.report_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    monitor = ModelMonitoring()
    monitor.initiate_monitoring()
