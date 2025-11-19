import os
import sys
import pandas as pd
from src.Retail_Sale_Intelligent_System.logger import logging
from src.Retail_Sale_Intelligent_System.exception import CustomException

# Evidently Imports (Industry Standard for Monitoring)
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently import ColumnMapping
except ImportError:
    print("Evidently library not installed. Please run: pip install evidently")
    sys.exit(1)

class ModelMonitoring:
    def __init__(self):
        # Paths
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.log_data_path = os.path.join("artifacts", "prediction_logs.csv")
        self.report_path = os.path.join("artifacts", "monitoring_report.html")

    def initiate_monitoring(self):
        logging.info("Starting Model Monitoring...")
        try:
            # 1. Check if logs exist
            if not os.path.exists(self.log_data_path):
                raise CustomException("No prediction logs found. Run the app and make predictions first.", sys)

            # 2. Load Data
            # Reference Data (Training Data)
            train_df = pd.read_csv(self.train_data_path)
            
            # Current Data (Production Logs)
            current_df = pd.read_csv(self.log_data_path)
            
            print(f"Training Data Shape: {train_df.shape}")
            print(f"Current Logged Data Shape: {current_df.shape}")

            # 3. Define Column Mapping
            # We need to tell Evidently which columns are what
            target = 'Profitable'
            prediction = 'Prediction'
            
            # Common columns between Train and Logs (excluding timestamps etc)
            numerical_features = ['Sales','Quantity','Discount','Ship_Duration','Log_Sales','Month_Num','Weekday_Num']
            categorical_features = ['Region','Segment','Category','Sub_Category','Ship_Mode']
            
            # Filter columns to ensure consistency
            # Note: Logs might not have the 'Profitable' target (Ground Truth), so we might skip TargetDrift
            # But we definitely check DataDrift (Features)
            
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = numerical_features
            column_mapping.categorical_features = categorical_features
            
            # If logs have Ground Truth (e.g. we updated them later), we can track Target Drift
            # For now, we focus on Data Drift (Input changes)
            
            # 4. Create Report
            logging.info("Generating Drift Report...")
            
            # Using DataDriftPreset to check feature distribution changes
            report = Report(metrics=[
                DataDriftPreset(), 
            ])

            report.run(reference_data=train_df, current_data=current_df, column_mapping=column_mapping)

            # 5. Save Report
            report.save_html(self.report_path)
            logging.info(f"Monitoring Report Saved at: {self.report_path}")
            print(f"\nReport Generated Successfully! Open this file in browser: {self.report_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    monitor = ModelMonitoring()
    monitor.initiate_monitoring()