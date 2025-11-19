import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.Retail_Sale_Intelligent_System.utils import save_object
from src.Retail_Sale_Intelligent_System.exception import CustomException
from src.Retail_Sale_Intelligent_System.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            # define features
            # REMOVED: 'Profit Margin (%)' from numerical list
            numerical_features = ['Sales','Quantity','Discount','Ship_Duration','Log_Sales','Month_Num','Weekday_Num']
            
            categorical_features = ['Region','Segment','Category','Sub_Category','Ship_Mode']
            
            num_transformer=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())
            ])
            cat_transformer=Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])             

            logging.info(f"Categorical Columns:{categorical_features}")
            logging.info(f"Numerical Columns:{numerical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num",num_transformer,numerical_features),
                    ("cat",cat_transformer,categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transormation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = 'Profitable'

            ## divide the train dataset to independent and dependent feature
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)