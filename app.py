from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.Retail_Sale_Intelligent_System.pipelines.prediction_pipeline import CustomData, PredictPipeline
import pickle
import os

app = Flask(__name__)

# Load the model and preprocessor
model = pickle.load(open('artifacts/model.pkl', 'rb'))
preprocessor = pickle.load(open('artifacts/preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # 1. Get data from form
            # REMOVED: Profit input
            order_date = pd.to_datetime(request.form.get('order_date'))
            ship_date = pd.to_datetime(request.form.get('ship_date'))
            sales = float(request.form.get('sales'))
            quantity = int(request.form.get('quantity'))
            discount = float(request.form.get('discount'))
            
            region = request.form.get('region')
            segment = request.form.get('segment')
            category = request.form.get('category')
            sub_category = request.form.get('sub_category')
            ship_mode = request.form.get('ship_mode')

            # 2. Feature Engineering 
            ship_duration = (ship_date - order_date).days
            # REMOVED: Profit Margin calculation
            log_sales = np.log1p(sales)
            
            weekday_num = order_date.weekday() + 1 
            month_num = order_date.month

            # 3. Prepare DataFrame
            # Matches new DataTransformation features
            data_dict = {
                'Sales': [sales],
                'Quantity': [quantity],
                'Discount': [discount],
                'Ship_Duration': [ship_duration],
                'Log_Sales': [log_sales],
                'Month_Num': [month_num],
                'Weekday_Num': [weekday_num],
                'Region': [region],
                'Segment': [segment],
                'Category': [category],
                'Sub_Category': [sub_category],
                'Ship_Mode': [ship_mode]
            }
            
            df = pd.DataFrame(data_dict)
            print("Input Data:\n", df)

            # 4. Transform and Predict
            data_scaled = preprocessor.transform(df)
            prediction = model.predict(data_scaled)
            
            # Prediction Result
            # Classification: 0 or 1
            result_value = prediction[0]
            
            if result_value == 1:
                result_text = "Profitable ✅"
                alert_class = "alert-success"
            else:
                result_text = "Not Profitable ❌"
                alert_class = "alert-danger"

            return render_template('index.html', results=result_text, alert_class=alert_class)

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', results=f"Error: {str(e)}", alert_class="alert-warning")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)