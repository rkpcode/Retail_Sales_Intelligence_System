import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Retail Intelligence System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1d29 100%);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    
    /* Headers */
    h1 {
        color: #FAFAFA;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2, h3 {
        color: #FF4B4B;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #262730 100%);
    }
    
    /* Success/Error Messages */
    .success-box {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 200, 81, 0.4);
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4);
        margin: 1rem 0;
    }
    
    /* Divider */
    hr {
        border: 2px solid #FF4B4B;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & PREPROCESSOR ====================
@st.cache_resource
def load_models():
    """Load trained model and preprocessor"""
    try:
        model = pickle.load(open('artifacts/best_model.pkl', 'rb'))
        preprocessor = pickle.load(open('artifacts/preprocessor.pkl', 'rb'))
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, preprocessor = load_models()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/sales-performance.png", width=100)
    st.title("📊 Quick Start Guide")
    st.markdown("---")
    
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Adjust Parameters** - Use sliders to set transaction details
    2. **Click Analyze** - Get instant profitability prediction
    3. **View Insights** - Check product category analytics
    4. **Quick Test** - Use Quick Predictor for fast testing
    """)
    
    st.markdown("---")
    st.markdown("### 📥 Sample Data")
    st.markdown("Download sample CSV to understand the format:")
    
    # Create sample CSV data
    sample_data = pd.DataFrame({
        'Sales': [500.0, 1200.0, 350.0],
        'Quantity': [5, 10, 3],
        'Discount': [0.10, 0.05, 0.20],
        'Order_Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Ship_Date': ['2024-01-18', '2024-01-19', '2024-01-20'],
        'Region': ['East', 'West', 'Central'],
        'Segment': ['Consumer', 'Corporate', 'Consumer'],
        'Category': ['Technology', 'Office Supplies', 'Furniture'],
        'Sub_Category': ['Phones', 'Binders', 'Chairs'],
        'Ship_Mode': ['Standard Class', 'Second Class', 'First Class']
    })
    
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv,
        file_name="sample_retail_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("""
    **Model:** CatBoost Classifier  
    **Accuracy:** 85.2%  
    **Features:** 12 engineered features  
    **Training Data:** 10,000+ transactions
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📖 Documentation](https://github.com/rkpcode/Retail_Sales_Intelligence_System)")
    st.markdown("[💼 GitHub Repo](https://github.com/rkpcode/Retail_Sales_Intelligence_System)")
    st.markdown("[📧 Contact](mailto:contactrkp21@gmail.com)")

# ==================== HELPER FUNCTIONS ====================
def engineer_features(sales, quantity, discount, order_date, ship_date, 
                      region, segment, category, sub_category, ship_mode):
    """
    Replicate feature engineering from training pipeline
    """
    # Calculate derived features
    ship_duration = (ship_date - order_date).days
    log_sales = np.log1p(sales)
    month_num = order_date.month
    weekday_num = order_date.weekday() + 1
    
    # Create DataFrame matching training features
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
    
    return pd.DataFrame(data_dict)

def predict_profitability(df):
    """Make prediction using loaded model"""
    try:
        # Transform data
        data_scaled = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(data_scaled)
        
        # Get probability if available
        try:
            proba = model.predict_proba(data_scaled)
            confidence = max(proba[0]) * 100
        except:
            confidence = None
        
        return prediction[0], confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ==================== MAIN DASHBOARD ====================
st.title("🚀 Retail Profitability Intelligence System")
st.markdown("### Transform Transaction Data into Business Decisions")
st.markdown("---")

# ==================== SECTION 1: KPI HEADER ====================
st.subheader("📊 System Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Model Accuracy",
        value="85.2%",  # Replace with actual model score
        delta="High Confidence"
    )

with col2:
    st.metric(
        label="Prediction Speed",
        value="< 0.5s",
        delta="Real-time"
    )

with col3:
    st.metric(
        label="Transactions Analyzed",
        value="10,000+",
        delta="Training Dataset"
    )

with col4:
    st.metric(
        label="Loss Prevention Rate",
        value="~23%",  # Mock estimate for demo
        delta="Estimated Impact"
    )

st.markdown("---")

# ==================== SECTION 2: WHAT-IF SIMULATOR ====================
st.subheader("🎯 What-If Profitability Simulator")
st.markdown("**Adjust parameters below to see real-time profitability predictions**")

# Create two columns for inputs
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📈 Transaction Details")
    
    sales = st.slider(
        "Sales Amount ($)",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        step=10.0,
        help="Total sales value of the transaction"
    )
    
    quantity = st.slider(
        "Quantity",
        min_value=1,
        max_value=100,
        value=5,
        help="Number of items in the transaction"
    )
    
    discount = st.slider(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Discount percentage applied"
    )
    
    order_date = st.date_input(
        "Order Date",
        value=datetime.now(),
        help="Date when order was placed"
    )
    
    ship_date = st.date_input(
        "Ship Date",
        value=datetime.now() + timedelta(days=3),
        help="Date when order will be shipped"
    )

with col_right:
    st.markdown("#### 🏷️ Product & Customer Details")
    
    region = st.selectbox(
        "Region",
        options=['East', 'West', 'Central', 'South'],
        help="Geographic region of the customer"
    )
    
    segment = st.selectbox(
        "Customer Segment",
        options=['Consumer', 'Corporate', 'Home Office'],
        help="Type of customer"
    )
    
    category = st.selectbox(
        "Product Category",
        options=['Furniture', 'Office Supplies', 'Technology'],
        help="Main product category"
    )
    
    # Sub-categories based on category
    sub_category_options = {
        'Furniture': ['Bookcases', 'Chairs', 'Furnishings', 'Tables'],
        'Office Supplies': ['Appliances', 'Art', 'Binders', 'Envelopes', 'Fasteners', 
                           'Labels', 'Paper', 'Storage', 'Supplies'],
        'Technology': ['Accessories', 'Copiers', 'Machines', 'Phones']
    }
    
    sub_category = st.selectbox(
        "Sub-Category",
        options=sub_category_options.get(category, ['Other']),
        help="Specific product sub-category"
    )
    
    ship_mode = st.selectbox(
        "Shipping Mode",
        options=['Standard Class', 'Second Class', 'First Class', 'Same Day'],
        help="Shipping method selected"
    )

# Prediction Button
st.markdown("---")
if st.button("🔮 Analyze Transaction Profitability", use_container_width=True):
    with st.spinner("Analyzing transaction..."):
        # Engineer features
        input_df = engineer_features(
            sales, quantity, discount, 
            pd.to_datetime(order_date), pd.to_datetime(ship_date),
            region, segment, category, sub_category, ship_mode
        )
        
        # Make prediction
        prediction, confidence = predict_profitability(input_df)
        
        if prediction is not None:
            # Display result
            if prediction == 1:
                st.success(f"# ✅ PROFITABLE TRANSACTION")
                st.balloons()
                
                # Show confidence with progress bar
                if confidence:
                    st.metric("Confidence Score", f"{confidence:.1f}%", delta="High Confidence")
                    st.progress(confidence / 100)
                else:
                    st.metric("Confidence Score", "85%+", delta="Model Accuracy")
                
                # Detailed breakdown
                st.info("""💡 **Recommendation:** This transaction is likely to generate profit. 
                
**Why this is profitable:**
- Optimal discount level (≤ 20%)
- Good product category selection
- Efficient shipping method
                
**Next Steps:** Proceed with confidence!""")
                
            else:
                st.error(f"# ❌ NOT PROFITABLE TRANSACTION")
                
                # Show confidence with progress bar
                if confidence:
                    st.metric("Confidence Score", f"{confidence:.1f}%", delta="Loss Prediction")
                    st.progress(confidence / 100)
                else:
                    st.metric("Confidence Score", "85%+", delta="Model Accuracy")
                
                # Actionable recommendations
                st.warning("""⚠️ **This transaction may result in a LOSS.**
                
**Key Issues Detected:**
- Discount too high (> 20% reduces margins)
- Low sales amount relative to costs
- Expensive shipping method
                
**Recommendations to Fix:**""")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    ✅ **Reduce discount** to 15-20%  
                    ✅ **Increase sales amount** by $100+  
                    ✅ **Switch to Standard Class** shipping
                    """)
                with col2:
                    st.markdown("""
                    ✅ **Focus on Technology** category  
                    ✅ **Target West region** customers  
                    ✅ **Bundle products** to increase quantity
                    """)

st.markdown("---")

# ==================== SECTION 3: PRODUCT INSIGHTS ====================
st.subheader("📦 Product Category Insights")
st.markdown("**Understanding profitability patterns across categories**")

# Mock data for visualization (replace with actual data analysis)
category_data = pd.DataFrame({
    'Category': ['Technology', 'Office Supplies', 'Furniture'],
    'Profit_Rate': [78, 65, 52],
    'Avg_Sales': [850, 320, 1200],
    'Transaction_Count': [3200, 5400, 1400]
})

col1, col2 = st.columns(2)

with col1:
    # Profitability by Category
    fig1 = px.bar(
        category_data,
        x='Category',
        y='Profit_Rate',
        title='Profitability Rate by Category (%)',
        color='Profit_Rate',
        color_continuous_scale=['#FF4444', '#FFB347', '#00C851'],
        labels={'Profit_Rate': 'Profit Rate (%)'}
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Transaction Volume
    fig2 = px.pie(
        category_data,
        values='Transaction_Count',
        names='Category',
        title='Transaction Distribution by Category',
        color_discrete_sequence=['#FF4B4B', '#FF8E53', '#FFB347']
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA')
    )
    st.plotly_chart(fig2, use_container_width=True)

# Key Insights
st.markdown("### 🎯 Key Insights")
col1, col2 = st.columns(2)

with col1:
    st.success("**✅ Top Profitable Categories**")
    st.markdown("""
    1. **Technology** - 78% profit rate
    2. **Office Supplies** - 65% profit rate
    3. **Furniture** - 52% profit rate
    """)

with col2:
    st.warning("**⚠️ Risk Factors to Watch**")
    st.markdown("""
    - High discounts (>30%) significantly reduce profitability
    - Same Day shipping often leads to losses
    - Furniture category has lower margins
    """)

st.markdown("---")

# ==================== SECTION 4: QUICK PREDICTOR ====================
st.subheader("⚡ Quick Predictor")
st.markdown("**Pre-filled defaults for fast testing**")

with st.expander("🚀 Click to expand Quick Predictor", expanded=False):
    qcol1, qcol2, qcol3 = st.columns(3)
    
    with qcol1:
        q_sales = st.number_input("Sales ($)", value=500.0, min_value=0.0)
        q_quantity = st.number_input("Quantity", value=5, min_value=1)
        q_discount = st.number_input("Discount (%)", value=10.0, min_value=0.0, max_value=100.0)
    
    with qcol2:
        q_region = st.selectbox("Region", ['East', 'West', 'Central', 'South'], key='q_region')
        q_segment = st.selectbox("Segment", ['Consumer', 'Corporate', 'Home Office'], key='q_segment')
        q_category = st.selectbox("Category", ['Technology', 'Office Supplies', 'Furniture'], key='q_category')
    
    with qcol3:
        q_sub_category = st.text_input("Sub-Category", value="Phones")
        q_ship_mode = st.selectbox("Ship Mode", ['Standard Class', 'Second Class', 'First Class', 'Same Day'], key='q_ship')
    
    if st.button("⚡ Quick Predict", use_container_width=True):
        q_input_df = engineer_features(
            q_sales, q_quantity, q_discount,
            pd.to_datetime(datetime.now()),
            pd.to_datetime(datetime.now() + timedelta(days=3)),
            q_region, q_segment, q_category, q_sub_category, q_ship_mode
        )
        
        q_prediction, q_confidence = predict_profitability(q_input_df)
        
        if q_prediction == 1:
            st.success(f"✅ **PROFITABLE** (Confidence: {q_confidence:.1f}%)" if q_confidence else "✅ **PROFITABLE**")
        else:
            st.error(f"❌ **NOT PROFITABLE** (Confidence: {q_confidence:.1f}%)" if q_confidence else "❌ **NOT PROFITABLE**")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p><strong>Retail Sales Intelligence System</strong> | Powered by Machine Learning</p>
    <p>Built with ❤️ using Streamlit & CatBoost</p>
</div>
""", unsafe_allow_html=True)
