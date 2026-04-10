# Databricks notebook source
# DBTITLE 1,Project Introduction
# MAGIC %md
# MAGIC # 🎯 AI-Powered Telco Customer Retention Analysis
# MAGIC
# MAGIC ## Project Overview
# MAGIC This notebook implements an end-to-end machine learning solution for predicting customer churn in the telecommunications industry, enhanced with AI-powered personalized retention strategies.
# MAGIC
# MAGIC ## Key Features
# MAGIC - **Churn Prediction**: Logistic Regression model to identify at-risk customers
# MAGIC - **AI-Powered Insights**: LLM-generated personalized retention strategies
# MAGIC - **Interactive Reporting**: Dynamic, responsive HTML reports with visualizations
# MAGIC - **Real-time Analysis**: Widget-based customer lookup system
# MAGIC
# MAGIC ## Tech Stack
# MAGIC - **Data Processing**: PySpark
# MAGIC - **ML Framework**: Spark MLlib
# MAGIC - **AI Integration**: Databricks Foundation Models (Meta Llama 3.1)
# MAGIC - **Visualization**: Plotly
# MAGIC - **UI**: Custom HTML/CSS with responsive design
# MAGIC
# MAGIC ## Dataset
# MAGIC Telco Customer Churn dataset from Kaggle containing customer demographics, services, and churn status.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Section 1: Setup
# MAGIC %md
# MAGIC ## 1. Environment Setup
# MAGIC Installing required packages for data processing, ML modeling, visualization, and AI integration.

# COMMAND ----------

# DBTITLE 1,Install Kaggle Integration
# Install kagglehub to download datasets from Kaggle
# pandas-datasets addon provides seamless integration with pandas DataFrames
%pip install kagglehub[pandas-datasets]

# COMMAND ----------

# DBTITLE 1,Install Visualization Libraries
# Install visualization libraries
# - plotly: Interactive charts and graphs
# - kaleido: Static image export for Plotly figures
%pip install plotly kaleido

# COMMAND ----------

# DBTITLE 1,Section 2: Imports
# MAGIC %md
# MAGIC ## 2. Import Libraries
# MAGIC Importing all necessary libraries for data processing, ML, and external integrations.

# COMMAND ----------

# DBTITLE 1,PySpark ML Imports
# PySpark imports for machine learning and data transformation
from pyspark.sql.types import DoubleType, IntegerType  # Data type casting
from pyspark.sql.functions import col  # Column operations
from pyspark.ml.feature import StringIndexer, VectorAssembler  # Feature engineering
from pyspark.ml.classification import LogisticRegression  # ML algorithm
from pyspark.ml import Pipeline  # ML pipeline for chaining transformations

# COMMAND ----------

# DBTITLE 1,External Libraries
# External library imports
import kagglehub  # Kaggle dataset integration
import pandas as pd  # Data manipulation and analysis

# COMMAND ----------

# DBTITLE 1,Section 3: Data Loading
# MAGIC %md
# MAGIC ## 3. Data Loading & Exploration
# MAGIC Download the Telco Customer Churn dataset from Kaggle and perform initial exploration.

# COMMAND ----------

# DBTITLE 1,Download & Load Dataset
# Download Telco Customer Churn dataset from Kaggle
# Dataset contains customer demographics, services, and churn status
dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print(f"Dataset downloaded to {dataset_path}")

# Load CSV data into pandas DataFrame for initial exploration
df = pd.read_csv(f"{dataset_path}/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display dataset dimensions
print(f"\nDataset shape: {df.shape}")

# Preview first 5 records
print(f"\nFirst 5 records:")
display(df.head())

# COMMAND ----------

# DBTITLE 1,Convert to PySpark DataFrame
# Convert pandas DataFrame to PySpark DataFrame for distributed processing
# PySpark enables scalable data processing and ML on large datasets
telcos_df = spark.createDataFrame(df)

# COMMAND ----------

# DBTITLE 1,Preview Data
# Display first 10 records of the PySpark DataFrame
# Visual inspection to understand data structure and content
telcos_df.limit(10).display()

# COMMAND ----------

# DBTITLE 1,Inspect Schema
# Print DataFrame schema to understand column names and data types
# Important for identifying columns that need type casting or transformation
telcos_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Data Cleaning & Type Casting
# Data Cleaning: Select relevant columns and cast TotalCharges to DoubleType
# TotalCharges may contain empty strings that need to be converted to numeric
# Using try_cast() to handle conversion errors gracefully (converts errors to null)
telcos_df = telcos_df.select(
    col("customerID"),           # Unique customer identifier
    col("gender"),               # Male/Female
    col("SeniorCitizen"),        # 0/1 indicator
    col("Partner"),              # Yes/No - has partner
    col("Dependents"),           # Yes/No - has dependents
    col("tenure"),               # Months with company
    col("PhoneService"),         # Yes/No
    col("MultipleLines"),        # Yes/No/No phone service
    col("InternetService"),      # DSL/Fiber optic/No
    col("OnlineSecurity"),       # Yes/No/No internet service
    col("OnlineBackup"),         # Yes/No/No internet service
    col("DeviceProtection"),     # Yes/No/No internet service
    col("TechSupport"),          # Yes/No/No internet service
    col("StreamingTV"),          # Yes/No/No internet service
    col("StreamingMovies"),      # Yes/No/No internet service
    col("Contract"),             # Month-to-month/One year/Two year
    col("PaperlessBilling"),     # Yes/No
    col("PaymentMethod"),        # Electronic check/Mailed check/Bank transfer/Credit card
    col("MonthlyCharges"),       # Monthly amount charged
    col("TotalCharges").try_cast(DoubleType()),  # Total amount charged (cast to double)
    col("Churn")                 # Target variable: Yes/No
)

# COMMAND ----------

# DBTITLE 1,Section 4: ML Model
# MAGIC %md
# MAGIC ## 4. Machine Learning Model Development
# MAGIC Building a churn prediction model using Logistic Regression with PySpark MLlib Pipeline.

# COMMAND ----------

# DBTITLE 1,Initialize Model (First Pass)
# Initialize Logistic Regression model
# - labelCol: Column containing the target variable (churn)
# - featuresCol: Column containing feature vector
lr = LogisticRegression(labelCol="Churn_idx", featuresCol="features")

# COMMAND ----------

# DBTITLE 1,Feature Engineering Pipeline
# STAGE 1: StringIndexer - Convert categorical variables to numeric indices
# Transforms string labels into numeric indices (0, 1, 2, ...)
# Required because ML algorithms work with numeric features
indexer = StringIndexer(
    inputCols=["gender", "InternetService", "Contract", "Churn"],
    outputCols=["gender_idx", "InternetService_idx", "Contract_idx", "Churn_idx"]
)

# STAGE 2: VectorAssembler - Combine multiple columns into a single feature vector
# ML algorithms require features in a single vector column
# handleInvalid="skip" - Skip rows with null values in input columns
assembler = VectorAssembler(
    inputCols=["tenure", "MonthlyCharges", "gender_idx", "InternetService_idx", "Contract_idx"],
    outputCol="features",
    handleInvalid="skip"
)

# COMMAND ----------

# DBTITLE 1,Define Logistic Regression
# STAGE 3: Logistic Regression model
# Binary classification algorithm to predict customer churn
# Outputs prediction (0/1) and probability vector [prob_no_churn, prob_churn]
lr = LogisticRegression(labelCol="Churn_idx", featuresCol="features")

# COMMAND ----------

# DBTITLE 1,Build ML Pipeline
# Create ML Pipeline with 3 stages:
# 1. StringIndexer: Convert categorical to numeric
# 2. VectorAssembler: Combine features into vector
# 3. LogisticRegression: Train classification model
# Pipeline ensures consistent preprocessing for training and prediction
pipeline = Pipeline(stages=[indexer, assembler, lr])

# COMMAND ----------

# DBTITLE 1,Train Model
# Train the model on the entire dataset
# In production, split into train/test sets for validation
# The fitted model can transform new data through all pipeline stages
model = pipeline.fit(telcos_df)

# COMMAND ----------

# DBTITLE 1,Generate Predictions
# Generate predictions on the same dataset
# Adds new columns: features, rawPrediction, probability, prediction
# - probability: [prob_not_churn, prob_churn]
# - prediction: 0 (no churn) or 1 (churn)
predictions = model.transform(telcos_df)

# COMMAND ----------

# DBTITLE 1,Preview Predictions
# Display first 10 predictions to verify model output
# Check columns: prediction, probability, and actual Churn label
predictions.limit(10).display()

# COMMAND ----------

# DBTITLE 1,Section 5: AI Integration
# MAGIC %md
# MAGIC ## 5. AI-Powered Retention Strategy Generation
# MAGIC Integrating Databricks Foundation Models (Meta Llama 3.1) to generate personalized retention strategies for at-risk customers.

# COMMAND ----------

# DBTITLE 1,Type Check
# Verify predictions DataFrame type
# Confirms we're working with PySpark DataFrame for distributed processing
type(predictions)

# COMMAND ----------

# DBTITLE 1,Prompt Engineering Function
def promt_method(customer_profile):
    """
    Creates a structured prompt for the LLM to generate personalized retention strategies.
    
    Args:
        customer_profile: PySpark Row object containing customer data
        
    Returns:
        str: Formatted prompt with customer profile and instructions for the LLM
    """
    # Format customer profile data cleanly (no Row object representation)
    # Extract individual fields and present them in a readable format
    profile_text = f"""
- Gender: {customer_profile['gender']}
- Senior Citizen: {'Yes' if customer_profile['SeniorCitizen'] == 1 else 'No'}
- Partner: {customer_profile['Partner']}
- Dependents: {customer_profile['Dependents']}
- Tenure: {customer_profile['tenure']} months
- Contract: {customer_profile['Contract']}
- Monthly Charges: ${customer_profile['MonthlyCharges']}
- Total Charges: ${customer_profile['TotalCharges'] if customer_profile['TotalCharges'] else 0}
- Internet Service: {customer_profile['InternetService']}
- Phone Service: {customer_profile['PhoneService']}
- Online Security: {customer_profile['OnlineSecurity']}
- Online Backup: {customer_profile['OnlineBackup']}
- Device Protection: {customer_profile['DeviceProtection']}
- Tech Support: {customer_profile['TechSupport']}
- Streaming TV: {customer_profile['StreamingTV']}
- Streaming Movies: {customer_profile['StreamingMovies']}
- Payment Method: {customer_profile['PaymentMethod']}
- Paperless Billing: {customer_profile['PaperlessBilling']}
- Predicted Churn Risk: {'High' if customer_profile['prediction'] == 1.0 else 'Low'}
"""
    
    # Return complete prompt with instructions for the LLM
    return f"""
You are a Telco retention expert. Here is a customer profile:
{profile_text}

Based on this customer profile, suggest personalized retention strategies that focus on the customer's needs and preferences. Ensure your recommendations are customer-centric and minimize financial impact to the company. Provide actionable suggestions that balance customer satisfaction with cost-effectiveness.

Important guidelines:
- Keep response under 100 words
- Start with "Based on Customer Profile"
- Use **bold** for key points (the system will render them with color automatically)
- Be specific and actionable
"""

# COMMAND ----------

# DBTITLE 1,LLM Integration Function
from openai import OpenAI

def generate_retention_strategy(customer_profile):
    """
    Generates AI-powered personalized retention strategy using Databricks Foundation Models.
    
    Args:
        customer_profile: PySpark Row object with customer data
        
    Returns:
        str: AI-generated retention strategy (under 100 words)
    """
    # Get Databricks API token for authentication
    DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    
    # Initialize OpenAI client with Databricks Foundation Model endpoint
    client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url="https://7474649014881769.ai-gateway.cloud.databricks.com/mlflow/v1"
    )
    
    # Create prompt using customer profile
    promt = promt_method(customer_profile)

    # Call Meta Llama 3.1 8B Instruct model via Databricks
    response = client.chat.completions.create(
        model="databricks-meta-llama-3-1-8b-instruct",
        messages=[{"role": "user", "content": promt}],
        max_tokens=3000
    )
    
    # Extract and return the generated strategy text
    return response.choices[0].message.content

# COMMAND ----------

# DBTITLE 1,Identify At-Risk Customers
# Identify at-risk customers (false negatives from business perspective)
# These customers haven't churned yet (Churn="No") but model predicts they will (prediction=1.0)
# These are high-priority targets for retention campaigns
at_risk_customers = predictions.filter(
    (col("Churn") == "No") & (col("prediction") == 1.0)
)

# Extract customer IDs for the first 50 at-risk customers
customer_ids = [row.customerID for row in at_risk_customers.select("customerID").limit(50).collect()]

# Display summary statistics
print(f"Found {at_risk_customers.count()} at-risk customers")
print(f"\nFirst 10 customer IDs: {customer_ids[:10]}")

# COMMAND ----------

# DBTITLE 1,Section 6: Interactive Reporting
# MAGIC %md
# MAGIC ## 6. Interactive Customer Analysis & Reporting
# MAGIC Widget-based system for dynamic customer lookup and comprehensive retention report generation.

# COMMAND ----------

# DBTITLE 1,Display At-Risk Customers
# Display list of at-risk customers for easy selection
# Shows customers who haven't churned yet but are predicted to churn
print("\n AT-Risk CUSTOMERS - Copy a Customer ID to analyze")
print("=" * 80)

# Filter for at-risk customers and select key attributes
at_risk_df = predictions.filter(
    (col("Churn") == "No") & (col("prediction") == 1.0)
).select(
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "probability"  # [prob_no_churn, prob_churn]
).orderBy(col("probability").desc())  # Sort by churn probability (highest risk first)

# Display summary and top 20 highest-risk customers
print(f"\nShowing top 20 highest-risk customers (out of {at_risk_df.count()} total)\n")
at_risk_df.limit(20).display()

# COMMAND ----------

# DBTITLE 1,Simple Customer Lookup (Legacy)
from pyspark.sql.functions import col

# Create a text widget for customer ID input
# Widget appears at the top of the notebook for easy access
dbutils.widgets.text("customer_id", "", "Customer ID")
selected_customer_id = dbutils.widgets.get("customer_id")

print(f"Analyzing customer: {selected_customer_id}\n")

# Fetch customer data from predictions DataFrame
sample_customer = predictions.filter((col("customerID") == selected_customer_id)).collect()

if sample_customer:
    sample_customer = sample_customer[0]  # Extract first row from list
    
    # Display customer details
    print(f"Customer Status:")
    print(f"    - Actual Churn: {sample_customer['Churn']}")
    print(f"    - Predicted Churn: {sample_customer['prediction']}")
    print(f"    - Tenure: {sample_customer['tenure']} months")
    print(f"    - Monthly Charges: ${sample_customer['MonthlyCharges']}")
    print(f"    - Contract: {sample_customer['Contract']}\n")

    # Generate AI-powered retention strategy
    print("=" * 60)
    print("PERSONALIZED RETENTION STRATEGY")
    print("=" * 60)
    strategy = generate_retention_strategy(sample_customer)
    print(strategy)
else:
    print(f"Customer ID '{selected_customer_id}' not found in the dataset.")
    print("\nTip: Run the cell above to see available at-risk customer IDs")

# COMMAND ----------

# DBTITLE 1,Interactive Customer Retention Intelligence Report
import plotly.graph_objects as go
import plotly.express as px
from pyspark.sql.functions import col, avg
from IPython.display import display, HTML
import numpy as np
import re

def convert_markdown_to_html(text):
    """
    Convert markdown formatting to HTML for better rendering in reports.
    Applies color styling to bold text for visual emphasis.
    
    Args:
        text: Markdown-formatted string
        
    Returns:
        str: HTML-formatted string with styled elements
    """
    # Convert **bold** to <strong> with color
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #667eea;">\1</strong>', text)
    
    # Convert *italic* to <em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert bullet points to HTML lists with styling
    lines = text.split('\n')
    in_list = False
    result = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                result.append('<ul style="margin: 10px 0; padding-left: 25px;">')
                in_list = True
            result.append(f'<li style="margin: 5px 0;">{stripped[2:]}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            if stripped:
                result.append(f'<p style="margin: 10px 0;">{stripped}</p>')
    
    if in_list:
        result.append('</ul>')
    
    return ''.join(result)

def generate_customer_retention_report(customer_id):
    """
    Generate a comprehensive, interactive HTML report for customer retention analysis.
    
    Features:
    - Responsive design with modern UI/UX
    - Interactive Plotly visualizations (Risk Score, Spending, Service Utilization)
    - AI-generated personalized retention strategy
    - Customer profile dashboard
    
    Args:
        customer_id: Customer ID to analyze
        
    Returns:
        None (displays HTML report in notebook output)
    """
    # Fetch customer data from predictions
    customer_data = predictions.filter(col("customerID") == customer_id).collect()
    
    if not customer_data:
        # Display error message if customer not found
        error_html = f"""
        <div style="background: #f8d7da; border-left: 5px solid #dc3545; padding: 20px; 
                    border-radius: 10px; margin: 20px 0;">
            <h3 style="color: #721c24; margin: 0 0 10px 0;">❌ Customer Not Found</h3>
            <p style="margin: 0; color: #721c24;">
                Customer ID '{customer_id}' not found in the dataset.
                <br><br>
                <strong>Tip:</strong> Check Cell 21 for available at-risk customer IDs.
            </p>
        </div>
        """
        display(HTML(error_html))
        return
    
    customer = customer_data[0]
    
    # Calculate dataset averages for comparison
    avg_stats = predictions.agg(
        avg("tenure").alias("avg_tenure"),
        avg("MonthlyCharges").alias("avg_monthly"),
        avg("TotalCharges").alias("avg_total")
    ).collect()[0]
    
    # Extract churn probability from model output
    churn_prob = float(customer['probability'][1]) * 100  # Convert to percentage
    
    # Determine risk level and color coding
    if churn_prob < 40:
        risk_level = "Low Risk"
        risk_color = "#28a745"  # Green
        risk_bg = "#d4edda"
    elif churn_prob < 70:
        risk_level = "Medium Risk"
        risk_color = "#ffc107"  # Yellow
        risk_bg = "#fff3cd"
    else:
        risk_level = "High Risk"
        risk_color = "#dc3545"  # Red
        risk_bg = "#f8d7da"
    
    # === VISUALIZATION 1: Churn Risk Gauge with Delta ===
    gauge_color = '#dc3545' if churn_prob > 70 else '#ffc107' if churn_prob > 40 else '#28a745'
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_prob,
        title={'text': "Churn Risk Score", 'font': {'size': 18, 'color': '#2c3e50'}},
        number={'suffix': '%', 'font': {'size': 45, 'color': gauge_color}},
        delta={'reference': 50, 'font': {'size': 16}},  # Compare to 50% threshold
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#bdc3c7"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ecf0f1",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},    # Low risk - green
                {'range': [40, 70], 'color': '#fff3cd'},   # Medium risk - yellow
                {'range': [70, 100], 'color': '#f8d7da'}   # High risk - red
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 4},
                'thickness': 0.8,
                'value': 60  # Warning threshold marker
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white',
        font={'family': "'Inter', 'Segoe UI', Roboto, sans-serif"}
    )
    
    gauge_html = fig_gauge.to_html(include_plotlyjs='cdn', div_id='gauge-chart', config={'responsive': True, 'displayModeBar': False})
    
    # === VISUALIZATION 2: Spending Analysis ===
    fig_spending = go.Figure()
    
    # Customer spending bars
    fig_spending.add_trace(go.Bar(
        name='Customer',
        x=['Monthly Charges', 'Total Charges'],
        y=[customer['MonthlyCharges'], customer['TotalCharges'] if customer['TotalCharges'] else 0],
        marker_color='#667eea',
        text=[f"${customer['MonthlyCharges']:.2f}", f"${customer['TotalCharges'] if customer['TotalCharges'] else 0:.2f}"],
        textposition='outside',
        textfont=dict(size=13, color='#2c3e50', weight='bold')
    ))
    
    # Average spending bars
    fig_spending.add_trace(go.Bar(
        name='Average',
        x=['Monthly Charges', 'Total Charges'],
        y=[avg_stats['avg_monthly'], avg_stats['avg_total'] if avg_stats['avg_total'] else 0],
        marker_color='#f093fb',
        text=[f"${avg_stats['avg_monthly']:.2f}", f"${avg_stats['avg_total'] if avg_stats['avg_total'] else 0:.2f}"],
        textposition='outside',
        textfont=dict(size=13, color='#7f8c8d')
    ))
    
    fig_spending.update_layout(
        title={'text': 'Customer Spending Analysis', 'font': {'size': 18, 'color': '#2c3e50'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='',
        yaxis_title='Amount ($)',
        height=320,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        margin=dict(l=40, r=40, t=50, b=40),
        bargap=0.3,
        font={'family': "'Inter', 'Segoe UI', Roboto, sans-serif"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_spending.update_xaxes(showgrid=False)
    fig_spending.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    
    spending_html = fig_spending.to_html(include_plotlyjs=False, div_id='spending-chart', config={'responsive': True, 'displayModeBar': False})
    
    # === VISUALIZATION 3: Service Utilization Profile (Polar Chart) ===
    services = {
        'Phone': 1 if customer['PhoneService'] == 'Yes' else 0,
        'Multiple Lines': 1 if customer['MultipleLines'] == 'Yes' else 0,
        'Internet': 1 if customer['InternetService'] != 'No' else 0,
        'Security': 1 if customer['OnlineSecurity'] == 'Yes' else 0,
        'Backup': 1 if customer['OnlineBackup'] == 'Yes' else 0,
        'Protection': 1 if customer['DeviceProtection'] == 'Yes' else 0,
        'Tech Support': 1 if customer['TechSupport'] == 'Yes' else 0,
        'Streaming TV': 1 if customer['StreamingTV'] == 'Yes' else 0,
        'Movies': 1 if customer['StreamingMovies'] == 'Yes' else 0
    }
    
    fig_services = go.Figure()
    
    fig_services.add_trace(go.Barpolar(
        r=list(services.values()),
        theta=list(services.keys()),
        marker=dict(color='#667eea', line=dict(color='white', width=2)),
        opacity=0.85,
        hovertemplate='<b>%{theta}</b><br>Status: %{r}<extra></extra>'
    ))
    
    fig_services.update_layout(
        title={'text': 'Service Utilization Profile', 'font': {'size': 20, 'color': '#2c3e50'}, 'x': 0.5, 'xanchor': 'center'},
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 1], ticktext=['No', 'Yes']),
            bgcolor='#f8f9fa',
            angularaxis=dict(tickfont=dict(size=11, color='#2c3e50'))
        ),
        height=380,
        paper_bgcolor='white',
        margin=dict(l=70, r=70, t=70, b=30),
        font={'family': "'Inter', 'Segoe UI', Roboto, sans-serif"}
    )
    
    services_html = fig_services.to_html(include_plotlyjs=False, div_id='services-chart', config={'responsive': True, 'displayModeBar': False})
    
    # Generate AI-powered retention strategy
    strategy = generate_retention_strategy(customer)
    strategy_html_content = convert_markdown_to_html(strategy)
    
    # === COMPLETE REPORT HTML ===
    # Combines all sections into a single responsive HTML output
    complete_report = f"""
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        .report-container {{
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f6fa;
        }}
        
        .header-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 25px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }}
        
        .main-title {{
            font-size: 36px;
            font-weight: 700;
            margin: 0 0 8px 0;
            letter-spacing: -0.5px;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        
        .subtitle {{
            font-size: 16px;
            opacity: 0.9;
            margin: 0;
            font-weight: 400;
        }}
        
        .customer-section {{
            background: white;
            border-radius: 16px;
            padding: 20px 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .customer-header {{
            color: #2c3e50;
            font-size: 20px;
            font-weight: 600;
            margin: 0;
            flex: 1;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        
        .risk-badge {{
            background: {risk_bg};
            color: {risk_color};
            padding: 10px 20px;
            border-radius: 30px;
            font-size: 16px;
            font-weight: 700;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 2px solid {risk_color};
        }}
        
        .profile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }}
        
        .profile-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 12px 15px;
            border-radius: 10px;
            border-left: 3px solid #667eea;
            transition: all 0.2s;
        }}
        
        .profile-card:hover {{
            transform: translateX(3px);
            border-left-color: #764ba2;
            box-shadow: 0 3px 10px rgba(102, 126, 234, 0.2);
        }}
        
        .profile-label {{
            color: #667eea;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .profile-value {{
            color: #2c3e50;
            font-size: 15px;
            font-weight: 500;
        }}
        
        .viz-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .viz-card {{
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .viz-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}
        
        .viz-card-full {{
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .viz-card-full:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}
        
        .strategy-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 30px;
            border-radius: 16px;
            color: white;
            box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
        }}
        
        .strategy-title {{
            margin: 0 0 20px 0;
            font-size: 26px;
            font-weight: 600;
            text-align: center;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        
        .strategy-content {{
            background: rgba(255,255,255,0.98);
            padding: 25px;
            border-radius: 12px;
            color: #2c3e50;
            font-size: 16px;
            line-height: 1.8;
            font-weight: 400;
        }}
        
        .strategy-content strong {{
            color: #667eea;
            font-weight: 700;
        }}
        
        @media (max-width: 768px) {{
            .main-title {{
                font-size: 26px;
            }}
            
            .subtitle {{
                font-size: 14px;
            }}
            
            .customer-section {{
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .profile-grid {{
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }}
            
            .viz-row {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    
    <div class="report-container">
        <!-- Header with Title (NO RISK BADGE) -->
        <div class="header-section">
            <h1 class="main-title">🎯 Customer Retention Intelligence Report</h1>
            <p class="subtitle">AI-Powered Analysis & Retention Strategy</p>
        </div>
        
        <!-- Customer Profile Section with Risk Badge on Right -->
        <div class="customer-section">
            <h2 class="customer-header">Customer: {customer_id}</h2>
            <div class="risk-badge">
                {risk_level}
            </div>
        </div>
        
        <div style="background: white; border-radius: 16px; padding: 20px 25px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div class="profile-grid">
                <div class="profile-card">
                    <div class="profile-label">Gender</div>
                    <div class="profile-value">{customer['gender']}</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Tenure</div>
                    <div class="profile-value">{customer['tenure']} months</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Contract Type</div>
                    <div class="profile-value">{customer['Contract']}</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Monthly Charges</div>
                    <div class="profile-value">${customer['MonthlyCharges']:.2f}</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Total Charges</div>
                    <div class="profile-value">${customer['TotalCharges'] if customer['TotalCharges'] else 0:.2f}</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Internet Service</div>
                    <div class="profile-value">{customer['InternetService']}</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Payment Method</div>
                    <div class="profile-value">{customer['PaymentMethod']}</div>
                </div>
                <div class="profile-card">
                    <div class="profile-label">Actual Churn</div>
                    <div class="profile-value">{customer['Churn']}</div>
                </div>
            </div>
        </div>
        
        <!-- Visualizations Row: Risk Score + Spending Analysis -->
        <div class="viz-row">
            <div class="viz-card">
                {gauge_html}
            </div>
            <div class="viz-card">
                {spending_html}
            </div>
        </div>
        
        <!-- Service Utilization (Full Width) -->
        <div class="viz-card-full">
            {services_html}
        </div>
        
        <!-- Personalized Retention Strategy -->
        <div class="strategy-card">
            <h2 class="strategy-title">💡 Personalized Retention Strategy</h2>
            <div class="strategy-content">
                {strategy_html_content}
            </div>
        </div>
    </div>
    """
    
    # Display complete report as single HTML output
    display(HTML(complete_report))



# COMMAND ----------

# DBTITLE 1,Generate Report (Dynamic)
# Get customer ID from widget (dynamic input)
selected_customer_id = dbutils.widgets.get("customer_id")

if selected_customer_id:
    # Generate and display interactive report
    generate_customer_retention_report(selected_customer_id)
else:
    # Show warning if no customer ID provided
    display(HTML("""
    <div style="background: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; 
                border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #856404; margin: 0 0 10px 0;">⚠️ No Customer ID Selected</h3>
        <p style="margin: 0; color: #856404;">
            Please enter a customer ID in the widget above, then run this cell again.
            <br><br>
            <strong>Tip:</strong> Check Cell 21 for available at-risk customer IDs.
        </p>
    </div>
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC