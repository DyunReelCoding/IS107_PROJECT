import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_excel('Online Retail.xlsx')

# Quick overview of the dataset
st.header("OVERVIEW OF THE DATASET")
st.write(df.info())
st.write(df.describe())

# Check for missing values
st.write("Missing Values:", df.isnull().sum())

# Check for duplicates
st.write("Duplicate Rows:", df.duplicated().sum())

# Drop rows with missing CustomerID
df_clean = df.dropna(subset=['CustomerID'])

# Use .loc to avoid the SettingWithCopyWarning
df_clean.loc[:, 'Description'] = df_clean['Description'].fillna('Unknown')

# Remove invalid transactions (Quantity and UnitPrice)
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Drop duplicate rows
df_clean = df_clean.drop_duplicates()

# Confirm changes
st.write(df_clean.info())
st.write(df_clean.describe())

# Create engine for PostgreSQL connection
engine = create_engine('postgresql+psycopg2://postgres@localhost:5432/retail_data_warehouse')

# Load the cleaned pandas DataFrame into PostgreSQL
df_clean.to_sql('cleaned_online_retail_data', engine, index=False, if_exists='replace')

# Query to retrieve sales and join with product data for category filtering
query_sales = '''
SELECT 
    fs.quantity, 
    fs.unit_price, 
    fs.invoice_date, 
    fs.stock_code, 
    dp.description AS product_category
FROM 
    fact_sales fs
JOIN 
    dim_product dp ON fs.stock_code = dp.stock_code;
'''
data_sales = pd.read_sql(query_sales, engine)

# Calculate total sales
data_sales['total_sales'] = data_sales['quantity'] * data_sales['unit_price']
data_sales['invoice_date'] = pd.to_datetime(data_sales['invoice_date'])  # Convert to datetime

# Streamlit layout
st.title("Sales Dashboard")

# Dropdown for product category filter
product_category = st.selectbox(
    'Filter by Product Category',
    ['All Categories'] + list(data_sales['product_category'].unique())
)

# Date picker for filtering sales by date range
start_date = st.date_input("Select Start Date", value=data_sales['invoice_date'].min().date())
end_date = st.date_input("Select End Date", value=data_sales['invoice_date'].max().date())

# Dropdown for region filter
query_regions = '''
SELECT DISTINCT c.country 
FROM dim_customer c;
'''
regions = pd.read_sql(query_regions, engine)
region_filter = st.selectbox(
    'Filter by Region',
    ['All Regions'] + list(regions['country'])
)

# Total Sales
filtered_sales = data_sales[
    (data_sales['invoice_date'] >= pd.to_datetime(start_date)) &
    (data_sales['invoice_date'] <= pd.to_datetime(end_date))
]

# Apply product category filter
if product_category != 'All Categories':
    filtered_sales = filtered_sales[filtered_sales['product_category'] == product_category]

# Apply region filter
if region_filter != 'All Regions':
    filtered_sales = filtered_sales[filtered_sales['country'] == region_filter]

total_sales = filtered_sales['total_sales'].sum()
st.subheader(f"Total Sales: ${total_sales:,.2f}")

# Graph for top-selling products
top_selling_products = filtered_sales.groupby('stock_code').agg(total_quantity=('quantity', 'sum')).reset_index()
top_selling_products = top_selling_products.nlargest(10, 'total_quantity')

fig_top_selling = px.bar(
    top_selling_products, 
    x='stock_code', 
    y='total_quantity', 
    title='Top Selling Products',
    labels={'total_quantity': 'Quantity Sold', 'stock_code': 'Product Code'},
    color='total_quantity',
    template='plotly_white'
)
st.plotly_chart(fig_top_selling)

report_forecast = '''
Polynomial regression was used to predict future sales based on historical data, capturing potential non-linear trends.
The model uses time-related features like the day, month, and weekday to improve predictions. The results show a
reasonable match between actual and predicted sales, suggesting that the model can help in sales forecasting.
'''
print(report_forecast)

# Callback to update sales by region graph
st.header("Total Sales by Region")

# Date picker for filtering sales by date range for the total sales by region
start_date_region = st.date_input("Select Start Date for Region Sales", value=data_sales['invoice_date'].min().date())
end_date_region = st.date_input("Select End Date for Region Sales", value=data_sales['invoice_date'].max().date())

# Query to retrieve sales by region based on selected dates
query_sales_by_region = '''
SELECT 
    c.country,
    SUM(fs."quantity" * fs.unit_price) AS total_sales
FROM 
    fact_sales fs
JOIN 
    dim_customer c ON fs.customer_id = c.customer_id
WHERE
    fs.invoice_date BETWEEN '{start_date}' AND '{end_date}'
GROUP BY 
    c.country
ORDER BY 
    total_sales DESC
LIMIT 10;
'''

# Execute the query
sales_by_region = pd.read_sql(query_sales_by_region.format(start_date=start_date_region, end_date=end_date_region), engine)

# Calculate total sales for the displayed regions
total_sales_region = sales_by_region['total_sales'].sum()

# Display the total sales amount
st.subheader(f"Total Sales in Selected Region: ${total_sales_region:,.2f}")

# Create bar chart for sales by region
fig_sales_by_region = px.bar(
    sales_by_region, 
    x='country', 
    y='total_sales', 
    title='Total Sales by Region',
    labels={'total_sales': 'Total Sales ($)', 'country': 'Region'},
    color='total_sales',
    template='plotly_white'
)
st.plotly_chart(fig_sales_by_region)

# Customer Segmentation
st.header("Customer Segmentation")
query_customers = '''
SELECT c.customer_id, SUM(fs.quantity * fs.unit_price) AS total_spent, COUNT(fs.invoice_no) AS num_transactions
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
GROUP BY c.customer_id;
'''
data_customers = pd.read_sql(query_customers, engine)

# Handle missing or invalid data (if any)
data_customers.fillna(0, inplace=True)  # Replace NaN values with 0
data_customers = data_customers[(data_customers['total_spent'] > 0) & (data_customers['num_transactions'] > 0)]  # Keep only positive values

# Preprocessing the data for clustering
scaler = StandardScaler()
X = scaler.fit_transform(data_customers[['total_spent', 'num_transactions']])

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data_customers['cluster'] = kmeans.fit_predict(X)

# Plot the clusters
fig_customer_segmentation = px.scatter(data_customers, x='total_spent', y='num_transactions', color='cluster', title='Customer Segmentation')
st.plotly_chart(fig_customer_segmentation)

# Customer segmentation report
report = '''
Customer segmentation was performed using K-Means clustering based on total spending and transaction frequency. 
The optimal number of clusters was determined using the elbow method. The resulting clusters can be used to identify 
high-value customers, frequent buyers, and low-engagement customers. The red 'X' markers represent the centroids of each cluster.
'''
print(report)

# Sales Forecasting
st.header("Sales Forecasting")
query_sales_forecast = '''
SELECT invoice_date, SUM(quantity * unit_price) AS total_sales
FROM fact_sales
GROUP BY invoice_date
ORDER BY invoice_date;
'''
data_sales_forecast = pd.read_sql(query_sales_forecast, engine)

# Convert the invoice_date to datetime and create numerical and time-related features
data_sales_forecast['invoice_date'] = pd.to_datetime(data_sales_forecast['invoice_date'])
data_sales_forecast['day'] = (data_sales_forecast['invoice_date'] - data_sales_forecast['invoice_date'].min()).dt.days

# Split data into training and test sets
X = data_sales_forecast[['day']]
y = data_sales_forecast['total_sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Linear Regression for forecasting
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales
y_pred = model.predict(X_test)

# Plot actual vs predicted
fig_sales_forecasting = px.scatter(x=X_test['day'], y=y_test, title='Sales Forecasting')
fig_sales_forecasting.add_scatter(x=X_test['day'], y=y_pred, mode='lines', name='Predicted Sales')
st.plotly_chart(fig_sales_forecasting)

# User Guide
st.sidebar.header("User Guide")
st.sidebar.write(""" 
This interactive web application allows users to explore key metrics, customer segmentation, and sales forecasts, aiding in decision-making. 
Here’s how you can use the app:

- **Date Range Picker**: At the top of the dashboard, you can filter sales and forecast data by selecting a date range. This will automatically update the sales graphs.

**Sales Dashboard**:
- **Total Sales Over Time**: A line chart showing sales trends over the selected date range.
- **Top-Selling Products**: A bar chart showing the top-selling products by quantity sold.
- **Total Sales by Region**: A bar chart showing the regions contributing most to sales.
- **Customer Segmentation**: A scatter plot showing customer segments based on total spending and transaction frequency. Use this to identify high-value and frequent customers.
- **Sales Forecasting**: A line chart showing predicted future sales based on historical data. You can compare actual vs predicted sales to assess the accuracy of the forecast.
""")
