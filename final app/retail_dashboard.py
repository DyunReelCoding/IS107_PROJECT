import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset and clean it
def load_and_clean_data():
    df = pd.read_excel('Online Retail.xlsx')

    # Quick overview of the dataset
    st.write(df.info())
    st.write(df.describe())

    # Check for missing values
    st.write("Missing values:")
    st.write(df.isnull().sum())

    # Check for duplicates
    st.write("Duplicate rows:")
    st.write(df.duplicated().sum())

    # Drop rows with missing CustomerID (optional)
    df_clean = df.dropna(subset=['CustomerID'])

    # Fill missing descriptions
    df_clean['Description'] = df_clean['Description'].fillna('Unknown')

    # Remove invalid transactions
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

    # Drop duplicate rows
    df_clean = df_clean.drop_duplicates()

    return df_clean

# Establish PostgreSQL connection
def create_db_engine():
    return create_engine('postgresql+psycopg2://postgres@localhost:5432/retail_data_warehouse')

# Load cleaned data into PostgreSQL
def load_data_to_db(df_clean, engine):
    df_clean.to_sql('cleaned_online_retail_data', engine, index=False, if_exists='replace')

# Main function to run the Streamlit app
def main():
    df_clean = load_and_clean_data()
    engine = create_db_engine()
    load_data_to_db(df_clean, engine)

    # Streamlit layout
    st.title("Retail Data BI Dashboard")

    # Date range picker
    st.header("Sales Dashboard")
    start_date = st.date_input("Start date", value=df_clean['InvoiceDate'].min())
    end_date = st.date_input("End date", value=df_clean['InvoiceDate'].max())

    # Query sales data
    query_sales = 'SELECT * FROM fact_sales;'
    data_sales = pd.read_sql(query_sales, engine)
    data_sales['total_sales'] = data_sales['quantity'] * data_sales['unit_price']

    # Filter sales data by date range
    filtered_sales = data_sales[
        (data_sales['invoice_date'] >= pd.to_datetime(start_date)) &
        (data_sales['invoice_date'] <= pd.to_datetime(end_date))
    ]

    # Total sales by date
    total_sales_by_date = filtered_sales.groupby('invoice_date')['total_sales'].sum().reset_index()
    total_sales_fig = px.line(total_sales_by_date, x='invoice_date', y='total_sales', title='Total Sales Over Time')
    st.plotly_chart(total_sales_fig)

    # Top products by quantity
    top_products = filtered_sales.groupby('stock_code')['quantity'].sum().nlargest(10).reset_index()
    top_products_fig = px.bar(top_products, x='stock_code', y='quantity', title='Top-Selling Products')
    st.plotly_chart(top_products_fig)

    # Sales by region
    query_sales_by_region = f'''
    SELECT c.country, SUM(fs.quantity * fs.unit_price) AS total_sales
    FROM fact_sales fs
    JOIN dim_customer c ON fs.customer_id = c.customer_id
    WHERE fs.invoice_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY c.country
    ORDER BY total_sales DESC
    LIMIT 10;
    '''
    sales_by_region = pd.read_sql(query_sales_by_region, engine)
    region_sales_fig = px.bar(sales_by_region, x='country', y='total_sales', title='Total Sales by Region')
    st.plotly_chart(region_sales_fig)

    # Customer Segmentation
    st.header("Customer Segmentation")
    query_customers = '''
    SELECT c.customer_id, SUM(fs.quantity * fs.unit_price) AS total_spent, COUNT(fs.invoice_no) AS num_transactions
    FROM fact_sales fs
    JOIN dim_customer c ON fs.customer_id = c.customer_id
    GROUP BY c.customer_id;
    '''
    data_customers = pd.read_sql(query_customers, engine)
    
    # Clustering (KMeans)
    scaler = StandardScaler()
    X = scaler.fit_transform(data_customers[['total_spent', 'num_transactions']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_customers['cluster'] = kmeans.fit_predict(X)

    # Plot clusters
    cluster_fig = px.scatter(data_customers, x='total_spent', y='num_transactions', color='cluster', title='Customer Segmentation')
    st.plotly_chart(cluster_fig)

    # Sales Forecasting
    st.header("Sales Forecasting")
    query_sales_forecast = '''
    SELECT invoice_date, SUM(quantity * unit_price) AS total_sales
    FROM fact_sales
    GROUP BY invoice_date
    ORDER BY invoice_date;
    '''
    data_sales_forecast = pd.read_sql(query_sales_forecast, engine)
    data_sales_forecast['invoice_date'] = pd.to_datetime(data_sales_forecast['invoice_date'])
    data_sales_forecast['day'] = (data_sales_forecast['invoice_date'] - data_sales_forecast['invoice_date'].min()).dt.days

    # Train Linear Regression for forecasting
    X = data_sales_forecast[['day']]
    y = data_sales_forecast['total_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future sales
    y_pred = model.predict(X_test)

    # Plot actual vs predicted
    fig = px.scatter(x=X_test['day'], y=y_test, title='Sales Forecasting')
    fig.add_traces(px.line(x=X_test['day'], y=y_pred).data)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
