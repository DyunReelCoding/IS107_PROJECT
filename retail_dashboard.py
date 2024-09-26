import pandas as pd
from sqlalchemy import create_engine
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset and clean it
def load_and_clean_data():
    df = pd.read_excel('Online Retail.xlsx')
    
    # Quick overview of the dataset
    print(df.info())
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())

    # Check for duplicates
    print(df.duplicated().sum())

    # Drop rows with missing CustomerID (optional)
    df_clean = df.dropna(subset=['CustomerID'])

    # Fill missing descriptions
    df_clean.loc[:, 'Description'] = df_clean['Description'].fillna('Unknown')

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

# Create Dash application for visualization
def create_dash_app(engine):
    app = dash.Dash(__name__)

    # Query sales data
    query_sales = 'SELECT * FROM fact_sales;'
    data_sales = pd.read_sql(query_sales, engine)
    data_sales['total_sales'] = data_sales['quantity'] * data_sales['unit_price']

    # Query customer data for segmentation
    query_customers = '''
    SELECT c.customer_id, SUM(fs.quantity * fs.unit_price) AS total_spent, COUNT(fs.invoice_no) AS num_transactions
    FROM fact_sales fs
    JOIN dim_customer c ON fs.customer_id = c.customer_id
    GROUP BY c.customer_id;
    '''
    data_customers = pd.read_sql(query_customers, engine)

    # Query sales data for forecasting
    query_sales_forecast = '''
    SELECT invoice_date, SUM(quantity * unit_price) AS total_sales
    FROM fact_sales
    GROUP BY invoice_date
    ORDER BY invoice_date;
    '''
    data_sales_forecast = pd.read_sql(query_sales_forecast, engine)
    data_sales_forecast['invoice_date'] = pd.to_datetime(data_sales_forecast['invoice_date'])
    data_sales_forecast['day'] = (data_sales_forecast['invoice_date'] - data_sales_forecast['invoice_date'].min()).dt.days

    # Layout of the app
    app.layout = html.Div([
        html.H1("Retail Data BI Dashboard"),
        
        # Sales Dashboard
        html.H2("Sales Dashboard"),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=data_sales['invoice_date'].min(),
            end_date=data_sales['invoice_date'].max(),
            display_format='YYYY-MM-DD'
        ),
        dcc.Graph(id='total-sales-graph'),
        dcc.Graph(id='top-products-graph'),
        dcc.Graph(id='region-sales-graph'),
        
        # Customer Segmentation
        html.H2("Customer Segmentation"),
        dcc.Graph(id='customer-segmentation-graph'),
        
        # Sales Forecasting
        html.H2("Sales Forecasting"),
        dcc.Graph(id='sales-forecasting-graph')
    ])

    # Callbacks to update visualizations
    @app.callback(
        [Output('total-sales-graph', 'figure'),
         Output('top-products-graph', 'figure'),
         Output('region-sales-graph', 'figure')],
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date')]
    )
    def update_sales_dashboard(start_date, end_date):
        # Filter sales by date range
        filtered_sales = data_sales[
            (data_sales['invoice_date'] >= start_date) &
            (data_sales['invoice_date'] <= end_date)
        ]
        
        # Total sales by date
        total_sales_by_date = filtered_sales.groupby('invoice_date')['total_sales'].sum().reset_index()
        total_sales_fig = px.line(total_sales_by_date, x='invoice_date', y='total_sales', title='Total Sales Over Time')
        
        # Top products by quantity
        top_products = filtered_sales.groupby('stock_code')['quantity'].sum().nlargest(10).reset_index()
        top_products_fig = px.bar(top_products, x='stock_code', y='quantity', title='Top-Selling Products')
        
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
        
        return total_sales_fig, top_products_fig, region_sales_fig

    # Callback for Customer Segmentation
    @app.callback(
        Output('customer-segmentation-graph', 'figure'),
        Input('date-picker-range', 'start_date')
    )
    def update_customer_segmentation(start_date):
        # Clustering (KMeans)
        scaler = StandardScaler()
        X = scaler.fit_transform(data_customers[['total_spent', 'num_transactions']])
        kmeans = KMeans(n_clusters=3, random_state=42)
        data_customers['cluster'] = kmeans.fit_predict(X)
        
        # Plot clusters
        fig = px.scatter(data_customers, x='total_spent', y='num_transactions', color='cluster', title='Customer Segmentation')
        return fig

    # Callback for Sales Forecasting
    @app.callback(
        Output('sales-forecasting-graph', 'figure'),
        Input('date-picker-range', 'start_date')
    )
    def update_sales_forecasting(start_date):
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
        return fig

    return app

# Main execution
if __name__ == '__main__':
    df_clean = load_and_clean_data()
    engine = create_db_engine()
    load_data_to_db(df_clean, engine)
    app = create_dash_app(engine)
    app.run_server(debug=True)

# User Guide for Retail BI Dashboard:
# This interactive web application allows users to explore key metrics, customer segmentation, and sales forecasts, aiding in decision-making. Here’s how you can use the app:
# 1. Date Range Picker: At the top of the dashboard, you can filter sales and forecast data by selecting a date range. This will automatically update the sales graphs.
# 2. Sales Dashboard:
#   - Total Sales Over Time: A line chart showing sales trends over the selected date range.
#   - Top-Selling Products: A bar chart showing the top-selling products by quantity sold.
#   - Total Sales by Region: A bar chart showing the regions contributing most to sales.
# 3. Customer Segmentation: A scatter plot showing customer segments based on total spending and transaction frequency. Use this to identify high-value and frequent customers.
# 4. Sales Forecasting: A line chart showing predicted future sales based on historical data. You can compare actual vs predicted sales to assess the accuracy of the forecast.
