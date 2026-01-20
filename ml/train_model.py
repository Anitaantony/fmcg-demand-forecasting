# ml/train_model.py
import sys
sys.path.append('..')

import mysql.connector
import pandas as pd
from predict import FMCGDemandForecaster
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='fmcg_forecasting'
    )

def train_and_save_forecasts():
    """Train models and save forecasts for all products"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get all active products and stores
    cursor.execute("SELECT product_id FROM products WHERE is_active = TRUE")
    products = cursor.fetchall()
    
    cursor.execute("SELECT store_id FROM stores WHERE is_active = TRUE LIMIT 5")
    stores = cursor.fetchall()
    
    total = len(products) * len(stores)
    current = 0
    
    for product in products:
        product_id = product['product_id']
        
        for store in stores:
            store_id = store['store_id']
            current += 1
            
            print(f"\n[{current}/{total}] Training: Product {product_id}, Store {store_id}")
            
            # Fetch historical sales data
            query = """
                SELECT sale_date, SUM(quantity_sold) as quantity
                FROM sales_transactions
                WHERE product_id = %s AND store_id = %s
                GROUP BY sale_date
                ORDER BY sale_date
            """
            df = pd.read_sql(query, conn, params=(product_id, store_id))
            
            if len(df) < 30:
                print(f"  ⚠ Insufficient data ({len(df)} records)")
                continue
            
            try:
                # Initialize forecaster
                forecaster = FMCGDemandForecaster(model_type='hw')
                
                # Prepare data
                series = forecaster.prepare_data(df, value_col='quantity')
                
                # Train model
                success = forecaster.train_holt_winters(series, seasonal_periods=7)
                
                if not success:
                    print(f"  ✗ Training failed")
                    continue
                
                # Generate forecast
                predictions, lower, upper = forecaster.forecast(steps=30)
                
                # Calculate inventory metrics
                metrics = forecaster.calculate_inventory_metrics(predictions)
                
                # Update product reorder metrics
                cursor.execute("""
                    UPDATE products 
                    SET safety_stock = %s, reorder_point = %s
                    WHERE product_id = %s
                """, (metrics['safety_stock'], metrics['reorder_point'], product_id))
                
                # Delete old forecasts
                cursor.execute("""
                    DELETE FROM forecast_results
                    WHERE product_id = %s AND store_id = %s
                """, (product_id, store_id))
                
                # Insert new forecasts
                start_date = datetime.now().date() + timedelta(days=1)
                for i, pred in enumerate(predictions):
                    forecast_date = start_date + timedelta(days=i)
                    cursor.execute("""
                        INSERT INTO forecast_results 
                        (store_id, product_id, forecast_date, predicted_demand,
                         confidence_interval_lower, confidence_interval_upper,
                         model_used, model_version)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        store_id, product_id, forecast_date,
                        float(pred), float(lower[i]), float(upper[i]),
                        'HoltWinters', '1.0'
                    ))
                
                # Insert inventory recommendations
                cursor.execute("""
                    INSERT INTO inventory_recommendations
                    (store_id, product_id, calculation_date, recommended_order_quantity,
                     reorder_point, safety_stock, economic_order_quantity,
                     average_daily_demand, demand_std_dev, service_level,
                     lead_time_days, stockout_risk)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    store_id, product_id, datetime.now().date(),
                    metrics['recommended_order_quantity'],
                    metrics['reorder_point'],
                    metrics['safety_stock'],
                    metrics['economic_order_quantity'],
                    metrics['average_daily_demand'],
                    metrics['demand_std_dev'],
                    metrics['service_level'],
                    metrics['lead_time_days'],
                    metrics['stockout_risk']
                ))
                
                conn.commit()
                print(f"  ✓ Success - Avg Demand: {metrics['average_daily_demand']:.1f}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                conn.rollback()
                continue
    
    cursor.close()
    conn.close()
    print("\n✓ All forecasts generated!")

if __name__ == "__main__":
    train_and_save_forecasts()