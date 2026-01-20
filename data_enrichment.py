"""
FMCG Data Enrichment Script
Converts Kaggle dataset to enriched FMCG database
"""

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker()

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',  # Change this
        database='fmcg_forecasting'
    )

# FMCG Product Master Data
FMCG_PRODUCTS = {
    1: {'name': 'Fresh Milk 1L', 'category': 'Dairy Products', 'brand': 'Amul', 'price': 60, 'cost': 45, 'shelf_life': 5, 'perishable': True, 'unit': 'litre'},
    2: {'name': 'Whole Wheat Bread', 'category': 'Food & Beverages', 'brand': 'Britannia', 'price': 45, 'cost': 30, 'shelf_life': 7, 'perishable': True, 'unit': 'pack'},
    3: {'name': 'Shampoo 200ml', 'category': 'Hair Care', 'brand': 'Pantene', 'price': 180, 'cost': 120, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    4: {'name': 'Detergent Powder 1kg', 'category': 'Detergents', 'brand': 'Surf Excel', 'price': 250, 'cost': 180, 'shelf_life': 365, 'perishable': False, 'unit': 'kg'},
    5: {'name': 'Toothpaste 150g', 'category': 'Oral Care', 'brand': 'Colgate', 'price': 85, 'cost': 55, 'shelf_life': 730, 'perishable': False, 'unit': 'tube'},
    6: {'name': 'Biscuits 200g', 'category': 'Snacks & Confectionery', 'brand': 'Parle-G', 'price': 20, 'cost': 12, 'shelf_life': 180, 'perishable': False, 'unit': 'pack'},
    7: {'name': 'Cooking Oil 1L', 'category': 'Food & Beverages', 'brand': 'Fortune', 'price': 180, 'cost': 140, 'shelf_life': 365, 'perishable': False, 'unit': 'litre'},
    8: {'name': 'Bath Soap 100g', 'category': 'Skin Care', 'brand': 'Lux', 'price': 40, 'cost': 25, 'shelf_life': 730, 'perishable': False, 'unit': 'bar'},
    9: {'name': 'Rice 1kg', 'category': 'Food & Beverages', 'brand': 'India Gate', 'price': 90, 'cost': 65, 'shelf_life': 365, 'perishable': False, 'unit': 'kg'},
    10: {'name': 'Tea Powder 250g', 'category': 'Beverages', 'brand': 'Tata Tea', 'price': 125, 'cost': 90, 'shelf_life': 365, 'perishable': False, 'unit': 'pack'},
    11: {'name': 'Sugar 1kg', 'category': 'Food & Beverages', 'brand': 'Madhur', 'price': 50, 'cost': 38, 'shelf_life': 730, 'perishable': False, 'unit': 'kg'},
    12: {'name': 'Coffee Powder 200g', 'category': 'Beverages', 'brand': 'Nescafe', 'price': 220, 'cost': 160, 'shelf_life': 365, 'perishable': False, 'unit': 'jar'},
    13: {'name': 'Floor Cleaner 1L', 'category': 'Cleaning Supplies', 'brand': 'Lizol', 'price': 150, 'cost': 100, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    14: {'name': 'Hand Wash 250ml', 'category': 'Personal Care', 'brand': 'Dettol', 'price': 75, 'cost': 50, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    15: {'name': 'Potato Chips 100g', 'category': 'Snacks & Confectionery', 'brand': "Lay's", 'price': 20, 'cost': 12, 'shelf_life': 90, 'perishable': False, 'unit': 'pack'},
    16: {'name': 'Soft Drink 2L', 'category': 'Beverages', 'brand': 'Coca-Cola', 'price': 90, 'cost': 60, 'shelf_life': 180, 'perishable': False, 'unit': 'bottle'},
    17: {'name': 'Mineral Water 1L', 'category': 'Beverages', 'brand': 'Bisleri', 'price': 20, 'cost': 10, 'shelf_life': 365, 'perishable': False, 'unit': 'bottle'},
    18: {'name': 'Butter 100g', 'category': 'Dairy Products', 'brand': 'Amul', 'price': 50, 'cost': 35, 'shelf_life': 30, 'perishable': True, 'unit': 'pack'},
    19: {'name': 'Cheese Slice 200g', 'category': 'Dairy Products', 'brand': 'Britannia', 'price': 120, 'cost': 85, 'shelf_life': 60, 'perishable': True, 'unit': 'pack'},
    20: {'name': 'Yogurt 400g', 'category': 'Dairy Products', 'brand': 'Mother Dairy', 'price': 40, 'cost': 28, 'shelf_life': 10, 'perishable': True, 'unit': 'cup'},
    21: {'name': 'Chocolate Bar 50g', 'category': 'Snacks & Confectionery', 'brand': 'Cadbury', 'price': 30, 'cost': 18, 'shelf_life': 365, 'perishable': False, 'unit': 'bar'},
    22: {'name': 'Noodles 200g', 'category': 'Food & Beverages', 'brand': 'Maggi', 'price': 40, 'cost': 25, 'shelf_life': 180, 'perishable': False, 'unit': 'pack'},
    23: {'name': 'Ketchup 500g', 'category': 'Food & Beverages', 'brand': 'Kissan', 'price': 110, 'cost': 75, 'shelf_life': 365, 'perishable': False, 'unit': 'bottle'},
    24: {'name': 'Baby Diapers 20pcs', 'category': 'Baby Care', 'brand': 'Pampers', 'price': 450, 'cost': 320, 'shelf_life': 730, 'perishable': False, 'unit': 'pack'},
    25: {'name': 'Baby Wipes 80pcs', 'category': 'Baby Care', 'brand': 'Himalaya', 'price': 120, 'cost': 80, 'shelf_life': 730, 'perishable': False, 'unit': 'pack'},
    26: {'name': 'Hair Oil 200ml', 'category': 'Hair Care', 'brand': 'Parachute', 'price': 95, 'cost': 65, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    27: {'name': 'Face Cream 50g', 'category': 'Skin Care', 'brand': 'Ponds', 'price': 180, 'cost': 120, 'shelf_life': 730, 'perishable': False, 'unit': 'jar'},
    28: {'name': 'Deodorant 150ml', 'category': 'Personal Care', 'brand': 'Axe', 'price': 220, 'cost': 150, 'shelf_life': 730, 'perishable': False, 'unit': 'can'},
    29: {'name': 'Sanitizer 500ml', 'category': 'Health & Hygiene', 'brand': 'Dettol', 'price': 180, 'cost': 120, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    30: {'name': 'Washing Bar 250g', 'category': 'Detergents', 'brand': 'Rin', 'price': 35, 'cost': 22, 'shelf_life': 730, 'perishable': False, 'unit': 'bar'},
    31: {'name': 'Dishwash Gel 500ml', 'category': 'Cleaning Supplies', 'brand': 'Vim', 'price': 95, 'cost': 65, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    32: {'name': 'Toilet Cleaner 500ml', 'category': 'Cleaning Supplies', 'brand': 'Harpic', 'price': 110, 'cost': 75, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    33: {'name': 'Incense Sticks 100g', 'category': 'Household Care', 'brand': 'Cycle', 'price': 50, 'cost': 30, 'shelf_life': 730, 'perishable': False, 'unit': 'pack'},
    34: {'name': 'Matchbox', 'category': 'Household Care', 'brand': 'Aim', 'price': 2, 'cost': 1, 'shelf_life': 1095, 'perishable': False, 'unit': 'box'},
    35: {'name': 'Candles 10pcs', 'category': 'Household Care', 'brand': 'Generic', 'price': 25, 'cost': 15, 'shelf_life': 1095, 'perishable': False, 'unit': 'pack'},
    36: {'name': 'Tissue Paper Roll', 'category': 'Household Care', 'brand': 'Origami', 'price': 120, 'cost': 80, 'shelf_life': 730, 'perishable': False, 'unit': 'roll'},
    37: {'name': 'Garbage Bags 30pcs', 'category': 'Household Care', 'brand': 'Shalimar', 'price': 85, 'cost': 55, 'shelf_life': 1095, 'perishable': False, 'unit': 'pack'},
    38: {'name': 'Mosquito Coil 10pcs', 'category': 'Household Care', 'brand': 'Mortein', 'price': 40, 'cost': 25, 'shelf_life': 730, 'perishable': False, 'unit': 'pack'},
    39: {'name': 'Air Freshener 250ml', 'category': 'Household Care', 'brand': 'Odonil', 'price': 150, 'cost': 100, 'shelf_life': 730, 'perishable': False, 'unit': 'can'},
    40: {'name': 'Battery AA 4pcs', 'category': 'Household Care', 'brand': 'Duracell', 'price': 120, 'cost': 80, 'shelf_life': 1825, 'perishable': False, 'unit': 'pack'},
    41: {'name': 'Corn Flakes 500g', 'category': 'Food & Beverages', 'brand': "Kellogg's", 'price': 200, 'cost': 140, 'shelf_life': 365, 'perishable': False, 'unit': 'box'},
    42: {'name': 'Jam 500g', 'category': 'Food & Beverages', 'brand': 'Kissan', 'price': 140, 'cost': 95, 'shelf_life': 365, 'perishable': False, 'unit': 'jar'},
    43: {'name': 'Honey 500g', 'category': 'Food & Beverages', 'brand': 'Dabur', 'price': 240, 'cost': 170, 'shelf_life': 730, 'perishable': False, 'unit': 'bottle'},
    44: {'name': 'Peanut Butter 400g', 'category': 'Food & Beverages', 'brand': 'Sundrop', 'price': 220, 'cost': 150, 'shelf_life': 365, 'perishable': False, 'unit': 'jar'},
    45: {'name': 'Pickle 500g', 'category': 'Food & Beverages', 'brand': 'Priya', 'price': 130, 'cost': 90, 'shelf_life': 365, 'perishable': False, 'unit': 'jar'},
    46: {'name': 'Sauce 500g', 'category': 'Food & Beverages', 'brand': 'Tops', 'price': 95, 'cost': 65, 'shelf_life': 365, 'perishable': False, 'unit': 'bottle'},
    47: {'name': 'Salt 1kg', 'category': 'Food & Beverages', 'brand': 'Tata', 'price': 22, 'cost': 15, 'shelf_life': 1095, 'perishable': False, 'unit': 'pack'},
    48: {'name': 'Spice Mix 100g', 'category': 'Food & Beverages', 'brand': 'MDH', 'price': 60, 'cost': 40, 'shelf_life': 365, 'perishable': False, 'unit': 'pack'},
    49: {'name': 'Dry Fruits 250g', 'category': 'Food & Beverages', 'brand': 'Khari', 'price': 350, 'cost': 250, 'shelf_life': 180, 'perishable': False, 'unit': 'pack'},
    50: {'name': 'Energy Drink 250ml', 'category': 'Beverages', 'brand': 'Red Bull', 'price': 125, 'cost': 85, 'shelf_life': 365, 'perishable': False, 'unit': 'can'}
}

STORE_NAMES = [
    'Central Supermart', 'QuickMart Express', 'FreshValue Store', 
    'Daily Needs Bazaar', 'Metro Grocery', 'Smart Shopper', 
    'Family Mart', 'Neighborhood Store', 'Big Basket Outlet', 
    'Reliance Fresh'
]

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 
          'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']

def insert_stores(conn, num_stores=10):
    """Insert store master data"""
    cursor = conn.cursor()
    
    for store_id in range(1, num_stores + 1):
        cursor.execute("""
            INSERT INTO stores (store_id, store_name, location, city, state, 
                              manager_name, contact_number, store_type, 
                              opening_date, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE store_name=VALUES(store_name)
        """, (
            store_id,
            STORE_NAMES[store_id - 1] if store_id <= len(STORE_NAMES) else f'Store {store_id}',
            fake.address(),
            CITIES[(store_id - 1) % len(CITIES)],
            'Maharashtra' if store_id <= 3 else 'Karnataka',
            fake.name(),
            fake.phone_number()[:15],
            random.choice(['supermarket', 'hypermarket', 'convenience']),
            datetime(2010, 1, 1) + timedelta(days=random.randint(0, 1000)),
            True
        ))
    
    conn.commit()
    print(f"✓ Inserted {num_stores} stores")

def insert_products(conn):
    """Insert product master data"""
    cursor = conn.cursor()
    
    # Get category mapping
    cursor.execute("SELECT category_id, category_name FROM categories")
    category_map = {name: id for id, name in cursor.fetchall()}
    
    for product_id, details in FMCG_PRODUCTS.items():
        category_id = category_map.get(details['category'], 1)
        
        cursor.execute("""
            INSERT INTO products (
                product_id, product_name, product_code, category_id, brand,
                unit_of_measure, unit_price, cost_price, shelf_life_days,
                minimum_stock_level, maximum_stock_level, reorder_point,
                safety_stock, lead_time_days, is_perishable, is_active
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                product_name=VALUES(product_name),
                unit_price=VALUES(unit_price)
        """, (
            product_id,
            details['name'],
            f'FMCG-{product_id:04d}',
            category_id,
            details['brand'],
            details['unit'],
            details['price'],
            details['cost'],
            details['shelf_life'],
            20 if details['perishable'] else 10,
            500 if details['perishable'] else 1000,
            50 if details['perishable'] else 30,
            30 if details['perishable'] else 15,
            2 if details['perishable'] else 5,
            details['perishable'],
            True
        ))
    
    conn.commit()
    print(f"✓ Inserted {len(FMCG_PRODUCTS)} products")

def import_sales_data(conn, csv_file='train.csv', batch_size=10000):
    """Import sales data from Kaggle CSV"""
    print("Loading Kaggle dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Map item IDs to our product range (1-50)
    unique_items = df['item'].unique()
    if len(unique_items) > 50:
        # Map original items to 1-50 range
        item_mapping = {item: (item % 50) + 1 for item in unique_items}
    else:
        item_mapping = {item: item for item in unique_items}
    
    df['product_id'] = df['item'].map(item_mapping)
    
    # Add pricing based on product
    df['unit_price'] = df['product_id'].map(
        lambda x: FMCG_PRODUCTS.get(x, {'price': 50})['price']
    )
    df['total_amount'] = df['sales'] * df['unit_price']
    
    # Rename columns to match database
    df = df.rename(columns={
        'date': 'sale_date',
        'store': 'store_id',
        'sales': 'quantity_sold'
    })
    
    # Select relevant columns
    df_insert = df[['store_id', 'product_id', 'sale_date', 
                    'quantity_sold', 'unit_price', 'total_amount']]
    
    cursor = conn.cursor()
    
    print(f"Importing {len(df_insert)} sales records...")
    
    # Batch insert for performance
    for i in range(0, len(df_insert), batch_size):
        batch = df_insert.iloc[i:i+batch_size]
        
        values = [tuple(x) for x in batch.values]
        
        cursor.executemany("""
            INSERT INTO sales_transactions 
            (store_id, product_id, sale_date, quantity_sold, unit_price, total_amount)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, values)
        
        conn.commit()
        print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} records")
    
    print(f"✓ Imported {len(df_insert)} sales transactions")
    
    return df_insert

def initialize_inventory(conn):
    """Create initial inventory records"""
    cursor = conn.cursor()
    
    # Get all store and product combinations
    cursor.execute("SELECT store_id FROM stores WHERE is_active = TRUE")
    stores = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT product_id FROM products WHERE is_active = TRUE")
    products = [row[0] for row in cursor.fetchall()]
    
    print(f"Initializing inventory for {len(stores)} stores × {len(products)} products...")
    
    for store_id in stores:
        for product_id in products:
            current_stock = random.randint(50, 500)
            
            cursor.execute("""
                INSERT INTO inventory (store_id, product_id, current_stock, 
                                     reserved_stock, last_restocked_date, 
                                     last_restocked_quantity)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE current_stock = VALUES(current_stock)
            """, (
                store_id,
                product_id,
                current_stock,
                random.randint(0, 10),
                datetime.now().date() - timedelta(days=random.randint(1, 30)),
                random.randint(100, 300)
            ))
    
    conn.commit()
    print(f"✓ Initialized {len(stores) * len(products)} inventory records")

def create_admin_user(conn):
    """Create default admin user"""
    import bcrypt
    
    cursor = conn.cursor()
    
    password = "admin123"
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (%s, %s, %s, %s, %s)
        """, ('admin', 'admin@fmcg.com', hashed, 'System Administrator', 'admin'))
        
        conn.commit()
        print("✓ Created admin user (username: admin, password: admin123)")
    except mysql.connector.IntegrityError:
        print("✓ Admin user already exists")

def main():
    """Main execution function"""
    print("="*60)
    print("FMCG Data Enrichment Script")
    print("="*60)
    
    conn = get_db_connection()
    
    try:
        # Step 1: Insert stores
        print("\n[1/5] Setting up stores...")
        insert_stores(conn, num_stores=10)
        
        # Step 2: Insert products
        print("\n[2/5] Setting up products...")
        insert_products(conn)
        
        # Step 3: Import sales data
        print("\n[3/5] Importing sales data...")
        import_sales_data(conn, csv_file='data/raw/train.csv')
        
        # Step 4: Initialize inventory
        print("\n[4/5] Initializing inventory...")
        initialize_inventory(conn)
        
        # Step 5: Create admin user
        print("\n[5/5] Creating admin user...")
        create_admin_user(conn)
        
        print("\n" + "="*60)
        print("✓ Data enrichment completed successfully!")
        print("="*60)
        
        # Display summary
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sales_transactions")
        sales_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM products")
        product_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stores")
        store_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(sale_date), MAX(sale_date) FROM sales_transactions")
        date_range = cursor.fetchone()
        
        print(f"\nDatabase Summary:")
        print(f"  Stores: {store_count}")
        print(f"  Products: {product_count}")
        print(f"  Sales Records: {sales_count:,}")
        print(f"  Date Range: {date_range[0]} to {date_range[1]}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()