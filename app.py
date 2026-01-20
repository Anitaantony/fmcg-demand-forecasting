"""
FMCG Demand Forecasting & Inventory Optimization System
Main Flask Application
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_mysqldb import MySQL
import bcrypt
import json
from datetime import datetime, timedelta
from functools import wraps
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'fmcg_secret_key_2024_change_in_production'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'  # CHANGE THIS
app.config['MYSQL_DB'] = 'fmcg_forecasting'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# ============================================================================
# AUTHENTICATION DECORATORS
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        
        # Hash password
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        cur = mysql.connection.cursor()
        try:
            cur.execute("""
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (%s, %s, %s, %s, %s)
            """, (username, email, hashed, full_name, 'viewer'))
            
            mysql.connection.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            mysql.connection.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')
        finally:
            cur.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT user_id, username, password_hash, role, full_name 
            FROM users WHERE username = %s AND is_active = TRUE
        """, [username])
        
        user = cur.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            
            # Update last login
            cur.execute("""
                UPDATE users SET last_login = NOW() WHERE user_id = %s
            """, [user['user_id']])
            mysql.connection.commit()
            
            # Log activity
            log_activity('login', 'users', user['user_id'], None, None)
            
            flash(f'Welcome back, {user["full_name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'danger')
        
        cur.close()
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    log_activity('logout', 'users', session.get('user_id'), None, None)
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

# ============================================================================
# DASHBOARD ROUTES
# ============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    cur = mysql.connection.cursor()
    
    # Get key metrics
    cur.execute("""
        SELECT 
            COUNT(DISTINCT product_id) as total_products,
            COUNT(DISTINCT store_id) as total_stores,
            SUM(current_stock) as total_stock_value,
            COUNT(CASE WHEN current_stock <= (
                SELECT reorder_point FROM products p 
                WHERE p.product_id = inventory.product_id
            ) THEN 1 END) as low_stock_items
        FROM inventory
    """)
    metrics = cur.fetchone()
    
    # Get recent sales (last 7 days)
    cur.execute("""
        SELECT 
            sale_date,
            SUM(total_amount) as daily_revenue,
            SUM(quantity_sold) as daily_quantity
        FROM sales_transactions
        WHERE sale_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        GROUP BY sale_date
        ORDER BY sale_date DESC
        LIMIT 7
    """)
    recent_sales = cur.fetchall()
    
    # Top selling products (last 30 days)
    cur.execute("""
        SELECT 
            p.product_name,
            p.brand,
            c.category_name,
            SUM(st.quantity_sold) as total_sold,
            SUM(st.total_amount) as revenue
        FROM sales_transactions st
        JOIN products p ON st.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        WHERE st.sale_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        GROUP BY st.product_id
        ORDER BY total_sold DESC
        LIMIT 10
    """)
    top_products = cur.fetchall()
    
    # Low stock alerts
    cur.execute("""
        SELECT 
            p.product_name,
            p.brand,
            s.store_name,
            i.current_stock,
            p.reorder_point,
            p.safety_stock
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        JOIN stores s ON i.store_id = s.store_id
        WHERE i.current_stock <= p.reorder_point
        ORDER BY (p.reorder_point - i.current_stock) DESC
        LIMIT 10
    """)
    low_stock_alerts = cur.fetchall()
    
    # Unread alerts
    cur.execute("""
        SELECT COUNT(*) as unread_count 
        FROM alerts 
        WHERE is_read = FALSE AND is_resolved = FALSE
    """)
    alerts_count = cur.fetchone()['unread_count']
    
    cur.close()
    
    return render_template('dashboard.html',
                         metrics=metrics,
                         recent_sales=recent_sales,
                         top_products=top_products,
                         low_stock_alerts=low_stock_alerts,
                         alerts_count=alerts_count)

# ============================================================================
# INVENTORY MANAGEMENT ROUTES
# ============================================================================

@app.route('/inventory')
@login_required
def inventory():
    cur = mysql.connection.cursor()
    
    # Get filters from query params
    store_filter = request.args.get('store', '')
    category_filter = request.args.get('category', '')
    search_query = request.args.get('search', '')
    
    # Build dynamic query
    query = """
        SELECT 
            i.inventory_id,
            s.store_name,
            p.product_name,
            p.brand,
            c.category_name,
            i.current_stock,
            i.reserved_stock,
            i.available_stock,
            p.reorder_point,
            p.safety_stock,
            p.unit_price,
            (i.current_stock * p.unit_price) as stock_value,
            i.last_restocked_date,
            CASE 
                WHEN i.current_stock <= p.safety_stock THEN 'Critical'
                WHEN i.current_stock <= p.reorder_point THEN 'Low'
                WHEN i.current_stock >= p.maximum_stock_level THEN 'Overstock'
                ELSE 'Normal'
            END as stock_status
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        JOIN stores s ON i.store_id = s.store_id
        JOIN categories c ON p.category_id = c.category_id
        WHERE 1=1
    """
    
    params = []
    
    if store_filter:
        query += " AND s.store_id = %s"
        params.append(store_filter)
    
    if category_filter:
        query += " AND c.category_id = %s"
        params.append(category_filter)
    
    if search_query:
        query += " AND (p.product_name LIKE %s OR p.brand LIKE %s)"
        params.extend([f'%{search_query}%', f'%{search_query}%'])
    
    query += " ORDER BY stock_status DESC, i.current_stock ASC LIMIT 100"
    
    cur.execute(query, params)
    inventory_items = cur.fetchall()
    
    # Get filter options
    cur.execute("SELECT store_id, store_name FROM stores WHERE is_active = TRUE")
    stores = cur.fetchall()
    
    cur.execute("SELECT category_id, category_name FROM categories")
    categories = cur.fetchall()
    
    cur.close()
    
    return render_template('inventory.html',
                         inventory=inventory_items,
                         stores=stores,
                         categories=categories)

@app.route('/inventory/update/<int:inventory_id>', methods=['POST'])
@login_required
def update_inventory(inventory_id):
    new_stock = request.form.get('current_stock')
    
    cur = mysql.connection.cursor()
    
    # Get old value for logging
    cur.execute("SELECT current_stock FROM inventory WHERE inventory_id = %s", [inventory_id])
    old_stock = cur.fetchone()
    
    # Update
    cur.execute("""
        UPDATE inventory 
        SET current_stock = %s,
            last_restocked_date = CURDATE(),
            last_restocked_quantity = %s - current_stock
        WHERE inventory_id = %s
    """, (new_stock, new_stock, inventory_id))
    
    mysql.connection.commit()
    
    # Log activity
    log_activity('update', 'inventory', inventory_id,
                json.dumps({'current_stock': old_stock['current_stock']}),
                json.dumps({'current_stock': new_stock}))
    
    cur.close()
    
    flash('Inventory updated successfully', 'success')
    return redirect(url_for('inventory'))

# ============================================================================
# FORECASTING ROUTES
# ============================================================================

@app.route('/forecast')
@login_required
def forecast():
    cur = mysql.connection.cursor()
    
    # Get products with forecasts
    cur.execute("""
        SELECT DISTINCT 
            p.product_id, 
            p.product_name,
            p.brand,
            c.category_name
        FROM products p
        JOIN forecast_results f ON p.product_id = f.product_id
        JOIN categories c ON p.category_id = c.category_id
        ORDER BY p.product_name
    """)
    products = cur.fetchall()
    
    # Get stores
    cur.execute("SELECT store_id, store_name FROM stores WHERE is_active = TRUE")
    stores = cur.fetchall()
    
    cur.close()
    
    return render_template('forecast.html', products=products, stores=stores)

@app.route('/forecast/run', methods=['POST'])
@login_required
def run_forecast():
    """Trigger forecast generation"""
    try:
        # Import and run the forecasting module
        from ml.train_model import train_and_save_forecasts
        
        train_and_save_forecasts()
        flash('Forecast generated successfully!', 'success')
    except Exception as e:
        flash(f'Error generating forecast: {str(e)}', 'danger')
    
    return redirect(url_for('forecast'))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/sales-trend')
@login_required
def api_sales_trend():
    """API endpoint for sales trend chart"""
    cur = mysql.connection.cursor()
    
    cur.execute("""
        SELECT 
            sale_date,
            SUM(quantity_sold) as total_quantity,
            SUM(total_amount) as total_revenue
        FROM sales_transactions
        WHERE sale_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        GROUP BY sale_date
        ORDER BY sale_date
    """)
    
    data = cur.fetchall()
    cur.close()
    
    return jsonify({
        'labels': [str(row['sale_date']) for row in data],
        'quantities': [int(row['total_quantity']) for row in data],
        'revenues': [float(row['total_revenue']) for row in data]
    })

@app.route('/api/forecast/<int:product_id>')
@login_required
def api_forecast(product_id):
    """API endpoint for product forecast"""
    store_id = request.args.get('store_id', 1)
    
    cur = mysql.connection.cursor()
    
    # Get forecast data
    cur.execute("""
        SELECT 
            forecast_date,
            predicted_demand,
            confidence_interval_lower,
            confidence_interval_upper,
            actual_sales
        FROM forecast_results
        WHERE product_id = %s AND store_id = %s
        ORDER BY forecast_date
        LIMIT 30
    """, (product_id, store_id))
    
    forecast_data = cur.fetchall()
    
    # Get historical sales
    cur.execute("""
        SELECT 
            sale_date,
            SUM(quantity_sold) as quantity
        FROM sales_transactions
        WHERE product_id = %s AND store_id = %s
              AND sale_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
        GROUP BY sale_date
        ORDER BY sale_date
    """, (product_id, store_id))
    
    historical_data = cur.fetchall()
    
    cur.close()
    
    return jsonify({
        'forecast': {
            'dates': [str(row['forecast_date']) for row in forecast_data],
            'predictions': [float(row['predicted_demand']) for row in forecast_data],
            'lower_bound': [float(row['confidence_interval_lower']) for row in forecast_data],
            'upper_bound': [float(row['confidence_interval_upper']) for row in forecast_data],
            'actuals': [float(row['actual_sales']) if row['actual_sales'] else None for row in forecast_data]
        },
        'historical': {
            'dates': [str(row['sale_date']) for row in historical_data],
            'sales': [int(row['quantity']) for row in historical_data]
        }
    })

@app.route('/api/inventory-recommendations/<int:product_id>')
@login_required
def api_inventory_recommendations(product_id):
    """API endpoint for inventory recommendations"""
    store_id = request.args.get('store_id', 1)
    
    cur = mysql.connection.cursor()
    
    cur.execute("""
        SELECT * FROM inventory_recommendations
        WHERE product_id = %s AND store_id = %s
        ORDER BY calculation_date DESC
        LIMIT 1
    """, (product_id, store_id))
    
    recommendation = cur.fetchone()
    cur.close()
    
    if recommendation:
        return jsonify(dict(recommendation))
    else:
        return jsonify({'error': 'No recommendations found'}), 404

# ============================================================================
# REPORTS ROUTES
# ============================================================================

@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html')

@app.route('/reports/sales')
@login_required
def sales_report():
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    cur = mysql.connection.cursor()
    
    # Sales by category
    cur.execute("""
        SELECT 
            c.category_name,
            COUNT(DISTINCT st.product_id) as product_count,
            SUM(st.quantity_sold) as total_quantity,
            SUM(st.total_amount) as total_revenue
        FROM sales_transactions st
        JOIN products p ON st.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        WHERE st.sale_date BETWEEN %s AND %s
        GROUP BY c.category_id
        ORDER BY total_revenue DESC
    """, (start_date, end_date))
    
    category_sales = cur.fetchall()
    
    # Sales by store
    cur.execute("""
        SELECT 
            s.store_name,
            s.city,
            SUM(st.quantity_sold) as total_quantity,
            SUM(st.total_amount) as total_revenue,
            COUNT(DISTINCT st.product_id) as unique_products
        FROM sales_transactions st
        JOIN stores s ON st.store_id = s.store_id
        WHERE st.sale_date BETWEEN %s AND %s
        GROUP BY s.store_id
        ORDER BY total_revenue DESC
    """, (start_date, end_date))
    
    store_sales = cur.fetchall()
    
    cur.close()
    
    return render_template('sales_report.html',
                         category_sales=category_sales,
                         store_sales=store_sales,
                         start_date=start_date,
                         end_date=end_date)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_activity(action, table_name, record_id, old_values, new_values):
    """Log user activity"""
    if 'user_id' not in session:
        return
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO audit_logs (user_id, action, table_name, record_id,
                                  old_values, new_values, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            session['user_id'],
            action,
            table_name,
            record_id,
            old_values,
            new_values,
            request.remote_addr
        ))
        mysql.connection.commit()
        cur.close()
    except:
        pass  # Don't fail if logging fails

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    mysql.connection.rollback()
    return render_template('500.html'), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)