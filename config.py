# config.py

"""
FMCG Forecasting System Configuration
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fmcg-dev-secret-key-2024-change-in-production'
    
    # MySQL Database Configuration
    MYSQL_HOST = os.environ.get('MYSQL_HOST') or 'localhost'
    MYSQL_USER = os.environ.get('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD') or 'your_password'
    MYSQL_DB = os.environ.get('MYSQL_DB') or 'fmcg_forecasting'
    MYSQL_CURSORCLASS = 'DictCursor'
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_TYPE = 'filesystem'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads/'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Forecasting Parameters
    FORECAST_HORIZON_DAYS = 30
    FORECAST_CONFIDENCE_LEVEL = 0.95
    DEFAULT_SERVICE_LEVEL = 0.95
    DEFAULT_LEAD_TIME_DAYS = 3
    
    # Inventory Parameters
    MIN_STOCK_MULTIPLIER = 1.5  # Minimum stock = avg_daily_demand * multiplier
    MAX_STOCK_MULTIPLIER = 30   # Maximum stock = avg_daily_demand * multiplier
    REORDER_MULTIPLIER = 2.0    # Reorder point multiplier
    
    # Model Training Parameters
    TRAIN_TEST_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    SEASONAL_PERIODS = 7  # Weekly seasonality
    
    # Alert Thresholds
    LOW_STOCK_THRESHOLD_PERCENT = 20  # Alert when stock < 20% of max
    CRITICAL_STOCK_DAYS = 7  # Alert when stock < 7 days of demand
    OVERSTOCK_THRESHOLD_PERCENT = 90  # Alert when stock > 90% of max
    
    # Pagination
    ITEMS_PER_PAGE = 50
    
    # Logging
    LOG_FILE = 'logs/fmcg_system.log'
    LOG_LEVEL = 'INFO'
    
    # Email Configuration (for alerts - optional)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # API Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "200 per hour"
    
    # Cache Configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Feature Flags
    ENABLE_EMAIL_ALERTS = False
    ENABLE_SMS_ALERTS = False
    ENABLE_AUTO_REORDER = False
    ENABLE_ADVANCED_ANALYTICS = True
    
    # Model Versions
    MODEL_VERSION = '1.0.0'
    SUPPORTED_ALGORITHMS = ['sarima', 'holt_winters', 'random_forest', 'gradient_boosting']
    DEFAULT_ALGORITHM = 'holt_winters'
    
    # Time Series Parameters
    MIN_DATA_POINTS = 30  # Minimum historical data points needed
    OUTLIER_THRESHOLD = 3  # Standard deviations for outlier detection
    
    # Business Rules
    BUSINESS_DAYS_PER_WEEK = 7
    BUSINESS_HOURS_PER_DAY = 24
    CURRENCY_SYMBOL = '₹'
    CURRENCY_CODE = 'INR'
    
    # Report Generation
    REPORT_FORMATS = ['pdf', 'excel', 'csv']
    DEFAULT_REPORT_FORMAT = 'excel'
    
    # Dashboard Refresh
    DASHBOARD_REFRESH_INTERVAL = 300  # seconds
    
    # Data Retention
    FORECAST_RETENTION_DAYS = 90
    AUDIT_LOG_RETENTION_DAYS = 365
    SALES_DATA_RETENTION_YEARS = 5


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    ENV = 'development'
    
    # Override for development
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    ENV = 'production'
    
    # Security enhancements for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Stricter rate limiting
    RATELIMIT_DEFAULT = "100 per hour"
    
    # Production database (use environment variables)
    MYSQL_HOST = os.environ.get('MYSQL_HOST')
    MYSQL_USER = os.environ.get('MYSQL_USER')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
    MYSQL_DB = os.environ.get('MYSQL_DB')
    
    # Enable production features
    ENABLE_EMAIL_ALERTS = True
    ENABLE_SMS_ALERTS = True


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    ENV = 'testing'
    
    # Use separate test database
    MYSQL_DB = 'fmcg_forecasting_test'
    
    # Disable security features for testing
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])


# Constants
class Constants:
    """Application constants"""
    
    # User Roles
    ROLE_ADMIN = 'admin'
    ROLE_MANAGER = 'manager'
    ROLE_ANALYST = 'analyst'
    ROLE_VIEWER = 'viewer'
    
    ROLES = [ROLE_ADMIN, ROLE_MANAGER, ROLE_ANALYST, ROLE_VIEWER]
    
    # Alert Types
    ALERT_LOW_STOCK = 'low_stock'
    ALERT_EXPIRY_WARNING = 'expiry_warning'
    ALERT_FORECAST_DEVIATION = 'forecast_deviation'
    ALERT_DEMAND_SPIKE = 'demand_spike'
    
    # Alert Severity
    SEVERITY_LOW = 'low'
    SEVERITY_MEDIUM = 'medium'
    SEVERITY_HIGH = 'high'
    SEVERITY_CRITICAL = 'critical'
    
    # Purchase Order Status
    PO_STATUS_PENDING = 'pending'
    PO_STATUS_APPROVED = 'approved'
    PO_STATUS_RECEIVED = 'received'
    PO_STATUS_CANCELLED = 'cancelled'
    
    # Payment Methods
    PAYMENT_CASH = 'cash'
    PAYMENT_CARD = 'card'
    PAYMENT_UPI = 'upi'
    PAYMENT_OTHER = 'other'
    
    # Store Types
    STORE_SUPERMARKET = 'supermarket'
    STORE_HYPERMARKET = 'hypermarket'
    STORE_CONVENIENCE = 'convenience'
    
    # Time Frequencies
    FREQ_DAILY = 'D'
    FREQ_WEEKLY = 'W'
    FREQ_MONTHLY = 'M'
    
    # Model Types
    MODEL_SARIMA = 'sarima'
    MODEL_HOLT_WINTERS = 'hw'
    MODEL_RANDOM_FOREST = 'rf'
    MODEL_GRADIENT_BOOSTING = 'gb'
    MODEL_AUTO = 'auto'