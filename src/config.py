# Bittensor related configuration
SUBTENSOR_NETWORK = "finney"  # Options: finney, local, mock

# Taostats API related configuration
TAOSTATS_API_URL = "https://api.taostats.io"
TAOSTATS_API_KEY = "your_api_key"  # Please replace with actual API key

# dTAO price prediction related configuration
PREDICTION_WINDOW = 30  # Predict for future 30 days
HISTORICAL_WINDOW = 60  # Use past 60 days of data for training
MAX_SUBNETS = 150  # Adjusted to handle 100+ subnets with room for expansion

# Interface configuration
DEFAULT_DISPLAY_LIMIT = 20  # Default to display top 20 subnets
SEARCH_ENABLED = True  # Enable search functionality
ENABLE_CACHING = True  # Enable data caching
CACHE_TIMEOUT = 300  # Cache timeout in seconds

# Model configuration
MODEL_CONFIGS = {
    'lstm': {
        'units': 50,
        'dropout': 0.2, 
        'epochs': 50,
        'batch_size': 32,
        'time_steps': 7
    },
    'arima': {
        'order': (5, 1, 0)
    },
    'xgboost': {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'num_boost_round': 100
    },
    'prophet': {
        'daily_seasonality': True
    }
}

# Default prediction model
DEFAULT_MODEL = 'random_forest' 