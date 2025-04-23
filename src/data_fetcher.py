import bittensor as bt
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional, Any
import random

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SUBTENSOR_NETWORK, TAOSTATS_API_URL, TAOSTATS_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data fetching class, responsible for getting data related to a subnet from Bittensor and Taostats
    """
    
    def __init__(self):
        self.subtensor = bt.subtensor(network=SUBTENSOR_NETWORK)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TAOSTATS_API_KEY}"
        }
    
    def get_subnets_list(self) -> List[Dict[str, Any]]:
        """Get a list of all subnets and their basic information"""
        try:
            subnet_info = []
            subnet_list = self.subtensor.get_subnets()
            
            for netuid in subnet_list:
                try:
                    subnet_data = self.subtensor.get_subnet_info(netuid=netuid)
                    subnet_info.append({
                        'netuid': netuid,
                        'name': f"Subnet {netuid}",
                        'emission': float(subnet_data.emission_value) if hasattr(subnet_data, 'emission_value') else 0.0,
                        'max_n': int(subnet_data.max_n) if hasattr(subnet_data, 'max_n') else 0,
                        'min_stake': float(subnet_data.min_stake) if hasattr(subnet_data, 'min_stake') else 0.0,
                    })
                except Exception as subnet_error:
                    logger.error(f"Failed to get subnet {netuid} information: {str(subnet_error)}")
            
            return subnet_info
        
        except Exception as e:
            logger.error(f"Failed to get subnet list: {str(e)}")
            return []
    
    def get_subnet_dtao_price(self, netuid: int) -> Optional[float]:
        """Get the current dTAO price for a specific subnet"""
        try:
            subnet_data = self.subtensor.get_subnet_info(netuid=netuid)
            
            # In the dTAO system, price is calculated via tau_in/alpha_in
            price = float(subnet_data.tau_in) / float(subnet_data.alpha_in) if hasattr(subnet_data, 'tau_in') and hasattr(subnet_data, 'alpha_in') else None
            
            # If unable to get directly from Bittensor, try Taostats
            if price <= 0:
                try:
                    price = self._get_taostats_subnet_price(netuid)
                except Exception as e:
                    logger.warning(f"Failed to get price from Taostats: {str(e)}")
            
            return price
        
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} dTAO price: {str(e)}")
            return None
    
    def _get_taostats_subnet_price(self, netuid: int) -> Optional[float]:
        """Get the dTAO price for a specific subnet from Taostats API"""
        try:
            url = f"{TAOSTATS_API_URL}/subnets/{netuid}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    return float(data['price'])
            
            logger.warning(f"Taostats API did not return subnet {netuid} price data")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} price from Taostats: {str(e)}")
            return None
    
    def get_historical_dtao_prices(self, netuid: int, days: int = 30) -> pd.DataFrame:
        """Get historical dTAO price data for a specific subnet over a period of time"""
        try:
            # Calculate start date from current date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format date in API required format
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            url = f"{TAOSTATS_API_URL}/subnets/{netuid}/historical-prices"
            params = {
                "start_date": start_str,
                "end_date": end_str
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'prices' in data and len(data['prices']) > 0:
                    # Convert API response data to DataFrame
                    df = pd.DataFrame(data['prices'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    return df
            
            # If unable to get data from API, generate mock data for testing
            return self._generate_mock_price_data(netuid, days)
            
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} historical prices: {str(e)}")
            # Return mock data in case of error
            return self._generate_mock_price_data(netuid, days)
    
    def _generate_mock_price_data(self, netuid: int, days: int) -> pd.DataFrame:
        """Generate mock price data for testing"""
        logger.warning(f"Generating mock price data for subnet {netuid}")
        
        # Get current price as baseline, if unavailable use random value
        current_price = self.get_subnet_dtao_price(netuid) or random.uniform(0.1, 10)
        
        # Generate past N days dates
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, 0, -1)]
        
        # Generate mock price data with some random fluctuations and slight upward trend
        # Base price fluctuates between 80%-120% of current price
        base_prices = [current_price * random.uniform(0.8, 1.2) for _ in range(days)]
        
        # Add slight trend
        for i in range(1, days):
            base_prices[i] = base_prices[i-1] * random.uniform(0.98, 1.03)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'price': base_prices,
            'volume': np.random.randint(1000, 10000, size=days),
            'market_cap': [p * np.random.randint(10000, 100000) for p in base_prices]
        })
        
        df.set_index('date', inplace=True)
        return df
    
    def get_subnet_metrics(self, netuid: int) -> Dict[str, Any]:
        """Get subnet-related metrics, including liquidity, validator count, miner count, etc."""
        try:
            subnet_data = self.subtensor.get_subnet_info(netuid=netuid)
            
            metrics = {
                'netuid': netuid,
                'tau_in': float(subnet_data.tau_in) if hasattr(subnet_data, 'tau_in') else 0.0,
                'alpha_in': float(subnet_data.alpha_in) if hasattr(subnet_data, 'alpha_in') else 0.0,
                'alpha_out': float(subnet_data.alpha_out) if hasattr(subnet_data, 'alpha_out') else 0.0,
                'total_stake': float(subnet_data.total_stake) if hasattr(subnet_data, 'total_stake') else 0.0,
                'emission': float(subnet_data.emission_value) if hasattr(subnet_data, 'emission_value') else 0.0,
                'active_validators': 0,
                'active_miners': 0,
                'tempo': int(subnet_data.tempo) if hasattr(subnet_data, 'tempo') else 0,
            }
            
            # Get active validators and miners count
            try:
                metagraph = self.subtensor.metagraph(netuid)
                metrics['active_validators'] = len([uid for uid in range(metagraph.n.item()) if metagraph.validator_permit[uid]])
                metrics['active_miners'] = len([uid for uid in range(metagraph.n.item()) if not metagraph.validator_permit[uid]])
            except Exception as mg_error:
                logger.error(f"Failed to get metagraph for subnet {netuid}: {str(mg_error)}")
            
            # Try to get additional metrics from Taostats
            try:
                taostats_metrics = self._get_taostats_subnet_metrics(netuid)
                if taostats_metrics:
                    metrics.update(taostats_metrics)
            except Exception as ts_error:
                logger.error(f"Failed to get additional metrics from Taostats: {str(ts_error)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get subnet {netuid} metrics: {str(e)}")
            return {
                'netuid': netuid,
                'error': str(e)
            }
    
    def _get_taostats_subnet_metrics(self, netuid: int) -> Dict[str, Any]:
        """Get additional subnet metrics from Taostats API"""
        try:
            url = f"{TAOSTATS_API_URL}/subnets/{netuid}/metrics"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get additional metrics from Taostats: {str(e)}")
            return {}


# Test code
if __name__ == "__main__":
    data_fetcher = DataFetcher()
    
    # Test getting subnet list
    subnets = data_fetcher.get_subnets_list()
    print(f"Found {len(subnets)} subnets")
    
    # Test getting first subnet info
    if subnets:
        first_subnet = subnets[0]['netuid']
        print(f"\nGetting info for subnet {first_subnet}")
        
        # Get current price
        price = data_fetcher.get_subnet_dtao_price(first_subnet)
        print(f"Current price: {price}")
        
        # Get historical price data
        history = data_fetcher.get_historical_dtao_prices(first_subnet, days=30)
        print(f"Historical data points: {len(history)}")
        
        # Get subnet metrics
        metrics = data_fetcher.get_subnet_metrics(first_subnet)
        print(f"Subnet metrics: {metrics}") 