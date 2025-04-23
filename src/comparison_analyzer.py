import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MAX_SUBNETS
from src.data_fetcher import DataFetcher
from src.price_predictor import PricePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class ComparisonAnalyzer:
    """
    Class for analyzing and comparing multiple subnets
    """
    
    def __init__(self, data_fetcher: DataFetcher, price_predictor: PricePredictor):
        self.data_fetcher = data_fetcher
        self.price_predictor = price_predictor
    
    def get_top_subnets_by_emission(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top subnets with highest emission values"""
        try:
            subnets = self.data_fetcher.get_subnets_list()
            
            # Sort by emission in descending order
            sorted_subnets = sorted(subnets, key=lambda x: x.get('emission', 0), reverse=True)
            
            # Return top N
            return sorted_subnets[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top subnets by emission: {str(e)}")
            return []
    
    def get_top_subnets_by_price(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top subnets with highest dTAO price"""
        try:
            subnets = self.data_fetcher.get_subnets_list()
            
            # Get price for each subnet
            for subnet in subnets:
                netuid = subnet.get('netuid')
                subnet['price'] = self.data_fetcher.get_subnet_dtao_price(netuid) or 0
            
            # Sort by price in descending order
            sorted_subnets = sorted(subnets, key=lambda x: x.get('price', 0), reverse=True)
            
            # Return top N
            return sorted_subnets[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top subnets by price: {str(e)}")
            return []
    
    def get_top_growth_subnets(self, days: int = 30, limit: int = 5) -> List[Dict[str, Any]]:
        """Get subnets with highest predicted growth potential"""
        try:
            subnets = self.data_fetcher.get_subnets_list()
            growth_data = []
            
            # For efficiency, limit number of subnets to process
            process_subnets = subnets[:min(len(subnets), MAX_SUBNETS)]
            
            for subnet in process_subnets:
                netuid = subnet.get('netuid')
                
                # Predict future prices
                prediction = self.price_predictor.predict_future_prices(netuid, days=days)
                if prediction.get('success', False):
                    growth_score = self._calculate_growth_score(prediction)
                    growth_data.append({
                        'netuid': netuid,
                        'name': subnet.get('name', f"Subnet {netuid}"),
                        'emission': subnet.get('emission', 0),
                        'current_price': prediction.get('current_price', 0),
                        'predicted_price': prediction.get('prediction', {}).get('predicted_price', pd.Series()).iloc[-1] if 'prediction' in prediction else 0,
                        'price_change_percent': prediction.get('price_change_percent', 0),
                        'growth_score': growth_score
                    })
            
            # Sort by growth score in descending order
            sorted_growth = sorted(growth_data, key=lambda x: x.get('growth_score', 0), reverse=True)
            
            # Return top N
            return sorted_growth[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top growth subnets: {str(e)}")
            return []
    
    def _calculate_growth_score(self, prediction: Dict[str, Any]) -> float:
        """Calculate growth potential score for a subnet"""
        # Base score: predicted price change percentage
        price_change = prediction.get('price_change_percent', 0)
        base_score = max(0, price_change) * 0.8
        
        # Model reliability adjustment: adjust score based on R²
        r2 = prediction.get('metrics', {}).get('train_r2', 0)
        reliability_factor = max(0.2, min(1.0, r2))
        
        # Final score
        return base_score * reliability_factor
    
    def compare_subnet_metrics(self, netuids: List[int]) -> pd.DataFrame:
        """Compare key metrics for multiple subnets"""
        try:
            metrics_list = []
            
            for netuid in netuids:
                metrics = self.data_fetcher.get_subnet_metrics(netuid)
                
                # Add price information
                price = self.data_fetcher.get_subnet_dtao_price(netuid) or 0
                metrics['price'] = price
                
                # Calculate additional ratios
                if metrics.get('tau_in', 0) > 0 and metrics.get('alpha_in', 0) > 0:
                    metrics['price_emission_ratio'] = price / max(0.0001, metrics.get('emission', 0))
                    metrics['liquidity_depth'] = metrics.get('tau_in', 0) / price
                
                metrics_list.append(metrics)
            
            # Create DataFrame
            df = pd.DataFrame(metrics_list)
            
            # Select important columns and sort
            columns = ['netuid', 'price', 'emission', 'price_emission_ratio', 
                      'tau_in', 'alpha_in', 'alpha_out', 'liquidity_depth',
                      'active_validators', 'active_miners', 'total_stake']
            
            df = df[[col for col in columns if col in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to compare subnet metrics: {str(e)}")
            return pd.DataFrame()
    
    def compare_price_predictions(self, netuids: List[int], days: int = 30) -> Dict[str, Any]:
        """Compare price predictions for multiple subnets"""
        try:
            predictions = {}
            
            for netuid in netuids:
                prediction = self.price_predictor.predict_future_prices(netuid, days=days)
                if prediction.get('success', False):
                    predictions[netuid] = prediction
            
            if not predictions:
                return {'success': False, 'error': 'No successful prediction results'}
            
            return {
                'success': True,
                'predictions': predictions,
                'days': days
            }
            
        except Exception as e:
            logger.error(f"Failed to compare price predictions: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def visualize_price_comparison(self, comparison_result: Dict[str, Any], 
                                  show_history: bool = True, 
                                  save_path: Optional[str] = None) -> None:
        """Visualize price prediction comparison for multiple subnets"""
        if not comparison_result.get('success', False):
            logger.error(f"Cannot visualize failed comparison: {comparison_result.get('error', 'Unknown error')}")
            return
        
        plt.figure(figsize=(12, 6))
        
        predictions = comparison_result.get('predictions', {})
        if not predictions:
            return
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, (netuid, pred) in enumerate(predictions.items()):
            if show_history:
                # Plot historical prices
                plt.plot(pred['full_data'].index, 
                         pred['full_data']['historical_price'], 
                         '--', 
                         color=colors[i % len(colors)], 
                         alpha=0.5,
                         label=f"Subnet {netuid} History")
            
            # Plot predicted prices
            pred_data = pred['full_data'][pred['full_data']['predicted_price'].notna()]
            plt.plot(pred_data.index, 
                     pred_data['predicted_price'], 
                     '-', 
                     color=colors[i % len(colors)],
                     linewidth=2,
                     label=f"Subnet {netuid} Prediction")
        
        # Add labels and title
        plt.title(f"{comparison_result['days']} Day Subnet dTAO Price Prediction Comparison")
        plt.xlabel('Date')
        plt.ylabel('Price (TAO)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add prediction summary
        summary_text = "Expected Price Change:\n"
        for netuid, pred in predictions.items():
            change = pred['price_change_percent']
            summary_text += f"Subnet {netuid}: {change:.2f}%\n"
        
        # Add text box
        plt.figtext(0.15, 0.15, summary_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison chart saved to {save_path}")
        else:
            plt.show()
    
    def generate_investment_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate investment recommendations based on analysis"""
        try:
            # Get subnets with highest growth potential
            growth_subnets = self.get_top_growth_subnets(limit=limit)
            
            # Get subnets with highest current price
            price_subnets = self.get_top_subnets_by_price(limit=limit)
            
            # Get subnets with highest emission
            emission_subnets = self.get_top_subnets_by_emission(limit=limit)
            
            # Merge all subnets and deduplicate
            all_subnets = []
            seen_netuids = set()
            
            for subnet_list in [growth_subnets, price_subnets, emission_subnets]:
                for subnet in subnet_list:
                    netuid = subnet.get('netuid')
                    if netuid not in seen_netuids:
                        all_subnets.append(subnet)
                        seen_netuids.add(netuid)
            
            # Score all subnets
            scored_subnets = []
            
            for subnet in all_subnets:
                netuid = subnet.get('netuid')
                
                # Get subnet metrics
                metrics = self.data_fetcher.get_subnet_metrics(netuid)
                
                # Calculate growth score (if available)
                growth_score = subnet.get('growth_score', 0)
                
                # Calculate price/emission ratio
                price = subnet.get('price', 0) or self.data_fetcher.get_subnet_dtao_price(netuid) or 0
                emission = metrics.get('emission', 0.001)  # Avoid division by zero
                price_emission_ratio = price / emission
                
                # Calculate liquidity score
                alpha_in = metrics.get('alpha_in', 0)
                alpha_out = metrics.get('alpha_out', 0.001)  # Avoid division by zero
                liquidity_score = alpha_in / alpha_out
                
                # Activity score
                active_score = (metrics.get('active_validators', 0) + metrics.get('active_miners', 0)) / 100
                
                # Combined score
                investment_score = (
                    growth_score * 0.4 +                   # Growth potential
                    price_emission_ratio * 0.2 +           # Price/emission ratio
                    liquidity_score * 0.2 +                # Liquidity score
                    active_score * 0.1 +                   # Activity score
                    metrics.get('emission', 0) * 0.1       # Emission
                )
                
                # Add score and reason
                recommendation = {
                    'netuid': netuid,
                    'name': subnet.get('name', f"Subnet {netuid}"),
                    'investment_score': investment_score,
                    'price': price,
                    'emission': metrics.get('emission', 0),
                    'price_change_percent': subnet.get('price_change_percent', 0),
                    'active_validators': metrics.get('active_validators', 0),
                    'active_miners': metrics.get('active_miners', 0),
                    'recommendation_reason': self._generate_recommendation_reason(subnet, metrics)
                }
                
                scored_subnets.append(recommendation)
            
            # Sort by investment score
            sorted_recommendations = sorted(scored_subnets, key=lambda x: x.get('investment_score', 0), reverse=True)
            
            # Return top N recommendations
            return sorted_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to generate investment recommendations: {str(e)}")
            return []
    
    def _generate_recommendation_reason(self, subnet: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Generate investment recommendation reason based on subnet data"""
        reasons = []
        
        # Growth potential
        price_change = subnet.get('price_change_percent', 0)
        if price_change > 10:
            reasons.append(f"High predicted growth potential ({price_change:.1f}%)")
        
        # Price/emission ratio
        price = subnet.get('price', 0)
        emission = metrics.get('emission', 0.001)
        price_emission_ratio = price / emission
        
        if price_emission_ratio < 1:
            reasons.append("Low price relative to emission")
        elif price_emission_ratio > 10:
            reasons.append("High price relative to emission")
        
        # Liquidity
        alpha_in = metrics.get('alpha_in', 0)
        alpha_out = metrics.get('alpha_out', 0)
        if alpha_in / max(alpha_out, 1) < 2:
            reasons.append("Shallow liquidity pool")
        elif alpha_in / max(alpha_out, 1) > 10:
            reasons.append("Deep liquidity pool")
        
        # Activity
        total_active = metrics.get('active_validators', 0) + metrics.get('active_miners', 0)
        if total_active > 50:
            reasons.append(f"High activity ({total_active} active participants)")
        
        # Emission
        if metrics.get('emission', 0) > 0.3:
            reasons.append(f"High block emission ({metrics.get('emission', 0):.3f} TAO)")
        
        if not reasons:
            reasons.append("Balanced overall metrics")
        
        return ", ".join(reasons)

# Test code
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from price_predictor import PricePredictor
    
    fetcher = DataFetcher()
    predictor = PricePredictor(fetcher)
    analyzer = ComparisonAnalyzer(fetcher, predictor)
    
    # Test getting top 3 subnets by emission
    print("Top 3 subnets by emission:")
    emission_subnets = analyzer.get_top_subnets_by_emission(limit=3)
    for subnet in emission_subnets:
        print(f"Subnet {subnet['netuid']}: emission = {subnet.get('emission', 0)}")
    
    # If there are top 3 subnets, get their metrics for comparison
    if emission_subnets:
        netuids = [s['netuid'] for s in emission_subnets]
        print("\nComparing subnet metrics:")
        metrics_df = analyzer.compare_subnet_metrics(netuids)
        print(metrics_df)
        
        # Compare price predictions
        print("\nComparing price predictions:")
        price_comparison = analyzer.compare_price_predictions(netuids, days=15)
        if price_comparison['success']:
            analyzer.visualize_price_comparison(price_comparison, show_history=True)
    
    # Test investment recommendations
    print("\nGenerating investment recommendations:")
    recommendations = analyzer.generate_investment_recommendations(limit=3)
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. Subnet {rec['netuid']} - Score: {rec['investment_score']:.2f}")
        print(f"   Reason: {rec['recommendation_reason']}")
        print(f"   Current Price: {rec.get('price', 0):.6f} TAO")
        print(f"   Predicted Change: {rec.get('price_change_percent', 0):.2f}%") 