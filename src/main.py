#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from typing import Dict, List, Any, Optional

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SUBTENSOR_NETWORK, PREDICTION_WINDOW, MAX_SUBNETS
from src.data_fetcher import DataFetcher
from src.price_predictor import PricePredictor
from src.comparison_analyzer import ComparisonAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dtao_predictor.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bittensor dTAO Price Prediction Tool', add_help=True)
    
    # Main command options
    parser.add_argument('--network', type=str, default=SUBTENSOR_NETWORK,
                        help=f'Bittensor network (default: {SUBTENSOR_NETWORK})')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # List subnets command
    list_parser = subparsers.add_parser('list', help='List subnet information')
    list_parser.add_argument('--limit', type=int, default=10,
                          help='Number of subnets to display (default: 10)')
    list_parser.add_argument('--sort', type=str, choices=['emission', 'price'], default='emission',
                           help='Sort method (default: emission)')
    
    # View subnet details command
    info_parser = subparsers.add_parser('info', help='View detailed subnet information')
    info_parser.add_argument('netuid', type=int, help='Subnet UID')
    
    # Predict subnet price command
    predict_parser = subparsers.add_parser('predict', help='Predict subnet dTAO price')
    predict_parser.add_argument('netuid', type=int, help='Subnet UID')
    predict_parser.add_argument('--days', type=int, default=PREDICTION_WINDOW,
                              help=f'Prediction days (default: {PREDICTION_WINDOW})')
    predict_parser.add_argument('--model', type=str, 
                              choices=['random_forest', 'linear', 'svr', 'lstm', 'arima', 'xgboost', 'prophet'], 
                              default='random_forest', help='Prediction model (default: random_forest)')
    predict_parser.add_argument('--save', type=str, help='Path to save prediction chart')
    
    # Compare subnets command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple subnets')
    compare_parser.add_argument('netuids', type=int, nargs='+', help='List of subnet UIDs')
    compare_parser.add_argument('--days', type=int, default=30,
                              help='Prediction days (default: 30)')
    compare_parser.add_argument('--save', type=str, help='Path to save comparison chart')
    
    # Investment advice command
    recommend_parser = subparsers.add_parser('recommend', help='Get investment advice')
    recommend_parser.add_argument('--limit', type=int, default=5,
                                help='Number of recommendations (default: 5)')
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    return args

def display_subnets_list(subnets: List[Dict[str, Any]]):
    """Format and display subnet list"""
    if not subnets:
        print("No subnets found")
        return
    
    print(f"\n{'=' * 70}")
    print(f"{'Subnet UID':<10} {'Name':<20} {'Price(TAO)':<15} {'Emission':<15} {'Max Nodes':<10}")
    print(f"{'-' * 70}")
    
    for subnet in subnets:
        netuid = subnet.get('netuid', 'N/A')
        name = subnet.get('name', f'Subnet {netuid}')
        price = subnet.get('price', 0)
        emission = subnet.get('emission', 0)
        max_n = subnet.get('max_n', 0)
        
        print(f"{netuid:<10} {name:<20} {price:<15.6f} {emission:<15.6f} {max_n:<10}")
    
    print(f"{'=' * 70}\n")

def display_subnet_info(metrics: Dict[str, Any], price: Optional[float] = None):
    """Format and display detailed subnet information"""
    if not metrics:
        print("Subnet information not found")
        return
    
    netuid = metrics.get('netuid', 'N/A')
    print(f"\n{'=' * 70}")
    print(f"Subnet {netuid} Detailed Information")
    print(f"{'-' * 70}")
    
    # Basic information
    print(f"Price(TAO): {price or 0:.6f}")
    print(f"Emission: {metrics.get('emission', 0):.6f}")
    
    # Liquidity pool information
    print(f"\nLiquidity Pool Information:")
    print(f"  TAO Reserve(tau_in): {metrics.get('tau_in', 0):.6f}")
    print(f"  Alpha Reserve(alpha_in): {metrics.get('alpha_in', 0):.6f}")
    print(f"  Alpha Circulation(alpha_out): {metrics.get('alpha_out', 0):.6f}")
    
    # Participant information
    print(f"\nParticipant Information:")
    print(f"  Active Validators: {metrics.get('active_validators', 0)}")
    print(f"  Active Miners: {metrics.get('active_miners', 0)}")
    
    # Other information
    print(f"\nOther Information:")
    print(f"  Total Stake: {metrics.get('total_stake', 0):.6f}")
    print(f"  Tempo: {metrics.get('tempo', 0)}")
    
    print(f"{'=' * 70}\n")

def display_prediction_result(prediction: Dict[str, Any]):
    """Format and display prediction results"""
    if not prediction.get('success', False):
        print(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
        return
    
    netuid = prediction.get('netuid', 'N/A')
    current_price = prediction.get('current_price', 0)
    predicted_df = prediction.get('prediction')
    if predicted_df is None or predicted_df.empty:
        print("No prediction data")
        return
    
    last_predicted = predicted_df['predicted_price'].iloc[-1]
    change_percent = prediction.get('price_change_percent', 0)
    
    print(f"\n{'=' * 70}")
    print(f"Subnet {netuid} dTAO Price Prediction")
    print(f"{'-' * 70}")
    print(f"Current price: {current_price:.6f} TAO")
    print(f"Predicted end price: {last_predicted:.6f} TAO")
    print(f"Expected change: {change_percent:.2f}%")
    print(f"Model: {prediction.get('model_name', 'unknown')}")
    
    # Display model performance metrics
    metrics = prediction.get('metrics', {})
    print(f"\nModel Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    
    # Display prediction trend
    print(f"\nPrediction Trend (every 7 days):")
    step = max(1, len(predicted_df) // 5)
    for i in range(0, len(predicted_df), step):
        date = predicted_df.index[i].strftime("%Y-%m-%d")
        price = predicted_df['predicted_price'].iloc[i]
        lower = predicted_df['lower_bound'].iloc[i]
        upper = predicted_df['upper_bound'].iloc[i]
        print(f"  {date}: {price:.6f} TAO (range: {lower:.6f} - {upper:.6f})")
    
    print(f"{'=' * 70}\n")

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Format and display investment recommendations"""
    if not recommendations:
        print("No investment recommendations available")
        return
    
    print(f"\n{'=' * 70}")
    print(f"dTAO Subnet Investment Recommendations")
    print(f"{'-' * 70}")
    
    for i, rec in enumerate(recommendations):
        netuid = rec.get('netuid', 'N/A')
        score = rec.get('investment_score', 0)
        reason = rec.get('recommendation_reason', 'None')
        price = rec.get('price', 0)
        change = rec.get('price_change_percent', 0)
        
        print(f"{i+1}. Subnet {netuid} - Investment Score: {score:.2f}")
        print(f"   Current Price: {price:.6f} TAO")
        print(f"   Predicted 30-day Change: {change:.2f}%")
        print(f"   Recommendation Reason: {reason}")
        print(f"   Active Validators: {rec.get('active_validators', 0)}    Active Miners: {rec.get('active_miners', 0)}")
        print()
    
    print(f"{'=' * 70}\n")

def main():
    """Main program entry"""
    args = parse_args()
    
    try:
        # Initialize modules
        data_fetcher = DataFetcher()
        price_predictor = PricePredictor(data_fetcher)
        analyzer = ComparisonAnalyzer(data_fetcher, price_predictor)
        
        # Execute corresponding operation based on subcommand
        if args.command == 'list':
            print(f"Getting subnet list (sort method: {args.sort})...")
            if args.sort == 'emission':
                subnets = analyzer.get_top_subnets_by_emission(limit=args.limit)
            else:  # price
                subnets = analyzer.get_top_subnets_by_price(limit=args.limit)
            display_subnets_list(subnets)
        
        elif args.command == 'info':
            print(f"Getting detailed information for subnet {args.netuid}...")
            metrics = data_fetcher.get_subnet_metrics(args.netuid)
            price = data_fetcher.get_subnet_dtao_price(args.netuid)
            display_subnet_info(metrics, price)
        
        elif args.command == 'predict':
            print(f"Predicting dTAO price for subnet {args.netuid} for the next {args.days} days...")
            prediction = price_predictor.predict_future_prices(
                args.netuid, days=args.days, model_name=args.model
            )
            display_prediction_result(prediction)
            
            if prediction.get('success', False):
                price_predictor.visualize_prediction(prediction, save_path=args.save)
                if args.save:
                    print(f"Prediction chart saved to: {args.save}")
        
        elif args.command == 'compare':
            print(f"Comparing dTAO price predictions for subnets {args.netuids}...")
            
            # First compare basic metrics
            metrics_df = analyzer.compare_subnet_metrics(args.netuids)
            if not metrics_df.empty:
                print("\nSubnet Metrics Comparison:")
                print(metrics_df.to_string())
            
            # Compare price predictions
            comparison = analyzer.compare_price_predictions(args.netuids, days=args.days)
            if comparison.get('success', False):
                print(f"\nSubnet Price Prediction Comparison (next {args.days} days):")
                for netuid, pred in comparison.get('predictions', {}).items():
                    price_change = pred.get('price_change_percent', 0)
                    print(f"  Subnet {netuid}: Expected change {price_change:.2f}%")
                
                analyzer.visualize_price_comparison(comparison, save_path=args.save)
                if args.save:
                    print(f"Comparison chart saved to: {args.save}")
            else:
                print(f"Comparison failed: {comparison.get('error', 'Unknown error')}")
        
        elif args.command == 'recommend':
            print(f"Generating top {args.limit} investment recommendations...")
            recommendations = analyzer.generate_investment_recommendations(limit=args.limit)
            display_recommendations(recommendations)
        
        return 0
    
    except KeyboardInterrupt:
        print("\nOperation interrupted")
        return 1
    except Exception as e:
        logger.error(f"Execution error: {str(e)}", exc_info=True)
        print(f"\nError occurred during execution: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 