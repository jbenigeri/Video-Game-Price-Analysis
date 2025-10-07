#!/usr/bin/env python3
"""
Create Steam Historical Prices CSV
=================================

This script takes the Kaggle Steam sales dataset and enriches it with
historical price data from the IsThereAnyDeal (ITAD) API.

Input:  steam_sales.csv (from Kaggle)
Output: steam_historical_prices.csv (enriched with ITAD price history)
"""

import pandas as pd
import time
import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path so we can import our integration module
sys.path.append(str(Path(__file__).parent))

from steam_itad_integration import SteamITADIntegration

def load_steam_sales_data(csv_path):
    """Load the Steam sales dataset from Kaggle"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from Steam sales dataset")
        return df
    except Exception as e:
        print(f"Error loading Steam sales data: {e}")
        return None

def get_unique_game_names(df):
    """Extract unique game names from the Steam dataset"""
    if 'Game Name' in df.columns:
        return df['Game Name'].unique()
    else:
        print("Error: 'Game Name' column not found in Steam dataset")
        return []

def create_historical_prices_csv(steam_df, output_path, api_key, max_games=None):
    """
    Create historical prices CSV by enriching Steam data with ITAD API
    
    Args:
        steam_df: DataFrame from steam_sales.csv
        output_path: Path for output CSV file
        api_key: ITAD API key
        max_games: Maximum number of games to process (for testing)
    
    Note: Uses German (DE) pricing to match EUR currency in Steam dataset
    """
    itad_client = SteamITADIntegration(api_key)
    unique_games = get_unique_game_names(steam_df)
    
    if max_games:
        unique_games = unique_games[:max_games]
        print(f"Processing first {max_games} games")
    
    all_price_data = []
    processed_count = 0
    error_count = 0
    
    print(f"Processing {len(unique_games)} games...")
    
    for i, game_name in enumerate(unique_games, 1):
        print(f"[{i:3d}/{len(unique_games)}] {game_name}")
        
        try:
            game_info = itad_client.search_game(game_name)
            if not game_info:
                error_count += 1
                continue
            
            game_id = game_info['id']
            price_history = itad_client.get_price_history(game_id, country="DE")
            
            if not price_history:
                error_count += 1
                continue
            
            for price_record in price_history:
                try:
                    deal = price_record.get('deal', {})
                    price_info = deal.get('price', {})
                    regular_info = deal.get('regular', {})
                    
                    price_data = {
                        'Game Name': game_name,
                        'ITAD Game ID': game_id,
                        'Date': price_record.get('timestamp', ''),
                        'Price (€)': price_info.get('amount', 0),
                        'Original Price (€)': regular_info.get('amount', 0),
                        'Discount%': deal.get('cut', 0),
                        'Currency': price_info.get('currency', 'EUR'),
                        'Shop Name': price_record.get('shop', {}).get('name', 'Steam'),
                        'Shop ID': price_record.get('shop', {}).get('id', 61)
                    }
                    all_price_data.append(price_data)
                except Exception:
                    continue
            
            processed_count += 1
            
        except Exception:
            error_count += 1
            continue
        
        time.sleep(0.25)  # Rate limiting
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(unique_games)} | Success: {processed_count} | Errors: {error_count} | Records: {len(all_price_data)}")
    
    if all_price_data:
        price_df = pd.DataFrame(all_price_data)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df = price_df.sort_values(['Game Name', 'Date'])
        price_df.to_csv(output_path, index=False)
        
        print(f"\nCreated: {output_path}")
        print(f"Records: {len(price_df)} | Games: {processed_count} | Errors: {error_count}")
        print(f"Date range: {price_df['Date'].min()} to {price_df['Date'].max()}")
        
        return price_df
    else:
        print("No price data collected")
        return None

def main():
    """Main function to run the historical prices creation"""
    parser = argparse.ArgumentParser(description='Create Steam Historical Prices CSV from ITAD API')
    parser.add_argument('--debug', action='store_true', 
                       help='Debug mode: process only first 5 games and add _test to output filename')
    args = parser.parse_args()
    
    STEAM_CSV_PATH = "data/steam_sales.csv"
    API_KEY = "fbe8607111c774e3c53089d038ac7494046873ca"
    
    if args.debug:
        OUTPUT_CSV_PATH = "data/steam_historical_prices_test.csv"
        max_games = 5
        print("DEBUG MODE: Processing first 5 games")
    else:
        OUTPUT_CSV_PATH = "data/steam_historical_prices.csv"
        max_games = None
        print("FULL MODE: Processing all games")
    
    if not os.path.exists(STEAM_CSV_PATH):
        print(f"Error: Steam sales CSV not found at: {STEAM_CSV_PATH}")
        return
    
    steam_df = load_steam_sales_data(STEAM_CSV_PATH)
    if steam_df is None:
        return
    
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    result_df = create_historical_prices_csv(steam_df, OUTPUT_CSV_PATH, API_KEY, max_games)
    
    if result_df is not None:
        print(f"SUCCESS: {OUTPUT_CSV_PATH}")
    else:
        print("FAILED: No data collected")

if __name__ == "__main__":
    main()
