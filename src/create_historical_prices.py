#!/usr/bin/env python3
"""
Create Steam Historical Prices CSV and Game-Level Analysis Data
=============================================================

This script takes the Kaggle Steam sales dataset and enriches it with
historical price data and game metadata from the IsThereAnyDeal (ITAD) API.

Input:  steam_sales.csv (from Kaggle)
Output: 
- steam_historical_prices.csv (time-series format: one row per price record)
- steam_games_analysis.csv (game-level format: one row per game with time-series as lists)

Features:
- Fetches historical price data for each game
- Retrieves game metadata: release dates, publishers, developers, tags
- Collects ratings: Steam and Metacritic scores
- Gathers player statistics: recent and peak player counts
- Creates both time-series and game-level data formats for different analysis needs
- Game-level format stores time-series data as JSON lists for easy analysis
"""

import pandas as pd
import time
import os
import sys
import argparse
import json
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
    
    Note: 
    - Uses German (DE) pricing to match EUR currency in Steam dataset
    - Fetches comprehensive game metadata from ITAD API including:
      * Release dates, publishers, developers, tags
      * Steam and Metacritic ratings
      * Recent and peak player counts
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
            
            # Get detailed game information including release date, publisher, ratings, etc.
            detailed_game_info = itad_client.get_game_info(game_id)
            
            # Extract various game information
            itad_release_date = None
            itad_publishers = None
            itad_developers = None
            itad_tags = None
            itad_steam_rating = None
            itad_metacritic_rating = None
            itad_players_recent = None
            itad_players_peak = None
            
            if detailed_game_info:
                itad_release_date = detailed_game_info.get('releaseDate')
                
                # Extract publishers
                if 'publishers' in detailed_game_info and detailed_game_info['publishers']:
                    itad_publishers = ', '.join([pub['name'] for pub in detailed_game_info['publishers']])
                
                # Extract developers
                if 'developers' in detailed_game_info and detailed_game_info['developers']:
                    itad_developers = ', '.join([dev['name'] for dev in detailed_game_info['developers']])
                
                # Extract tags
                if 'tags' in detailed_game_info and detailed_game_info['tags']:
                    itad_tags = ', '.join(detailed_game_info['tags'])
                
                # Extract ratings
                if 'reviews' in detailed_game_info and detailed_game_info['reviews']:
                    for review in detailed_game_info['reviews']:
                        if review.get('source') == 'Steam':
                            itad_steam_rating = review.get('score')
                        elif review.get('source') == 'Metascore':
                            itad_metacritic_rating = review.get('score')
                
                # Extract player counts
                if 'players' in detailed_game_info and detailed_game_info['players']:
                    itad_players_recent = detailed_game_info['players'].get('recent')
                    itad_players_peak = detailed_game_info['players'].get('peak')
                
                print(f"    → Release date: {itad_release_date}")
                print(f"    → Publishers: {itad_publishers}")
                print(f"    → Steam rating: {itad_steam_rating}")
            else:
                print(f"    → Game info: Not found")
            
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
                        'Shop ID': price_record.get('shop', {}).get('id', 61),
                        'ITAD Release Date': itad_release_date,
                        'ITAD Publishers': itad_publishers,
                        'ITAD Developers': itad_developers,
                        'ITAD Tags': itad_tags,
                        'ITAD Steam Rating': itad_steam_rating,
                        'ITAD Metacritic Rating': itad_metacritic_rating,
                        'ITAD Players Recent': itad_players_recent,
                        'ITAD Players Peak': itad_players_peak
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
        
        # Check how many games have various ITAD data
        games_with_release_dates = price_df[price_df['ITAD Release Date'].notna()]['Game Name'].nunique()
        games_with_publishers = price_df[price_df['ITAD Publishers'].notna()]['Game Name'].nunique()
        games_with_ratings = price_df[price_df['ITAD Steam Rating'].notna()]['Game Name'].nunique()
        games_with_tags = price_df[price_df['ITAD Tags'].notna()]['Game Name'].nunique()
        
        print(f"Games with release dates: {games_with_release_dates}/{processed_count}")
        print(f"Games with publishers: {games_with_publishers}/{processed_count}")
        print(f"Games with Steam ratings: {games_with_ratings}/{processed_count}")
        print(f"Games with tags: {games_with_tags}/{processed_count}")
        
        return price_df
    else:
        print("No price data collected")
        return None

def transform_to_game_level(df):
    """
    Transform time-series data to game-level data
    
    Args:
        df: DataFrame with historical price data
        
    Returns:
        DataFrame with one row per game
    """
    print("Transforming to game-level data...")
    
    # Group by game and aggregate data
    game_data = []
    
    for game_name, group in df.groupby('Game Name'):
        # Static metadata (take first non-null value for each field)
        static_data = {
            'Game Name': game_name,
            'ITAD Game ID': group['ITAD Game ID'].iloc[0] if 'ITAD Game ID' in group.columns else None,
            'ITAD Release Date': group['ITAD Release Date'].dropna().iloc[0] if group['ITAD Release Date'].notna().any() else None,
            'ITAD Publishers': group['ITAD Publishers'].dropna().iloc[0] if group['ITAD Publishers'].notna().any() else None,
            'ITAD Developers': group['ITAD Developers'].dropna().iloc[0] if group['ITAD Developers'].notna().any() else None,
            'ITAD Tags': group['ITAD Tags'].dropna().iloc[0] if group['ITAD Tags'].notna().any() else None,
            'ITAD Steam Rating': group['ITAD Steam Rating'].dropna().iloc[0] if group['ITAD Steam Rating'].notna().any() else None,
            'ITAD Metacritic Rating': group['ITAD Metacritic Rating'].dropna().iloc[0] if group['ITAD Metacritic Rating'].notna().any() else None,
            'ITAD Players Recent': group['ITAD Players Recent'].dropna().iloc[0] if group['ITAD Players Recent'].notna().any() else None,
            'ITAD Players Peak': group['ITAD Players Peak'].dropna().iloc[0] if group['ITAD Players Peak'].notna().any() else None,
            # Static price and shop info (these don't change over time)
            'Original Price (€)': group['Original Price (€)'].iloc[0] if 'Original Price (€)' in group.columns else None,
            'Shop Name': group['Shop Name'].iloc[0] if 'Shop Name' in group.columns else None,
            'Shop ID': group['Shop ID'].iloc[0] if 'Shop ID' in group.columns else None,
        }
        
        # Time-series data as lists (sorted by date)
        group_sorted = group.sort_values('Date')
        
        time_series_data = {
            'Price History (€)': group_sorted['Price (€)'].tolist(),
            'Discount History (%)': group_sorted['Discount%'].tolist(),
            'Price Dates': group_sorted['Date'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        }
        
        # Calculate summary statistics
        summary_stats = {
            'Total Price Records': len(group),
            'Date Range Start': group_sorted['Date'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'Date Range End': group_sorted['Date'].max().strftime('%Y-%m-%d %H:%M:%S'),
            'Min Price (€)': group_sorted['Price (€)'].min(),
            'Max Price (€)': group_sorted['Price (€)'].max(),
            'Min Original Price (€)': group_sorted['Original Price (€)'].min(),
            'Max Original Price (€)': group_sorted['Original Price (€)'].max(),
            'Min Discount (%)': group_sorted['Discount%'].min(),
            'Max Discount (%)': group_sorted['Discount%'].max(),
            'Avg Discount (%)': group_sorted['Discount%'].mean(),
            'Current Price (€)': group_sorted['Price (€)'].iloc[-1] if len(group_sorted) > 0 else None,
            'Current Original Price (€)': group_sorted['Original Price (€)'].iloc[-1] if len(group_sorted) > 0 else None,
            'Current Discount (%)': group_sorted['Discount%'].iloc[-1] if len(group_sorted) > 0 else None,
        }
        
        # Combine all data
        game_record = {**static_data, **time_series_data, **summary_stats}
        game_data.append(game_record)
    
    return pd.DataFrame(game_data)

def save_game_level_data(df, output_path):
    """Save the game-level data to CSV"""
    try:
        # Convert lists to JSON strings for CSV compatibility
        df_copy = df.copy()
        
        # Convert list columns to JSON strings
        list_columns = [
            'Price History (€)', 'Discount History (%)', 'Price Dates'
        ]
        
        for col in list_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        
        df_copy.to_csv(output_path, index=False)
        print(f"Saved game-level data to: {output_path}")
        print(f"Games: {len(df_copy)} | Columns: {len(df_copy.columns)}")
        
        return df_copy
    except Exception as e:
        print(f"Error saving game-level data: {e}")
        return None

def create_analysis_summary(df):
    """Create a summary of the transformed data"""
    print("\n=== GAME-LEVEL DATA SUMMARY ===")
    print(f"Total games: {len(df)}")
    print(f"Games with release dates: {df['ITAD Release Date'].notna().sum()}")
    print(f"Games with publishers: {df['ITAD Publishers'].notna().sum()}")
    print(f"Games with Steam ratings: {df['ITAD Steam Rating'].notna().sum()}")
    print(f"Games with tags: {df['ITAD Tags'].notna().sum()}")
    
    print(f"\nPrice data coverage:")
    print(f"Games with price history: {df['Total Price Records'].gt(0).sum()}")
    print(f"Average price records per game: {df['Total Price Records'].mean():.1f}")
    print(f"Max price records for a game: {df['Total Price Records'].max()}")
    
    print(f"\nDate range:")
    print(f"Earliest price: {df['Date Range Start'].min()}")
    print(f"Latest price: {df['Date Range End'].max()}")
    
    print(f"\nDiscount statistics:")
    print(f"Games with discounts: {df['Max Discount (%)'].gt(0).sum()}")
    print(f"Average max discount: {df['Max Discount (%)'].mean():.1f}%")
    print(f"Highest discount: {df['Max Discount (%)'].max():.1f}%")

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
        GAME_LEVEL_CSV_PATH = "data/steam_games_analysis_test.csv"
        max_games = 5
        print("DEBUG MODE: Processing first 5 games")
    else:
        OUTPUT_CSV_PATH = "data/steam_historical_prices.csv"
        GAME_LEVEL_CSV_PATH = "data/steam_games_analysis.csv"
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
        
        # Transform to game-level data
        print("\n" + "="*50)
        print("CREATING GAME-LEVEL ANALYSIS DATA")
        print("="*50)
        
        game_level_df = transform_to_game_level(result_df)
        
        if game_level_df is not None and len(game_level_df) > 0:
            # Save game-level data
            game_result = save_game_level_data(game_level_df, GAME_LEVEL_CSV_PATH)
            
            if game_result is not None:
                # Create summary
                create_analysis_summary(game_result)
                print(f"\nSUCCESS: {GAME_LEVEL_CSV_PATH}")
            else:
                print("FAILED: Could not save game-level data")
        else:
            print("FAILED: Could not transform to game-level data")
    else:
        print("FAILED: No data collected")

if __name__ == "__main__":
    main()
