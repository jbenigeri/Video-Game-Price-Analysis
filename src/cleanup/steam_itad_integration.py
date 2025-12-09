#!/usr/bin/env python3
"""
Steam ITAD Integration - Production Ready

This module provides functions to integrate Steam game data with IsThereAnyDeal API
for comprehensive price analysis. All functions are focused on Steam-only data.

Author: Video Game Price Analysis Project
"""

import requests
import json
from datetime import datetime, timedelta

# API Configuration
API_KEY = "fbe8607111c774e3c53089d038ac7494046873ca"
BASE_URL = "https://api.isthereanydeal.com"

class SteamITADIntegration:
    """Main class for Steam-ITAD integration."""
    
    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key or API_KEY
        self.base_url = BASE_URL
    
    def search_game(self, game_name, results=5):
        """Search for a game by name and return the first result.
        
        Args:
            game_name (str): Name of the game to search for
            results (int): Maximum number of results to return (default: 5)
            
        Returns:
            dict: Game data with ID, title, etc. or None if not found
        """
        url = f"{self.base_url}/games/search/v1"
        params = {
            'key': self.api_key,
            'title': game_name,
            'results': results
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data[0] if data else None
        return None
    
    def get_game_prices(self, game_id, country="DE"):
        """Get current Steam prices for a game.
        
        Args:
            game_id (str): ITAD game ID
            country (str): Country code (default: 'DE')
            
        Returns:
            dict: Current price data or None if not found
        """
        url = f"{self.base_url}/games/prices/v3"
        params = {
            'key': self.api_key,
            'country': country,
            'shops': '61'  # Steam ID
        }
        payload = [game_id]
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, params=params, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data[0] if data else None
        return None
    
    def get_game_overview(self, game_id, country="DE"):
        """Get Steam price overview for a game.
        
        Args:
            game_id (str): ITAD game ID
            country (str): Country code (default: 'DE')
            
        Returns:
            dict: Price overview data or None if not found
        """
        url = f"{self.base_url}/games/overview/v2"
        params = {
            'key': self.api_key,
            'country': country,
            'shops': '61'  # Steam ID
        }
        payload = [game_id]
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, params=params, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['prices'][0] if data.get('prices') else None
        return None
    
    def get_price_history(self, game_id, country="DE", since=None):
        """Get Steam price history for a game.
        
        Args:
            game_id (str): ITAD game ID
            country (str): Country code (default: 'DE')
            since (str): Start date in ISO format (default: None for full history)
            
        Returns:
            list: Price history records or empty list if not found
        """
        url = f"{self.base_url}/games/history/v2"
        params = {
            'key': self.api_key,
            'id': game_id,
            'country': country,
            'shops': '61'  # Steam ID
        }
        
        if since:
            params['since'] = since
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return []
    
    def get_price_history_since_date(self, game_id, start_date, country="DE"):
        """Get Steam price history since a specific date.
        
        Args:
            game_id (str): ITAD game ID
            start_date (str): Start date in ISO format (e.g., '2020-01-01T00:00:00')
            country (str): Country code (default: 'DE')
            
        Returns:
            list: Price history records since the specified date
        """
        return self.get_price_history(game_id, country=country, since=start_date)
    
    def get_multiple_games_prices(self, game_ids):
        """Get current Steam prices for multiple games.
        
        Args:
            game_ids (list): List of ITAD game IDs
            
        Returns:
            list: List of price data for each game
        """
        url = f"{self.base_url}/games/prices/v3"
        params = {
            'key': self.api_key,
            'shops': '61'  # Steam ID
        }
        payload = game_ids
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, params=params, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        return []
    
    def get_multiple_games_overview(self, game_ids):
        """Get Steam price overview for multiple games.
        
        Args:
            game_ids (list): List of ITAD game IDs
            
        Returns:
            list: List of overview data for each game
        """
        url = f"{self.base_url}/games/overview/v2"
        params = {
            'key': self.api_key,
            'shops': '61'  # Steam ID
        }
        payload = game_ids
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, params=params, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('prices', [])
        return []
    
    def get_game_info(self, game_id):
        """Get detailed game information including release date.
        
        Args:
            game_id (str): ITAD game ID
            
        Returns:
            dict: Game information including release date or None if not found
        """
        url = f"{self.base_url}/games/info/v2"
        params = {
            'key': self.api_key,
            'id': game_id  # This should work as a single ID
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # The API returns the game data directly, not wrapped in a 'data' object
            if data and data.get('id') == game_id:
                return data
        return None


# Example usage functions
def example_basic_usage():
    """Example: Basic usage of Steam ITAD integration."""
    print("=== Basic Usage Example ===")
    
    # Initialize the integration
    itad = SteamITADIntegration()
    
    # Search for a game
    print("1. Searching for 'Cyberpunk 2077'...")
    game = itad.search_game("Cyberpunk 2077")
    if not game:
        print("Game not found!")
        return
    
    print(f"✅ Found: {game['title']} (ID: {game['id']})")
    
    # Get current prices
    print("\n2. Getting current Steam prices...")
    prices = itad.get_game_prices(game['id'])
    if prices:
        deals = prices.get('deals', [])
        if deals:
            deal = deals[0]  # Should be Steam only
            print(f"   Current Price: ${deal['price']['amount']} {deal['price']['currency']}")
            print(f"   Regular Price: ${deal['regular']['amount']} {deal['regular']['currency']}")
            print(f"   Discount: {deal['cut']}%")
        else:
            print("   No current Steam deals")
    
    # Get price overview
    print("\n3. Getting price overview...")
    overview = itad.get_game_overview(game['id'])
    if overview:
        current = overview.get('current', {})
        if current:
            print(f"   Current: ${current['price']['amount']} {current['price']['currency']}")
            print(f"   Regular: ${current['regular']['amount']} {current['regular']['currency']}")
            print(f"   Discount: {current['cut']}%")


def example_price_history():
    """Example: Getting price history for a game."""
    print("\n=== Price History Example ===")
    
    itad = SteamITADIntegration()
    
    # Search for a game
    game = itad.search_game("Rust")
    if not game:
        print("Game not found!")
        return
    
    print(f"Game: {game['title']} (ID: {game['id']})")
    
    # Get full price history since release
    print("\n1. Getting full Steam price history since release...")
    history = itad.get_price_history(game['id'])
    
    if history:
        print(f"✅ Found {len(history)} price changes")
        
        # Show date range
        if len(history) > 1:
            oldest = history[-1]['timestamp']
            newest = history[0]['timestamp']
            print(f"   Date range: {oldest} to {newest}")
        
        # Show recent changes
        print("\n   Recent price changes (last 5):")
        for i, change in enumerate(history[:5], 1):
            price = change['deal']['price']['amount']
            currency = change['deal']['price']['currency']
            discount = change['deal']['cut']
            timestamp = change['timestamp']
            print(f"   {i}. {timestamp}: ${price} {currency} ({discount}% off)")
    else:
        print("❌ No price history found")


def example_europe_prices():
    """Example: Getting European Steam prices."""
    print("\n=== European Prices Example ===")
    
    itad = SteamITADIntegration()
    
    # Search for a game
    game = itad.search_game("Cyberpunk 2077")
    if not game:
        print("Game not found!")
        return
    
    print(f"Game: {game['title']} (ID: {game['id']})")
    
    # Get EU price history
    print("\n1. Getting EU Steam price history...")
    history_eu = itad.get_price_history(game['id'], country="DE")
    
    if history_eu:
        print(f"✅ Found {len(history_eu)} EU price changes")
        
        # Show recent EU prices
        print("\n   Recent EU price changes (last 3):")
        for i, change in enumerate(history_eu[:3], 1):
            price = change['deal']['price']['amount']
            currency = change['deal']['price']['currency']
            discount = change['deal']['cut']
            timestamp = change['timestamp']
            print(f"   {i}. {timestamp}: {price} {currency} ({discount}% off)")
    else:
        print("❌ No EU price history found")
    
    # Get current EU prices
    print("\n2. Getting current EU Steam prices...")
    prices_eu = itad.get_game_prices(game['id'])
    if prices_eu:
        deals = prices_eu.get('deals', [])
        if deals:
            deal = deals[0]
            print(f"   Current EU Price: {deal['price']['amount']} {deal['price']['currency']}")
            print(f"   Regular EU Price: {deal['regular']['amount']} {deal['regular']['currency']}")
            print(f"   Discount: {deal['cut']}%")


def example_custom_date_range():
    """Example: Getting price history for a custom date range."""
    print("\n=== Custom Date Range Example ===")
    
    itad = SteamITADIntegration()
    
    # Search for a game
    game = itad.search_game("Rust")
    if not game:
        print("Game not found!")
        return
    
    print(f"Game: {game['title']} (ID: {game['id']})")
    
    # Get price history for last 2 years
    print("\n1. Getting Steam price history for last 2 years...")
    two_years_ago = (datetime.now() - timedelta(days=730)).isoformat()
    history_2y = itad.get_price_history_since_date(game['id'], two_years_ago)
    
    if history_2y:
        print(f"✅ Found {len(history_2y)} price changes in last 2 years")
        
        # Show price changes
        print("\n   Price changes in last 2 years:")
        for i, change in enumerate(history_2y[:5], 1):
            price = change['deal']['price']['amount']
            currency = change['deal']['price']['currency']
            discount = change['deal']['cut']
            timestamp = change['timestamp']
            print(f"   {i}. {timestamp}: ${price} {currency} ({discount}% off)")
    else:
        print("❌ No price history found for last 2 years")


def example_multiple_games():
    """Example: Getting data for multiple games at once."""
    print("\n=== Multiple Games Example ===")
    
    itad = SteamITADIntegration()
    
    # Search for multiple games
    game_names = ["Cyberpunk 2077", "Rust", "PEAK"]
    game_ids = []
    
    print("1. Searching for multiple games...")
    for name in game_names:
        game = itad.search_game(name)
        if game:
            game_ids.append(game['id'])
            print(f"   ✅ {name}: {game['id']}")
        else:
            print(f"   ❌ {name}: Not found")
    
    if not game_ids:
        print("No games found!")
        return
    
    # Get prices for all games
    print(f"\n2. Getting Steam prices for {len(game_ids)} games...")
    prices_data = itad.get_multiple_games_prices(game_ids)
    
    print(f"✅ Retrieved prices for {len(prices_data)} games")
    
    # Display results
    print("\n3. Price Summary:")
    for game_data in prices_data:
        game_id = game_data['id']
        deals = game_data.get('deals', [])
        
        if deals:
            deal = deals[0]  # Steam deal
            price = deal['price']['amount']
            currency = deal['price']['currency']
            discount = deal['cut']
            print(f"   Game {game_id}: ${price} {currency} ({discount}% off)")
        else:
            print(f"   Game {game_id}: No Steam deals available")


def main():
    """Run all examples."""
    print("Steam ITAD Integration - Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_price_history()
        example_europe_prices()
        example_custom_date_range()
        example_multiple_games()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\n📚 Usage Summary:")
        print("   - Search games: itad.search_game('Game Name')")
        print("   - Get prices: itad.get_game_prices(game_id)")
        print("   - Get overview: itad.get_game_overview(game_id)")
        print("   - Get history: itad.get_price_history(game_id)")
        print("   - EU prices: itad.get_price_history(game_id, country='DE')")
        print("   - Custom date: itad.get_price_history_since_date(game_id, '2020-01-01')")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
