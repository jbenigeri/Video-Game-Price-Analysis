# Steam ITAD Integration - Usage Examples

This guide shows how to use the `steam_itad_integration.py` module to get Steam game prices and price history from the IsThereAnyDeal API.

## Quick Start

```python
from steam_itad_integration import SteamITADIntegration

# Initialize
itad = SteamITADIntegration()

# Search for a game
game = itad.search_game("Cyberpunk 2077")
print(f"Found: {game['title']} (ID: {game['id']})")
```

## Basic Examples

### 1. Get Current Steam Prices

```python
# Search and get current prices
game = itad.search_game("Cyberpunk 2077")
prices = itad.get_game_prices(game['id'])

if prices and prices.get('deals'):
    deal = prices['deals'][0]  # Steam deal
    print(f"Current: ${deal['price']['amount']} {deal['price']['currency']}")
    print(f"Regular: ${deal['regular']['amount']} {deal['regular']['currency']}")
    print(f"Discount: {deal['cut']}%")
```

### 2. Get Price Overview

```python
# Get price overview (current + historical low)
overview = itad.get_game_overview(game['id'])

if overview:
    current = overview['current']
    lowest = overview['lowest']
    
    print(f"Current: ${current['price']['amount']} {current['price']['currency']}")
    print(f"Lowest Ever: ${lowest['price']['amount']} {lowest['price']['currency']}")
    print(f"Bundled: {overview['bundled']} times")
```

### 3. Get Full Price History (Since Release)

```python
# Get complete price history since release
history = itad.get_price_history(game['id'])

print(f"Found {len(history)} price changes")
for change in history[:5]:  # Show last 5
    price = change['deal']['price']['amount']
    discount = change['deal']['cut']
    timestamp = change['timestamp']
    print(f"{timestamp}: ${price} ({discount}% off)")
```

### 4. Get European Prices

```python
# Get EU Steam prices
history_eu = itad.get_price_history(game['id'], country="DE")

print(f"Found {len(history_eu)} EU price changes")
for change in history_eu[:3]:  # Show last 3
    price = change['deal']['price']['amount']
    currency = change['deal']['price']['currency']
    print(f"{change['timestamp']}: {price} {currency}")
```

### 5. Get Price History Since Specific Date

```python
# Get price history since specific date
from datetime import datetime, timedelta

# Last 2 years
two_years_ago = (datetime.now() - timedelta(days=730)).isoformat()
history_2y = itad.get_price_history_since_date(game['id'], two_years_ago)

# Since specific date
history_since = itad.get_price_history_since_date(game['id'], '2020-01-01T00:00:00')

print(f"Found {len(history_2y)} changes in last 2 years")
print(f"Found {len(history_since)} changes since 2020")
```

### 6. Multiple Games at Once

```python
# Search multiple games
game_names = ["Cyberpunk 2077", "Rust", "PEAK"]
game_ids = []

for name in game_names:
    game = itad.search_game(name)
    if game:
        game_ids.append(game['id'])

# Get prices for all games
prices_data = itad.get_multiple_games_prices(game_ids)

for game_data in prices_data:
    deals = game_data.get('deals', [])
    if deals:
        deal = deals[0]  # Steam deal
        print(f"Game {game_data['id']}: ${deal['price']['amount']}")
```

## Advanced Examples

### 7. Compare US vs EU Prices

```python
# Get both US and EU prices
history_us = itad.get_price_history(game['id'], country="US")
history_eu = itad.get_price_history(game['id'], country="DE")

print(f"US changes: {len(history_us)}")
print(f"EU changes: {len(history_eu)}")

# Compare latest prices
if history_us and history_eu:
    us_latest = history_us[0]['deal']['price']['amount']
    eu_latest = history_eu[0]['deal']['price']['amount']
    print(f"Latest US: ${us_latest}")
    print(f"Latest EU: €{eu_latest}")
```

### 8. Find Lowest Price Ever

```python
# Get full history and find lowest price
history = itad.get_price_history(game['id'])

if history:
    lowest = min(history, key=lambda x: x['deal']['price']['amount'])
    lowest_price = lowest['deal']['price']['amount']
    lowest_date = lowest['timestamp']
    print(f"Lowest price ever: ${lowest_price} on {lowest_date}")
```

### 9. Track Price Trends

```python
# Analyze price patterns
history = itad.get_price_history(game['id'])

if len(history) > 10:
    # Get last 10 price changes
    recent = history[:10]
    
    # Count sales vs regular prices
    sales = [p for p in recent if p['deal']['cut'] > 0]
    regular = [p for p in recent if p['deal']['cut'] == 0]
    
    print(f"Recent sales: {len(sales)}")
    print(f"Recent regular prices: {len(regular)}")
    
    # Calculate average sale discount
    if sales:
        avg_discount = sum(s['deal']['cut'] for s in sales) / len(sales)
        print(f"Average sale discount: {avg_discount:.1f}%")
```

## Running the Examples

To run all examples:

```bash
python steam_itad_integration.py
```

This will execute all the example functions and show you the output.

## Error Handling

```python
try:
    game = itad.search_game("Non-existent Game")
    if not game:
        print("Game not found")
    
    prices = itad.get_game_prices(game['id'])
    if not prices:
        print("No price data available")
        
except Exception as e:
    print(f"Error: {e}")
```

## Rate Limiting

The API has rate limits, so for multiple requests, add delays:

```python
import time

for game_name in game_names:
    game = itad.search_game(game_name)
    # Process game data...
    time.sleep(1)  # 1 second delay between requests
```

## Country Codes

Common country codes for different regions:

- `US` - United States (USD)
- `DE` - Germany (EUR)
- `GB` - United Kingdom (GBP)
- `FR` - France (EUR)
- `CA` - Canada (CAD)
- `AU` - Australia (AUD)

## Notes

- All functions return Steam-only data
- Price history is returned in chronological order (newest first)
- Dates are in ISO format (e.g., '2025-09-18T19:31:08+02:00')
- Prices include both amount and currency information
- Discount percentages are included when available
