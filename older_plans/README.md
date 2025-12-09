# Steam Video Game Price Analysis

A comprehensive data science project for analyzing Steam video game prices using the IsThereAnyDeal API.

## Project Overview

This project integrates Steam game data with the IsThereAnyDeal API to provide:
- Real-time Steam price data
- Historical price analysis since game release
- Multi-currency support (US, EU, etc.)
- Price trend analysis and patterns

## Files

- `steam_itad_integration.py` - Main integration module with all API functions
- `USAGE_EXAMPLES.md` - Comprehensive usage guide with examples
- `requirements.txt` - Python dependencies

## Quick Start

```python
from steam_itad_integration import SteamITADIntegration

# Initialize
itad = SteamITADIntegration()

# Search for a game
game = itad.search_game("Cyberpunk 2077")

# Get current Steam prices
prices = itad.get_game_prices(game['id'])

# Get full price history since release
history = itad.get_price_history(game['id'])

# Get European prices
history_eu = itad.get_price_history(game['id'], country="DE")
```

## Features

- **Steam-Only Data**: All functions return Steam-specific pricing
- **Full Price History**: Complete price data since game release
- **Multi-Currency**: Support for US, EU, and other regional pricing
- **Batch Processing**: Get data for multiple games at once
- **Easy Integration**: Simple API for your analysis projects

## Documentation

See `USAGE_EXAMPLES.md` for detailed examples and usage patterns.

## Author

Jacob benigeri - Video Game Price Analysis Project
