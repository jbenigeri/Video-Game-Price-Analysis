# Steam Historical Prices CSV Creator

## 📋 Overview

This script takes the Kaggle Steam sales dataset and enriches it with historical price data from the IsThereAnyDeal (ITAD) API.

## 🔄 Data Flow

```
steam_sales.csv → Game Names → ITAD API → Historical Prices → steam_historical_prices.csv
```

## 📊 Input/Output

### Input: `steam_sales.csv`
- Source: Kaggle Steam dataset
- Contains: Game names, basic sales data
- Location: `data/steam_sales.csv`

### Output: `steam_historical_prices.csv`
- Contains: Historical price data for each game
- Columns:
  - `Game Name`: Original game name from Steam dataset
  - `ITAD Game ID`: ITAD game ID
  - `Date`: Price change timestamp
  - `Price (€)`: Current price in EUR
  - `Original Price (€)`: Regular/launch price in EUR
  - `Discount%`: Discount percentage (0-100)
  - `Currency`: Currency code (EUR)
  - `Shop Name`: Store name (Steam)
  - `Shop ID`: Store ID (61 for Steam)

## 🚀 Usage

### Debug Mode (Testing)
```bash
# Activate conda environment
conda activate jacob

# Run with first 5 games (debug mode)
python src/create_historical_prices.py --debug
```
**Output:** `data/steam_historical_prices_test.csv`

### Full Processing
```bash
# Process all games in the dataset
python src/create_historical_prices.py
```
**Output:** `data/steam_historical_prices.csv`

## ⚙️ Configuration

Edit these variables in `src/create_historical_prices.py`:

```python
STEAM_CSV_PATH = "data/steam_sales.csv"  # Input file path
API_KEY = "your_itad_api_key_here"  # Your ITAD API key
```

**Note:** Output path is automatically set based on debug mode:
- Debug mode: `data/steam_historical_prices_test.csv`
- Full mode: `data/steam_historical_prices.csv`

## 📈 Expected Output

The script will:
1. Load Steam sales data
2. Extract unique game names
3. For each game (5 in debug mode, all in full mode):
   - Search ITAD API for game ID
   - Fetch historical price data
   - Process and store results
4. Create final CSV with all historical prices
5. Show progress and statistics

### Debug Mode Output:
```
🐛 DEBUG MODE: Processing first 5 games only
Processing 5 games...
[  1/  5] Processing: PEAK
    Found game ID: abc123...
    Found 12 price records
    Successfully processed: PEAK
...
```

### Full Mode Output:
```
🚀 FULL MODE: Processing all games
Processing 2545 games...
[  1/2545] Processing: PEAK
...
```

## ⚠️ Important Notes

- **Rate Limiting**: 1 second delay between API calls
- **Error Handling**: Games not found in ITAD are logged and skipped
- **Testing Mode**: Default processes only first 5 games
- **API Key**: Make sure your ITAD API key is valid
- **Data Directory**: Ensure `data/` directory exists

## 🔍 Troubleshooting

### Common Issues:
1. **"Steam sales CSV not found"**: Download Kaggle dataset to `data/` directory
2. **"Game not found in ITAD"**: Some games may not be in ITAD database
3. **"No price history found"**: Game exists but has no price data
4. **API errors**: Check your API key and internet connection

### Progress Monitoring:
- Shows progress every 10 games
- Displays success/error counts
- Shows total price records collected
- Displays sample output data

## 📊 Sample Output

```
🚀 Steam Historical Prices CSV Creator
==================================================
Loading Steam sales data from: data/steam_sales.csv
Loaded 1000 records from Steam sales dataset
Found 500 unique games

🧪 TESTING MODE: Processing first 5 games only

============================================================
CREATING STEAM HISTORICAL PRICES CSV
============================================================
Processing 5 games...
------------------------------------------------------------
[  1/  5] Processing: Cyberpunk 2077
    ✅ Found game ID: 01849783-6a26-7147-ab32-71804ca47e8e
    📊 Found 45 price records
    ✅ Successfully processed: Cyberpunk 2077
...
```

## 🎯 Next Steps

After running the script:
1. Check `data/steam_historical_prices.csv` for results
2. Use this data for predictive modeling
3. Analyze price trends and discount patterns
4. Integrate with economic data for inflation adjustment
