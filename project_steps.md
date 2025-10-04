# Project Steps - Video Game Price Analysis

## Overview
This document outlines the step-by-step implementation of "What Drives Video Game Prices? Inflation, Discounts, and Console Cycles" with specific code snippets and implementation details.

## Phase 1: Data Collection & Preparation (Week 1)

### Step 1.1: Set up Project Structure
```bash
# Create directory structure
mkdir -p data/{raw,processed}
mkdir -p src/{data,analysis,modeling,visualization}
mkdir -p notebooks
mkdir -p results/{plots,models,reports}
```

### Step 1.2: Download Kaggle Datasets
```python
# src/data/download_datasets.py
import kagglehub
import pandas as pd
import os

def download_steam_dataset():
    """Download Steam sales historical dataset from Kaggle"""
    # Download Steam dataset
    kagglehub.dataset_download('benjaminlundkvist/steam-sales-historical-dataset')
    
    # Load and explore
    steam_df = pd.read_csv('~/.cache/kagglehub/datasets/benjaminlundkvist/steam-sales-historical-dataset/versions/5/steam_sales.csv')
    
    print(f"Steam dataset shape: {steam_df.shape}")
    print(f"Date range: {steam_df['date'].min()} to {steam_df['date'].max()}")
    print(f"Columns: {list(steam_df.columns)}")
    
    return steam_df

def download_metacritic_dataset():
    """Download Metacritic dataset from Kaggle"""
    # Download Metacritic dataset
    kagglehub.dataset_download('metacritic-game-scores')
    
    # Load and explore
    metacritic_df = pd.read_csv('~/.cache/kagglehub/datasets/metacritic-game-scores/versions/1/metacritic_games.csv')
    
    print(f"Metacritic dataset shape: {metacritic_df.shape}")
    print(f"Columns: {list(metacritic_df.columns)}")
    
    return metacritic_df

if __name__ == "__main__":
    steam_data = download_steam_dataset()
    metacritic_data = download_metacritic_dataset()
```

### Step 1.3: Collect Economic Data
```python
# src/data/economic_data.py
import pandas as pd
import requests
import yfinance as yf

def get_eu_cpi_data():
    """Download EU Consumer Price Index data from Eurostat API"""
    # Eurostat API endpoint for EU CPI
    url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/PRC_HICP_MIDX"
    params = {
        'format': 'json',
        'lang': 'en',
        'geo': 'EU27_2020',  # EU27 countries
        'coicop': 'CP00',    # All-items HICP
        'time': '2010-01:2024-12'
    }
    
    response = requests.get(url, params=params)
    cpi_data = response.json()
    
    # Process Eurostat data
    cpi_df = pd.DataFrame(cpi_data['value'])
    cpi_df['date'] = pd.to_datetime(cpi_df['time'])
    cpi_df['cpi'] = pd.to_numeric(cpi_df['values'])
    cpi_df = cpi_df[['date', 'cpi']].dropna()
    
    return cpi_df

def get_eu_inflation_data_alternative():
    """Alternative method using FRED for EU inflation data"""
    # FRED API endpoint for EU CPI
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'CPHPTT01EZM661N',  # EU HICP
        'api_key': 'YOUR_FRED_API_KEY',
        'file_type': 'json',
        'limit': 1000
    }
    
    response = requests.get(url, params=params)
    cpi_data = response.json()
    
    # Convert to DataFrame
    cpi_df = pd.DataFrame(cpi_data['observations'])
    cpi_df['date'] = pd.to_datetime(cpi_df['date'])
    cpi_df['cpi'] = pd.to_numeric(cpi_df['value'])
    
    return cpi_df

def get_console_release_dates():
    """Define console release dates for event study"""
    console_releases = {
        'PlayStation 5': '2020-11-12',
        'Xbox Series X': '2020-11-10',
        'Xbox Series S': '2020-11-10',
        'Nintendo Switch': '2017-03-03',
        'PlayStation 4': '2013-11-15',
        'Xbox One': '2013-11-22',
        'Nintendo Switch OLED': '2021-10-08'
    }
    
    return pd.DataFrame(list(console_releases.items()), 
                       columns=['console', 'release_date'])

def get_eu_exchange_rates():
    """Download EUR/USD exchange rates for currency conversion if needed"""
    # Using yfinance for EUR/USD exchange rate
    eur_usd = yf.download("EURUSD=X", start="2010-01-01", end="2024-12-31")
    eur_usd = eur_usd['Close'].reset_index()
    eur_usd.columns = ['date', 'eur_usd_rate']
    
    return eur_usd

def get_eu_gdp_data():
    """Download EU GDP data for economic context"""
    # FRED API for EU GDP
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'CLVMNACSCAB1GQEZ',  # EU GDP
        'api_key': 'YOUR_FRED_API_KEY',
        'file_type': 'json',
        'limit': 1000
    }
    
    response = requests.get(url, params=params)
    gdp_data = response.json()
    
    gdp_df = pd.DataFrame(gdp_data['observations'])
    gdp_df['date'] = pd.to_datetime(gdp_df['date'])
    gdp_df['gdp'] = pd.to_numeric(gdp_df['value'])
    
    return gdp_df
```

### Step 1.4: Data Cleaning and Standardization
```python
# src/data/data_cleaning.py
import pandas as pd
import numpy as np
from datetime import datetime

def clean_steam_data(steam_df):
    """Clean and standardize Steam dataset"""
    # Convert date column
    steam_df['date'] = pd.to_datetime(steam_df['date'])
    
    # Remove duplicates
    steam_df = steam_df.drop_duplicates()
    
    # Handle missing values
    steam_df = steam_df.dropna(subset=['price', 'date'])
    
    # Standardize currency (Steam prices are in EUR)
    steam_df['price_eur'] = steam_df['price']  # Steam prices are in EUR
    
    # Add game tier classification based on EUR prices
    steam_df['game_tier'] = pd.cut(steam_df['price_eur'], 
                                  bins=[0, 10, 30, 60, float('inf')], 
                                  labels=['Indie', 'Mid-tier', 'AAA', 'Premium'])
    
    return steam_df

def clean_metacritic_data(metacritic_df):
    """Clean and standardize Metacritic dataset"""
    # Convert release date
    metacritic_df['release_date'] = pd.to_datetime(metacritic_df['release_date'])
    
    # Handle missing values
    metacritic_df = metacritic_df.dropna(subset=['release_date', 'metascore'])
    
    # Standardize platform names
    platform_mapping = {
        'PC': 'PC',
        'PlayStation 4': 'PS4',
        'PlayStation 5': 'PS5',
        'Xbox One': 'Xbox One',
        'Xbox Series X': 'Xbox Series X',
        'Nintendo Switch': 'Switch'
    }
    
    metacritic_df['platform'] = metacritic_df['platform'].map(platform_mapping)
    
    return metacritic_df

def merge_datasets(steam_df, metacritic_df):
    """Merge Steam and Metacritic datasets"""
    # Merge on game name and approximate release date
    merged_df = pd.merge_asof(
        steam_df.sort_values('date'),
        metacritic_df.sort_values('release_date'),
        left_on='date',
        right_on='release_date',
        by='name',
        direction='backward',
        tolerance=pd.Timedelta(days=30)
    )
    
    return merged_df
```

## Phase 2: Descriptive Analysis (Week 2)

### Step 2.1: Inflation Adjustment
```python
# src/analysis/inflation_analysis.py
import pandas as pd
import numpy as np

def adjust_for_inflation_eu(df, cpi_df, base_year=2025):
    """Adjust EU prices for inflation using EU CPI data"""
    # Get base year CPI
    base_cpi = cpi_df[cpi_df['date'].dt.year == base_year]['cpi'].iloc[0]
    
    # Merge CPI data
    df['year'] = df['date'].dt.year
    cpi_df['year'] = cpi_df['date'].dt.year
    df = df.merge(cpi_df[['year', 'cpi']], on='year', how='left')
    
    # Calculate inflation adjustment factor
    df['inflation_factor'] = base_cpi / df['cpi']
    
    # Adjust prices (assuming prices are in EUR)
    df['price_adjusted_eur'] = df['price_eur'] * df['inflation_factor']
    
    # Optional: Convert to USD for comparison
    if 'eur_usd_rate' in df.columns:
        df['price_adjusted_usd'] = df['price_adjusted_eur'] * df['eur_usd_rate']
    
    return df

def convert_eur_to_usd(df, exchange_rates_df):
    """Convert EUR prices to USD using historical exchange rates"""
    # Merge exchange rate data
    df = df.merge(exchange_rates_df, on='date', how='left')
    
    # Forward fill missing exchange rates
    df['eur_usd_rate'] = df['eur_usd_rate'].fillna(method='ffill')
    
    # Convert prices
    df['price_usd'] = df['price_eur'] * df['eur_usd_rate']
    
    return df

def calculate_price_half_life(df):
    """Calculate time to reach specific discount thresholds"""
    results = []
    
    for game in df['name'].unique():
        game_data = df[df['name'] == game].sort_values('date')
        launch_price = game_data['price_adjusted'].iloc[0]
        
        # Calculate discount percentages
        game_data['discount_pct'] = (game_data['price_adjusted'] - launch_price) / launch_price * 100
        
        # Find time to reach -30%, -50%, -70% discounts
        for threshold in [-30, -50, -70]:
            threshold_data = game_data[game_data['discount_pct'] <= threshold]
            if not threshold_data.empty:
                days_to_threshold = (threshold_data['date'].iloc[0] - game_data['date'].iloc[0]).days
                results.append({
                    'game': game,
                    'threshold': threshold,
                    'days_to_threshold': days_to_threshold,
                    'launch_price': launch_price
                })
    
    return pd.DataFrame(results)
```

### Step 2.2: Price Trend Visualization
```python
# src/visualization/price_trends.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_inflation_adjusted_prices(df):
    """Plot inflation-adjusted launch prices by year and platform"""
    plt.figure(figsize=(15, 8))
    
    # Group by year and platform
    yearly_prices = df.groupby(['year', 'platform'])['price_adjusted_eur'].median().reset_index()
    
    # Create line plot
    for platform in yearly_prices['platform'].unique():
        platform_data = yearly_prices[yearly_prices['platform'] == platform]
        plt.plot(platform_data['year'], platform_data['price_adjusted_eur'], 
                marker='o', label=platform, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Inflation-Adjusted Price (2025 EUR)')
    plt.title('Inflation-Adjusted Launch Prices by Platform Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/inflation_adjusted_prices.png', dpi=300)
    plt.show()

def plot_discount_half_life(half_life_df):
    """Plot discount half-life analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    thresholds = [-30, -50, -70]
    for i, threshold in enumerate(thresholds):
        threshold_data = half_life_df[half_life_df['threshold'] == threshold]
        
        # Box plot of days to threshold
        sns.boxplot(data=threshold_data, y='days_to_threshold', ax=axes[i])
        axes[i].set_title(f'Days to {threshold}% Discount')
        axes[i].set_ylabel('Days')
    
    plt.tight_layout()
    plt.savefig('results/plots/discount_half_life.png', dpi=300)
    plt.show()

def plot_price_curves(df, sample_games=10):
    """Plot price curves for sample games"""
    plt.figure(figsize=(15, 10))
    
    # Select sample games
    sample_games_list = df['name'].unique()[:sample_games]
    
    for i, game in enumerate(sample_games_list):
        game_data = df[df['name'] == game].sort_values('date')
        
        plt.subplot(2, 5, i+1)
        plt.plot(game_data['date'], game_data['price_adjusted_eur'], linewidth=2)
        plt.title(game[:20] + '...' if len(game) > 20 else game)
        plt.xticks(rotation=45)
        plt.ylabel('Price (2025 EUR)')
    
    plt.tight_layout()
    plt.savefig('results/plots/sample_price_curves.png', dpi=300)
    plt.show()
```

## Phase 3: Predictive Modeling (Week 3)

### Data Sources for Historical Prices

#### Primary Source: Kaggle Steam Dataset
- **Comprehensive historical data** for thousands of games
- **Price changes over time** for each game
- **Long time period** (2010-2024+)
- **Consistent data structure** across all games

#### Supplementary Source: ITAD API
- **Validation** - Cross-checking Steam prices from Kaggle
- **Additional games** - Games not in the Kaggle dataset
- **Real-time data** - Current prices for recent games
- **Cross-store comparison** - Steam vs other stores

#### When to Use ITAD API:
```python
# src/data/itad_integration.py
from steam_itad_integration import SteamITADIntegration

def enhance_kaggle_data_with_itad(kaggle_df, api_key):
    """Enhance Kaggle data with ITAD API for specific games"""
    itad_client = SteamITADIntegration(api_key)
    
    # Use ITAD for:
    # 1. Games missing from Kaggle dataset
    # 2. Recent games (2024+) not in Kaggle
    # 3. Validation of price data
    # 4. Cross-store price comparison
    
    enhanced_games = []
    for game_name in kaggle_df['name'].unique()[:10]:  # Sample first 10
        try:
            # Search for game in ITAD
            game_info = itad_client.search_game(game_name)
            if game_info:
                # Get price history
                price_history = itad_client.get_game_price_history(game_info['id'])
                enhanced_games.append({
                    'game_name': game_name,
                    'itad_id': game_info['id'],
                    'price_history': price_history
                })
        except Exception as e:
            print(f"Error processing {game_name}: {e}")
    
    return enhanced_games

def create_combined_dataset(kaggle_df, itad_enhanced_data):
    """Combine Kaggle and ITAD data for comprehensive analysis"""
    # Start with Kaggle data as base
    combined_df = kaggle_df.copy()
    
    # Add ITAD data for enhanced games
    for game_data in itad_enhanced_data:
        game_name = game_data['game_name']
        itad_history = game_data['price_history']
        
        if itad_history:
            # Convert ITAD data to same format as Kaggle
            itad_df = pd.DataFrame(itad_history)
            itad_df['name'] = game_name
            itad_df['date'] = pd.to_datetime(itad_df['timestamp'])
            itad_df['price_eur'] = itad_df['deal']['price']['amount']
            
            # Append to combined dataset
            combined_df = pd.concat([combined_df, itad_df[['name', 'date', 'price_eur']]], 
                                  ignore_index=True)
    
    return combined_df
```

### What We're Predicting

#### 1. Time-to-Event Predictions (Survival Analysis)
- **Time to first −50% discount** - How many days after release until a game reaches 50% off
- **Time to first −30% discount** - Earlier discount threshold
- **Time to first −70% discount** - Deep discount threshold

#### 2. Price Level Predictions (Regression)
- **Price after 180 days** - What will the price be 6 months after release
- **Price after 365 days** - What will the price be 1 year after release
- **Maximum discount within first 12 months** - What's the deepest discount in the first year

#### 3. Classification Predictions
- **Discount probability** - Will a game reach 30%/50%/70% discount within X days
- **Price tier prediction** - Will a game become "budget" (<€10) within a year
- **Console cycle impact** - Will console launches affect discount timing

### Step 3.1: Feature Engineering
```python
# src/modeling/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_features(df):
    """Create features for predictive modeling"""
    # Time-based features
    df['days_since_release'] = (df['date'] - df['release_date']).dt.days
    df['months_since_release'] = df['days_since_release'] / 30.44
    
    # Price-based features
    df['launch_price'] = df.groupby('name')['price_adjusted_eur'].transform('first')
    df['current_discount'] = (df['price_adjusted_eur'] - df['launch_price']) / df['launch_price'] * 100
    
    # Console cycle features
    df['console_cycle_indicator'] = 0
    for console, release_date in console_releases.items():
        release_date = pd.to_datetime(release_date)
        mask = (df['release_date'] >= release_date - pd.Timedelta(days=90)) & \
               (df['release_date'] <= release_date + pd.Timedelta(days=90))
        df.loc[mask, 'console_cycle_indicator'] = 1
    
    # Review score features
    df['metascore_bin'] = pd.cut(df['metascore'], 
                                bins=[0, 60, 80, 100], 
                                labels=['Low', 'Medium', 'High'])
    
    # Publisher features
    df['publisher_size'] = df.groupby('publisher')['name'].transform('nunique')
    
    return df

def prepare_modeling_data(df):
    """Prepare data for machine learning models"""
    # Select features
    feature_cols = [
        'days_since_release', 'months_since_release', 'launch_price',
        'metascore', 'console_cycle_indicator', 'publisher_size',
        'platform', 'genre', 'game_tier'
    ]
    
    # Encode categorical variables
    le_platform = LabelEncoder()
    le_genre = LabelEncoder()
    le_tier = LabelEncoder()
    
    df['platform_encoded'] = le_platform.fit_transform(df['platform'])
    df['genre_encoded'] = le_genre.fit_transform(df['genre'])
    df['tier_encoded'] = le_tier.fit_transform(df['game_tier'])
    
    # Create target variables
    df['time_to_50_discount'] = df.groupby('name')['current_discount'].apply(
        lambda x: (x <= -50).idxmax() if (x <= -50).any() else np.nan
    )
    
    df['max_discount_12m'] = df.groupby('name')['current_discount'].transform(
        lambda x: x.rolling(window=365, min_periods=1).min()
    )
    
    return df, feature_cols
```

### Step 3.2: Survival Analysis
```python
# src/modeling/survival_analysis.py
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd

def survival_analysis(df):
    """Perform survival analysis for time to discount"""
    # Prepare survival data
    survival_data = df.groupby('name').agg({
        'days_since_release': 'max',
        'current_discount': 'min',
        'metascore': 'first',
        'launch_price': 'first',
        'console_cycle_indicator': 'first',
        'publisher_size': 'first',
        'platform_encoded': 'first',
        'genre_encoded': 'first',
        'tier_encoded': 'first'
    }).reset_index()
    
    # Create event indicator (reached -50% discount)
    survival_data['event'] = (survival_data['current_discount'] <= -50).astype(int)
    survival_data['duration'] = survival_data['days_since_release']
    
    # Fit Cox proportional hazards model
    cph = CoxPHFitter()
    cph.fit(survival_data[['duration', 'event', 'metascore', 'launch_price', 
                          'console_cycle_indicator', 'publisher_size', 
                          'platform_encoded', 'genre_encoded', 'tier_encoded']],
            duration_col='duration', event_col='event')
    
    # Print results
    print("Cox Proportional Hazards Model Results:")
    print(cph.summary)
    
    # Calculate concordance index
    c_index = concordance_index(survival_data['duration'], 
                               -cph.predict_partial_hazard(survival_data), 
                               survival_data['event'])
    print(f"Concordance Index: {c_index:.3f}")
    
    return cph

def plot_survival_curves(cph, survival_data):
    """Plot survival curves for different groups"""
    import matplotlib.pyplot as plt
    
    # Plot survival curves by platform
    plt.figure(figsize=(12, 8))
    
    for platform in survival_data['platform_encoded'].unique():
        platform_data = survival_data[survival_data['platform_encoded'] == platform]
        cph.plot_partial_hazards(platform_data, 
                                columns=['platform_encoded'],
                                title=f'Survival Curves by Platform')
    
    plt.tight_layout()
    plt.savefig('results/plots/survival_curves.png', dpi=300)
    plt.show()
```

### Step 3.3: Tree-Based Models
```python
# src/modeling/tree_models.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

def train_tree_models(df, feature_cols):
    """Train tree-based models for price prediction"""
    # Prepare features and targets
    X = df[feature_cols].fillna(0)
    y = df['max_discount_12m'].fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Train Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    
    # Evaluate models
    for model, name in [(rf, 'Random Forest'), (gb, 'Gradient Boosting')]:
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} Results:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.3f}")
        print()
    
    return rf, gb

def plot_feature_importance(rf, gb, feature_cols):
    """Plot feature importance for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Random Forest importance
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[0].barh(rf_importance['feature'], rf_importance['importance'])
    axes[0].set_title('Random Forest Feature Importance')
    axes[0].set_xlabel('Importance')
    
    # Gradient Boosting importance
    gb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1].barh(gb_importance['feature'], gb_importance['importance'])
    axes[1].set_title('Gradient Boosting Feature Importance')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('results/plots/feature_importance.png', dpi=300)
    plt.show()

def plot_shap_values(rf, X_test, feature_cols):
    """Plot SHAP values for model interpretability"""
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
    plt.tight_layout()
    plt.savefig('results/plots/shap_summary.png', dpi=300)
    plt.show()
```

## Phase 4: Event Study Analysis (Week 4)

### Step 4.1: Console Launch Impact Analysis
```python
# src/analysis/event_study.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_console_impact(df, console_releases):
    """Analyze impact of console launches on game pricing"""
    results = []
    
    for console, release_date in console_releases.items():
        release_date = pd.to_datetime(release_date)
        
        # Define event windows
        pre_window = pd.Timedelta(days=180)
        post_window = pd.Timedelta(days=180)
        
        # Find games released around console launch
        console_games = df[
            (df['release_date'] >= release_date - pre_window) &
            (df['release_date'] <= release_date + post_window)
        ].copy()
        
        if len(console_games) > 0:
            # Calculate discount rates before and after console launch
            console_games['days_from_console'] = (
                console_games['release_date'] - release_date
            ).dt.days
            
            # Group by time periods
            pre_console = console_games[console_games['days_from_console'] < 0]
            post_console = console_games[console_games['days_from_console'] >= 0]
            
            if len(pre_console) > 0 and len(post_console) > 0:
                pre_discount = pre_console['current_discount'].mean()
                post_discount = post_console['current_discount'].mean()
                
                results.append({
                    'console': console,
                    'release_date': release_date,
                    'pre_discount': pre_discount,
                    'post_discount': post_discount,
                    'discount_change': post_discount - pre_discount,
                    'n_games': len(console_games)
                })
    
    return pd.DataFrame(results)

def plot_event_study(df, console_releases):
    """Plot event study results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Discount rates around console launches
    for console, release_date in console_releases.items():
        release_date = pd.to_datetime(release_date)
        
        # Get games around console launch
        console_games = df[
            (df['release_date'] >= release_date - pd.Timedelta(days=180)) &
            (df['release_date'] <= release_date + pd.Timedelta(days=180))
        ]
        
        if len(console_games) > 0:
            console_games['days_from_console'] = (
                console_games['release_date'] - release_date
            ).dt.days
            
            # Plot discount rates
            axes[0, 0].scatter(console_games['days_from_console'], 
                             console_games['current_discount'], 
                             alpha=0.6, label=console)
    
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Days from Console Launch')
    axes[0, 0].set_ylabel('Discount Rate (%)')
    axes[0, 0].set_title('Discount Rates Around Console Launches')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Price evolution comparison
    # (Additional plotting code for price evolution)
    
    plt.tight_layout()
    plt.savefig('results/plots/event_study.png', dpi=300)
    plt.show()
```

## Phase 5: Final Analysis & Reporting (Week 5)

### Step 5.1: Generate Final Report
```python
# src/reporting/generate_report.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_summary_statistics(df):
    """Generate summary statistics for the final report"""
    summary = {
        'total_games': df['name'].nunique(),
        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        'platforms': df['platform'].nunique(),
        'publishers': df['publisher'].nunique(),
        'avg_launch_price': df.groupby('name')['price_adjusted_eur'].first().mean(),
        'avg_max_discount': df.groupby('name')['current_discount'].min().mean(),
        'console_cycle_games': df[df['console_cycle_indicator'] == 1]['name'].nunique()
    }
    
    return summary

def create_executive_summary(summary_stats, model_results):
    """Create executive summary for the report"""
    summary = f"""
# Executive Summary

## Key Findings

### Dataset Overview
- **Total Games Analyzed**: {summary_stats['total_games']:,}
- **Time Period**: {summary_stats['date_range']}
- **Platforms Covered**: {summary_stats['platforms']}
- **Publishers**: {summary_stats['publishers']:,}

### Price Analysis
- **Average Launch Price**: €{summary_stats['avg_launch_price']:.2f}
- **Average Maximum Discount**: {summary_stats['avg_max_discount']:.1f}%
- **Games Around Console Launches**: {summary_stats['console_cycle_games']:,}

### Model Performance
- **Survival Model C-Index**: {model_results.get('c_index', 'N/A')}
- **Random Forest R²**: {model_results.get('rf_r2', 'N/A')}
- **Gradient Boosting R²**: {model_results.get('gb_r2', 'N/A')}

## Key Insights
1. **Inflation Impact**: Real launch prices have [X]% change over time
2. **Discount Patterns**: Games reach 50% discount in [X] days on average
3. **Console Cycles**: Console launches [do/don't] significantly impact pricing
4. **Predictive Factors**: [Top 3 factors] are most important for price prediction

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]
"""
    
    return summary

def generate_final_report():
    """Generate the complete final report"""
    # Load processed data
    df = pd.read_csv('data/processed/merged_data.csv')
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(df)
    
    # Load model results
    model_results = pd.read_csv('results/models/model_performance.csv')
    
    # Create executive summary
    exec_summary = create_executive_summary(summary_stats, model_results)
    
    # Save report
    with open('results/reports/final_report.md', 'w') as f:
        f.write(exec_summary)
    
    print("Final report generated successfully!")
    print(f"Report saved to: results/reports/final_report.md")

if __name__ == "__main__":
    generate_final_report()
```

### Step 5.2: Create Interactive Dashboard
```python
# src/visualization/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_dashboard():
    """Create interactive Streamlit dashboard"""
    st.set_page_config(page_title="Video Game Price Analysis", layout="wide")
    
    st.title("🎮 Video Game Price Analysis Dashboard")
    
    # Load data
    df = pd.read_csv('data/processed/merged_data.csv')
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    platform_filter = st.sidebar.multiselect(
        "Platform", 
        options=df['platform'].unique(),
        default=df['platform'].unique()
    )
    
    genre_filter = st.sidebar.multiselect(
        "Genre", 
        options=df['genre'].unique(),
        default=df['genre'].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df['platform'].isin(platform_filter)) &
        (df['genre'].isin(genre_filter))
    ]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{filtered_df['name'].nunique():,}")
    
    with col2:
        st.metric("Avg Launch Price", f"€{filtered_df.groupby('name')['price_adjusted_eur'].first().mean():.2f}")
    
    with col3:
        st.metric("Avg Max Discount", f"{filtered_df.groupby('name')['current_discount'].min().mean():.1f}%")
    
    with col4:
        st.metric("Console Cycle Games", f"{filtered_df[filtered_df['console_cycle_indicator'] == 1]['name'].nunique():,}")
    
    # Charts
    st.subheader("Price Trends Over Time")
    
    # Price trend chart
    yearly_prices = filtered_df.groupby(['year', 'platform'])['price_adjusted_eur'].median().reset_index()
    
    fig = px.line(yearly_prices, x='year', y='price_adjusted_eur', color='platform',
                  title='Inflation-Adjusted Launch Prices by Platform')
    st.plotly_chart(fig, use_container_width=True)
    
    # Discount analysis
    st.subheader("Discount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Discount distribution
        discount_data = filtered_df.groupby('name')['current_discount'].min().reset_index()
        fig = px.histogram(discount_data, x='current_discount', 
                          title='Distribution of Maximum Discounts')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Platform comparison
        platform_discounts = filtered_df.groupby('platform')['current_discount'].min().reset_index()
        fig = px.bar(platform_discounts, x='platform', y='current_discount',
                     title='Average Maximum Discount by Platform')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    create_dashboard()
```

## Implementation Checklist

### Week 1: Data Collection & Preparation
- [ ] Set up project structure
- [ ] Download Kaggle datasets
- [ ] Collect economic data (CPI, console releases)
- [ ] Clean and merge datasets
- [ ] Create inflation-adjusted prices

### Week 2: Descriptive Analysis
- [ ] Plot inflation-adjusted price trends
- [ ] Calculate discount half-lives
- [ ] Analyze price patterns by platform/genre
- [ ] Create initial visualizations

### Week 3: Predictive Modeling
- [ ] Engineer features for ML models
- [ ] Implement survival analysis
- [ ] Train tree-based models
- [ ] Evaluate model performance
- [ ] Generate SHAP explanations

### Week 4: Event Study
- [ ] Analyze console launch impacts
- [ ] Create event study visualizations
- [ ] Quantify console cycle effects
- [ ] Compare pre/post console launch pricing

### Week 5: Final Analysis & Reporting
- [ ] Generate summary statistics
- [ ] Create executive summary
- [ ] Build interactive dashboard
- [ ] Finalize visualizations
- [ ] Write final report

## File Structure
```
Video-Game-Price-Analysis/
├── data/
│   ├── raw/
│   │   ├── steam_sales.csv
│   │   ├── metacritic_games.csv
│   │   ├── cpi_data.csv
│   │   └── console_releases.csv
│   └── processed/
│       ├── merged_data.csv
│       └── inflation_adjusted.csv
├── src/
│   ├── data/
│   │   ├── download_datasets.py
│   │   ├── economic_data.py
│   │   └── data_cleaning.py
│   ├── analysis/
│   │   ├── inflation_analysis.py
│   │   └── event_study.py
│   ├── modeling/
│   │   ├── feature_engineering.py
│   │   ├── survival_analysis.py
│   │   └── tree_models.py
│   ├── visualization/
│   │   ├── price_trends.py
│   │   └── dashboard.py
│   └── reporting/
│       └── generate_report.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_descriptive_analysis.ipynb
│   ├── 03_predictive_modeling.ipynb
│   └── 04_event_study.ipynb
├── results/
│   ├── plots/
│   ├── models/
│   └── reports/
├── requirements.txt
├── README.md
├── project_proposal.md
├── project_steps.md
└── todo_list.md
```

This comprehensive project plan provides detailed code snippets and implementation steps for each phase of the video game price analysis project, following the methodology outlined in the project proposal.
