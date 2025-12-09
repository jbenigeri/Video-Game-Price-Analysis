# Video Game Price Analysis

A data science project analyzing Steam video game pricing patterns, with a focus on the impact of inflation and COVID-19 on game prices in the European market (2015-2024).

📖 **[View the Interactive Analysis](https://jbenigeri.github.io/Video-Game-Price-Analysis/)**

## Project Goal

This project investigates whether video game prices have kept pace with inflation and how the COVID-19 pandemic affected pricing strategies on Steam. By adjusting historical prices to 2024 euros using the Harmonised Index of Consumer Prices (HICP), we compare "real" prices across years to understand market trends and price tier shifts.

### Key Research Questions

1. **Have video game prices kept pace with inflation?**
2. **Did COVID-19 cause a structural shift in game pricing?**
3. **How has market composition changed across price tiers?**
4. **Does platform availability (Windows/Linux/MacOS) affect pricing?**

---

## Project Structure

```
Video-Game-Price-Analysis/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Data files
│   ├── steam_sales.csv       # Main Steam sales dataset
│   ├── steam_sales_post_2015.csv  # Filtered clean dataset used for analysis (2015+)
│   ├── all_items_hicp.csv    # Inflation data (HICP)
│   └── cleanup/              # Additional/processed data
├── src/                      # Notebooks and scripts
│   ├── nb1__exploratory_data_analysis.ipynb
│   ├── nb2__prices_analysis.ipynb
│   ├── images/               # Generated plots/figures
│   │   ├── nb1/              # Figures from notebook 1
│   │   └── nb2/              # Figures from notebook 2
│   └── cleanup/              # Older/backup files
└── older_plans/              # Older planning documents
```

---

## Notebooks

### [1. Exploratory Data Analysis](https://jbenigeri.github.io/Video-Game-Price-Analysis/nb1-exploratory-data-analysis/)

**Purpose:** Initial exploration of the Steam sales dataset.

**What it covers:**
- Data loading and cleaning (column renaming, type conversions)
- Correlation analysis between price, ratings, reviews, discounts, and platform support
- Platform availability analysis (Windows, Linux, MacOS)
- Price distribution comparisons across platforms (Windows-only vs cross-platform games)
- Discount distribution and patterns
- Game release trends over time (2006-2025)
- Price range × release year cross-tabulation
- Filtering to create a clean 2015-2025 dataset for further analysis

**Key findings:**
- Windows-exclusive games tend to be higher-priced (AAA segment)
- Cross-platform games (Linux/MacOS) are typically lower-priced indie titles
- Platform availability has no significant relationship with discount behavior

---

### [2. Price & Inflation Analysis](https://jbenigeri.github.io/Video-Game-Price-Analysis/nb2-prices-analysis/)

**Purpose:** In-depth analysis of video game pricing with inflation adjustment.

**What it covers:**
- Inflation adjustment using HICP (Harmonised Index of Consumer Prices)
- Nominal vs real (inflation-adjusted) price comparisons
- COVID-19 impact analysis on game pricing (2020-2022)
- Price tier composition over time
- Psychological price point analysis (€9.99, €19.99, €59.99, etc.)
- Statistical modeling of price trends

**Key questions addressed:**
- Are games becoming more or less affordable in real terms?
- Did the pandemic cause permanent pricing shifts?
- Are publishers moving toward premium or budget pricing?

---

## Data Sources

### `steam_sales.csv`
- **Source:** Steam store data (scraped/collected)
- **Contents:** 2,543 games with pricing, ratings, reviews, discounts, platform availability, and release dates
- **Fields:** Game Name, Rating, #Reviews, Discount%, Price (€), Original Price (€), Release Date, Windows, Linux, MacOS, Fetched At

### `steam_sales_post_2015.csv`
- **Source:** Filtered from `steam_sales.csv`
- **Contents:** 683 games released 2015 or later
- **Purpose:** Cleaner dataset for trend analysis where Steam had mature market presence

### `all_items_hicp.csv`
- **Source:** [Eurostat HICP Database](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_aind/default/table?lang=en)
- **Contents:** Harmonised Index of Consumer Prices (2004-2024)
- **Fields:** Year, European Union, Germany, United States
- **Base year:** 2015 = 100
- **Purpose:** Adjust historical game prices to 2024 euros for "real" price comparisons

### `data/cleanup/` (not used in final analysis)
- Historical price data from [IsThereAnyDeal API](https://isthereanydeal.com/)
- Additional HICP datasets (recreation/culture, games/toys/hobbies categories)

<!-- ### `data/cleanup/steam_historical_prices.csv`
- **Source:** [IsThereAnyDeal API](https://isthereanydeal.com/)
- **Contents:** Historical price tracking for games over time
- **Purpose:** Track price changes, sales, and discount patterns over a game's lifetime

### `data/cleanup/` (Additional HICP data)
- `recreation_and_culture_hicp.csv` - HICP for recreation/culture sector
- `games_toys_and_hobbies_hicp.csv` - HICP for games/toys/hobbies category
- **Purpose:** Alternative inflation indices for sector-specific comparisons -->

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Video-Game-Price-Analysis.git
   cd Video-Game-Price-Analysis
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n vgpa python=3.11
   conda activate vgpa
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks**
   ```bash
   jupyter notebook
   ```
   Open the notebooks in order:
   - [1. Exploratory Data Analysis](https://jbenigeri.github.io/Video-Game-Price-Analysis/nb1-exploratory-data-analysis/) (`src/nb1__exploratory_data_analysis.ipynb`)
   - [2. Price & Inflation Analysis](https://jbenigeri.github.io/Video-Game-Price-Analysis/nb2-prices-analysis/) (`src/nb2__prices_analysis.ipynb`)

---

## Key Technologies

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib / seaborn** - Visualization
- **Jupyter Notebook** - Interactive analysis


