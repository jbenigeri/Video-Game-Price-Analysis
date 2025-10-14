# Video Game Tier Approaches

## Overview

This document outlines the specific approaches for creating video game tiers to analyze discount patterns and pricing strategies. The strategy is divided into two main components:

1. **Price-based tiers** for answering questions about discount behavior
2. **Non-price features** for machine learning prediction of time-to-discount and discount percentage

### Important Data Filtering Note

**All analyses in this document use games released in 2015 or later.** Earlier years have very little data, which would skew our results and make statistical comparisons unreliable. By filtering to 2015+, we ensure:
- Sufficient sample sizes in each year for robust analysis
- More consistent data quality (older Steam data can be incomplete)
- Better inflation adjustment accuracy (our HICP data is most reliable for recent years)
- Focus on modern pricing strategies relevant to today's market

---

## Part 1: Price-Based Tiers (For Discount Analysis)

### Primary Approach: Inflation-Adjusted Percentile Tiers

This is our main tiering method for analyzing discount patterns.

#### Steps:

1. **Merge inflation data with game data**
   - Extract release year from `Release Date`
   - Join with Recreation and Culture HICP data by year
   - Handle games released in years outside inflation data range (extrapolate or use nearest year)

```python
import pandas as pd
import numpy as np

# Load data
df_games = pd.read_csv('data/steam_sales.csv')
df_inflation = pd.read_csv('data/all_items_hicp.csv')

# Parse release date and extract year
df_games['Release Date'] = pd.to_datetime(df_games['Release Date'], format='%d %b, %Y')
df_games['release_year'] = df_games['Release Date'].dt.year

# Filter to games released in 2015 or later (earlier years have very little data)
print(f"Total games before filtering: {len(df_games)}")
df_games = df_games[df_games['release_year'] >= 2015].copy()
print(f"Total games after filtering (2015+): {len(df_games)}")

print("\nGames per year:")
print(df_games['release_year'].value_counts().sort_index())

# Prepare inflation data (assuming you'll use European Union column)
df_inflation = df_inflation.rename(columns={'European Union': 'HICP'})

# Merge with inflation data
df_games = df_games.merge(df_inflation[['Year', 'HICP']], 
                          left_on='release_year', 
                          right_on='Year', 
                          how='left')

# Handle missing inflation data (games outside the inflation data range)
# Option 1: Use nearest available year
df_games['HICP'] = df_games.groupby('release_year')['HICP'].transform(
    lambda x: x.fillna(method='ffill').fillna(method='bfill')
)

# Option 2: Use the first/last available HICP value for out-of-range years
min_year = df_inflation['Year'].min()
max_year = df_inflation['Year'].max()
hicp_min = df_inflation[df_inflation['Year'] == min_year]['HICP'].values[0]
hicp_max = df_inflation[df_inflation['Year'] == max_year]['HICP'].values[0]

df_games.loc[df_games['release_year'] < min_year, 'HICP'] = hicp_min
df_games.loc[df_games['release_year'] > max_year, 'HICP'] = hicp_max
```

2. **Calculate inflation adjustment factor**
   - Choose base year (e.g., 2025)
   - For each game: `adjustment_factor = HICP_2025 / HICP_release_year`
   - Apply: `adjusted_price = original_price × adjustment_factor`

```python
# Choose base year for adjustment
BASE_YEAR = 2025

# Get base year HICP
base_hicp = df_inflation[df_inflation['Year'] == BASE_YEAR]['HICP'].values[0]

# Calculate adjustment factor
df_games['adjustment_factor'] = base_hicp / df_games['HICP']

# Apply inflation adjustment to original price
# Remove € symbol and convert to float if needed
df_games['Original Price (€)'] = df_games['Original Price (€)'].astype(float)
df_games['adjusted_price'] = df_games['Original Price (€)'] * df_games['adjustment_factor']

print(f"Base year: {BASE_YEAR}, Base HICP: {base_hicp}")
print(f"Price adjustment example:")
print(df_games[['Game Name', 'release_year', 'Original Price (€)', 
                'adjusted_price']].head(10))
```

3. **Calculate percentiles on adjusted prices**
   - Compute percentiles: 25th, 50th, 75th percentiles
   - Create quartile tiers:
     - **Tier 1 (Budget):** 0-25th percentile
     - **Tier 2 (Mid-Low):** 25-50th percentile
     - **Tier 3 (Mid-High):** 50-75th percentile
     - **Tier 4 (Premium):** 75-100th percentile

```python
# Calculate percentiles
p25 = df_games['adjusted_price'].quantile(0.25)
p50 = df_games['adjusted_price'].quantile(0.50)
p75 = df_games['adjusted_price'].quantile(0.75)

print(f"Percentile thresholds:")
print(f"25th percentile: €{p25:.2f}")
print(f"50th percentile: €{p50:.2f}")
print(f"75th percentile: €{p75:.2f}")

# Create price tiers using pd.cut
df_games['price_tier'] = pd.cut(df_games['adjusted_price'],
                                 bins=[0, p25, p50, p75, np.inf],
                                 labels=['Budget', 'Mid-Low', 'Mid-High', 'Premium'],
                                 include_lowest=True)

# Check tier distribution
print("\nPrice tier distribution:")
print(df_games['price_tier'].value_counts().sort_index())
```

4. **Create yearly tier assignments**
   - Calculate percentiles separately for each release year
   - This captures relative market positioning within each year
   - Creates `yearly_tier` variable

```python
# Calculate percentiles within each release year
def assign_yearly_tier(group):
    """Assign tiers based on percentiles within each year"""
    p25 = group['adjusted_price'].quantile(0.25)
    p50 = group['adjusted_price'].quantile(0.50)
    p75 = group['adjusted_price'].quantile(0.75)
    
    conditions = [
        group['adjusted_price'] <= p25,
        (group['adjusted_price'] > p25) & (group['adjusted_price'] <= p50),
        (group['adjusted_price'] > p50) & (group['adjusted_price'] <= p75),
        group['adjusted_price'] > p75
    ]
    choices = ['Budget', 'Mid-Low', 'Mid-High', 'Premium']
    
    return pd.Series(np.select(conditions, choices), index=group.index)

df_games['yearly_tier'] = df_games.groupby('release_year').apply(
    assign_yearly_tier
).reset_index(level=0, drop=True)

# Compare overall vs yearly tiers
print("\nYearly tier distribution:")
print(df_games['yearly_tier'].value_counts().sort_index())

# Example: Compare tiers for games from different years
print("\nExample: Same price, different years, different yearly tiers:")
print(df_games[['Game Name', 'release_year', 'adjusted_price', 
                'price_tier', 'yearly_tier']].sample(10))
```

#### Why This Approach?

- **Inflation adjustment:** A €60 game in 2015 is very different from €60 in 2025 in real purchasing power terms
- **Percentiles ensure balanced samples:** Each tier has roughly equal number of games for statistical analysis
- **Distribution-agnostic:** Works regardless of price distribution skew
- **Captures relative positioning:** Games are compared to their contemporaries
- **Temporal validity:** Controls for overall market price trends over time

---

### Secondary Approach: Price Distribution Visualization + Natural Breaks

Before finalizing price tiers, visualize the actual distribution to validate cutoffs.

#### Steps:

1. **Create histogram of inflation-adjusted prices**
   - Plot distribution of adjusted `Original Price (€)`
   - Use appropriate bin sizes (e.g., €5 bins)
   - Add kernel density estimate overlay

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# Create histogram with KDE overlay
# HISTOGRAM: Shows the frequency distribution of prices in bins
# KDE (Kernel Density Estimate): Smooth curve that estimates the probability density
# KDE is useful for seeing the overall shape without the "choppiness" of histogram bars
plt.subplot(1, 2, 1)
plt.hist(df_games['adjusted_price'], bins=50, alpha=0.7, color='skyblue', 
         edgecolor='black', density=True)
df_games['adjusted_price'].plot(kind='kde', color='darkblue', linewidth=2)
plt.xlabel('Inflation-Adjusted Price (€)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Inflation-Adjusted Game Prices', fontsize=14, fontweight='bold')
plt.xlim(0, df_games['adjusted_price'].quantile(0.95))  # Limit to 95th percentile for clarity

# Add percentile lines
for q, label in [(0.25, '25th'), (0.50, '50th'), (0.75, '75th')]:
    val = df_games['adjusted_price'].quantile(q)
    plt.axvline(val, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(val, plt.ylim()[1]*0.9, f'{label}\n€{val:.2f}', 
             ha='center', color='red', fontweight='bold')

plt.subplot(1, 2, 2)
# Log scale for better visibility of lower prices
plt.hist(df_games['adjusted_price'], bins=100, alpha=0.7, color='lightcoral', 
         edgecolor='black')
plt.xlabel('Inflation-Adjusted Price (€)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xlim(0, df_games['adjusted_price'].quantile(0.95))

plt.tight_layout()
plt.savefig('price_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nPrice Distribution Summary:")
print(df_games['adjusted_price'].describe())
```

2. **Identify natural breaks in the distribution**
   - Look for gaps or valleys in the distribution
   - Check if common psychological price points appear:
     - €9.99 (indie/budget threshold)
     - €19.99 (mid-indie threshold)
     - €29.99 (premium indie / budget AAA)
     - €59.99 (standard AAA)

```python
# Define psychological price points
psych_prices = [9.99, 19.99, 29.99, 39.99, 49.99, 59.99, 69.99]

# Count games within ±€2 of each psychological price point
print("\nGames clustered around psychological price points:")
for price in psych_prices:
    count = len(df_games[(df_games['adjusted_price'] >= price - 2) & 
                         (df_games['adjusted_price'] <= price + 2)])
    pct = (count / len(df_games)) * 100
    print(f"€{price:6.2f}: {count:5d} games ({pct:5.2f}%)")

# Visualize clustering around psychological prices
plt.figure(figsize=(12, 6))
plt.hist(df_games['adjusted_price'], bins=200, alpha=0.6, color='steelblue', 
         edgecolor='black')

# Add vertical lines for psychological prices
for price in psych_prices:
    plt.axvline(price, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(price, plt.ylim()[1]*0.95, f'€{price:.0f}', 
             rotation=90, ha='right', va='top', color='red', fontweight='bold')

plt.xlabel('Inflation-Adjusted Price (€)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Price Distribution with Psychological Price Points', fontsize=14, fontweight='bold')
plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('psychological_price_clustering.png', dpi=300, bbox_inches='tight')
plt.show()
```

3. **Compare percentile cutoffs vs. natural breaks**
   - Overlay percentile thresholds on the histogram
   - If percentile cutoffs align with natural breaks → use percentiles
   - If psychological price points show clearer separation → consider using them instead

```python
# Get percentile values
p25 = df_games['adjusted_price'].quantile(0.25)
p50 = df_games['adjusted_price'].quantile(0.50)
p75 = df_games['adjusted_price'].quantile(0.75)

# Compare percentiles with psychological prices
print("\nComparison: Percentiles vs Psychological Prices")
print(f"{'Percentile':<15} {'Value (€)':<12} {'Nearest Psych Price':<20}")
print("-" * 50)
print(f"{'25th':<15} {p25:<12.2f} {'€9.99 or €19.99':<20}")
print(f"{'50th':<15} {p50:<12.2f} {'€29.99 or €39.99':<20}")
print(f"{'75th':<15} {p75:<12.2f} {'€49.99 or €59.99':<20}")

# Visualize comparison
plt.figure(figsize=(14, 6))
plt.hist(df_games['adjusted_price'], bins=100, alpha=0.5, color='gray', 
         edgecolor='black', label='Price Distribution')

# Percentile lines
for q, label, color in [(0.25, '25th', 'blue'), (0.50, '50th', 'green'), 
                         (0.75, '75th', 'purple')]:
    val = df_games['adjusted_price'].quantile(q)
    plt.axvline(val, color=color, linestyle='-', linewidth=2.5, 
                label=f'{label} percentile: €{val:.2f}', alpha=0.8)

# Psychological price lines
for price in psych_prices:
    plt.axvline(price, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

plt.xlabel('Inflation-Adjusted Price (€)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Percentile Cutoffs vs Psychological Price Points', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('percentiles_vs_psychological_prices.png', dpi=300, bbox_inches='tight')
plt.show()

# Decision criterion: alignment check
print("\nAlignment Check:")
threshold = 3  # €3 tolerance
for q, label in [(0.25, '25th'), (0.50, '50th'), (0.75, '75th')]:
    val = df_games['adjusted_price'].quantile(q)
    closest_psych = min(psych_prices, key=lambda x: abs(x - val))
    diff = abs(val - closest_psych)
    aligned = "✓ ALIGNED" if diff < threshold else "✗ NOT ALIGNED"
    print(f"{label} percentile (€{val:.2f}) vs €{closest_psych:.2f}: "
          f"diff = €{diff:.2f} {aligned}")
```

4. **Create alternative tier variable if needed**
   - `price_tier_natural`: Based on distribution-informed cutoffs
   - Compare results between percentile and natural break approaches

```python
# Create natural break tiers based on psychological prices
# Adjust these based on your distribution analysis above
natural_breaks = [0, 14.99, 24.99, 44.99, np.inf]  # Example breaks
natural_labels = ['Budget', 'Mid-Low', 'Mid-High', 'Premium']

df_games['price_tier_natural'] = pd.cut(df_games['adjusted_price'],
                                         bins=natural_breaks,
                                         labels=natural_labels,
                                         include_lowest=True)

# Compare tier assignments
print("\nTier Distribution Comparison:")
print("\nPercentile-based tiers:")
print(df_games['price_tier'].value_counts().sort_index())
print(f"\nTotal: {df_games['price_tier'].value_counts().sum()}")

print("\nNatural break tiers:")
print(df_games['price_tier_natural'].value_counts().sort_index())
print(f"\nTotal: {df_games['price_tier_natural'].value_counts().sum()}")

# Crosstab to see agreement
print("\nCrosstab: Percentile vs Natural Break Tiers")
crosstab = pd.crosstab(df_games['price_tier'], 
                       df_games['price_tier_natural'], 
                       margins=True)
print(crosstab)

# Calculate agreement rate
agreement = (df_games['price_tier'] == df_games['price_tier_natural']).sum()
agreement_rate = agreement / len(df_games) * 100
print(f"\nAgreement rate: {agreement_rate:.2f}%")

# Visualize side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_games['price_tier'].value_counts().sort_index().plot(kind='bar', 
    ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Percentile-Based Tiers', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tier', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

df_games['price_tier_natural'].value_counts().sort_index().plot(kind='bar', 
    ax=axes[1], color='coral', edgecolor='black')
axes[1].set_title('Natural Break Tiers', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tier', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('tier_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Why This Approach?

- **Data-driven validation:** Ensures tiers reflect actual market structure
- **Psychological pricing insight:** Games may cluster around .99 price points
- **Flexibility:** Allows switching between percentile and fixed-price tiers if one proves more meaningful
- **Transparency:** Visualization makes tier definitions interpretable

---

## Part 2: Research Questions and Tier Definitions

Each research question requires specific tier definitions optimized for that analysis.

### Question 1: Do expensive games discount more or less?

**Tier Definition:** Inflation-adjusted price percentiles (or natural breaks if validated)

**Steps:**

1. Use the primary approach (inflation-adjusted percentiles) described above
2. Assign each game to a price tier
3. Calculate average discount % by tier

```python
# Assuming 'price_tier' has been created from Part 1

# Calculate discount statistics by price tier
discount_by_tier = df_games.groupby('price_tier')['Discount%'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)

print("Discount Statistics by Price Tier:")
print(discount_by_tier)

# Calculate additional metrics
print("\nMean adjusted price by tier:")
print(df_games.groupby('price_tier')['adjusted_price'].mean().round(2))
```

4. Test for statistical significance (ANOVA, Kruskal-Wallis)

```python
from scipy import stats

# Remove NaN values
df_clean = df_games[['price_tier', 'Discount%']].dropna()

# Convert discount % to absolute values (remove negative sign if present)
df_clean['abs_discount'] = df_clean['Discount%'].abs()

# Get discount values for each tier
tier_groups = [df_clean[df_clean['price_tier'] == tier]['abs_discount'].values 
               for tier in ['Budget', 'Mid-Low', 'Mid-High', 'Premium']]

# Test for normality (Shapiro-Wilk test on a sample)
# SHAPIRO-WILK TEST: Tests if data follows a normal (bell curve) distribution
# Null hypothesis: Data is normally distributed
# If p-value > 0.05: Data is likely normal → can use parametric tests (ANOVA)
# If p-value < 0.05: Data is NOT normal → use non-parametric tests (Kruskal-Wallis)
print("\nNormality Tests (sample of 1000 per tier):")
for i, tier in enumerate(['Budget', 'Mid-Low', 'Mid-High', 'Premium']):
    sample = tier_groups[i][:1000] if len(tier_groups[i]) > 1000 else tier_groups[i]
    stat, p = stats.shapiro(sample)
    print(f"{tier}: p-value = {p:.4f} {'(normal)' if p > 0.05 else '(not normal)'}")

# Kruskal-Wallis test (non-parametric alternative to ANOVA)
# KRUSKAL-WALLIS TEST: Tests if 3+ groups have different distributions
# Non-parametric = doesn't assume normal distribution (robust to outliers and skewed data)
# Null hypothesis: All groups have the same distribution
# If p-value < 0.05: At least one group is significantly different
# Use this when data is NOT normally distributed
h_stat, p_value = stats.kruskal(*tier_groups)
print(f"\nKruskal-Wallis H-test:")
print(f"H-statistic: {h_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} "
      f"between tiers (α=0.05)")

# ANOVA (parametric test, for comparison)
# ONE-WAY ANOVA: Tests if 3+ groups have different means
# Parametric = assumes normal distribution and equal variances
# If p-value < 0.05: At least one group mean is significantly different
# More powerful than Kruskal-Wallis IF assumptions are met
f_stat, p_value_anova = stats.f_oneway(*tier_groups)
print(f"\nOne-way ANOVA F-test:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value_anova:.6f}")

# Post-hoc pairwise comparisons (if significant)
# MANN-WHITNEY U TEST: Compares two groups (non-parametric)
# Used after Kruskal-Wallis to find which specific pairs differ
# BONFERRONI CORRECTION: Adjusts significance level for multiple comparisons
# Divides α (0.05) by number of tests to control false positives
if p_value < 0.05:
    print("\nPost-hoc pairwise Mann-Whitney U tests (Bonferroni corrected):")
    tiers = ['Budget', 'Mid-Low', 'Mid-High', 'Premium']
    n_comparisons = len(tiers) * (len(tiers) - 1) / 2
    alpha_corrected = 0.05 / n_comparisons
    
    for i in range(len(tiers)):
        for j in range(i + 1, len(tiers)):
            u_stat, p_val = stats.mannwhitneyu(tier_groups[i], tier_groups[j], 
                                               alternative='two-sided')
            sig = "***" if p_val < alpha_corrected else ""
            print(f"{tiers[i]} vs {tiers[j]}: p = {p_val:.6f} {sig}")
```

5. Visualize with boxplots or violin plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
df_clean = df_games[['price_tier', 'Discount%']].dropna()
df_clean['abs_discount'] = df_clean['Discount%'].abs()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Box plot
# BOX PLOT: Shows distribution using quartiles
# - Box: 25th to 75th percentile (middle 50% of data)
# - Line in box: Median (50th percentile)
# - Whiskers: Extend to 1.5×IQR or min/max
# - Dots: Outliers beyond whiskers
# Good for: Comparing medians and spread across groups
sns.boxplot(data=df_clean, x='price_tier', y='abs_discount', 
            palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Discount Distribution by Price Tier (Box Plot)', 
                     fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Price Tier', fontsize=12)
axes[0, 0].set_ylabel('Discount %', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Violin plot
# VIOLIN PLOT: Combines box plot with KDE (density curve on both sides)
# - Width at each point: How many data points at that value
# - Wider = more data points at that discount level
# - Shows full distribution shape (box plots hide this)
# Good for: Seeing if distributions are bimodal (two peaks) or skewed
sns.violinplot(data=df_clean, x='price_tier', y='abs_discount', 
               palette='muted', ax=axes[0, 1])
axes[0, 1].set_title('Discount Distribution by Price Tier (Violin Plot)', 
                     fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Price Tier', fontsize=12)
axes[0, 1].set_ylabel('Discount %', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Bar plot with error bars (mean ± std)
tier_stats = df_clean.groupby('price_tier')['abs_discount'].agg(['mean', 'std'])
tier_stats.plot(kind='bar', y='mean', yerr='std', ax=axes[1, 0], 
                color='steelblue', edgecolor='black', capsize=5, legend=False)
axes[1, 0].set_title('Mean Discount by Price Tier (± 1 SD)', 
                     fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Price Tier', fontsize=12)
axes[1, 0].set_ylabel('Mean Discount %', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Histogram overlays
# OVERLAID HISTOGRAMS: Multiple distributions on same plot with transparency
# - density=True: Normalizes so areas sum to 1 (makes groups comparable)
# - alpha=0.5: Transparency so we can see overlapping distributions
# Good for: Directly comparing distribution shapes across groups
for tier in ['Budget', 'Mid-Low', 'Mid-High', 'Premium']:
    data = df_clean[df_clean['price_tier'] == tier]['abs_discount']
    axes[1, 1].hist(data, bins=30, alpha=0.5, label=tier, density=True)
axes[1, 1].set_title('Discount Distribution Overlays by Tier', 
                     fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Discount %', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('discount_by_price_tier_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary interpretation
print("\n=== INTERPRETATION ===")
mean_by_tier = df_clean.groupby('price_tier')['abs_discount'].mean()
print("\nMean discount by tier:")
for tier, mean_disc in mean_by_tier.items():
    print(f"{tier:12s}: {mean_disc:6.2f}%")

if mean_by_tier['Premium'] > mean_by_tier['Budget']:
    print("\n→ Expensive (Premium) games discount MORE than budget games")
else:
    print("\n→ Expensive (Premium) games discount LESS than budget games")
```

**Why this approach:**
- Inflation adjustment ensures "expensive" is measured in real terms
- Percentiles provide balanced sample sizes for comparison
- Captures whether high-price games use discounts more/less aggressively
- Statistical tests confirm if observed differences are meaningful
- Multiple visualizations reveal distribution patterns beyond just means

---

### Question 2: Do popular games discount differently?

**Tier Definition:** Review-volume based tiers

**Steps:**

1. **Clean review count data**
   - Remove commas from `#Reviews` column
   - Convert to numeric
   - Handle missing values

```python
# Clean the #Reviews column
df_games['#Reviews'] = df_games['#Reviews'].str.replace(',', '')
df_games['review_count'] = pd.to_numeric(df_games['#Reviews'], errors='coerce')

# Check for missing values
print(f"Missing review counts: {df_games['review_count'].isna().sum()}")
print(f"Zero review counts: {(df_games['review_count'] == 0).sum()}")

# Handle missing values (fill with 0 or median)
df_games['review_count'] = df_games['review_count'].fillna(0)

print("\nReview count statistics:")
print(df_games['review_count'].describe())
```

2. **Create log-transformed review variable**
   - `log_reviews = log10(#Reviews + 1)`  (add 1 to handle zeros)
   - This accounts for heavy right-skew in review counts

```python
import numpy as np

# Log transform (add 1 to handle zeros)
# LOG TRANSFORMATION: Compresses large values, spreads out small values
# Why use it: Review counts are heavily right-skewed (few games have millions of reviews)
# log10(1000) = 3, log10(10000) = 4, log10(100000) = 5
# Makes the distribution more "normal" and easier to work with
# Adding 1 prevents log(0) which is undefined
df_games['log_reviews'] = np.log10(df_games['review_count'] + 1)

# Visualize the transformation
# Note: We use 2 subplots side-by-side to compare distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
axes[0].hist(df_games['review_count'], bins=100, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Review Count', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Original Review Count Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, df_games['review_count'].quantile(0.95))

# Log-transformed distribution
axes[1].hist(df_games['log_reviews'], bins=50, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('Log10(Review Count + 1)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Log-Transformed Review Count Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('review_count_transformation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nLog-transformed review statistics:")
print(df_games['log_reviews'].describe())
```

3. **Define popularity tiers using percentiles of log_reviews**
   - **Niche:** 0-25th percentile of log_reviews
   - **Mid-Market:** 25-50th percentile
   - **Popular:** 50-75th percentile
   - **Blockbuster:** 75-100th percentile

```python
# Calculate percentiles on log-transformed reviews
p25_log = df_games['log_reviews'].quantile(0.25)
p50_log = df_games['log_reviews'].quantile(0.50)
p75_log = df_games['log_reviews'].quantile(0.75)

print("Log-review percentiles:")
print(f"25th: {p25_log:.2f} (≈{10**p25_log:.0f} reviews)")
print(f"50th: {p50_log:.2f} (≈{10**p50_log:.0f} reviews)")
print(f"75th: {p75_log:.2f} (≈{10**p75_log:.0f} reviews)")

# Create popularity tiers
df_games['popularity_tier'] = pd.cut(df_games['log_reviews'],
                                     bins=[-np.inf, p25_log, p50_log, p75_log, np.inf],
                                     labels=['Niche', 'Mid-Market', 'Popular', 'Blockbuster'])

print("\nPopularity tier distribution:")
print(df_games['popularity_tier'].value_counts().sort_index())
```

4. **Alternative: Fixed thresholds based on domain knowledge**
   - **Niche:** <1,000 reviews
   - **Small:** 1,000-10,000 reviews
   - **Medium:** 10,000-100,000 reviews
   - **Large:** >100,000 reviews

```python
# Create fixed threshold tiers
df_games['popularity_tier_fixed'] = pd.cut(df_games['review_count'],
                                           bins=[0, 1000, 10000, 100000, np.inf],
                                           labels=['Niche', 'Small', 'Medium', 'Large'],
                                           include_lowest=True)

print("\nFixed threshold popularity tier distribution:")
print(df_games['popularity_tier_fixed'].value_counts().sort_index())

# Compare the two approaches
# CROSSTAB: Creates a contingency table showing how two categorical variables relate
# Rows = one tier method, Columns = other tier method
# Cell values = count of games in both categories
# margins=True adds row/column totals
# Good for: Seeing how much two tier definitions agree
comparison = pd.crosstab(df_games['popularity_tier'], 
                        df_games['popularity_tier_fixed'], 
                        margins=True)
print("\nComparison: Percentile vs Fixed Thresholds")
print(comparison)
```

5. **Analyze discount patterns by popularity tier**

```python
# Discount analysis by popularity
discount_by_popularity = df_games.groupby('popularity_tier')['Discount%'].agg([
    'count', 'mean', 'median', 'std'
]).round(2)

print("\nDiscount Statistics by Popularity Tier:")
print(discount_by_popularity)

# Statistical test
df_clean_pop = df_games[['popularity_tier', 'Discount%']].dropna()
df_clean_pop['abs_discount'] = df_clean_pop['Discount%'].abs()

pop_groups = [df_clean_pop[df_clean_pop['popularity_tier'] == tier]['abs_discount'].values 
              for tier in ['Niche', 'Mid-Market', 'Popular', 'Blockbuster']]

h_stat, p_value = stats.kruskal(*pop_groups)
print(f"\nKruskal-Wallis test for popularity tiers:")
print(f"H-statistic: {h_stat:.4f}, p-value: {p_value:.6f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df_clean_pop, x='popularity_tier', y='abs_discount', 
            palette='viridis', ax=axes[0])
axes[0].set_title('Discount by Popularity Tier', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Popularity Tier', fontsize=12)
axes[0].set_ylabel('Discount %', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

sns.violinplot(data=df_clean_pop, x='popularity_tier', y='abs_discount', 
               palette='plasma', ax=axes[1])
axes[1].set_title('Discount Distribution by Popularity', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Popularity Tier', fontsize=12)
axes[1].set_ylabel('Discount %', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('discount_by_popularity_tier.png', dpi=300, bbox_inches='tight')
plt.show()

# Special case: High price + Low reviews (potential failures)
df_games['is_potential_failure'] = (
    (df_games['price_tier'].isin(['Mid-High', 'Premium'])) & 
    (df_games['popularity_tier'] == 'Niche')
)

print(f"\nPotential failures (high price + low reviews): {df_games['is_potential_failure'].sum()}")
print("\nDiscount for potential failures:")
print(df_games.groupby('is_potential_failure')['Discount%'].agg(['mean', 'median', 'count']))
```

**Why this approach:**
- Review count is strong proxy for sales volume and marketing reach
- Log transformation handles extreme outliers (e.g., games with 1M+ reviews)
- Percentiles ensure equal sample sizes, fixed thresholds align with market intuition
- Separates "popular" (high visibility) from "expensive" (high price)

**Special consideration:**
- Some big-budget games tank and have few reviews → these are interesting outliers
- Flag these for separate analysis: high price tier + low review tier

---

### Question 3: Do quality games follow different patterns?

**Tier Definition:** Rating-based tiers (with price and review interactions)

**Steps:**

1. **Create rating tiers**
   - **Poor:** Rating < 5.0
   - **Mixed:** Rating 5.0-6.9
   - **Positive:** Rating 7.0-7.9
   - **Excellent:** Rating 8.0+

```python
# Create rating tiers
df_games['rating_tier'] = pd.cut(df_games['Rating'],
                                 bins=[0, 5.0, 7.0, 8.0, 10.0],
                                 labels=['Poor', 'Mixed', 'Positive', 'Excellent'],
                                 include_lowest=True)

print("Rating tier distribution:")
print(df_games['rating_tier'].value_counts().sort_index())

print("\nMean rating by tier:")
print(df_games.groupby('rating_tier')['Rating'].mean())

# Visualize rating distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df_games['Rating'].hist(bins=50, color='gold', edgecolor='black', alpha=0.7)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Game Rating Distribution', fontsize=14, fontweight='bold')
plt.axvline(5.0, color='red', linestyle='--', linewidth=2, label='Tier boundaries')
plt.axvline(7.0, color='red', linestyle='--', linewidth=2)
plt.axvline(8.0, color='red', linestyle='--', linewidth=2)
plt.legend()

plt.subplot(1, 2, 2)
df_games['rating_tier'].value_counts().sort_index().plot(kind='bar', 
    color=['firebrick', 'orange', 'yellowgreen', 'darkgreen'], edgecolor='black')
plt.xlabel('Rating Tier', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Game Count by Rating Tier', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('rating_tier_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

2. **Create interaction tiers: Quality × Price**
   - **Value:** High rating (7+) × Low/Mid price (bottom 50% percentile)
   - **Premium Quality:** High rating (7+) × High price (top 50% percentile)
   - **Risky:** Low rating (<6) × High price (top 50% percentile)
   - **Budget/Experimental:** Low rating (<6) × Low price (bottom 50% percentile)

```python
# Define high/low quality and price
df_games['is_high_quality'] = df_games['Rating'] >= 7.0
df_games['is_low_quality'] = df_games['Rating'] < 6.0
df_games['is_high_price'] = df_games['adjusted_price'] >= df_games['adjusted_price'].median()

# Create Quality × Price interaction tiers
def assign_quality_price_tier(row):
    if pd.isna(row['Rating']) or pd.isna(row['adjusted_price']):
        return None
    if row['is_high_quality'] and not row['is_high_price']:
        return 'Value'
    elif row['is_high_quality'] and row['is_high_price']:
        return 'Premium Quality'
    elif row['is_low_quality'] and row['is_high_price']:
        return 'Risky'
    elif row['is_low_quality'] and not row['is_high_price']:
        return 'Budget/Experimental'
    else:
        return 'Mid-Quality'

df_games['quality_price_tier'] = df_games.apply(assign_quality_price_tier, axis=1)

print("\nQuality × Price tier distribution:")
print(df_games['quality_price_tier'].value_counts())

# Analyze discounts by quality-price tier
print("\nDiscount by Quality × Price tier:")
print(df_games.groupby('quality_price_tier')['Discount%'].agg(['mean', 'median', 'count']).round(2))
```

3. **Create interaction tiers: Quality × Popularity**
   - **Hidden Gem:** High rating (7+) × Low reviews (bottom 25% percentile)
   - **Acclaimed Hit:** High rating (7+) × High reviews (top 25% percentile)
   - **Controversial:** Low rating (<6) × High reviews (top 25% percentile)
   - **Obscure:** Low rating (<6) × Low reviews (bottom 25% percentile)

```python
# Define high/low popularity
review_p25 = df_games['review_count'].quantile(0.25)
review_p75 = df_games['review_count'].quantile(0.75)

df_games['is_low_popularity'] = df_games['review_count'] <= review_p25
df_games['is_high_popularity'] = df_games['review_count'] >= review_p75

# Create Quality × Popularity interaction tiers
def assign_quality_popularity_tier(row):
    if pd.isna(row['Rating']) or pd.isna(row['review_count']):
        return None
    if row['is_high_quality'] and row['is_low_popularity']:
        return 'Hidden Gem'
    elif row['is_high_quality'] and row['is_high_popularity']:
        return 'Acclaimed Hit'
    elif row['is_low_quality'] and row['is_high_popularity']:
        return 'Controversial'
    elif row['is_low_quality'] and row['is_low_popularity']:
        return 'Obscure'
    else:
        return 'Mid-Tier'

df_games['quality_popularity_tier'] = df_games.apply(assign_quality_popularity_tier, axis=1)

print("\nQuality × Popularity tier distribution:")
print(df_games['quality_popularity_tier'].value_counts())

# Analyze discounts
print("\nDiscount by Quality × Popularity tier:")
print(df_games.groupby('quality_popularity_tier')['Discount%'].agg(['mean', 'median', 'count']).round(2))
```

4. **Analyze discount patterns across quality tiers and interactions**

```python
# Comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Discount by Rating Tier
df_clean_rating = df_games[['rating_tier', 'Discount%']].dropna()
df_clean_rating['abs_discount'] = df_clean_rating['Discount%'].abs()
sns.boxplot(data=df_clean_rating, x='rating_tier', y='abs_discount', 
            palette='RdYlGn', ax=axes[0, 0])
axes[0, 0].set_title('Discount by Rating Tier', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Rating Tier', fontsize=12)
axes[0, 0].set_ylabel('Discount %', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Discount by Quality × Price
df_clean_qp = df_games[['quality_price_tier', 'Discount%']].dropna()
df_clean_qp['abs_discount'] = df_clean_qp['Discount%'].abs()
sns.boxplot(data=df_clean_qp, x='quality_price_tier', y='abs_discount', 
            palette='Set3', ax=axes[0, 1])
axes[0, 1].set_title('Discount by Quality × Price Tier', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Quality × Price Tier', fontsize=12)
axes[0, 1].set_ylabel('Discount %', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Discount by Quality × Popularity
df_clean_qpop = df_games[['quality_popularity_tier', 'Discount%']].dropna()
df_clean_qpop['abs_discount'] = df_clean_qpop['Discount%'].abs()
sns.boxplot(data=df_clean_qpop, x='quality_popularity_tier', y='abs_discount', 
            palette='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Discount by Quality × Popularity Tier', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Quality × Popularity Tier', fontsize=12)
axes[1, 0].set_ylabel('Discount %', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Heatmap: Rating vs Price (mean discount)
# HEATMAP: 2D grid where color represents a numeric value
# - Rows = Rating tiers, Columns = Price tiers
# - Cell color = Mean discount (darker = higher discount)
# - annot=True: Shows actual numbers in cells
# - Good for: Seeing interaction effects between two categorical variables
# Example: "Do high-quality, high-price games discount more than low-quality, high-price games?"
pivot_table = df_games.pivot_table(
    values='Discount%', 
    index='rating_tier', 
    columns='price_tier', 
    aggfunc=lambda x: x.abs().mean()
)
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Mean Discount: Rating × Price Tiers', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Price Tier', fontsize=12)
axes[1, 1].set_ylabel('Rating Tier', fontsize=12)

plt.tight_layout()
plt.savefig('discount_by_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical tests
print("\nStatistical Tests:")
print("\n1. Rating Tier:")
rating_groups = [df_clean_rating[df_clean_rating['rating_tier'] == tier]['abs_discount'].values 
                 for tier in ['Poor', 'Mixed', 'Positive', 'Excellent']]
h_stat, p_val = stats.kruskal(*rating_groups)
print(f"   Kruskal-Wallis: H={h_stat:.4f}, p={p_val:.6f}")

print("\n2. Quality × Price Tier:")
qp_tiers = df_clean_qp['quality_price_tier'].unique()
qp_groups = [df_clean_qp[df_clean_qp['quality_price_tier'] == tier]['abs_discount'].values 
             for tier in qp_tiers if tier is not None]
h_stat, p_val = stats.kruskal(*qp_groups)
print(f"   Kruskal-Wallis: H={h_stat:.4f}, p={p_val:.6f}")

# Key insights
print("\n=== KEY INSIGHTS ===")
mean_by_rating = df_clean_rating.groupby('rating_tier')['abs_discount'].mean()
print("\nMean discount by rating tier:")
for tier, mean_disc in mean_by_rating.items():
    print(f"  {tier:12s}: {mean_disc:6.2f}%")

mean_by_qp = df_clean_qp.groupby('quality_price_tier')['abs_discount'].mean().sort_values(ascending=False)
print("\nMean discount by Quality × Price tier (sorted):")
for tier, mean_disc in mean_by_qp.items():
    print(f"  {tier:20s}: {mean_disc:6.2f}%")
```

**Why this approach:**
- Rating captures perceived quality/reception
- Interaction with price reveals "value proposition" strategies
- Interaction with reviews separates quality from popularity
- Helps identify if quality games can maintain price or need to discount
- Tests whether "bad but popular" games discount differently than "good but obscure" games

---

### Question 4: Is there a lifecycle effect?

**Tier Definition:** Time-since-release tiers

**Steps:**

1. **Calculate game age**
   - `age_days = Fetched_At - Release_Date` (in days)
   - `age_months = age_days / 30.44` (average days per month)
   - `age_years = age_months / 12`

```python
# Parse dates
df_games['Fetched At'] = pd.to_datetime(df_games['Fetched At'])
df_games['Release Date'] = pd.to_datetime(df_games['Release Date'], format='%d %b, %Y')

# Calculate age
df_games['age_days'] = (df_games['Fetched At'] - df_games['Release Date']).dt.days
df_games['age_months'] = df_games['age_days'] / 30.44
df_games['age_years'] = df_games['age_months'] / 12

print("Game age statistics:")
print(df_games[['age_days', 'age_months', 'age_years']].describe())

# Visualize age distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_games['age_months'], bins=100, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Age (months)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Game Ages', fontsize=14, fontweight='bold')
axes[0].axvline(3, color='red', linestyle='--', alpha=0.7, label='3 months')
axes[0].axvline(12, color='orange', linestyle='--', alpha=0.7, label='12 months')
axes[0].axvline(24, color='green', linestyle='--', alpha=0.7, label='24 months')
axes[0].axvline(48, color='purple', linestyle='--', alpha=0.7, label='48 months')
axes[0].legend()

axes[1].hist(df_games['age_years'], bins=50, color='coral', edgecolor='black')
axes[1].set_xlabel('Age (years)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution of Game Ages (Years)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('game_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

2. **Create lifecycle tiers**
   - **Launch Window:** 0-3 months
   - **New Release:** 3-12 months
   - **Established:** 12-24 months
   - **Mature:** 24-48 months
   - **Catalog/Legacy:** 48+ months

```python
# Create lifecycle stage tiers
df_games['lifecycle_tier'] = pd.cut(df_games['age_months'],
                                    bins=[0, 3, 12, 24, 48, np.inf],
                                    labels=['Launch Window', 'New Release', 'Established', 
                                           'Mature', 'Catalog/Legacy'],
                                    include_lowest=True)

print("\nLifecycle tier distribution:")
print(df_games['lifecycle_tier'].value_counts().sort_index())

print("\nMean age by lifecycle tier:")
print(df_games.groupby('lifecycle_tier')['age_months'].mean().round(1))

# Visualize
df_games['lifecycle_tier'].value_counts().sort_index().plot(kind='bar', 
    figsize=(10, 6), color='teal', edgecolor='black')
plt.xlabel('Lifecycle Stage', fontsize=12)
plt.ylabel('Number of Games', fontsize=12)
plt.title('Game Count by Lifecycle Stage', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('lifecycle_tier_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

3. **Alternative: Percentile-based age tiers**
   - Use quartiles of `age_months` for equal sample sizes

```python
# Create percentile-based age tiers
age_p25 = df_games['age_months'].quantile(0.25)
age_p50 = df_games['age_months'].quantile(0.50)
age_p75 = df_games['age_months'].quantile(0.75)

print(f"\nAge percentiles (months):")
print(f"25th: {age_p25:.1f} months ({age_p25/12:.1f} years)")
print(f"50th: {age_p50:.1f} months ({age_p50/12:.1f} years)")
print(f"75th: {age_p75:.1f} months ({age_p75/12:.1f} years)")

df_games['lifecycle_tier_percentile'] = pd.cut(df_games['age_months'],
                                                bins=[-np.inf, age_p25, age_p50, age_p75, np.inf],
                                                labels=['Very New', 'New', 'Established', 'Old'])

print("\nPercentile-based lifecycle tier distribution:")
print(df_games['lifecycle_tier_percentile'].value_counts().sort_index())

# Compare fixed vs percentile approaches
comparison = pd.crosstab(df_games['lifecycle_tier'], 
                        df_games['lifecycle_tier_percentile'], 
                        margins=True)
print("\nComparison: Fixed vs Percentile Lifecycle Tiers")
print(comparison)
```

4. **Analyze discount patterns by lifecycle stage**
   - Track discount % over time
   - Identify when discounts typically begin
   - Test if older games discount more aggressively

```python
# Discount analysis by lifecycle
discount_by_lifecycle = df_games.groupby('lifecycle_tier')['Discount%'].agg([
    'count', 'mean', 'median', 'std'
]).round(2)

print("\nDiscount Statistics by Lifecycle Stage:")
print(discount_by_lifecycle)

# Statistical test
df_clean_lifecycle = df_games[['lifecycle_tier', 'Discount%']].dropna()
df_clean_lifecycle['abs_discount'] = df_clean_lifecycle['Discount%'].abs()

lifecycle_groups = [df_clean_lifecycle[df_clean_lifecycle['lifecycle_tier'] == tier]['abs_discount'].values 
                    for tier in ['Launch Window', 'New Release', 'Established', 'Mature', 'Catalog/Legacy']]

h_stat, p_value = stats.kruskal(*lifecycle_groups)
print(f"\nKruskal-Wallis test for lifecycle stages:")
print(f"H-statistic: {h_stat:.4f}, p-value: {p_value:.6f}")

# Visualization: Multiple perspectives
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Box plot by lifecycle stage
sns.boxplot(data=df_clean_lifecycle, x='lifecycle_tier', y='abs_discount', 
            palette='magma', ax=axes[0, 0])
axes[0, 0].set_title('Discount by Lifecycle Stage', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Lifecycle Stage', fontsize=12)
axes[0, 0].set_ylabel('Discount %', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Scatter plot: Age vs Discount
# SCATTER PLOT: Each point = one game, shows relationship between two continuous variables
# - X-axis = Age, Y-axis = Discount
# - Color = Price (viridis colormap: purple=low, yellow=high)
# - alpha=0.3: Transparency to see overlapping points
# Good for: Seeing if there's a relationship (correlation) between age and discount
sample = df_games.sample(min(5000, len(df_games)))  # Sample for visibility
axes[0, 1].scatter(sample['age_months'], sample['Discount%'].abs(), 
                   alpha=0.3, c=sample['adjusted_price'], cmap='viridis', s=20)
axes[0, 1].set_xlabel('Age (months)', fontsize=12)
axes[0, 1].set_ylabel('Discount %', fontsize=12)
axes[0, 1].set_title('Discount vs Game Age (colored by price)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim(0, df_games['age_months'].quantile(0.95))

# Add trend line
# POLYNOMIAL REGRESSION (degree 2): Fits a curve (parabola) to show the trend
# Degree 2 allows for curved relationships (e.g., discounts accelerate over time)
# Red dashed line shows the "average" discount at each age
z = np.polyfit(df_games['age_months'].dropna(), df_games['Discount%'].abs().dropna(), 2)
p = np.poly1d(z)
x_trend = np.linspace(0, df_games['age_months'].quantile(0.95), 100)
axes[0, 1].plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
axes[0, 1].legend()

# 3. Mean discount over time (binned)
df_games['age_bins'] = pd.cut(df_games['age_months'], bins=20)
discount_over_time = df_games.groupby('age_bins')['Discount%'].apply(lambda x: x.abs().mean())
bin_centers = [interval.mid for interval in discount_over_time.index]
axes[1, 0].plot(bin_centers, discount_over_time.values, marker='o', 
                linewidth=2, markersize=6, color='darkblue')
axes[1, 0].set_xlabel('Age (months)', fontsize=12)
axes[1, 0].set_ylabel('Mean Discount %', fontsize=12)
axes[1, 0].set_title('Mean Discount Over Game Age', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 4. Distribution of discounts by lifecycle (violin)
sns.violinplot(data=df_clean_lifecycle, x='lifecycle_tier', y='abs_discount', 
               palette='rocket', ax=axes[1, 1])
axes[1, 1].set_title('Discount Distribution by Lifecycle', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Lifecycle Stage', fontsize=12)
axes[1, 1].set_ylabel('Discount %', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('discount_by_lifecycle_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Key insights
print("\n=== KEY INSIGHTS ===")
mean_by_lifecycle = df_clean_lifecycle.groupby('lifecycle_tier')['abs_discount'].mean()
print("\nMean discount by lifecycle stage:")
for tier, mean_disc in mean_by_lifecycle.items():
    print(f"  {tier:15s}: {mean_disc:6.2f}%")

# Correlation between age and discount
# CORRELATION: Measures linear relationship between two variables (-1 to +1)
# -1 = perfect negative (as one goes up, other goes down)
#  0 = no linear relationship
# +1 = perfect positive (both go up/down together)
# Typically: |r| > 0.7 = strong, 0.3-0.7 = moderate, < 0.3 = weak
correlation = df_games[['age_months', 'Discount%']].corr().iloc[0, 1]
print(f"\nCorrelation between age and discount: {correlation:.4f}")

if correlation > 0:
    print("→ Older games tend to have LARGER discounts (positive lifecycle effect)")
else:
    print("→ Older games tend to have SMALLER discounts (negative lifecycle effect)")

# When do discounts typically begin?
early_games = df_games[df_games['age_months'] <= 6]
early_discount_pct = (early_games['Discount%'].abs() > 0).sum() / len(early_games) * 100
print(f"\nGames with discounts in first 6 months: {early_discount_pct:.1f}%")

mature_games = df_games[df_games['age_months'] > 24]
mature_discount_pct = (mature_games['Discount%'].abs() > 0).sum() / len(mature_games) * 100
print(f"Games with discounts after 24 months: {mature_discount_pct:.1f}%")
```

**Why this approach:**
- Lifecycle stage strongly affects pricing strategy
- New releases rarely discount (except failures)
- Older games often enter permanent discount territory
- Fixed time periods align with industry conventions (launch, first year, catalog)
- This is a time-based control variable for all other analyses

**Special consideration:**
- Early Access games may follow different patterns
- Games with major updates/DLC may "reset" lifecycle
- Consider flagging these edge cases

---

## Part 3: Non-Price Features (For Machine Learning)

These features are used as predictors in ML models for:
- **Target 1:** Time to first discount
- **Target 2:** Discount percentage

### Feature Set Overview

These are **features**, not tiers. They go into predictive models as-is or with minimal transformation.

#### 1. Review Volume Features

**Variables:**
- `review_count`: Raw number of reviews
- `log_review_count`: Log-transformed reviews
- `review_velocity`: Reviews per day since release (if temporal data available)
- `review_percentile`: Percentile rank within release year cohort

**Why useful for ML:**
- Proxy for sales/popularity
- Captures marketing reach
- Big games with few reviews are especially informative (failure signal)
- Can predict if publishers discount faster to recoup losses

**Special case to investigate:**
- High price + Low reviews → Will it be discounted faster?
- Create interaction feature: `(price_tier == 'high') & (review_tier == 'low')`

---

#### 2. Rating Features

**Variables:**
- `rating`: Average rating (raw)
- `rating_category`: Categorical (Poor/Mixed/Positive/Excellent)
- `rating_percentile`: Percentile rank within release year
- `rating_review_ratio`: Rating weighted by review count (if needed)

**Why useful for ML:**
- Quality signal
- Bad ratings may trigger earlier discounts
- Good ratings may allow price maintenance
- Can combine with price: High price + Good rating = different strategy than High price + Bad rating

**Interaction features for ML:**
- `rating × price_tier`
- `rating × review_count`
- `(rating < 6) & (price > median_price)` → risky launch indicator

---

#### 3. Platform Availability Features

**Variables:**
- `platform_count`: Number of platforms (1-3)
- `is_windows`, `is_linux`, `is_macos`: Binary indicators
- `is_multiplatform`: 1 if available on 2+ platforms
- `is_windows_exclusive`: 1 if Windows-only

**Why useful for ML:**
- Windows-only often indicates AAA games (resource-intensive)
- Multi-platform suggests established indie or major studio
- Best AAA games often not on Mac/Linux initially
- Platform availability correlates with development budget

**Note on interpretation:**
- Best games not playable on Mac → typically AAA titles with high system requirements
- This is a proxy for game tier, not a direct quality signal

---

#### 4. Time-Since-Release Features

**Variables:**
- `age_days`: Days since release
- `age_months`: Months since release
- `age_years`: Years since release
- `age_category`: Categorical lifecycle stage
- `is_new_release`: 1 if <6 months old
- `is_catalog`: 1 if >36 months old

**Why useful for ML:**
- Critical predictor of discount timing and magnitude
- Lifecycle stage determines pricing strategy
- Older games almost always discount more
- Can create polynomial features: `age²`, `age³` to capture non-linear effects

---

#### 5. Temporal and Seasonal Features (for advanced ML)

**Variables:**
- `release_month`: Month of year (1-12)
- `release_quarter`: Q1-Q4
- `is_holiday_release`: Released near major holiday
- `days_to_next_holiday`: Days until next major sale event
- `days_since_console_release`: Days since nearest console launch

**Why useful for ML:**
- Holiday releases follow different discount patterns
- Games released before Steam sales may discount differently
- Console releases create market pressure on PC prices

---

## Summary: Analysis Strategy

### For Discount Pattern Analysis (Descriptive/Causal):
1. Use **inflation-adjusted percentile price tiers** as primary tier definition
2. Create **question-specific tiers** for each research question
3. Validate with **price distribution visualization**
4. Test robustness across **multiple tier definitions**

### For Predictive Modeling (ML):
1. Use **non-price features as raw/transformed predictors**
2. Include **interaction features** (price × reviews, rating × price, etc.)
3. Let the model learn which features matter most
4. Use tiers as **categorical features** if needed (e.g., one-hot encode lifecycle stage)

### Key Principle:
**Price tiers answer "what" and "why" questions. Features predict "when" and "how much".**


