# Video Game Tier Approaches

## Overview

This document outlines various approaches for creating video game tiers to analyze discount patterns and pricing strategies. Given the available data (Steam sales data with prices, ratings, reviews, and inflation data), we can create tiers using multiple methodologies.

---

## A. Price-Based Tiering Methods

### 1. Inflation-Adjusted Price Tiers ⭐ **RECOMMENDED**

**Why this makes sense:**
- Controls for temporal price changes in the economy
- Makes games from 2015 comparable to games from 2025
- Essential for longitudinal analysis
- You already have the inflation data!

**Approach:**
- Adjust all `Original Price (€)` values to a common base year (e.g., 2025 euros)
- Define tiers based on inflation-adjusted prices:
  - **Budget/Indie:** €0-15 (adjusted)
  - **Mid-tier:** €15-35 (adjusted)
  - **Premium:** €35-60 (adjusted)
  - **AAA/Deluxe:** €60+ (adjusted)

**Advantages:**
- Fair comparison across years
- Accounts for economic changes
- Better for predictive modeling

---

### 2. Percentile-Based Tiers ⭐ **RECOMMENDED**

**Why this makes sense:**
- Distribution-agnostic (works regardless of price skew)
- Ensures balanced sample sizes across tiers
- Captures relative market positioning
- Robust to outliers

**Approach:**
- Calculate percentiles of `Original Price (€)` (either raw or inflation-adjusted)
- Define tiers:
  - **Tier 1 (Budget):** 0-25th percentile
  - **Tier 2 (Mid-Low):** 25-50th percentile
  - **Tier 3 (Mid-High):** 50-75th percentile
  - **Tier 4 (Premium):** 75-100th percentile

**Advantages:**
- Equal sample sizes for statistical analysis
- Adapts to market distribution
- No arbitrary cutoffs

---

### 3. Hybrid: Inflation-Adjusted Percentiles ⭐⭐ **BEST APPROACH**

**Why this is optimal:**
- Combines benefits of both methods
- Adjusts for inflation THEN uses percentiles
- Most robust and interpretable

**Steps:**
1. Adjust all prices for inflation to 2025 euros
2. Calculate percentiles on adjusted prices
3. Define tiers based on these percentiles
4. Optionally add "natural breaks" at common price points (€9.99, €19.99, €29.99, €59.99)

---

## B. Non-Price Variables for Tiering

Based on the available dataset, here are additional variables you can use:

### 1. Review Volume (#Reviews) ⭐⭐

**Why it matters:**
- Strong proxy for market reach and marketing budget
- AAA games typically have 100K+ reviews
- Indie games often have <10K reviews

**Suggested tiers:**
- **Niche/Indie:** <5,000 reviews
- **Mid-Market:** 5,000-50,000 reviews
- **Popular:** 50,000-200,000 reviews
- **Blockbuster:** 200,000+ reviews

---

### 2. Rating Score (Rating) ⭐

**Why it matters:**
- Indicates quality/reception
- Can combine with price for "value tiers"

**Example combined approach:**
- **Premium Quality:** High price + High rating (€40+, 7.5+)
- **Value Games:** Low/Mid price + High rating (€0-30, 7.5+)
- **Experimental:** High price + Low rating (€40+, <6.0)
- **Budget/Casual:** Low price + Mid rating

---

### 3. Platform Availability (Windows/Linux/MacOS)

**Why it matters:**
- AAA games often Windows-only initially
- Indies more likely cross-platform
- Can indicate development resources

**Tiers:**
- **Windows-only:** Often AAA exclusives or early access
- **Multi-platform (3/3):** Often established indies or large studios
- **Mac/Linux exclusivity:** Rare, niche titles

---

### 4. Time Since Release (derived from Release Date)

**Why it matters:**
- New releases command premium pricing
- Older games often budget-priced
- Controls for product lifecycle

**Tiers:**
- **New Release:** 0-6 months
- **Recent:** 6-18 months
- **Established:** 18-36 months
- **Catalog:** 36+ months

---

### 5. Multi-Dimensional Composite Score ⭐⭐⭐ **MOST SOPHISTICATED**

**Combine multiple factors:**

```python
Tier Score = (
    0.4 × Inflation_Adjusted_Price_Percentile +
    0.3 × Review_Volume_Percentile +
    0.2 × Rating_Score_Normalized +
    0.1 × Platform_Count_Factor
)
```

Then use quartiles/quintiles of this composite score.

**Advantages:**
- Captures multiple dimensions of "tier"
- More nuanced than price alone
- Can be validated against external classifications

---

## C. External Classification (if available)

You could potentially enrich your data with:

### Publisher Size
(e.g., from SteamDB or Wikipedia)
- **Major publishers:** EA, Ubisoft, Activision → AAA
- **Mid-tier:** Devolver, Paradox → Mid-tier
- **Self-published:** → Indie

### Game Genre
(could scrape from Steam)
- Some genres command premium pricing (AAA RPGs)
- Others typically budget (visual novels, casual)

### Steam Tags
(if you can access via API)
- Tags like "Indie," "AAA," "Early Access"

---

## Recommended Approach for This Project

Given the project goals and available data, here's the recommended strategy:

### Primary Method: Inflation-Adjusted Percentile Tiers
1. Adjust `Original Price (€)` for inflation using your HICP data
2. Calculate quartiles (or quintiles) on adjusted prices
3. This becomes your main tier variable

### Secondary Method: Review-Volume Tiers
1. Create tiers based on `#Reviews` (logged scale)
2. Use this to validate/supplement price-based tiers
3. Analyze interaction effects (e.g., do high-review, low-price games discount differently?)

### Exploratory: Composite Tier
1. Create a multi-dimensional tier for robustness checks
2. Compare discount patterns across different tiering methods
3. Use in sensitivity analysis

---

## Why This Matters for Discount Analysis

Different tier definitions will help you answer:

- Do **expensive games** discount more or less? (Price tiers)
- Do **popular games** discount differently? (Review-volume tiers)
- Do **quality games** follow different patterns? (Rating tiers)
- Is there a **lifecycle effect**? (Age tiers)

**Important Note:** The inflation adjustment is crucial because a €60 game in 2015 is very different from a €60 game in 2025 in real terms!

---

## Implementation Considerations

### Data Available
- **Price data:** `Price (€)`, `Original Price (€)`
- **Quality metrics:** `Rating`, `#Reviews`
- **Temporal data:** `Release Date`, `Fetched At`
- **Platform data:** `Windows`, `Linux`, `MacOS`
- **Inflation data:** Recreation and Culture HICP indices by year

### Next Steps
1. Clean and prepare the inflation data for merging with game data
2. Implement inflation adjustment function
3. Create multiple tier variables using different approaches
4. Compare tier distributions and validate against known AAA/indie titles
5. Analyze discount patterns across different tier definitions
6. Document which tier definition best explains discount behavior

---

## Expected Outcomes

By implementing multiple tiering approaches, you can:
1. Identify which factors most strongly predict discount patterns
2. Control for confounding variables in predictive models
3. Provide robust, multi-faceted analysis of pricing strategies
4. Validate findings across different tier definitions
5. Answer both "absolute" questions (fixed price thresholds) and "relative" questions (market positioning)

