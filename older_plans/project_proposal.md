# Project Proposal

## Title: What Drives Video Game Prices? Inflation, Discounts, and Console Cycles

### 1. Motivation and Background

Video games are a multibillion-dollar industry, yet prices behave in complex ways. New releases often debut at premium price points, followed by steep discounts, bundles, and resale dynamics. Meanwhile, macroeconomic forces (inflation, consumer spending) and industry shocks (new console launches) may also shape pricing trends.

This project aims to uncover how video game prices evolve over time, what factors predict faster or deeper price drops, and whether major industry events — like console releases — accelerate discounting.

By combining economic analysis (inflation-adjusted comparisons), predictive modeling (discount dynamics), and event study methods (console cycles), this project demonstrates an end-to-end data science workflow with both technical and practical insights.

### 2. Research Goals

#### Inflation-Adjusted Trends
- Compare launch and resale prices across decades after adjusting for inflation.
- Assess whether "real" launch prices have truly risen.

#### Predictive Modeling of Discounts
- Predict time to first major discount and maximum discount depth within a year of release.
- Identify which factors — genre, publisher, platform, review scores, initial MSRP — best predict price erosion.

#### Impact of Console Cycles
- Estimate whether major console launches (e.g., PS5, Xbox Series X|S, Switch 2) change discount behavior for games released around those events.

### 3. Data Sources

- **Price Data (PC & Console)**: Steam historical price datasets (Kaggle), IsThereAnyDeal API for cross-store histories, PriceCharting for console resale values.
- **Game Metadata**: Metacritic datasets (genre, platform, scores, release dates).
- **Economic Data**: U.S. Consumer Price Index (CPI) from Bureau of Labor Statistics / FRED for inflation adjustment.
- **Industry Events**: Public records of console release dates.

### 4. Methodology & Steps

#### A. Data Preparation
- Collect raw datasets from multiple sources (Steam/Kaggle, ITAD API, PriceCharting, Metacritic).
- Standardize currencies, clean inconsistencies, and merge datasets by game title + release date.
- Adjust all nominal prices to 2025 USD using CPI.

#### B. Descriptive Analysis (Economic Lens)
- Plot inflation-adjusted launch prices by year and platform.
- Compute the "half-life" of prices: median time to reach −30%, −50%, and −70% of launch price.
- Compare across publishers, franchises, and genres.

#### C. Predictive Modeling (Machine Learning Lens)

**Targets:**
- Time to first −50% discount.
- Maximum discount within first 12 months.
- Price after 180 days.

**Features:**
- Metadata (platform, genre, publisher, review scores, franchise status).
- Initial MSRP, early discounting behavior.
- Console-cycle indicator (released ±90 days from a major launch).

**Models:**
- Survival models (Cox PH) for time-to-discount.
- Tree-based methods (Random Forest, Gradient Boosting) for regression/classification tasks.

**Evaluation:**
- Train/test split by release year to mimic real forecasting.
- Metrics: C-index for survival, MAE/RMSE for regression, classification accuracy for discount thresholds.
- SHAP values for model interpretability.

#### D. Event Study (Industry Shock Lens)
- Define event windows around console launches (e.g., ±6 months).
- Estimate the incremental probability and magnitude of discounts during these windows, controlling for game age and genre.
- Visualize event-time effects (discounts relative to months from console release).

### 5. Expected Outcomes

- **Economic Insight**: Evidence of how real launch prices and discount half-lives have changed across platforms and eras.
- **Predictive Models**: Accurate, interpretable models that highlight key drivers of discounting (e.g., genre, publisher reputation, review scores).
- **Industry Analysis**: Clear quantification of how console launches affect pricing dynamics.

#### Deliverables:
- GitHub repository with reproducible code and datasets.
- Jupyter notebooks demonstrating the workflow.
- Visualizations (price curves, survival plots, SHAP diagrams, event-time graphs).
- A short write-up (blog-style or PDF) summarizing findings.

### 6. Timeline (5 Weeks)

- **Week 1**: Data collection & cleaning; CPI inflation adjustments.
- **Week 2**: Descriptive analysis & initial visualizations.
- **Week 3**: Predictive modeling (survival + regression).
- **Week 4**: Event study & model interpretation.
- **Week 5**: Finalize visuals, write-up, and repository.

### 7. Contribution to MS Application

This project demonstrates:

- Ability to formulate real-world questions as data science problems.
- Competence across the full analytics pipeline: acquisition → cleaning → analysis → modeling → communication.
- Application of both statistical methods (survival, event study) and machine learning (tree ensembles with SHAP).
- A narrative that connects economics, predictive analytics, and industry relevance, making it engaging to reviewers.

