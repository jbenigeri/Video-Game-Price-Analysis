# Todo List - Video Game Price Analysis Project

## Jacob's Tasks

### 1. Set up project structure
- [x] Download the Kaggle datasets (Steam and Metacritic)
- [x] Place them in a `data/` directory
- [x] Create a `src/` directory for code
- [x] Add a `requirements.txt` file at the root of the repository

### 2. Environment setup
- [x] Create and activate a conda environment
- [x] Install dependencies from `requirements.txt`

### 3. Data exploration
- [x] Create a Jupyter notebook
- [x] Load the Kaggle datasets with pandas
- [x] Perform basic exploration (e.g., find min and max dates in the Steam dataset to define the time range for inflation data)

### 4. Additional data collection
- [x] Gather console release dates within the identified time window
- [x] Gather inflation data for the same time window (EU data)
- [ ] Consider game tiers to avoid skewed inflation (cheap independent games should be eliminated)
- [x] Need dates for the console release dates (ask chat and then validate)
- [x] We know the reviews include non-eu reviews, and that's okay with us even if we use EU price data
- [ ] Before a code block, what are you doing and after say what you learned if anything

### 5. Steam Sales Analysis - Reproduce Notebook Plots
- [ ] Recreate all plots from [Benjamin Lundkvist's Steam Sales EDA Notebook](https://www.kaggle.com/code/benjaminlundkvist/starter-steam-sales-historical-dataset)
- [ ] For each plot: Add a short explanation of what is being plotted
- [ ] For each plot: Provide a brief analysis or takeaway
- [ ] For each plot: Update plots with nicer formatting (consistent titles, labels, colors, figure sizes, and improved readability)

### 6. Steam Sales Analysis - Discount Distribution Analysis
- [ ] Analyze the distribution of discounts
- [ ] Explain why we see a peak at 50% but not much around it
- [ ] Note that 70–80%, 50%, and 20–30% are the main discount ranges
- [ ] Provide a simple, intuitive explanation — this is not a trick question

### 7. Steam Sales Analysis - Game Releases Per Year
- [ ] Plot the number of game releases per year
- [ ] For predictive modeling: data should be restricted to releases before a certain cutoff year
- [ ] For EDA: include all available years to visualize long-term trends

### 8. Steam Sales Analysis - Correlation Heat Map Analysis
- [ ] Examine whether higher ratings or more reviews correlate with larger discounts
- [ ] Analyze platform correlations (Windows independent of Mac/Linux, Mac-Linux correlation)
- [ ] Explain price and original_price correlation and why it makes sense
- [ ] Discuss why these variables are important and how they might interact with discounts
- [ ] Drill down on relationships between game tiers and discounts

### 9. Steam Sales Analysis - Game Tiers Definition
- [ ] Define game tiers by price range
- [ ] Define game tiers by external classification (e.g., publisher labels, AAA vs indie)
- [ ] Create tier-based analysis framework for discount patterns

### 10. Prepare and organize data
- [ ] Write down the console release dates and inflation data
- [ ] Use ChatGPT to convert them into CSV format
- [ ] Add these CSVs to the `data/` directory

### 11. Version control
- [ ] Push all changes (datasets, notebook, requirements, CSV files, etc.) to GitHub

### 12. Notion setup
- [ ] Create a Notion account (no further setup needed yet)

## Simon's Tasks

### 1. Investigate ITAD API integration
- [x] Investigate whether the Kaggle Steam dataset can be leveraged to fetch historical game price data using the IsThereAnyDeal API
- [x] Test API connectivity and authentication
- [x] Create Steam-focused integration functions
- [x] Build production-ready integration module (`steam_itad_integration.py`)
- [x] Create comprehensive usage examples and documentation
- [x] Clean up project structure and remove unnecessary files

### 2. Steam Sales Analysis - Time-based Effects
- [ ] Explore how time to holiday affects discount behavior
- [ ] Explore how time to console release affects discount behavior
- [ ] Analyze seasonal patterns in game pricing
- [ ] Investigate correlation between release timing and discount strategies

## Completed Tasks ✅

### Simon's Completed Work:
- [x] **API Investigation**: Successfully investigated ITAD API capabilities
- [x] **Steam Integration**: Built complete Steam-focused integration with all major functions
- [x] **Production Code**: Created `steam_itad_integration.py` with full functionality
- [x] **Documentation**: Created `USAGE_EXAMPLES.md` with comprehensive examples
- [x] **Project Structure**: Cleaned up and organized project files
- [x] **Steam-Only Focus**: All functions return Steam-only data for consistency
- [x] **Multi-Currency Support**: US, EU, and other regional pricing support
- [x] **Historical Data**: Full price history since game release
- [x] **Batch Processing**: Multiple games at once functionality

## Next Steps

### Immediate (Week 1):
1. **Jacob**: Set up project structure and download Kaggle datasets
2. **Jacob**: Create conda environment and install dependencies
3. **Jacob**: Begin data exploration in Jupyter notebook
4. **Simon**: Assist with data integration and API usage

### Short-term (Week 2-3):
1. **Jacob**: Complete data collection and preparation
2. **Jacob**: Gather console release dates and inflation data
3. **Simon**: Help with data integration and preprocessing
4. **Both**: Begin descriptive analysis and visualization

### Medium-term (Week 4-5):
1. **Both**: Implement predictive modeling
2. **Both**: Conduct event study analysis
3. **Both**: Create final visualizations and write-up

## Project Status

- **Steam ITAD Integration**: ✅ Complete and production-ready
- **Project Structure**: ✅ Clean and organized
- **Documentation**: ✅ Comprehensive usage guide
- **Data Collection**: 🔄 Ready to begin (Jacob's task)
- **Analysis Phase**: ⏳ Pending data collection

## Notes

- The Steam ITAD integration is fully functional and ready for data collection
- All functions are Steam-focused for consistency with the project goals
- The integration supports both US and EU pricing for comprehensive analysis
- Historical data goes back to game release dates for complete price analysis

