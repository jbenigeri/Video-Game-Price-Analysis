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
- [ ] Gather console release dates within the identified time window
- [ ] Gather inflation data for the same time window (EU data)
- [ ] Consider game tiers to avoid skewed inflation (cheap independent games should be eliminated)
- [ ] Need dates for the console release dates (ask chat and then validate)
- [ ] We know the reviews include non-eu reviews, and that's okay with us even if we use EU price data


### 5. Prepare and organize data
- [ ] Write down the console release dates and inflation data
- [ ] Use ChatGPT to convert them into CSV format
- [ ] Add these CSVs to the `data/` directory

### 6. Version control
- [ ] Push all changes (datasets, notebook, requirements, CSV files, etc.) to GitHub

### 7. Notion setup
- [ ] Create a Notion account (no further setup needed yet)

## Simon's Task

### 1. Investigate ITAD API integration
- [x] Investigate whether the Kaggle Steam dataset can be leveraged to fetch historical game price data using the IsThereAnyDeal API
- [x] Test API connectivity and authentication
- [x] Create Steam-focused integration functions
- [x] Build production-ready integration module (`steam_itad_integration.py`)
- [x] Create comprehensive usage examples and documentation
- [x] Clean up project structure and remove unnecessary files

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
