# AI Stock Prediction Web App

![Streamlit App]((https://brodynelly-ai-stockpredictionweb-main-fmirxg.streamlit.app/))

A powerful web application that uses machine learning to forecast stock prices. Built with Streamlit, Prophet, and yfinance.

![App Screenshot](https://github.com/user-attachments/assets/811e77e5-5fc6-49c4-a769-971bd5098c17)

## Features

- **Real-time Stock Data**: Fetch the latest stock data from Yahoo Finance
- **Interactive Visualizations**: Explore historical stock prices with interactive charts
- **AI-Powered Forecasting**: Predict future stock prices using Facebook's Prophet algorithm
- **Customizable Predictions**: Adjust the forecast period from 1 to 4 years
- **Data Transparency**: View the raw data and forecasted values
- **Component Analysis**: Understand trend and seasonal factors affecting stock prices

## How It Works

This application uses:
- **yfinance**: To download historical stock data
- **Prophet**: A time series forecasting model developed by Facebook Research
- **Streamlit**: For the interactive web interface
- **Plotly**: For creating interactive visualizations

The forecasting model analyzes historical patterns in stock prices, including:
- Long-term trends
- Seasonal patterns
- Weekly cycles
- Holiday effects

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/brodynelly/AI_StockPredictionWeb.git
   cd AI_StockPredictionWeb
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App Locally

```bash
streamlit run main.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. Select a stock from the dropdown menu (AAPL, GOOG, MSFT, GME)
2. Adjust the prediction period using the slider (1-4 years)
3. View historical data and forecasted prices
4. Explore the forecast components to understand patterns
5. Use the "Clear cached stock data" button to refresh the data

## Deployment

This app can be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app

## Troubleshooting

- **Data Loading Issues**: If you encounter data loading errors, try clearing the cache using the button in the app
- **Missing Dependencies**: Ensure all packages in requirements.txt are installed
- **Deployment Errors**: Check that your environment has all necessary dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Facebook Prophet](https://facebook.github.io/prophet/) for the forecasting algorithm
- [yfinance](https://github.com/ranaroussi/yfinance) for providing access to Yahoo Finance data
- [Streamlit](https://streamlit.io/) for the web app framework
