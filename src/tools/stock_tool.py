"""Stock information tool using Alpha Vantage API."""

import os
import requests
from langchain_core.tools import tool


@tool
def stock_info(symbol: str) -> str:
    """
    Get real-time stock information and intraday prices for a given stock symbol.

    Use this tool when users ask about stock prices, stock market data, or company stock information.

    Args:
        symbol: The stock ticker symbol (e.g., "IBM", "AAPL", "GOOGL", "MSFT")

    Returns:
        Current stock price information and recent intraday data
    """
    api_key = os.getenv("STOCK_API_KEY")

    if not api_key:
        return "Error: STOCK_API_KEY not configured. Please add your Alpha Vantage API key to the .env file."

    # Get intraday time series data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            return f"Error: Invalid stock symbol '{symbol}'. Please check the ticker symbol and try again."

        if "Note" in data:
            return "Error: API rate limit reached. Please try again later or upgrade your Alpha Vantage API key."

        if "Time Series (5min)" not in data:
            return f"Error: Unable to fetch data for symbol '{symbol}'. Please verify the symbol is correct."

        # Extract metadata and latest data point
        metadata = data.get("Meta Data", {})
        time_series = data["Time Series (5min)"]

        # Get the most recent timestamp
        latest_timestamp = sorted(time_series.keys(), reverse=True)[0]
        latest_data = time_series[latest_timestamp]

        # Format the response
        result = f"""Stock Information for {symbol}:

Last Updated: {latest_timestamp}
Open: ${float(latest_data['1. open']):.2f}
High: ${float(latest_data['2. high']):.2f}
Low: ${float(latest_data['3. low']):.2f}
Close: ${float(latest_data['4. close']):.2f}
Volume: {int(latest_data['5. volume']):,}

Data Source: Alpha Vantage (5-minute intervals)
Last Refreshed: {metadata.get('3. Last Refreshed', 'N/A')}
"""

        return result

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error fetching stock data: {str(e)}"
    except Exception as e:
        return f"Error processing stock data: {str(e)}"
