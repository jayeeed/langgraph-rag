"""Stock market data tool using Alpha Vantage API."""

import os
import requests
from typing import Literal, Optional
from langchain_core.tools import tool


@tool
def stock_info(
    symbol: str,
    function: Literal[
        "TIME_SERIES_INTRADAY",
        "TIME_SERIES_DAILY",
        "TIME_SERIES_DAILY_ADJUSTED",
        "TIME_SERIES_WEEKLY",
        "TIME_SERIES_WEEKLY_ADJUSTED",
        "TIME_SERIES_MONTHLY",
        "TIME_SERIES_MONTHLY_ADJUSTED",
        "GLOBAL_QUOTE",
    ] = "TIME_SERIES_DAILY",
    interval: Optional[Literal["1min", "5min", "15min", "30min", "60min"]] = None,
    outputsize: Literal["compact", "full"] = "compact",
) -> str:
    """
    Get stock market data from Alpha Vantage API.

    Use this tool to get stock prices, trading volumes, and historical data for any global equity.

    Args:
        symbol: Stock ticker symbol (e.g., 'IBM', 'AAPL', 'TSCO.LON')
        function: Type of data - GLOBAL_QUOTE for latest price, TIME_SERIES_DAILY for daily data,
                  TIME_SERIES_INTRADAY for minute-level data, etc.
        interval: Required for intraday data - '1min', '5min', '15min', '30min', or '60min'
        outputsize: 'compact' for latest 100 points or 'full' for complete history (20+ years)

    Returns:
        Formatted stock data including prices, volumes, and changes
    """
    api_key = os.getenv("STOCK_API_KEY")

    if not api_key:
        return "Error: STOCK_API_KEY environment variable not set"

    # Build parameters
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "datatype": "json",
    }

    # Add function-specific parameters
    if function == "TIME_SERIES_INTRADAY":
        if not interval:
            return "Error: interval is required for intraday data (1min, 5min, 15min, 30min, or 60min)"
        params["interval"] = interval
        params["outputsize"] = outputsize
    elif function != "GLOBAL_QUOTE":
        params["outputsize"] = outputsize

    try:
        response = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Check for errors
        if "Error Message" in data:
            return f"Error: {data['Error Message']}"
        if "Note" in data:
            return f"API Limit: {data['Note']}"

        # Format response
        return _format_stock_data(data, function)

    except requests.exceptions.RequestException as e:
        return f"API Request Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _format_stock_data(data: dict, function: str) -> str:
    """Format stock data for readability."""
    # Find the data keys
    meta_key = next((k for k in data.keys() if "Meta Data" in k), None)
    ts_key = next(
        (k for k in data.keys() if "Time Series" in k or "Global Quote" in k), None
    )

    if not ts_key:
        return str(data)

    time_series = data[ts_key]

    # Format quote data
    if function == "GLOBAL_QUOTE":
        result = f"**{time_series.get('01. symbol', 'N/A')} Stock Quote**\n\n"
        result += f"Price: ${time_series.get('05. price', 'N/A')}\n"
        result += f"Change: {time_series.get('09. change', 'N/A')} ({time_series.get('10. change percent', 'N/A')})\n"
        result += f"Open: ${time_series.get('02. open', 'N/A')}\n"
        result += f"High: ${time_series.get('03. high', 'N/A')}\n"
        result += f"Low: ${time_series.get('04. low', 'N/A')}\n"
        result += f"Volume: {time_series.get('06. volume', 'N/A')}\n"
        result += f"Previous Close: ${time_series.get('08. previous close', 'N/A')}\n"
        result += (
            f"Latest Trading Day: {time_series.get('07. latest trading day', 'N/A')}"
        )
        return result

    # Format time series data
    metadata = data.get(meta_key, {})
    result = f"**{metadata.get('2. Symbol', 'N/A')} Stock Data**\n\n"
    result += f"Last Refreshed: {metadata.get('3. Last Refreshed', 'N/A')}\n\n"

    # Show latest 5 data points
    items = list(time_series.items())[:5]
    result += "Latest Data:\n"
    for timestamp, values in items:
        result += f"\n{timestamp}:\n"
        for key, value in values.items():
            clean_key = key.split(". ", 1)[-1] if ". " in key else key
            result += f"  {clean_key}: {value}\n"

    if len(time_series) > 5:
        result += f"\n... and {len(time_series) - 5} more data points available"

    return result
