def calculate_percentage_change(previous: float, current: float) -> str:
    """
    Calculate the percentage change between two numbers.
    Args:
        previous: The value from the earlier period.
        current: The value from the current period.
    Returns:
        A string representing the percentage change, e.g., '6.88%'.
    """
    if previous == 0:
        return "N/A (previous value is zero)"
    
    change = ((current - previous) / abs(previous)) * 100
    return f"{change:.2f}%"

def calculate_margin(revenue: float, profit: float) -> str:
    """
    Calculate a profit or operating margin.
    Args:
        revenue: Total revenue.
        profit: Operating income or net income.
    Returns:
        A string representing the margin percentage, e.g., '21.50%'.
    """
    if revenue == 0:
        return "N/A (revenue is zero)"
    
    margin = (profit / revenue) * 100
    return f"{margin:.2f}%"

def calculate_ratio(numerator: float, denominator: float) -> str:
    """
    Calculate a simple ratio.
    Args:
        numerator: The top number of the ratio.
        denominator: The bottom number of the ratio.
    Returns:
        A string representing the ratio, e.g., '1.50x'.
    """
    if denominator == 0:
        return "N/A (denominator is zero)"
    
    ratio = numerator / denominator
    return f"{ratio:.2f}x"

def calculate_cagr(beginning_val: float, ending_val: float, periods: int) -> str:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    Args:
        beginning_val: Initial value at beginning period.
        ending_val: Final value at ending period.
        periods: Number of years/periods.
    Returns:
        A string representing CAGR percentage, e.g., '14.25%'.
    """
    if beginning_val <= 0 or ending_val <= 0 or periods <= 0:
        return "N/A (values must be positive and periods > 0)"
    
    cagr = ((ending_val / beginning_val) ** (1.0 / periods) - 1.0) * 100
    return f"{cagr:.2f}%"

def format_financial_amount(amount: float, currency: str = "$") -> str:
    """
    Format large financial amounts into readable strings (billions, millions, etc.).
    Args:
        amount: Raw numerical amount (typically in millions or exact dollars).
        currency: Currency symbol (default '$').
    Returns:
        Readable string, e.g., '$245.12B' or '$50.00M'.
    """
    abs_amt = abs(amount)
    sign = "-" if amount < 0 else ""
    if abs_amt >= 1e9:
        return f"{sign}{currency}{abs_amt / 1e9:.2f}B"
    elif abs_amt >= 1e6:
        return f"{sign}{currency}{abs_amt / 1e6:.2f}M"
    elif abs_amt >= 1e3:
        return f"{sign}{currency}{abs_amt / 1e3:.2f}K"
    else:
        return f"{sign}{currency}{abs_amt:.2f}"
