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
        A string representing the ratio, e.g., '1.5x'.
    """
    if denominator == 0:
        return "N/A (denominator is zero)"
    
    ratio = numerator / denominator
    return f"{ratio:.2f}x"
