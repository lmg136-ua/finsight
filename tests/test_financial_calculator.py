import pytest
from app.tools.financial_calculator import (
    calculate_percentage_change, 
    calculate_margin, 
    calculate_ratio,
    calculate_cagr,
    format_financial_amount
)

def test_calculate_percentage_change():
    assert calculate_percentage_change(100, 110) == "10.00%"
    assert calculate_percentage_change(100, 90) == "-10.00%"
    assert calculate_percentage_change(0, 100) == "N/A (previous value is zero)"

def test_calculate_margin():
    assert calculate_margin(1000, 200) == "20.00%"
    assert calculate_margin(500, 50) == "10.00%"
    assert calculate_margin(0, 100) == "N/A (revenue is zero)"

def test_calculate_ratio():
    assert calculate_ratio(150, 50) == "3.00x"
    assert calculate_ratio(100, 0) == "N/A (denominator is zero)"

def test_calculate_cagr():
    # 100 to 144 in 2 periods = 20%
    assert calculate_cagr(100, 144, 2) == "20.00%"
    assert calculate_cagr(0, 100, 2) == "N/A (values must be positive and periods > 0)"
    assert calculate_cagr(100, 100, 0) == "N/A (values must be positive and periods > 0)"

def test_format_financial_amount():
    assert format_financial_amount(245_120_000_000) == "$245.12B"
    assert format_financial_amount(50_500_000) == "$50.50M"
    assert format_financial_amount(15_000) == "$15.00K"
    assert format_financial_amount(-1_200_000_000) == "-$1.20B"
