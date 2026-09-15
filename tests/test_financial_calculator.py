import pytest
from app.tools.financial_calculator import calculate_percentage_change, calculate_margin, calculate_ratio

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
