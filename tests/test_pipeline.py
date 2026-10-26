"""
Unit and Integration Tests for Meridian Retail Logistics (MRL) Pipeline
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

def test_raw_data_dimensions():
    assert os.path.exists("data/raw_logistics_sample.csv"), "Raw dataset missing!"
    raw_df = pd.read_csv("data/raw_logistics_sample.csv")
    assert raw_df.shape[0] == 5075, f"Expected 5,075 raw rows, got {raw_df.shape[0]}"
    assert raw_df.shape[1] == 18, f"Expected 18 raw columns, got {raw_df.shape[1]}"

def test_cleaned_data_integrity():
    assert os.path.exists("data/cleaned_logistics_sample.csv"), "Cleaned dataset missing!"
    df = pd.read_csv("data/cleaned_logistics_sample.csv")
    assert df.shape[0] == 5000, f"Expected 5,000 clean rows, got {df.shape[0]}"
    assert df.shape[1] >= 35, f"Expected at least 35 columns, got {df.shape[1]}"
    assert (df["quantity"] < 0).sum() == 0, "Found unresolved negative quantities!"
    assert df["unit_price"].isna().sum() == 0, "Found unresolved missing unit prices!"
    assert df["shipping_cost_capped"].max() <= 86.5, "Outlier capping on shipping_cost failed!"

def test_champion_model_inference():
    model_path = "src/models/champion_model.joblib"
    assert os.path.exists(model_path), "Champion model file missing!"
    model = joblib.load(model_path)
    
    # Test sample quote
    sample = pd.DataFrame([{
        "order_weight_kg_capped": 10.5,
        "quantity": 3,
        "discount_pct": 0.10,
        "shipping_mode": "Standard Class",
        "product_category": "Electronics"
    }])
    pred = model.predict(sample)
    assert len(pred) == 1
    assert 5.0 <= pred[0] <= 120.0, f"Unreasonable predicted shipping cost: ${pred[0]:.2f}"

def test_route_optimization():
    from src.routing_engine import run_route_comparison
    summary, sweep_r, cw_r, depot, coords = run_route_comparison(seed=42)
    assert summary["Distance Reduction (%)"] > 15.0, "Route optimization did not save enough distance!"
    assert summary["Clarke-Wright Km"] < summary["Dispatcher Sweep Km"]
