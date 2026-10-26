"""
Meridian Retail Logistics (MRL) - Data Generator
Reproducible generation of the 5,075-row raw order extract for MRL.
Schema-aligned with DataCo Smart Supply Chain dataset with realistic faults injected.
Fixed random seed = 42 for exact reproducibility.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_raw_logistics_data(seed=42, output_path="data/raw_logistics_sample.csv"):
    rng = np.random.default_rng(seed)
    
    n_base = 5000
    
    # 1. Order Dates: spanning 2024-01-01 to 2025-10-01 (approx 639 days)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 10, 1)
    total_days = (end_date - start_date).days
    
    # Generate daily order counts roughly 40-70 per week (approx 7.8 orders/day)
    # Uniform / slightly Poisson arrival across the period
    day_offsets = np.sort(rng.integers(0, total_days + 1, size=n_base))
    order_dates = [start_date + timedelta(days=int(d)) for d in day_offsets]
    
    # 2. Identifiers
    order_ids = [f"ORD-{100000 + i}" for i in range(n_base)]
    customer_pool = [f"CUST-{rng.integers(1000, 3500)}" for _ in range(n_base)]
    
    # 3. Categorical Dimensions matching Week 3 Appendix A proportions
    # Shipping modes: Standard (55.4%), Second (21.0%), First (15.8%), Same Day (7.8%)
    mode_choices = ["Standard Class", "Second Class", "First Class", "Same Day"]
    mode_probs = [0.554, 0.210, 0.158, 0.078]
    shipping_modes = rng.choice(mode_choices, size=n_base, p=mode_probs)
    
    # Distribution Centres: DC-East-01 (34.1%), DC-Central-02 (33.1%), DC-West-03 (32.8%)
    dc_choices = ["DC-East-01", "DC-Central-02", "DC-West-03"]
    dc_probs = [0.341, 0.331, 0.328]
    dc_ids = rng.choice(dc_choices, size=n_base, p=dc_probs)
    
    # Product Categories & Unit Price Distributions
    cat_choices = ["Apparel", "Electronics", "Home Goods", "Sporting Goods", "Office Supplies"]
    cat_probs = [0.289, 0.218, 0.205, 0.176, 0.112]
    product_categories = rng.choice(cat_choices, size=n_base, p=cat_probs)
    
    # Regions: East (25.7%), West (25.0%), Central (24.5%), South (24.8%)
    region_choices = ["East", "West", "Central", "South"]
    region_probs = [0.257, 0.250, 0.245, 0.248]
    customer_regions = rng.choice(region_choices, size=n_base, p=region_probs)
    
    # Customer Segment: Consumer (52%), Corporate (30%), Home Office (18%)
    segment_choices = ["Consumer", "Corporate", "Home Office"]
    segment_probs = [0.52, 0.30, 0.18]
    customer_segments = rng.choice(segment_choices, size=n_base, p=segment_probs).tolist()
    
    # 4. Product SKUs, Quantities, and Pricing
    # Quantity: 1 to 7 units, discrete uniform (mean ~ 4.03)
    quantities = rng.integers(1, 8, size=n_base).tolist()
    
    # Unit prices based on category:
    # Electronics high ($150-$500), Home Goods ($50-$200), Sporting ($40-$150), Apparel ($20-$80), Office ($10-$50)
    unit_prices = []
    sku_ids = []
    for i, cat in enumerate(product_categories):
        sku_num = rng.integers(100, 999)
        sku_ids.append(f"SKU-{cat[:3].upper()}-{sku_num}")
        if cat == "Electronics":
            p = rng.uniform(120.0, 480.0)
        elif cat == "Home Goods":
            p = rng.uniform(40.0, 180.0)
        elif cat == "Sporting Goods":
            p = rng.uniform(30.0, 140.0)
        elif cat == "Apparel":
            p = rng.uniform(15.0, 75.0)
        else: # Office Supplies
            p = rng.uniform(10.0, 45.0)
        unit_prices.append(round(p, 2))
        
    # Discount percentage: 0%, 5%, 10%, 15%, 20%
    discounts = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n_base, p=[0.40, 0.20, 0.20, 0.10, 0.10])
    
    # 5. Order Weight (kg): right-skewed lognormal distribution (median ~6.36 kg, mean ~9.44 kg, tail up to 55 kg)
    # Lognormal parameters: meanlog ~ 1.85, sdlog ~ 0.75
    weights = np.round(rng.lognormal(mean=1.85, sigma=0.75, size=n_base), 2)
    weights = np.clip(weights, 0.28, 58.0)
    
    # 6. Transit & Delivery Dates
    # Promised transit times by mode:
    # Same Day: 1 day; First Class: 2 days; Second Class: 3 days; Standard Class: 5 days
    mode_lead_map = {"Same Day": 1, "First Class": 2, "Second Class": 3, "Standard Class": 5}
    promised_dates = [order_dates[i] + timedelta(days=mode_lead_map[shipping_modes[i]]) for i in range(n_base)]
    
    # Delivery delay: 58.6% on-time (delay <= 0), 41.4% late (delay 1 to 18)
    # Mean delay ~0.68, median 0
    delays = []
    for _ in range(n_base):
        if rng.random() < 0.586:
            # On-time: delay is 0 or -1 (early)
            d = 0 if rng.random() < 0.75 else -1
        else:
            # Late: exponential right tail, bounded between 1 and 18
            d = int(np.clip(rng.geometric(p=0.45), 1, 18))
        delays.append(d)
        
    actual_dates = [promised_dates[i] + timedelta(days=delays[i]) for i in range(n_base)]
    late_delivery_risks = [1 if d > 0 else 0 for d in delays]
    
    # 7. True Shipping Cost and Contaminated Carrier Bill
    # Cost formula from rate card:
    # Base fee + weight_rate * weight + category handling + noise
    # Standard: base $10.00, $1.50/kg
    # Second:   base $15.00, $1.85/kg
    # First:    base $22.00, $2.35/kg
    # Same Day: base $32.00, $2.75/kg
    # Plus noise std ~ 2.50
    base_fee_map = {"Standard Class": 10.0, "Second Class": 15.0, "First Class": 22.0, "Same Day": 32.0}
    per_kg_map   = {"Standard Class": 1.52, "Second Class": 1.86, "First Class": 2.38, "Same Day": 2.76}
    
    true_costs = []
    for i in range(n_base):
        m = shipping_modes[i]
        c = base_fee_map[m] + per_kg_map[m] * weights[i] + rng.normal(0, 2.5)
        # Small category handling adjustments
        if product_categories[i] == "Electronics": c += 2.0
        elif product_categories[i] == "Office Supplies": c -= 1.0
        true_costs.append(max(2.50, round(c, 2)))
        
    shipping_costs = list(true_costs)
    is_error_flag = [False] * n_base
    
    # Inject 130 mis-keyed bills (2.6%) inflated 6.1x to 14.0x
    # 102 in training set (first 4,000 orders), 28 in test set (last 1,000 orders)
    train_err_indices = rng.choice(range(4000), size=102, replace=False)
    test_err_indices  = rng.choice(range(4000, 5000), size=28, replace=False)
    err_indices = np.concatenate([train_err_indices, test_err_indices])
    
    for idx in err_indices:
        multiplier = rng.uniform(6.1, 14.0)
        shipping_costs[idx] = round(shipping_costs[idx] * multiplier, 2)
        is_error_flag[idx] = True
        
    # Order Profit: ~15% margin before freight
    profits = []
    for i in range(n_base):
        val = unit_prices[i] * quantities[i] * (1 - discounts[i])
        prof = round(val * rng.uniform(0.12, 0.18), 2)
        profits.append(prof)
        
    # Build initial clean DataFrame
    df = pd.DataFrame({
        "order_id": order_ids,
        "order_date": order_dates,
        "customer_id": customer_pool,
        "customer_segment": customer_segments,
        "customer_region": customer_regions,
        "dc_id": dc_ids,
        "product_category": product_categories,
        "sku_id": sku_ids,
        "quantity": quantities,
        "unit_price": unit_prices,
        "discount_pct": discounts,
        "shipping_mode": shipping_modes,
        "shipping_cost": shipping_costs,
        "order_weight_kg": weights,
        "promised_delivery_date": promised_dates,
        "actual_delivery_date": actual_dates,
        "late_delivery_risk": late_delivery_risks,
        "profit_per_order": profits,
        "_true_cost": true_costs,
        "_is_error": is_error_flag
    })
    
    # -------------------------------------------------------------
    # Deliberately Inject Realistic Week 2 Data Quality Issues
    # -------------------------------------------------------------
    
    # A. 60 Negative Quantities (sign error)
    neg_q_idx = rng.choice(range(n_base), size=60, replace=False)
    for idx in neg_q_idx:
        df.loc[idx, "quantity"] = -abs(df.loc[idx, "quantity"])
        
    # B. 53 Impossible Delivery Dates (delivery precedes order date)
    date_err_idx = rng.choice(range(n_base), size=53, replace=False)
    for idx in date_err_idx:
        df.loc[idx, "actual_delivery_date"] = df.loc[idx, "order_date"] - timedelta(days=int(rng.integers(1, 5)))
        
    # C. Missing Values
    # 156 missing unit_price
    na_price_idx = rng.choice(range(n_base), size=156, replace=False)
    df.loc[na_price_idx, "unit_price"] = np.nan
    
    # 190 missing customer_segment
    na_seg_idx = rng.choice(range(n_base), size=190, replace=False)
    for idx in na_seg_idx:
        df.loc[idx, "customer_segment"] = np.nan
        
    # 279 missing actual_delivery_date (57 in-transit, 222 lost/delayed)
    na_del_idx = rng.choice(range(n_base), size=279, replace=False)
    for idx in na_del_idx:
        df.loc[idx, "actual_delivery_date"] = pd.NaT
        
    # D. Inconsistent Categorical Formatting (12 variations each)
    def dirty_text(val):
        choice = rng.integers(0, 4)
        if choice == 0: return val.lower()
        if choice == 1: return val.upper()
        if choice == 2: return f"  {val}  "
        return val
        
    df["shipping_mode"] = [dirty_text(v) if rng.random() < 0.25 else v for v in df["shipping_mode"]]
    df["customer_region"] = [dirty_text(v) if rng.random() < 0.25 else v for v in df["customer_region"]]
    
    # E. Duplicate Rows:
    # 74 exact duplicates
    dupe_indices = rng.choice(range(n_base), size=74, replace=False)
    dupes_exact = df.iloc[dupe_indices].copy()
    
    # 1 soft duplicate (divergent row sharing same order_id)
    soft_dupe = df.iloc[rng.integers(0, n_base)].copy()
    soft_dupe["shipping_cost"] = round(soft_dupe["shipping_cost"] + 1.25, 2)
    
    # Concatenate to create the 5,075-row raw extract
    raw_df = pd.concat([df, dupes_exact, pd.DataFrame([soft_dupe])], ignore_index=True)
    
    # Shuffle slightly while keeping general chronology
    raw_df = raw_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Separate ground truth audit columns from raw operational extract
    ground_truth = raw_df[["order_id", "_true_cost", "_is_error"]].copy()
    ground_truth.drop_duplicates(subset=["order_id"], inplace=True)
    ground_truth.to_csv("data/ground_truth_audit.csv", index=False)
    
    clean_raw_df = raw_df.drop(columns=["_true_cost", "_is_error"])
    clean_raw_df.to_csv(output_path, index=False)
    
    print(f"Generated raw dataset: {clean_raw_df.shape[0]} rows, {clean_raw_df.shape[1]} columns at {output_path}")
    print(f"Exported ground truth audit dataset to data/ground_truth_audit.csv")
    return clean_raw_df

if __name__ == "__main__":
    generate_raw_logistics_data()
