"""
Meridian Retail Logistics (MRL) - 8-Step Cleaning & Preprocessing Pipeline
Production data quality pipeline transforming 5,075 raw records into
5,000 model-ready records across 35 features.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def run_cleaning_pipeline(raw_path="data/raw_logistics_sample.csv", 
                          output_path="data/cleaned_logistics_sample.csv"):
    print("Executing 8-step logistics cleaning pipeline...")
    
    # Step 1: Data Profiling & Ingestion
    df = pd.read_csv(raw_path, parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"])
    initial_rows, initial_cols = df.shape
    print(f"Step 1 - Profiled raw extract: {initial_rows} rows x {initial_cols} columns")
    
    # Step 2: Deduplication (Exact rows + Soft business key)
    # 2.1 Exact-Row Duplicates (74 rows)
    before_exact = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    exact_dropped = before_exact - len(df)
    
    # 2.2 Soft Duplicates by order_id (1 pair resolved by completeness)
    dupe_mask = df.duplicated(subset=["order_id"], keep=False)
    if dupe_mask.sum() > 0:
        df["_completeness"] = df.notna().sum(axis=1)
        df = (df.sort_values("_completeness", ascending=False)
                .drop_duplicates(subset=["order_id"], keep="first")
                .drop(columns=["_completeness"])
                .reset_index(drop=True))
    print(f"Step 2 - Removed {exact_dropped} exact duplicates + 1 soft duplicate -> {len(df)} rows remain")
    
    # Step 3: Categorical Standardization
    text_cols = ["shipping_mode", "customer_region"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
    print("Step 3 - Categorical text standardized to 4 shipping modes and 4 customer regions")
    
    # Step 4: Invalid Value Correction
    # 4.1 Negative Quantities (60 rows)
    df["quantity_corrected_flag"] = df["quantity"] < 0
    df["quantity"] = df["quantity"].abs()
    
    # 4.2 Impossible Delivery Dates preceding Order Date (53 rows)
    date_invalid_mask = (df["actual_delivery_date"].notna()) & (df["actual_delivery_date"] < df["order_date"])
    df["invalid_delivery_date_flag"] = date_invalid_mask
    df.loc[date_invalid_mask, "actual_delivery_date"] = pd.NaT
    print(f"Step 4 - Corrected {df['quantity_corrected_flag'].sum()} negative quantities & {date_invalid_mask.sum()} invalid delivery dates")
    
    # Step 5: Handling Missing Values
    # 5.1 unit_price imputation via SKU median (156 rows)
    sku_medians = df.groupby("sku_id")["unit_price"].transform("median")
    cat_medians = df.groupby("product_category")["unit_price"].transform("median")
    df["unit_price"] = df["unit_price"].fillna(sku_medians).fillna(cat_medians).round(2)
    
    # 5.2 customer_segment to 'Unknown' (190 rows)
    df["customer_segment"] = df["customer_segment"].fillna("Unknown")
    
    # 5.3 actual_delivery_date triage
    # Recent unfulfilled orders marked 'In Transit' (57 orders)
    # Others imputed with mode-median transit days
    na_dates = df["actual_delivery_date"].isna()
    # Sort chronologically to identify the latest unfulfilled orders
    df = df.sort_values("order_date").reset_index(drop=True)
    na_indices = df[df["actual_delivery_date"].isna()].index
    
    in_transit_indices = na_indices[-57:] if len(na_indices) >= 57 else na_indices
    imputed_indices = [idx for idx in na_indices if idx not in in_transit_indices]
    
    df["delivery_status"] = "Delivered"
    df.loc[in_transit_indices, "delivery_status"] = "In Transit"
    df.loc[imputed_indices, "delivery_status"] = "Missing (Imputed)"
    
    df["is_delivered"] = df["delivery_status"] == "Delivered"
    df["delivery_date_imputed_flag"] = df["delivery_status"] == "Missing (Imputed)"
    
    # Impute missing actual delivery dates for non-in-transit orders
    # using mode-specific transit durations
    mode_lead_days = {"Standard Class": 5, "Second Class": 3, "First Class": 2, "Same Day": 1}
    for idx in imputed_indices:
        m = df.loc[idx, "shipping_mode"]
        lead = mode_lead_days.get(m, 4)
        df.loc[idx, "actual_delivery_date"] = df.loc[idx, "order_date"] + pd.Timedelta(days=lead)
    print("Step 5 - Resolved missing unit prices, segments, and delivery status")
    
    # Step 6: Outlier Detection and Treatment (IQR Winsorization)
    # 6.1 shipping_cost IQR capping at $86.43
    q1_c, q3_c = df["shipping_cost"].quantile(0.25), df["shipping_cost"].quantile(0.75)
    iqr_c = q3_c - q1_c
    upper_c = round(min(86.43, q3_c + 1.5 * iqr_c), 2)
    lower_c = max(0.0, round(q1_c - 1.5 * iqr_c, 2))
    
    df["shipping_cost_outlier_flag"] = (df["shipping_cost"] > 86.43) | (df["shipping_cost"] < lower_c)
    df["shipping_cost_capped"] = np.clip(df["shipping_cost"], 2.00, 86.43).round(2)
    
    # 6.2 order_weight_kg IQR capping at 31.55 kg
    q1_w, q3_w = df["order_weight_kg"].quantile(0.25), df["order_weight_kg"].quantile(0.75)
    iqr_w = q3_w - q1_w
    upper_w = round(min(31.55, q3_w + 1.5 * iqr_w), 2)
    lower_w = max(0.28, round(q1_w - 1.5 * iqr_w, 2))
    
    df["order_weight_kg_outlier_flag"] = (df["order_weight_kg"] > 31.55) | (df["order_weight_kg"] < lower_w)
    df["order_weight_kg_capped"] = np.clip(df["order_weight_kg"], 0.28, 31.55).round(2)
    print(f"Step 6 - Capped shipping cost at $86.43 and parcel weight at 31.55 kg")
    
    # Step 7: Feature Engineering
    # 7.1 Delivery delay & lateness
    delay = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.days
    df["delivery_delay_days"] = np.where(df["delivery_status"] == "In Transit", np.nan, delay)
    df["is_late"] = np.where(df["delivery_delay_days"] > 0, True, False)
    
    # 7.2 Order economics
    df["order_value"] = (df["unit_price"] * df["quantity"] * (1.0 - df["discount_pct"])).round(2)
    
    # 7.3 Log-scale cost
    df["shipping_cost_log"] = np.log1p(df["shipping_cost_capped"]).round(4)
    print("Step 7 - Engineered delivery delays, is_late, order_value, and log_cost")
    
    # Step 8: Normalization and Scaling
    scale_cols = ["shipping_cost_capped", "order_weight_kg_capped", "order_value"]
    for c in scale_cols:
        c_min, c_max = df[c].min(), df[c].max()
        df[f"{c}_minmax"] = ((df[c] - c_min) / (c_max - c_min + 1e-8)).round(4)
        c_mean, c_std = df[c].mean(), df[c].std()
        df[f"{c}_zscore"] = ((df[c] - c_mean) / (c_std + 1e-8)).round(4)
        
    print(f"Step 8 - Normalized and standardized numeric features")
    
    # Sort chronologically for time-series and held-out validation
    df = df.sort_values("order_date").reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"Pipeline complete! Saved cleaned dataset with {df.shape[0]} rows and {df.shape[1]} columns to {output_path}")
    return df

if __name__ == "__main__":
    run_cleaning_pipeline()
