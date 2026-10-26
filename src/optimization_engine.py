"""
Meridian Retail Logistics (MRL) - Operational Optimization Engine
Implements the core prescriptive optimization levers:
1. Freight-bill anomaly audit
2. Shipping-mode reassignment counterfactuals
3. SLA promise-date calibration
4. ABC safety-stock tiering
"""

import numpy as np
import pandas as pd
from scipy import stats

def run_freight_audit(test_predictions_path="data/test_orders_with_predictions.csv", 
                      k_threshold=4.0):
    """
    Evaluates carrier invoice audit detector using predicted cost vs billed cost.
    k_threshold: ratio of billed_cost / predicted_cost to flag for manual review.
    """
    df = pd.read_csv(test_predictions_path)
    # Residual ratio
    ratio = df["shipping_cost"] / np.maximum(df["predicted_cost"], 1.0)
    flagged = ratio >= k_threshold
    
    # Ground truth evaluation
    real_errors = df["_is_error"]
    
    tp = (flagged & real_errors).sum()
    fp = (flagged & ~real_errors).sum()
    fn = (~flagged & real_errors).sum()
    tn = (~flagged & ~real_errors).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Financial recovery
    # Dollar overbilling in flagged true errors
    overbilled_recovered = (df.loc[flagged & real_errors, "shipping_cost"] - df.loc[flagged & real_errors, "_true_cost"]).sum()
    total_overbilling = (df.loc[real_errors, "shipping_cost"] - df.loc[real_errors, "_true_cost"]).sum()
    
    audit_summary = {
        "k_threshold": k_threshold,
        "total_test_orders": len(df),
        "total_billed_spend": round(df["shipping_cost"].sum(), 2),
        "total_overbilling": round(total_overbilling, 2),
        "total_flagged": int(flagged.sum()),
        "real_errors_caught": int(tp),
        "false_alarms": int(fp),
        "missed_errors": int(fn),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "dollars_recovered": round(overbilled_recovered, 2)
    }
    return audit_summary, df.assign(audit_ratio=ratio, audit_flagged=flagged)

def run_mode_reassignment(test_predictions_path="data/test_orders_with_predictions.csv", 
                          champion_model=None):
    """
    Simulates moving 10%, 15%, 20%, 25% of non-Standard orders down one mode tier.
    Same Day -> First Class; First Class -> Second Class; Second Class -> Standard Class.
    """
    import joblib
    if champion_model is None:
        champion_model = joblib.load("src/models/champion_model.joblib")
        
    df = pd.read_csv(test_predictions_path)
    leg = df[~df["_is_error"]].copy()
    
    tier_down_map = {
        "Same Day": "First Class",
        "First Class": "Second Class",
        "Second Class": "Standard Class"
    }
    
    # Identify eligible orders
    eligible_mask = leg["shipping_mode"].isin(tier_down_map.keys())
    eligible_orders = leg[eligible_mask].copy()
    
    # Counterfactual features
    cf_orders = eligible_orders.copy()
    cf_orders["shipping_mode"] = cf_orders["shipping_mode"].map(tier_down_map)
    
    NUM = ["order_weight_kg_capped", "quantity", "discount_pct"]
    CAT = ["shipping_mode", "product_category"]
    
    current_preds = eligible_orders["predicted_cost"].values
    downgrade_preds = champion_model.predict(cf_orders[NUM + CAT])
    
    unit_savings = current_preds - downgrade_preds
    eligible_orders["unit_saving"] = np.maximum(0.0, unit_savings)
    
    # Total potential savings if 100% shifted
    total_eligible = len(eligible_orders)
    avg_saving_per_shifted_order = eligible_orders["unit_saving"].mean()
    
    results = []
    for adoption_pct in [0.10, 0.15, 0.20, 0.25]:
        shifted_count = int(round(total_eligible * adoption_pct))
        # Top-saving orders or random sample
        projected_saving = round(shifted_count * avg_saving_per_shifted_order, 2)
        spend_share = round((projected_saving / leg["shipping_cost"].sum()) * 100, 2)
        results.append({
            "Adoption Rate": f"{int(adoption_pct * 100)}%",
            "Orders Shifted (per 1k)": shifted_count,
            "Total Savings ($)": projected_saving,
            "Logistics Spend Saved (%)": spend_share
        })
        
    return pd.DataFrame(results), eligible_orders

def run_promise_calibration(data_path="data/cleaned_logistics_sample.csv"):
    """
    Computes out-of-sample on-time rate as delivery promise buffer (padding days) increases.
    """
    df = pd.read_csv(data_path, parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"])
    # Filter delivered orders
    delivered = df[df["delivery_delay_days"].notna()].copy()
    
    split = 4000
    test = delivered.iloc[split:].copy()
    
    modes = ["All Modes", "Standard Class", "Second Class", "First Class", "Same Day"]
    results = []
    
    for pad in range(0, 6):
        row = {"Buffer Days": f"+{pad} Days"}
        # Current delay: actual - promised
        # Padded delay: actual - (promised + pad) = delay - pad
        # On-time condition: delay - pad <= 0
        padded_delay = test["delivery_delay_days"] - pad
        row["All Modes"] = round((padded_delay <= 0).mean() * 100, 1)
        
        for m in ["Standard Class", "Second Class", "First Class", "Same Day"]:
            m_sub = test[test["shipping_mode"] == m]
            if len(m_sub) > 0:
                row[m] = round(((m_sub["delivery_delay_days"] - pad) <= 0).mean() * 100, 1)
            else:
                row[m] = np.nan
        results.append(row)
        
    return pd.DataFrame(results)

def run_safety_stock_abc(data_path="data/cleaned_logistics_sample.csv"):
    """
    Compares Uniform 95% service level vs Tiered ABC safety stock policies.
    """
    df = pd.read_csv(data_path)
    
    # SKU economics
    sku_stats = df.groupby(["sku_id", "product_category"]).agg(
        total_demand=("quantity", "sum"),
        mean_unit_price=("unit_price", "mean"),
        order_count=("order_id", "count")
    ).reset_index()
    
    sku_stats["annual_value"] = sku_stats["total_demand"] * sku_stats["mean_unit_price"]
    sku_stats = sku_stats.sort_values("annual_value", ascending=False).reset_index(drop=True)
    
    sku_stats["cum_value"] = sku_stats["annual_value"].cumsum()
    total_val = sku_stats["annual_value"].sum()
    sku_stats["cum_pct"] = sku_stats["cum_value"] / total_val
    
    # ABC Tiering
    def assign_tier(cum):
        if cum <= 0.80: return "A"
        if cum <= 0.95: return "B"
        return "C"
        
    sku_stats["abc_tier"] = sku_stats["cum_pct"].apply(assign_tier)
    
    # Lead time L = 4 days, daily demand std dev approx 25% of mean
    lead_time = 4.0
    holding_cost_rate = 0.20 # 20% annual holding cost
    
    # Z factors: 90% -> 1.282, 95% -> 1.645, 99% -> 2.326
    z_map_tiered = {"A": 2.326, "B": 1.645, "C": 1.282}
    z_uniform = 1.645
    
    sku_stats["demand_std"] = np.maximum(1.0, sku_stats["total_demand"] / 92.0 * 0.40)
    sku_stats["sigma_L"] = sku_stats["demand_std"] * np.sqrt(lead_time)
    
    sku_stats["ss_uniform_units"] = z_uniform * sku_stats["sigma_L"]
    sku_stats["ss_uniform_cost"] = sku_stats["ss_uniform_units"] * sku_stats["mean_unit_price"] * holding_cost_rate
    
    sku_stats["z_tiered"] = sku_stats["abc_tier"].map(z_map_tiered)
    sku_stats["ss_tiered_units"] = sku_stats["z_tiered"] * sku_stats["sigma_L"]
    sku_stats["ss_tiered_cost"] = sku_stats["ss_tiered_units"] * sku_stats["mean_unit_price"] * holding_cost_rate
    
    summary = {
        "uniform_total_ss_value": round((sku_stats["ss_uniform_units"] * sku_stats["mean_unit_price"]).sum(), 2),
        "uniform_holding_cost_yr": round(sku_stats["ss_uniform_cost"].sum(), 2),
        "tiered_total_ss_value": round((sku_stats["ss_tiered_units"] * sku_stats["mean_unit_price"]).sum(), 2),
        "tiered_holding_cost_yr": round(sku_stats["ss_tiered_cost"].sum(), 2),
        "net_holding_cost_diff_yr": round(sku_stats["ss_tiered_cost"].sum() - sku_stats["ss_uniform_cost"].sum(), 2),
        "sku_counts_by_tier": sku_stats["abc_tier"].value_counts().to_dict()
    }
    return summary, sku_stats

if __name__ == "__main__":
    audit, _ = run_freight_audit()
    print("Freight Audit Summary:", audit)
    reassign, _ = run_mode_reassignment()
    print("\nMode Reassignment:\n", reassign)
    calib = run_promise_calibration()
    print("\nPromise Calibration:\n", calib)
