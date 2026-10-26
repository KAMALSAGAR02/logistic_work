"""
Meridian Retail Logistics (MRL) - Shipping Cost Prediction Models
Fits candidate regressors on the 4,000 training orders and evaluates them
on the 1,000 held-out test orders against audited ground truth.
Exports the champion Gradient Boosting model and model comparison metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def logwrap(reg):
    """Wrap regressor to fit on log(cost) and auto-exponentiate predictions."""
    return TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)

def evaluate_models(data_path="data/cleaned_logistics_sample.csv", 
                    audit_path="data/ground_truth_audit.csv",
                    model_dir="src/models"):
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load data and sort chronologically
    df = pd.read_csv(data_path, parse_dates=["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    
    # Merge ground truth error indicators
    gt = pd.read_csv(audit_path)
    df = df.merge(gt[["order_id", "_true_cost", "_is_error"]], on="order_id", how="left")
    
    # 2. Time-based split: 4,000 train / 1,000 test
    split = 4000
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    
    # Features available at order time
    NUM = ["order_weight_kg_capped", "quantity", "discount_pct"]
    CAT = ["shipping_mode", "product_category"]
    
    prep_tree = ColumnTransformer([
        ("n", "passthrough", NUM),
        ("c", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT)
    ])
    
    prep_lin = ColumnTransformer([
        ("n", StandardScaler(), NUM),
        ("c", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT)
    ])
    
    X_train = train[NUM + CAT]
    y_train = train["shipping_cost"] # Contaminated training bills
    
    # Legitimate test orders (orders without mis-keyed bill errors)
    test_leg = test[~test["_is_error"]].copy()
    X_test_leg = test_leg[NUM + CAT]
    y_test_leg = test_leg["_true_cost"].values
    
    # 3. Fit Candidate Models
    models = {}
    
    # A. Naive Baseline (training median)
    train_median = y_train.median()
    
    # B. OLS Linear Regression
    lr = Pipeline([("prep", prep_lin), ("reg", LinearRegression())])
    lr.fit(X_train, y_train)
    models["Linear Regression (OLS)"] = lr
    
    # C. Robust Huber Linear
    huber = Pipeline([("prep", prep_lin), ("reg", logwrap(HuberRegressor(epsilon=1.35, max_iter=500)))])
    huber.fit(X_train, y_train)
    models["Robust Huber Linear"] = huber
    
    # D. Decision Tree (log cost)
    dt = Pipeline([("prep", prep_tree), ("reg", logwrap(DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42)))])
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt
    
    # E. Random Forest Surrogate (Tree Ensemble via LightGBM Extra Trees / Bagging)
    rf = Pipeline([("prep", prep_tree), ("reg", logwrap(lgb.LGBMRegressor(
        boosting_type="rf",
        n_estimators=100,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        bagging_fraction=0.8,
        feature_fraction=0.8,
        random_state=42,
        verbose=-1
    )))])
    rf.fit(X_train, y_train)
    models["Random Forest (Tree Ensemble)"] = rf
    
    # F. Gradient Boosting Champion (L1 absolute loss, log cost, 300 trees, rate 0.05, depth 3)
    gb = Pipeline([("prep", prep_tree), ("reg", logwrap(lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=8,
        random_state=42,
        verbose=-1
    )))])
    gb.fit(X_train, y_train)
    models["Gradient Boosting (Champion)"] = gb
    
    # 4. Evaluate on Held-Out Legitimate Period
    results = []
    
    # Baseline score
    pred_base = np.full(len(y_test_leg), train_median)
    results.append({
        "Model": "Naive: Training Median",
        "RMSE ($)": round(mean_squared_error(y_test_leg, pred_base) ** 0.5, 2),
        "MAE ($)": round(mean_absolute_error(y_test_leg, pred_base), 2),
        "R²": round(r2_score(y_test_leg, pred_base), 3),
        "MdAPE (%)": round(100 * np.median(np.abs(y_test_leg - pred_base) / y_test_leg), 1),
        "Within 10%": round(100 * (np.abs(y_test_leg - pred_base) / y_test_leg <= 0.10).mean(), 1),
        "Within 20%": round(100 * (np.abs(y_test_leg - pred_base) / y_test_leg <= 0.20).mean(), 1),
    })
    
    for name, m in models.items():
        preds = m.predict(X_test_leg)
        rmse = round(mean_squared_error(y_test_leg, preds) ** 0.5, 2)
        mae = round(mean_absolute_error(y_test_leg, preds), 2)
        r2 = round(r2_score(y_test_leg, preds), 3)
        mdape = round(100 * np.median(np.abs(y_test_leg - preds) / y_test_leg), 1)
        w10 = round(100 * (np.abs(y_test_leg - preds) / y_test_leg <= 0.10).mean(), 1)
        w20 = round(100 * (np.abs(y_test_leg - preds) / y_test_leg <= 0.20).mean(), 1)
        results.append({
            "Model": name,
            "RMSE ($)": rmse,
            "MAE ($)": mae,
            "R²": r2,
            "MdAPE (%)": mdape,
            "Within 10%": w10,
            "Within 20%": w20
        })
        
    res_df = pd.DataFrame(results)
    print("\n=== Model Performance on Legitimate Test Orders ===")
    print(res_df.to_string(index=False))
    
    # Save comparison table
    res_df.to_csv("data/final_model_comparison.csv", index=False)
    
    # Save Champion Model
    model_save_path = os.path.join(model_dir, "champion_model.joblib")
    joblib.dump(gb, model_save_path)
    print(f"\nSaved champion Gradient Boosting model to {model_save_path}")
    
    # Attach predictions to full test set (all 1,000 orders) for freight bill audit & optimization
    test["predicted_cost"] = gb.predict(test[NUM + CAT]).round(2)
    test.to_csv("data/test_orders_with_predictions.csv", index=False)
    print("Saved test set predictions to data/test_orders_with_predictions.csv")
    
    return gb, res_df

if __name__ == "__main__":
    evaluate_models()
