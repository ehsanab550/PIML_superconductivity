# -*- coding: utf-8 -*-
"""
@author: EHSAN_ab
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load and prepare data (replace with your path of supercon and ICSD dataset with features)
df = pd.read_csv("path_to_supecon_dataset" ,  encoding='unicode_escape' )
df_ICSD = pd.read_csv("path_to_ICSD_dataset" ,  encoding='unicode_escape' )

X = df.iloc[:,4:80]
y = df['target']

X_I = df_ICSD.iloc[:,4:80]
y_I = df_ICSD['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# ================================
# Model Training and Evaluation
# ================================

def evaluate_model(model, X, y):
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return r2, mae, rmse

# 1. XGBoost Model
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

xgb_r2, xgb_mae, xgb_rmse = evaluate_model(xgb_model, X_test, y_test)
print("\n=== XGBoost Metrics ===")
print(f'R2: {xgb_r2:.4f}')
print(f'MAE: {xgb_mae:.4f}')
print(f'RMSE: {xgb_rmse:.4f}')

# 2. TabNet Model
# Data preparation
y_train_tab = y_train.to_numpy().reshape(-1, 1)
y_test_tab = y_test.to_numpy().reshape(-1, 1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model configuration
tabnet = TabNetRegressor(
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type="entmax",
    verbose=0
)

# Training
tabnet.fit(
    X_train=X_train_scaled,
    y_train=y_train_tab,
    eval_set=[(X_train_scaled, y_train_tab), (X_test_scaled, y_test_tab)],
    eval_name=["train", "val"],
    max_epochs=200,
    patience=30,
    batch_size=128,
    virtual_batch_size=64,
    num_workers=0
)

# Predictions
tabnet_train_pred = tabnet.predict(X_train_scaled).flatten()
tabnet_test_pred = tabnet.predict(X_test_scaled).flatten()

# Metrics
tabnet_r2, tabnet_mae, tabnet_rmse = evaluate_model(tabnet, X_test_scaled, y_test_tab.flatten())
print("\n=== TabNet Metrics ===")
print(f'R2: {tabnet_r2:.4f}')
print(f'MAE: {tabnet_mae:.4f}')
print(f'RMSE: {tabnet_rmse:.4f}')

# 3. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_r2, rf_mae, rf_rmse = evaluate_model(rf_model, X_test, y_test)
print("\n=== Random Forest Metrics ===")
print(f'R2: {rf_r2:.4f}')
print(f'MAE: {rf_mae:.4f}')
print(f'RMSE: {rf_rmse:.4f}')

# ================================
# Plotting Functions (Identical Style)
# ================================

def plot_predictions(y_train, y_test, train_pred, test_pred, model_name, filename):
    # Convert to flat arrays
    def to_flat_array(x):
        return np.array(x).ravel()
    
    y_train = to_flat_array(y_train)
    train_pred = to_flat_array(train_pred)
    y_test = to_flat_array(y_test)
    test_pred = to_flat_array(test_pred)

    # Set plot style
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='brown', linewidth=2)
    plt.rc('grid', color='lightgray', linestyle='--')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot training and test predictions
    ax.scatter(y_train, train_pred, 
               c='#1f77b4',  # Blue
               edgecolors=(0.5, 0.1, 0.8),
               s=120,
               alpha=0.8,
               label="Training Set")

    ax.scatter(y_test, test_pred,
               c='#d62728',  # Red
               edgecolors=(0.5, 0.1, 0.8),
               marker='s',  # Square markers
               s=140,
               alpha=0.9,
               label="Test Set")

    # Set equal axis limits
    all_vals = np.concatenate([y_train, train_pred, y_test, test_pred])
    buffer = 0.1 * (np.nanmax(all_vals) - np.nanmin(all_vals))
    min_val = np.nanmin(all_vals) - buffer
    max_val = np.nanmax(all_vals) + buffer
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Axis labels
    ax.set_xlabel(r'Experimental $T_{\mathrm{c}}$ (K$^{\circ}$)', 
                 fontsize=28, labelpad=15)
    ax.set_ylabel(r'Predicted $T_{\mathrm{c}}$ (K$^{\circ}$)', 
                 fontsize=28, labelpad=15)

    # Perfect prediction line
    ax.plot([min_val, max_val], [min_val, max_val],
           color='#2ca02c', lw=3, linestyle='--',
           label='Perfect Prediction')

    # Regression line (using training data)
    b, a = np.polyfit(y_train, train_pred, 1)
    xseq = np.linspace(min_val, max_val, 100)
    ax.plot(xseq, a + b*xseq, 
           color='k', lw=2.5, 
           label=f"{model_name} Fit: $T_c^{{pred}} = {b:.2f}T_c^{{exp}} + {a:.2f}$")

    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Metrics box
    metrics_text = (f'Training Metrics:\n'
                   f'R² = {train_r2:.2f}\n'
                   f'MAE = {train_mae:.2f} K\n'
                   f'RMSE = {train_rmse:.2f} K\n\n'
                   f'Test Metrics:\n'
                   f'R² = {test_r2:.2f}\n'
                   f'MAE = {test_mae:.2f} K\n'
                   f'RMSE = {test_rmse:.2f} K')

    ax.text(0.05, 0.95, metrics_text,
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=18,
           bbox=dict(boxstyle='round', 
                     facecolor='lightyellow',
                     edgecolor='goldenrod',
                     alpha=0.9))

    # Legend
    ax.legend(
        prop={'size': 16},
        loc='lower right',
        frameon=True,
        facecolor='lightyellow',
        edgecolor='goldenrod',
        borderpad=1.2,
        labelspacing=1.0
    )

    # Grid and ticks
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=22)

    plt.tight_layout()
    plt.savefig(f'{filename}.jpg', dpi=600, bbox_inches='tight')
    plt.show()

# ================================
# Generate Plots for All Models
# ================================

print("\nGenerating XGBoost Plot...")
plot_predictions(y_train, y_test, 
                xgb_train_pred, xgb_test_pred, 
                "XGBoost", "xgboost_tc_prediction")

print("\nGenerating TabNet Plot...")
plot_predictions(y_train, y_test, 
                tabnet_train_pred, tabnet_test_pred, 
                "TabNet", "tabnet_tc_prediction")

print("\nGenerating Random Forest Plot...")
plot_predictions(y_train, y_test, 
                rf_train_pred, rf_test_pred, 
                "Random Forest", "rf_tc_prediction")
