# -*- coding: utf-8 -*-
"""
@author: EHSAN_ab
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import shap
import numpy as np
#from xgboost import XGBRegressor


plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'olive'
plt.rcParams['axes.linewidth'] = 2


# Load data
df = pd.read_csv(r"path to the dataset with features", encoding='unicode_escape')
X = df.iloc[:, 3:90]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Calculate safe sample size
sample_size = min(500, len(X_train))
X_train_sampled = X_train.sample(sample_size, random_state=42)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values_obj = explainer(X_train_sampled, check_additivity=False)
shap_values = shap_values_obj.values

# Define physics-informed features
physics_informed_features = [
    "O/Cu_ratio", "Tc_avg","Fe_fraction", "TM_fraction", "Fe/(Fe+O)_ratio",
    "Magnetic_avg", "Lambda_avg", "CuO2_planes", "Apical_O_dist",
    "Hole_doping", "Jahn_Teller", "Perovskite_distortion", "Charge_reservoir",
    "Fe_layer_sep", "Chalcogen_height", "Pnictogen_ratio", "Fe_Fe_distance",
    "Magnetic_coupling", "Structural_anisotropy", "Rare_earth_radius",
    "Phonon_freq_avg", "DOS_Fermi", "Isotope_effect", "BCS_gap_ratio",
    "Covalent_character", "Electron_conc", "Atomic_size_var", "Melting_point_avg",
    "Ionicity_index", "Crystal_complexity", "Weighted_Z", "Valence_imbalance",
    "Lattice_stability"
]

# Create feature importance table
importance_df = pd.DataFrame({
    'feature': X_train_sampled.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

# Get the top features for the plot based on SHAP importance
top_features = importance_df.head(10)['feature'].tolist()

# Define plotting function with correct feature ordering
def plot_custom_shap(shap_values, X_data, top_features, highlight_features, 
                     cmap='jet', highlight_color='darkred', bg_color='white'):
    """
    Custom SHAP plot with actual feature names ordered by importance
    """
    num_features_to_show = len(top_features)
    
    # Create a new DataFrame with only the top features in importance order
    X_top = X_data[top_features]
    
    # Reorder SHAP values to match top features importance order
    feature_indices = [list(X_data.columns).index(f) for f in top_features]
    shap_top = shap_values[:, feature_indices]
    
    plt.figure(figsize=(13, 10))
    shap.summary_plot(
        shap_top, 
        X_top, 
        show=False,
        max_display=num_features_to_show,
        cmap=cmap,
        plot_size=None,
        feature_names=top_features  # Ensure correct names
    )
    
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    
    # Highlight specified features
    ytick_labels = ax.get_yticklabels()
    present_highlight = []
    
    for label in ytick_labels:
        text = label.get_text()
        if text in highlight_features:
            label.set_color(highlight_color)
            label.set_fontweight('bold')
            label.set_fontstyle('italic')
            label.set_fontsize(20)
            present_highlight.append(text)
    
    # Check for missing features
    missing = set(highlight_features) - set(present_highlight)
    if missing:
        print(f"Note: These features not in top {num_features_to_show}: {', '.join(missing)}")
        print("Consider increasing the number of features shown")

    # Axis styling
    ax.tick_params(axis='x', colors='black', labelsize=20)
    ax.tick_params(axis='y', colors='black', labelsize=20)
    
    # Grid customization
    plt.grid(True, which='major', axis='y', 
             linestyle='--', color='lightgray', alpha=0.7)
    
    # Border styling
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Title and labels
    plt.title(f"Top {num_features_to_show} Feature Impacts", 
             fontsize=22, pad=15, color='black', fontname='Times New Roman')
    plt.xlabel("SHAP Value Impact", fontsize=24, color='black', fontname='Times New Roman')
    
    plt.tight_layout()
    plt.savefig('path to the image folder to save', dpi=600, bbox_inches='tight')
    plt.show()

# Generate the plot with consistent ordering
plot_custom_shap(
    shap_values,
    X_train_sampled,
    top_features=top_features,
    highlight_features=physics_informed_features,
    cmap='jet',
    highlight_color='darkred',
    bg_color='white'
)

# Add physics-informed flag to importance table
importance_df['physics_informed'] = importance_df['feature'].apply(
    lambda x: 'Yes' if x in physics_informed_features else 'No'
)

# Save to CSV and print top 20
importance_df.to_csv('path to resluts folder', index=False)
print("\nTop 20 Features by SHAP Value Impact:")
print(importance_df.head(20).to_string(index=False))

# Additional analysis: Physics-informed features in top 20
physics_in_top = [feat for feat in physics_informed_features if feat in top_features]

print("\nPhysics-informed features in top 10:")
if physics_in_top:
    for i, feat in enumerate(physics_in_top):
        rank = top_features.index(feat) + 1
        print(f"{rank}. {feat}")
else:
    print("No physics-informed features in top 20")
