# SuperConNet: Physics-Informed Superconductor Prediction
<img width="1021" height="683" alt="nano-banana-2025-11-29T07-31-58" src="https://github.com/user-attachments/assets/140e8d71-c515-46b5-8595-31341f77e646" />

Python implementation for universal superconducting transition temperature (Tc) prediction. Integrates physics-informed descriptors with ML classification/regression to achieve high-accuracy Tc forecasting across material families. Includes SHAP interpretability for quantum mechanism analysis.

## Usage Workflow

### Step : Feature Engineering
Run first to generate physics-informed (PI) and structural descriptors:
```bash
python FEATURES_(PI_&_structural).py --input_data ./data/SupeCon.csv
