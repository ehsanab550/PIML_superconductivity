# SuperConNet: Physics-Informed Superconductor Prediction

Python implementation for universal superconducting transition temperature (Tc) prediction. Integrates physics-informed descriptors with ML classification/regression to achieve high-accuracy Tc forecasting across material families. Includes SHAP interpretability for quantum mechanism analysis.

## Usage Workflow

### Step : Feature Engineering
Run first to generate physics-informed (PI) and structural descriptors:
```bash
python FEATURES_(PI_&_structural).py --input_data ./data/SupeCon.csv
