# SuperConNet: Physics-Informed Superconductor Prediction
<img width="333" height="331" alt="image" src="https://github.com/user-attachments/assets/5587fd6d-e8ac-450f-bc12-ec981df72d10" />

Python implementation for universal superconducting transition temperature (Tc) prediction. Integrates physics-informed descriptors with ML classification/regression to achieve high-accuracy Tc forecasting across material families. Includes SHAP interpretability for quantum mechanism analysis.

## Usage Workflow

### Step : Feature Engineering
Run first to generate physics-informed (PI) and structural descriptors:
```bash
python FEATURES_(PI_&_structural).py --input_data ./data/SupeCon.csv
