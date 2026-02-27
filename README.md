# SuperConNet: Physics-Informed Superconductor Prediction
<img width="1021" height="683" alt="nano-banana-2025-11-29T07-31-58" src="https://github.com/user-attachments/assets/140e8d71-c515-46b5-8595-31341f77e646" />

Python implementation for universal superconducting transition temperature (Tc) prediction. Integrates physics-informed descriptors with ML classification/regression to achieve high-accuracy Tc forecasting across material families. Includes SHAP interpretability for quantum mechanism analysis.

## Usage Workflow

### Step : Feature Engineering
Run first to generate physics-informed (PI) and structural descriptors:
```bash
python FEATURES_(PI_&_structural).py --input_data ./data/SupeCon.csv
```
## 📝 Citation
If you use this code or data in your research, please cite the following paper:

> Alibagheri, Ehsan, et al. "A physics-informed machine learning framework for unified prediction of superconducting transition temperatures." *Materials Today Physics* 60 (2026): 101971. DOI: [10.1016/j.mtphys.2025.101971](https://doi.org/10.1016/j.mtphys.2025.101971)
>
> BibTeX entry:
> ```bibtex
> @article{alibagheri2026,
>   title={A physics-informed machine learning framework for unified prediction of superconducting transition temperatures},
>   author={Alibagheri, Ehsan, et al.},
>   journal={Materials Today Physics},
>   year={2026},
>   doi={10.1016/j.mtphys.2025.101971}
> }
> ```

### 📄 License
This project is available for academic use. Please cite the paper if you use the code. For commercial use, please contact the author.

For questions or issues, please open an issue on GitHub.
