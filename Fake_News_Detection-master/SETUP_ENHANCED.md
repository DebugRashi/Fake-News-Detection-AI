# Setup for enhanced experiments

The enhanced experiment script needs compiled scientific Python packages
(`numpy`, `pandas`, `scikit-learn`, plus optional `lime` and `shap`).

Use Python 3.11 or Python 3.12 for the smoothest setup. Python 3.14 is still
too new for many machine learning packages and can produce incompatible
compiled-package errors.

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-enhanced.txt
python enhanced_liar_experiments.py
```

If SHAP is slow or hard to install, the script still works with LIME and the
built-in Logistic Regression explanation fallback.
