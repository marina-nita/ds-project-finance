# Data Science Project: Finance Dataset Analysis

🗓️ **December 2022**

This repository contains a modular, logically separated Python project developed as part of a Data Science analysis focused on a financial dataset. The project was originally implemented in a Jupyter notebook and has been split into clean, reusable Python scripts.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `01_setup_imports.py` | All necessary libraries for data processing, visualization, and machine learning. |
| `02_install_and_mount.py` | Setup for Google Colab (mounting Google Drive, installing libraries). |
| `03_data_loading.py` | Loading the CSV data from Google Drive. |
| `04_data_inspection.py` | Initial exploration: `.info()`, `.head()`, etc. |
| `05_data_cleaning.py` | Cleaning operations: removing unnecessary columns, handling missing values. |
| `06_visualization.py` | Exploratory Data Analysis using Matplotlib, Seaborn, and PtitPrince. |
| `07_preprocessing.py` | Feature encoding and preprocessing steps. |
| `08_misc.py` | Miscellaneous code that doesn't fall into the above categories. |

---

## 💡 Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn, PtitPrince
- scikit-learn
- Google Colab

---

## 🧠 Objective

Analyze and understand patterns in financial data, prepare it for modeling, and visualize meaningful insights using clean, reproducible code.

---

## ✅ How to Use

1. Run the scripts in logical order (01 to 07) or combine into a Jupyter Notebook.
2. Ensure required libraries are installed (`pip install -r requirements.txt`).
3. Place dataset CSV at the expected path or modify the path in `03_data_loading.py`.



