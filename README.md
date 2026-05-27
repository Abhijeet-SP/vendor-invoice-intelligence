# Vendor Invoice Intelligence Portal

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vendor-invoice-intelligence-fxnuyhveiykegmqnyxzent.streamlit.app/)

An AI-powered internal analytics portal built with Streamlit that leverages machine learning to forecast freight costs and detect risky vendor invoices — reducing financial leakage and manual workload.

**Live App:** https://vendor-invoice-intelligence-fxnuyhveiykegmqnyxzent.streamlit.app/

---

## Features

### 1. Freight Cost Prediction
- Predicts freight cost from **Quantity** and **Invoice Dollars**
- Trained on three regression models: Linear Regression, Decision Tree, and Random Forest
- Best model selected automatically by lowest Mean Absolute Error (MAE)
- Supports budgeting, forecasting, and vendor negotiations

### 2. Invoice Manual Approval Flagging
- Classifies whether a vendor invoice should be **flagged for manual review**
- Detects abnormal cost, freight, or delivery patterns
- Built with a Random Forest Classifier tuned via GridSearchCV (5-fold CV, F1 scoring)
- Input features: Invoice Quantity, Invoice Dollars, Freight Cost, Total Item Quantity, Total Item Dollars

---

## Project Structure

```
Invoice_System/
├── app.py                                  # Streamlit web application
├── requirements.txt
├── data/
│   └── inventory.db                        # SQLite database (vendor_invoice + purchases tables)
├── models/
│   ├── freight_cost_regressor.pkl          # Trained regression model
│   ├── invoice_flag_classifier.pkl         # Trained classification model
│   └── invoice_scaler.pkl                  # StandardScaler for classification features
├── freight_cost_estimation/
│   ├── data_preprocessing.py               # Load, split data for regression
│   ├── modeling_evaluation.py              # Train & evaluate regression models
│   └── train.py                            # Training pipeline entry point
├── invoice_flagging_classification/
│   ├── data_preprocessing.py               # SQL joins, feature engineering, labelling
│   ├── modeling_evaluation.py              # Train & evaluate classification models
│   └── train.py                            # Training pipeline entry point
├── inference/
│   ├── predict_freight.py                  # Inference wrapper for freight cost
│   └── predict_invoice_flagging.py         # Inference wrapper for invoice flag
└── notebooks/
    ├── predicting_freight_cost.ipynb
    └── invoice_flagging.ipynb
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Streamlit |
| Data | SQLite (via `sqlite3`) |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Invoice_System.git
cd Invoice_System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the Models

If you want to retrain from scratch using the SQLite database:

```bash
# Freight cost regression model
cd freight_cost_estimation
python train.py

# Invoice flagging classification model
cd ../invoice_flagging_classification
python train.py
```

Pre-trained models are already included in the `models/` directory.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How It Works

### Freight Cost Prediction
1. Data is loaded from the `vendor_invoice` table in the SQLite database
2. Features: `Quantity`, `Dollars` → Target: `Freight`
3. Three models are trained and compared; the best (lowest MAE) is saved
4. The Streamlit UI takes user inputs and returns a predicted freight cost

### Invoice Risk Flagging
1. Data is loaded via a SQL join between `vendor_invoice` and `purchases` tables
2. A binary risk label (`flag_invoice`) is engineered based on:
   - Dollar discrepancy > $5 between invoice and purchase order
   - Average receiving delay > 10 days
3. Features are scaled with `StandardScaler`
4. A Random Forest Classifier is trained with hyperparameter tuning
5. The app predicts `MANUAL APPROVAL REQUIRED` or `SAFE for Auto-Approval`

---

## Model Performance

| Task | Model | Metric |
|---|---|---|
| Freight Cost Prediction | Best of LR / DT / RF (by MAE) | MAE, RMSE, R² |
| Invoice Flagging | Random Forest (GridSearchCV) | F1-Score, Accuracy |

Detailed training logs are printed to the console during the training runs.

---

## Database Schema

The SQLite database (`data/inventory.db`) contains two main tables:

- **`vendor_invoice`** — PONumber, Quantity, Dollars, Freight, InvoiceDate, PODate, PayDate
- **`purchases`** — PONumber, Brand, Quantity, Dollars, PODate, ReceivingDate

---

## License

This project is intended for portfolio and educational purposes.
