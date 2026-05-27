from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
# Paths
MODEL_PATH = BASE_DIR / "models" / "invoice_flag_classifier.pkl"
SCALER_PATH = BASE_DIR / "models" / "invoice_scaler.pkl"


def load_model(model_path=MODEL_PATH):
    """
    Load trained invoice flagging model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    return model


def load_scaler(scaler_path=SCALER_PATH):
    """
    Load trained scaler.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    return scaler


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new invoices.
    Parameters
    ----------
    input_data : dict
    Returns
    -------
    pd.DataFrame
    """

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)
    scaled_input = scaler.transform(input_df)
    predictions = model.predict(scaled_input)
    input_df["Predicted_Flag"] = predictions
    return input_df


if __name__ == "__main__":

    # Example test input
    sample_data = {
        "invoice_quantity": [100, 200, 700],
        "invoice_dollars": [15000, 22000, 27560],
        "Freight": [500, 700, 570],
        "total_item_quantity": [95, 190, 129],
        "total_item_dollars": [14990, 21000, 32091]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)