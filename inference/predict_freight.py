from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "freight_cost_regressor.pkl"

def load_model(model_path:str=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    """
    Load trained freight cost prediction model.
    """
    model = joblib.load(model_path)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    Parameters
    ----------
    input_data : dict
    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model=load_model()
    input_df=pd.DataFrame(input_data)
    input_df['Predicted_Freight']=model.predict(input_df).round()
    return input_df

if __name__=="__main__":
    # Example inference run (local testing)
    sample_data={
        "Quantity":[18,90,20,67],
        "Dollars":[1850,9000,2000,6791]
    }
    prediction=predict_freight_cost(sample_data)
    print(prediction)