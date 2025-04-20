from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Input schema with all 83 features
class LaunchFeatures(BaseModel):
    FlightNumber: float
    PayloadMass: float
    Flights: float
    Block: float
    ReusedCount: float
    Orbit_ES_L1: float
    Orbit_GEO: float
    Orbit_GTO: float
    Orbit_HEO: float
    Orbit_ISS: float
    Orbit_LEO: float
    Orbit_MEO: float
    Orbit_PO: float
    Serial_B1058: float
    Serial_B1059: float
    Serial_B1060: float
    Serial_B1062: float
    GridFins_False: float
    GridFins_True: float
    Reused_False: float
    Reused_True: float
    Legs_False: float
    Legs_True: float
    Extra_Feature_1: float
    Extra_Feature_2: float
    Extra_Feature_3: float
    Extra_Feature_4: float
    Extra_Feature_5: float
    Extra_Feature_6: float
    Extra_Feature_7: float
    Extra_Feature_8: float
    Extra_Feature_9: float
    Extra_Feature_10: float
    Extra_Feature_11: float
    Extra_Feature_12: float
    Extra_Feature_13: float
    Extra_Feature_14: float
    Extra_Feature_15: float
    Extra_Feature_16: float
    Extra_Feature_17: float
    Extra_Feature_18: float
    Extra_Feature_19: float
    Extra_Feature_20: float
    Extra_Feature_21: float
    Extra_Feature_22: float
    Extra_Feature_23: float
    Extra_Feature_24: float
    Extra_Feature_25: float
    Extra_Feature_26: float
    Extra_Feature_27: float
    Extra_Feature_28: float
    Extra_Feature_29: float
    Extra_Feature_30: float
    Extra_Feature_31: float
    Extra_Feature_32: float
    Extra_Feature_33: float
    Extra_Feature_34: float
    Extra_Feature_35: float
    Extra_Feature_36: float
    Extra_Feature_37: float
    Extra_Feature_38: float
    Extra_Feature_39: float
    Extra_Feature_40: float
    Extra_Feature_41: float
    Extra_Feature_42: float
    Extra_Feature_43: float
    Extra_Feature_44: float
    Extra_Feature_45: float
    Extra_Feature_46: float
    Extra_Feature_47: float
    Extra_Feature_48: float
    Extra_Feature_49: float
    Extra_Feature_50: float
    Extra_Feature_51: float
    Extra_Feature_52: float
    Extra_Feature_53: float
    Extra_Feature_54: float
    Extra_Feature_55: float
    Extra_Feature_56: float
    Extra_Feature_57: float
    Extra_Feature_58: float
    Extra_Feature_59: float
    Extra_Feature_60: float


app = FastAPI()

@app.get("/")
def root():
    return {"message": "SpaceX Model Inference API is live!"}

@app.post("/predict")
def predict(features: LaunchFeatures):
    input_data = np.array([list(features.dict().values())])
    prediction = model.predict(input_data)[0]
    return {
        "prediction": int(prediction),
        "label": "Will Land" if prediction == 1 else "Will Not Land"
    }