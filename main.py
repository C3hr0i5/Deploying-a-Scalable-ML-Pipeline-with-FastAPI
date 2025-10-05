import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# --- Load artifacts ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoder.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

encoder = load_model(ENCODER_PATH)
model = load_model(MODEL_PATH)

# TODO: create a RESTful API using FastAPI
app = FastAPI(title="Census Income API")

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Hello from the Census Income API!"}

# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    try:
        # DO NOT MODIFY: turn the Pydantic model into a dict.
        data_dict = data.dict()
        # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
        # The data has names with hyphens and Python does not allow those as variable names.
        # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
        data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
        data = pd.DataFrame.from_dict(data)

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        data_processed, _, _, _ = process_data(
            data=data,                      # use data as data input
            categorical_features=cat_features,
            label=None,
            training=False,                 # use training = False
            encoder=encoder                 # do not need to pass lb as input
        )

        _inference = inference(model, data_processed)  # array of 0/1
        return {"result": apply_label(_inference)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
