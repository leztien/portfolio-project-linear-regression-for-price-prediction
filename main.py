
"""
Deployment of the model with FastAPI (demo)

main.py
$ pip install "fastapi[all]"
$ uvicorn main:app --reload

http://127.0.0.1:8000/estimate?longitude=-122.5&latitude=23

Example POST request from the bash console:

curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{
    "longitude": -122, 
    "latitude": 37,
    "ocean_proximity": "NEAR BAY"
    }' \
  http://localhost:8000/estimate
"""

# imports for FastAPI
import uvicorn  # not needed if: $ uvicorn main:app --reload
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# imports for the pickled model
import joblib
from src.feature_engineering import ClusterSimilarity, make_ratio, feature_names_out, make_ratio_pipline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Load your model
MODEL_PATH = "final_model.pkl"
model = joblib.load(MODEL_PATH)


app = FastAPI()


@app.get("/")
async def root():
    return HTMLResponse("Hello World")


class Data(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float | None = None
    total_rooms: float | None = None
    total_bedrooms: float | None = None
    population: float | None = None
    households: float | None = None
    median_income: float | None = None
    ocean_proximity: str | None = None


@app.post("/estimate")
async def estimate(data: Data):
    from pandas import Series
    df = Series(data.dict()).to_frame().T
    estimate, = model.predict(df)
    return {'estimate': estimate}


# Estimate from longitude and latitude
@app.get("/estimate", response_class=HTMLResponse)
async def estimate(longitude: float, latitude: float):
    """http://127.0.0.1:8000/estimate?longitude=-122.5&latitude=23"""
    from pandas import Series
    data = Data(longitude=longitude, latitude=latitude).dict()
    df = Series(data).to_frame().T
    estimate, = model.predict(df)
    return HTMLResponse(f"Our estimate: {estimate}")



# comment this out when deploying
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# or bash: $ uvicorn main:app --reload
