import math
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


EXPECTED_INPUT_COUNT = 224
FORECAST_HORIZON = 30
SPOT = 100.0
STRIKE = 100.0
RISK_FREE_RATE = 0.05
TIME_TO_MATURITY = 1.0


class PredictRequest(BaseModel):
    values: List[float] = Field(..., min_length=EXPECTED_INPUT_COUNT, max_length=EXPECTED_INPUT_COUNT)


class PricePoint(BaseModel):
    day: int
    call: float
    put: float


class PredictResponse(BaseModel):
    predicted_curve: List[float]
    price_forecast: List[PricePoint]
    put_call_prices: List[dict]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_scholes(sigma: float, s: float = SPOT, k: float = STRIKE, t: float = TIME_TO_MATURITY, r: float = RISK_FREE_RATE):
    sigma = max(float(sigma), 1e-6)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    call = s * _normal_cdf(d1) - k * math.exp(-r * t) * _normal_cdf(d2)
    put = k * math.exp(-r * t) * _normal_cdf(-d2) - s * _normal_cdf(-d1)
    return call, put


def _smooth_curve(values: List[float]) -> List[float]:
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - 2)
        end = min(len(values), idx + 3)
        window = values[start:end]
        smoothed.append(sum(window) / len(window))
    return smoothed


def _forecast_prices(predicted_curve: List[float], horizon: int = FORECAST_HORIZON) -> List[PricePoint]:
    center = len(predicted_curve) // 2
    anchor_sigma = float(predicted_curve[center])
    near_sigma = float(predicted_curve[max(center - 8, 0)])
    far_sigma = float(predicted_curve[min(center + 8, len(predicted_curve) - 1)])
    slope = (far_sigma - near_sigma) / 16.0
    trend = slope * 0.12

    points: List[PricePoint] = []
    for day in range(1, horizon + 1):
        seasonal = 0.0035 * math.sin(day / 3.0)
        sigma_t = max(anchor_sigma + trend * day + seasonal, 1e-6)
        call_price, put_price = _black_scholes(sigma_t)
        points.append(PricePoint(day=day, call=call_price, put=put_price))
    return points


app = FastAPI(title="QVolution Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predicted_curve = _smooth_curve(req.values)
    forecast = _forecast_prices(predicted_curve, horizon=FORECAST_HORIZON)

    mean_call = sum(point.call for point in forecast) / len(forecast)
    mean_put = sum(point.put for point in forecast) / len(forecast)

    return PredictResponse(
        predicted_curve=predicted_curve,
        price_forecast=forecast,
        put_call_prices=[
            {"name": "Call", "price": mean_call},
            {"name": "Put", "price": mean_put},
        ],
    )