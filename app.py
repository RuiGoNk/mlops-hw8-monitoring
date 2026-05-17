from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
import logging
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Service")

# Метрики Prometheus
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5])
REQUEST_COUNT = Counter('http_requests_total', 'Total requests', ['method', 'endpoint', 'status'])

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return {"status": "ok", "version": "v1.0.0"}

@app.post("/predict")
async def predict(request: PredictRequest):
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
    
    # Имитация инференса
    time.sleep(0.05)  # 50ms latency
    
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    
    return {"prediction": 0, "latency_ms": latency * 1000}
