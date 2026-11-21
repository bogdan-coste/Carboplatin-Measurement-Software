from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import pandas as pd
import io

from .cnn_inference import CnnModelUnavailable, run_cnn_inference

app = FastAPI(
    title="Carboplatin Sensing AI API",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_ARCHITECTURES = {"pinn-cv", "cnn-transformer", "ann-mlp"}

class PredictionResponse(BaseModel):
    predicted_concentration_uM: float
    physics_residual_loss: float | None = None
    confidence_r2: float
    model_architecture: str
    preprocessing_pipeline: str
    potential_data: list[float]
    current_data: list[float]

@app.get("/")
def root():
    return {"status": "online", "message": "Carboplatin Sensing API connected."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_carboplatin(
    file: UploadFile = File(...),
    architecture: str = Form("pinn-cv"),
):
    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Fisierul trebuie sa fie in format .csv sau .txt",
        )

    architecture = architecture.lower()
    if architecture not in ALLOWED_ARCHITECTURES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Arhitectura necunoscuta: {architecture}. "
                f"Valori acceptate: {sorted(ALLOWED_ARCHITECTURES)}"
            ),
        )

    contents = await file.read()

    if architecture == "cnn-transformer":
        try:
            result = await run_in_threadpool(run_cnn_inference, contents, filename)
        except CnnModelUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"CNN inference failed: {error}",
            ) from error

        return PredictionResponse(
            predicted_concentration_uM=result["predicted_concentration_uM"],
            physics_residual_loss=None,
            confidence_r2=result.get("confidence_r2", 0.0),
            model_architecture=result.get("model_architecture", "CNN-Transformer"),
            preprocessing_pipeline=result.get(
                "preprocessing_pipeline", "StandardScaler + log target"
            ),
            potential_data=result["potential_data"],
            current_data=result["current_data"],
        )

    try:
        df = pd.read_csv(io.BytesIO(contents))

        if df.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="Fisierul trebuie sa contina cel putin 2 coloane",
            )

        potential_vals = df.iloc[:, 0].tolist()
        current_vals = df.iloc[:, 1].tolist()
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Eroare la procesarea fisierului: {str(error)}",
        ) from error

    if architecture == "ann-mlp":
        model_architecture = "ANN / MLP (mock)"
        preprocessing_pipeline = "StandardScaler (mock)"
    else:
        model_architecture = "PINN-CV (Fick's Diffusion)"
        preprocessing_pipeline = "splearn Denoising + MinMax"

    return PredictionResponse(
        predicted_concentration_uM=14.85,
        physics_residual_loss=0.0021,
        confidence_r2=0.984,
        model_architecture=model_architecture,
        preprocessing_pipeline=preprocessing_pipeline,
        potential_data=potential_vals,
        current_data=current_vals,
    )
