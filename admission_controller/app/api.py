from typing import List

from fastapi import FastAPI, File, UploadFile, status
from fastapi_health import health

from .admission_control_state import AdmissionControlState
from .constants import *
from .model import *

# Class manages the current admission controller state
ac_state = AdmissionControlState()

app = FastAPI()

app.add_api_route("/health", health([ac_state.health]))


@app.on_event("startup")
async def startup_event():
    ac_state.start_servers()


@app.on_event("shutdown")
def shutdown_event():
    ac_state.stop_servers()


@app.post("/upload/{model_type}", status_code=status.HTTP_201_CREATED)
async def register_model(
    model_name: str,
    model_type: ModelType,
    files: List[UploadFile] = File(...),
    version: str = "1",
):
    """
    User uploads a model and it's corresponding triton model configuration.
    Optionally user can also upload the offline profile data generated by model analyzer.

    Model is created in model repo path specified in model analyzer config.
    """

    return ac_state.register_model(model_name, model_type, files, version)


@app.post("/load", status_code=status.HTTP_201_CREATED)
async def load_model(load_request: ModelLoadRequest):
    return ac_state.load_model(load_request)


@app.post("/unload/{model_name}", status_code=status.HTTP_200_OK)
async def unload_model(model_name: str):
    ac_state.unload_model(model_name)
    return f"{model_name} unloaded"


@app.get("/stats/system", status_code=status.HTTP_200_OK)
async def get_system_stats():
    return ac_state.system_metrics()
