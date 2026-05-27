from fastapi import FastAPI
from contextlib import asynccontextmanager
from mangum import Mangum
from api.database import init_db, close_db
from api.routers import patients, schedules, dispensing, sync, medications, kvs_live, risk_notification, notifications, auth_routes, slots_routes
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    close_db()

app = FastAPI(
    title="Drug Dispenser API",
    version="0.1.0",
    lifespan=lifespan,
    redirect_slashes=False,
    root_path="/default"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/patients", tags=["Patients"])
app.include_router(schedules.router, prefix="/schedules", tags=["Schedules"])
app.include_router(dispensing.router, prefix="/dispensing-logs", tags=["Dispensing"])
app.include_router(sync.router, prefix="/sync", tags=["Sync"])
app.include_router(medications.router, prefix="/medications", tags=["Medications"])
app.include_router(kvs_live.router, prefix="/kvs-live", tags=["KVS Live"])
app.include_router(risk_notification.router, prefix="/risk", tags=["Risk"])
app.include_router(notifications.router, prefix="/notifications", tags=["Notifications"])
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
app.include_router(slots_routes.router, prefix="/slots", tags=["Slots"])

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

def handler(event, context):
    print(f"[DEBUG] event: {event}")
    mangum_handler = Mangum(app, lifespan="off", api_gateway_base_path="/default")
    return mangum_handler(event, context)