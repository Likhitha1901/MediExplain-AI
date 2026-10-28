from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes import router


ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "null"
]

app = FastAPI(
    title="MediExplain AI",
    description="LLM and RAG-Based Intelligent Medical Information System",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Ensure 500 exceptions still retain proper CORS headers
    to local frontend origins rather than being blocked by the browser.
    """
    origin = request.headers.get("origin")
    headers = {}
    if origin and (origin in ALLOWED_ORIGINS or origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"

    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error_type": "server_error",
                "message": "An internal server error occurred.",
                "error": str(exc)
            }
        },
        headers=headers
    )


app.include_router(router)


@app.get("/")
def home():
    return {
        "message": "Welcome to MediExplain AI",
        "status": "Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
