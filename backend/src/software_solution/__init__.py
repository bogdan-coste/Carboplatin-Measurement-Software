"""Software solution for carboplatin detection from electrochemical data."""

__all__ = ["main"]


def main() -> None:
    """Run the FastAPI application with uvicorn."""
    import uvicorn

    from .main import app

    uvicorn.run(app, host="127.0.0.1", port=8000)
