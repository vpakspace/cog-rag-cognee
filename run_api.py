"""Launch the FastAPI server."""
import uvicorn
from dotenv import load_dotenv

from cog_rag_cognee.config import get_settings

if __name__ == "__main__":
    load_dotenv()
    settings = get_settings()
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
