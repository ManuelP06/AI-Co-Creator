from fastapi import FastAPI
from app.database import Base, engine
from app.routers import upload_router
from app.routers import shots_router
from app.routers import analysis_router
from app.routers import transcription_router 
from app.routers import editor_router
from app.routers import renderer_router
Base.metadata.create_all(bind=engine)

app = FastAPI(title="winview.ai")

app.include_router(upload_router.router)
app.include_router(shots_router.router)
app.include_router(transcription_router.router) 
app.include_router(analysis_router.router) 
app.include_router(editor_router.router)
app.include_router(renderer_router.router)
# uvicorn app.main:app --reload

