from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from activeLearning import router

# Initiating the router
app = FastAPI()

# App configs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

app.include_router(router, prefix="/api")