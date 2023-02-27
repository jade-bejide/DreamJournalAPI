from urllib import response
from fastapi import Depends, HTTPException, FastAPI, status, Request, Header
from fastapi.responses import JSONResponse
from fastapi_utils.cbv import cbv
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.inferring_router import InferringRouter

from entrymodel import JournalEntry, Interpretation
import lda_run as dreamModel

app = FastAPI()
router = InferringRouter()

@cbv(router)
class System:
    @router.get("/")
    def root(self):
        return {"Welcome": "Dream Journal"}

    @router.post("/entry", response_model=Interpretation)
    def postEntry(self, entry: JournalEntry):
        print(entry.contents)
        return {'topic': dreamModel.lda_dreamModel(entry.contents)}

    


app.include_router(router)

origins = [
    "https://jadesolabejide.dev",
    "https://jade-bejide.github.io",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

