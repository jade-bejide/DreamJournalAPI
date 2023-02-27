from pydantic import BaseModel

class JournalEntry(BaseModel):
    contents: str

class Interpretation(BaseModel):
    topic: str