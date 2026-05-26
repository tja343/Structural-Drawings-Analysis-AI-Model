from pydantic import BaseModel
from typing import List
from app.schemas.engineering import EngineeringOutputSchema

class InferenceResponse(BaseModel):
    status: str
    message: str
    data: EngineeringOutputSchema

class BatchInferenceResponse(BaseModel):
    status: str
    message: str
    processed_count: int
    data: List[EngineeringOutputSchema]
