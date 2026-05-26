from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class ParsedSemanticSchema(BaseModel):
    bar_type: Optional[str] = None
    diameter: Optional[int] = None
    spacing: Optional[int] = None
    quantity: Optional[int] = None
    layer: Optional[str] = None

class AnnotationSchema(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    parsed: ParsedSemanticSchema
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    association_confidence: float = Field(ge=0.0, le=1.0)

class StructuralElementSchema(BaseModel):
    type: str # e.g., "rebar_region", "beam"
    bbox: BoundingBoxSchema
    detection_confidence: float = Field(ge=0.0, le=1.0)
    annotations: List[AnnotationSchema] = []

class EngineeringOutputSchema(BaseModel):
    drawing_id: str
    overall_confidence: float = Field(ge=0.0, le=1.0)
    elements: List[StructuralElementSchema]
