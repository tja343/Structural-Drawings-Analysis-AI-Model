from pydantic import BaseModel, Field
from typing import Dict, List, Optional

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
    direction: Optional[int] = None

class AnnotationSchema(BaseModel):
    bbox: BoundingBoxSchema
    text: str
    normalized_text: Optional[str] = None
    parsed: ParsedSemanticSchema
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    association_confidence: float = Field(ge=0.0, le=1.0)

class StructuralElementSchema(BaseModel):
    id: Optional[str] = None
    type: str # e.g., "rebar_region", "beam"
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    bbox: BoundingBoxSchema
    detection_confidence: float = Field(ge=0.0, le=1.0)
    annotations: List[AnnotationSchema] = Field(default_factory=list)

class EngineeringOutputSchema(BaseModel):
    schema_version: str = "1.1"
    drawing_id: str
    overall_confidence: float = Field(ge=0.0, le=1.0)
    summary: Dict[str, int] = Field(default_factory=dict)
    elements: List[StructuralElementSchema]
