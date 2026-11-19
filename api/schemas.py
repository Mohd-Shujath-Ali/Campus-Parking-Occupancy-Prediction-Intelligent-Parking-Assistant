from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    zone_id: str = Field(..., description="LPU parking zone identifier (e.g. 'ZONE_BLOCK1', 'ZONE_MAINPARK_NORTH')")
    horizon_minutes: int = Field(15, description="Prediction horizon: 15, 30, or 60 minutes")
    current_count: int = Field(..., description="Observed or deduplicated vehicle count at time t")
    is_exam_day: bool = Field(False, description="Flag indicating whether exam session regime is active")
    is_class_transition: bool = Field(False, description="Flag indicating rush surge window around lecture bells")
    timestamp: Optional[str] = Field(None, description="ISO timestamp (defaults to current time)")

class PredictionResponse(BaseModel):
    zone_id: str
    zone_name: str
    horizon_minutes: int
    predicted_vehicle_count: int
    predicted_low: Optional[int] = None    # 80% prediction interval
    predicted_high: Optional[int] = None
    operational_capacity: int
    available_spaces: int
    occupancy_rate: float
    saturation_probability: float
    status: str
    confidence_level: str
    lag_source: str = "estimated"  # observed | partial | estimated

class RecommendationRequest(BaseModel):
    destination: str = Field(..., description="Campus destination block (e.g. 'Block_26_CSE', 'Block_1', 'Food_Square')")
    horizon_minutes: int = Field(15, description="Arrival time offset in minutes")

class ZoneRankItem(BaseModel):
    zone_id: str
    name: str
    category: str
    score: float
    predicted_count: int
    operational_capacity: int
    available_spaces: int
    occupancy_rate: float
    saturation_probability: float
    walking_distance_meters: float
    walking_time_minutes: float
    status: str
    camera_age_seconds: float

class RecommendationResponse(BaseModel):
    destination: str
    horizon_minutes: int
    recommended_zones: List[ZoneRankItem]
    counterfactual_tip: Optional[str] = None


class IngestRequest(BaseModel):
    zone_id: str = Field(..., description="Zone the camera belongs to")
    count: int = Field(..., ge=0, le=2000, description="Deduplicated vehicle count in this camera's ROI")
    camera_id: Optional[str] = Field(None, description="Defaults to the zone's configured camera")
    timestamp: Optional[str] = Field(None, description="ISO 8601; defaults to now (IST)")
    source: str = Field("camera", description="camera | manual | simulator")
