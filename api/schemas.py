from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    zone_id: str = Field(..., description="LPU parking zone identifier (e.g. 'ZONE_BLOCK1', 'ZONE_MAINPARK_NORTH')")
    horizon_minutes: int = Field(15, description="Prediction horizon: 15, 30, or 60 minutes")
    current_count: int = Field(..., description="Observed or deduplicated vehicle count at time t")
    is_exam_day: bool = Field(False, description="Flag indicating whether exam session regime is active")
    is_class_transition: bool = Field(False, description="Flag indicating rush surge window around lecture bells")
    timestamp: Optional[str] = Field(None, description="ISO timestamp (defaults to current time)")
    lag_counts: Optional[Dict[str, int]] = Field(None, description="Optional historical counts e.g. {'15m': 120, '30m': 110}")

class PredictionResponse(BaseModel):
    zone_id: str
    zone_name: str
    horizon_minutes: int
    predicted_vehicle_count: int
    operational_capacity: int
    available_spaces: int
    occupancy_rate: float
    saturation_probability: float
    status: str
    confidence_level: str
    stream_frozen: Optional[bool] = Field(False, description="Flag indicating invariant video stream frames")

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

class NavigateRequest(BaseModel):
    zone_id: str = Field(..., description="Target zone identifier selected by driver")
