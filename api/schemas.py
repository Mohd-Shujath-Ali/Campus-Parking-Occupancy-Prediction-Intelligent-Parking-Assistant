from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    zone_id: str = Field(..., description="LPU parking zone identifier (e.g. 'ZONE_BLOCK1', 'ZONE_MAINPARK_NORTH')")
    horizon_minutes: int = Field(15, description="Minutes ahead, 5 to 60 (interpolated between the trained 15/30/60 horizons)")
    current_count: int = Field(..., description="Observed or deduplicated vehicle count at time t")
    is_exam_day: Optional[bool] = Field(None, description="Override the campus calendar; omit to use it")
    is_class_transition: Optional[bool] = Field(None, description="Override the bell schedule; omit to use it")
    timestamp: Optional[str] = Field(None, description="ISO timestamp (defaults to current time)")
    lag_counts: Optional[Dict[str, int]] = Field(None, description="Optional historical counts e.g. {'15m': 120, '30m': 110}")

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
    p_free: Optional[float] = None    # chance a free slot exists at that time (calibrated on held-out data)
    status: str
    confidence_level: str
    lag_source: str = "estimated"  # observed | partial | estimated
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
    status: str   # AVAILABLE | FILLING_FAST | SATURATED, or a warden override: CLOSED | FULL
    camera_age_seconds: float
    override_reason: Optional[str] = None
    p_free: Optional[float] = None            # chance of a free slot on arrival
    likely_full: bool = False                 # p_free below the warning threshold
    predicted_low: Optional[int] = None       # 80% range for the vehicle count on arrival
    predicted_high: Optional[int] = None
    confidence: Optional[str] = None          # high | medium | low | demo
    count_source: Optional[str] = None        # camera | simulated | profile
    forecast_basis: Optional[str] = None      # live | blended | calendar (how far ahead the arrival is)

class RecommendationResponse(BaseModel):
    destination: str                                # reference block the walking distances come from
    horizon_minutes: int
    recommended_zones: List[ZoneRankItem]
    counterfactual_tip: Optional[str] = None
    requested_block: Optional[str] = None           # block the user asked for (or their home block)
    requested_block_name: Optional[str] = None
    destination_source: Optional[str] = None        # "home_block" | "selected"
    uses_adjacent_block: bool = False               # True when ranked from an adjacent reference block
    data_mode: Optional[str] = None                 # demo | live | estimated: where the counts behind this come from
    conditions: Optional[Dict[str, Any]] = None     # bell window / exam day / holiday / events at the arrival time
    arrival_time: Optional[str] = None              # campus-local arrival time the forecast is for
    forecast_basis: Optional[str] = None            # live | blended | calendar


class IngestRequest(BaseModel):
    zone_id: str = Field(..., description="Zone the camera belongs to")
    count: int = Field(..., ge=0, le=2000, description="Deduplicated vehicle count in this camera's ROI")
    camera_id: Optional[str] = Field(None, description="Defaults to the zone's configured camera")
    timestamp: Optional[str] = Field(None, description="ISO 8601; defaults to now (IST)")
    source: str = Field("camera", description="camera | manual | simulator")


class NavigateRequest(BaseModel):
    zone_id: str = Field(..., description="Target zone identifier selected by driver")
