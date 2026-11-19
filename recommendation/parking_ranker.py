import yaml
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional

from utils.config import load_model_config


class InFlightDispatchTracker:
    """
    Tracks active in-flight driver dispatches towards parking zones with a 15-minute TTL.
    Prevents recommendation herd collapse where multiple drivers arriving simultaneously
    are all routed to the same currently-vacant zone until it abruptly overflows.
    """
    def __init__(self, ttl_seconds: int = 900):
        self.ttl_seconds = ttl_seconds
        self._dispatches: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def record_dispatch(self, zone_id: str, max_capacity: int = 150) -> int:
        now = time.time()
        with self._lock:
            if zone_id not in self._dispatches:
                self._dispatches[zone_id] = []
            active = [t for t in self._dispatches[zone_id] if t > (now - self.ttl_seconds)]
            # Edge-case safety cap: maximum active in-flight buffer is 35% of lot capacity
            if len(active) < max(5, int(max_capacity * 0.35)):
                active.append(now)
            self._dispatches[zone_id] = active
            return len(active)

    def get_in_flight_count(self, zone_id: str) -> int:
        now = time.time()
        cutoff = now - self.ttl_seconds
        with self._lock:
            if zone_id not in self._dispatches:
                return 0
            active = [t for t in self._dispatches[zone_id] if t > cutoff]
            self._dispatches[zone_id] = active
            return len(active)


class ParkingRanker:
    """
    Multi-criteria parking recommendation engine with Anti-Herding Stochastic Traffic Dispersion.
    Balances predicted availability, walking distance, camera freshness, and saturation risk.
    """
    def __init__(self, config_path: str = "configs/zones.yaml", seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.zones = {z["id"]: z for z in cfg["zones"]}
        # model_config.yaml is the single source of truth for ranking parameters
        self.weights = load_model_config()["recommendation"]
        # in-flight driver dispatches (POST /api/navigate) reduce a lot's effective availability
        self.dispatch_tracker = InFlightDispatchTracker(ttl_seconds=900)

    def rank_zones(
        self,
        destination: str,
        predictions: Dict[str, Dict[str, Any]], # zone_id -> {predicted_count, available_spaces, occupancy_rate, saturation_prob, camera_age_seconds}
        horizon_minutes: int = 15,
        apply_anti_herding: bool = True
    ) -> List[Dict[str, Any]]:
        scored_zones = []
        max_dist = 1200.0 # meters normalization anchor

        for z_id, z_meta in self.zones.items():
            pred = predictions.get(z_id, {})
            pred_count = pred.get("predicted_count", 0)
            cap = z_meta["operational_capacity"]
            in_flight = self.dispatch_tracker.get_in_flight_count(z_id)
            avail_spaces = max(0, cap - pred_count - in_flight)
            effective_count = pred_count + in_flight
            occ_rate = effective_count / cap if cap > 0 else 1.0
            sat_prob = pred.get("saturation_prob", 0.0)
            cam_age = pred.get("camera_age_seconds", 30.0)

            # Retrieve walking distance to destination (with case-insensitive / space tolerance)
            dest_distances = z_meta.get("walking_distances_meters", {})
            dest_map = {k.lower().replace(" ", "_"): v for k, v in dest_distances.items()}
            norm_dest = destination.strip().lower().replace(" ", "_")
            dist = dest_map.get(norm_dest, dest_distances.get(destination, 600.0))
            walk_time_mins = round(dist / 80.0, 1) # ~80m/min walking speed

            # Multi-criteria scoring
            avail_score = max(0.0, 1.0 - occ_rate)
            dist_score = max(0.0, 1.0 - (dist / max_dist))
            freshness_penalty = 0.3 if cam_age > self.weights["camera_stale_seconds"] else 0.0
            saturation_risk_penalty = (
                0.5 if sat_prob > self.weights["saturation_hard_exclude_threshold"] else 0.0
            )

            total_score = (
                self.weights["availability_weight"] * avail_score +
                self.weights["distance_weight"] * dist_score -
                self.weights["freshness_penalty_weight"] * freshness_penalty -
                saturation_risk_penalty
            )
            # Edge-case clamp: keep score positive for display elegance
            bounded_score = round(max(0.01, float(total_score)), 4)

            status = "AVAILABLE" if occ_rate < 0.75 else "FILLING_FAST" if occ_rate < 0.90 else "SATURATED"

            scored_zones.append({
                "zone_id": z_id,
                "name": z_meta["name"],
                "category": z_meta["category"],
                "score": bounded_score,
                "predicted_count": pred_count,
                "operational_capacity": cap,
                "available_spaces": avail_spaces,
                "occupancy_rate": round(float(occ_rate), 3),
                "saturation_probability": round(float(sat_prob), 3),
                "walking_distance_meters": dist,
                "walking_time_minutes": walk_time_mins,
                "status": status,
                "camera_age_seconds": cam_age
            })

        # Sort descending by utility score
        scored_zones.sort(key=lambda x: x["score"], reverse=True)

        # Anti-Herding Traffic Dispersion: zones whose score is within `delta` of
        # the best are near-ties. Instead of sending every driver to the same
        # lot, pick the lead zone among them with softmax(score / beta)
        # probabilities, so demand spreads across near-equivalent lots.
        if apply_anti_herding and len(scored_zones) >= 2:
            delta = self.weights["anti_herding_score_delta_threshold"]
            beta = self.weights["anti_herding_beta"]
            top = scored_zones[0]["score"]
            k = sum(1 for z in scored_zones if top - z["score"] <= delta)
            if k >= 2:
                scores = np.array([z["score"] for z in scored_zones[:k]])
                logits = (scores - scores.max()) / beta
                probs = np.exp(logits) / np.exp(logits).sum()
                lead = int(self.rng.choice(k, p=probs))
                scored_zones.insert(0, scored_zones.pop(lead))

        return scored_zones
