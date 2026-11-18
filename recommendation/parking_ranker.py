import yaml
import numpy as np
from typing import List, Dict, Any, Optional

from utils.config import load_model_config

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
            avail_spaces = max(0, cap - pred_count)
            occ_rate = pred_count / cap if cap > 0 else 1.0
            sat_prob = pred.get("saturation_prob", 0.0)
            cam_age = pred.get("camera_age_seconds", 30.0)

            # Retrieve walking distance to destination
            dist = z_meta.get("walking_distances_meters", {}).get(destination, 600.0)
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

            status = "AVAILABLE" if occ_rate < 0.75 else "FILLING_FAST" if occ_rate < 0.90 else "SATURATED"

            scored_zones.append({
                "zone_id": z_id,
                "name": z_meta["name"],
                "category": z_meta["category"],
                "score": round(float(total_score), 4),
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
