import numpy as np
from typing import List, Dict, Any, Tuple

class MultiCameraOccupancyCounter:
    """
    Fuses observations across multiple cameras covering a single parking zone,
    performs spatial deduplication in the ground coordinate system, and filters noise.
    """
    def __init__(self, zone_id: str, operational_capacity: int, distance_threshold_meters: float = 2.5):
        """
        Args:
            zone_id: Identifier for the parking ground (e.g. 'ZONE_A').
            operational_capacity: Agreed practical capacity.
            distance_threshold_meters: Max euclidean distance on the ground plane between two 
                                       detections from different cameras to consider them the same car.
        """
        self.zone_id = zone_id
        self.operational_capacity = operational_capacity
        self.distance_threshold = distance_threshold_meters

    def deduplicate_cross_camera(self, multi_cam_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs Global Ground-Plane Non-Maximum Suppression (NMS) across pooled camera detections.
        
        Args:
            multi_cam_detections: List of detections from all cameras covering this lot.
                                  Each dict contains: 'ground_coords', 'confidence', 'camera_id', 'bbox'.
        Returns:
            List of unique, deduplicated vehicle detections.
        """
        if not multi_cam_detections:
            return []

        # Sort detections by confidence score descending
        sorted_dets = sorted(multi_cam_detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        
        retained = []
        suppressed_indices = set()

        for i in range(len(sorted_dets)):
            if i in suppressed_indices:
                continue

            current = sorted_dets[i]
            retained.append(current)

            coords1 = current.get("ground_coords")
            if coords1 is None or len(coords1) < 2:
                continue
            gx1, gy1 = coords1[0], coords1[1]

            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed_indices:
                    continue

                candidate = sorted_dets[j]
                coords2 = candidate.get("ground_coords")
                if coords2 is None or len(coords2) < 2:
                    continue
                gx2, gy2 = coords2[0], coords2[1]

                # Calculate ground Euclidean distance in meters
                dist = np.sqrt((gx1 - gx2) ** 2 + (gy1 - gy2) ** 2)

                # If within the vehicle footprint radius, suppress candidate as a duplicate
                if dist < self.distance_threshold:
                    suppressed_indices.add(j)

        return retained

    def compute_zone_occupancy(
        self, 
        multi_cam_detections: List[Dict[str, Any]], 
        timestamp: str
    ) -> Dict[str, Any]:
        """
        Calculates the true deduplicated vehicle count, occupancy rate, and saturation alerts.
        """
        deduped = self.deduplicate_cross_camera(multi_cam_detections)
        vehicle_count = len(deduped)
        occupancy_rate = (vehicle_count / self.operational_capacity) if self.operational_capacity > 0 else 0.0
        
        is_saturated = occupancy_rate >= 0.90
        is_overflowing = vehicle_count > self.operational_capacity
        
        active_cameras = list(set(d.get("camera_id", "unknown") for d in multi_cam_detections))

        return {
            "timestamp": timestamp,
            "zone_id": self.zone_id,
            "raw_detection_count": len(multi_cam_detections),
            "deduplicated_count": vehicle_count,
            "operational_capacity": self.operational_capacity,
            "occupancy_rate": round(occupancy_rate, 4),
            "is_saturated": is_saturated,
            "is_overflowing": is_overflowing,
            "reporting_cameras": active_cameras,
            "duplicates_removed": len(multi_cam_detections) - vehicle_count
        }
