import pytest
import numpy as np
from computer_vision.roi_manager import ROIManager
from computer_vision.occupancy_counter import MultiCameraOccupancyCounter

def test_roi_ground_contact_point_filtering():
    # Define a 500x500 box as the ROI polygon
    roi = [[100, 100], [500, 100], [500, 500], [100, 500]]
    manager = ROIManager(roi_polygon=roi)

    # Car 1: Completely inside (bbox: [200, 200, 300, 300] -> contact pt: [250, 300])
    # Car 2: Centroid inside, but tires outside below ROI (bbox: [200, 450, 300, 550] -> contact pt: [250, 550])
    # Car 3: Completely outside
    detections = [
        {"bbox": [200, 200, 300, 300], "confidence": 0.95, "camera_id": "CAM1"},
        {"bbox": [200, 450, 300, 550], "confidence": 0.90, "camera_id": "CAM1"},
        {"bbox": [20, 20, 50, 50], "confidence": 0.85, "camera_id": "CAM1"},
    ]

    filtered = manager.filter_detections(detections)
    assert len(filtered) == 1
    assert filtered[0]["confidence"] == 0.95

def test_cross_camera_deduplication():
    counter = MultiCameraOccupancyCounter(zone_id="ZONE_A", operational_capacity=100, distance_threshold_meters=2.5)

    # Simulate Car A detected by both CAM1 and CAM2 in the overlapping region
    # CAM1 sees Car A at ground (10.0m, 15.0m) with conf 0.92
    # CAM2 sees Car A at ground (10.4m, 15.3m) with conf 0.88 (distance approx 0.5m apart)
    # CAM2 sees Car B at ground (25.0m, 30.0m) with conf 0.95
    multi_cam_dets = [
        {"ground_coords": (10.0, 15.0), "confidence": 0.92, "camera_id": "CAM1"},
        {"ground_coords": (10.4, 15.3), "confidence": 0.88, "camera_id": "CAM2"},
        {"ground_coords": (25.0, 30.0), "confidence": 0.95, "camera_id": "CAM2"}
    ]

    summary = counter.compute_zone_occupancy(multi_cam_dets, timestamp="2026-11-12 10:00:00")

    # Raw was 3, but Car A was seen twice. Deduplicated should be exactly 2 vehicles!
    assert summary["raw_detection_count"] == 3
    assert summary["deduplicated_count"] == 2
    assert summary["duplicates_removed"] == 1
    assert summary["reporting_cameras"] == ["CAM1", "CAM2"] or summary["reporting_cameras"] == ["CAM2", "CAM1"]

def test_cross_camera_missing_ground_coords_resilience():
    counter = MultiCameraOccupancyCounter(zone_id="ZONE_B", operational_capacity=50)
    # Detections without ground_coords should not crash deduplication
    dets = [
        {"confidence": 0.90, "camera_id": "CAM1"},
        {"ground_coords": None, "confidence": 0.85, "camera_id": "CAM2"},
        {"ground_coords": (5.0, 10.0), "confidence": 0.95, "camera_id": "CAM1"}
    ]
    summary = counter.compute_zone_occupancy(dets, timestamp="2026-11-12 11:00:00")
    assert summary["deduplicated_count"] == 3
    assert summary["duplicates_removed"] == 0
