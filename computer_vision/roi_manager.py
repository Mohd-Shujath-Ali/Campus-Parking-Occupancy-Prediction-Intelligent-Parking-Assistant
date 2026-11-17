import numpy as np
import cv2
from typing import List, Tuple, Dict, Any

class ROIManager:
    """
    Manages camera-specific Region of Interest (ROI) polygons and ground-plane projections.
    """
    def __init__(self, roi_polygon: List[List[int]], homography_matrix: np.ndarray = None):
        """
        Args:
            roi_polygon: List of [x, y] pixel coordinates forming the convex/concave boundary.
            homography_matrix: Optional 3x3 homography matrix mapping image coordinates (u, v) 
                               to ground-plane coordinates (X, Y in meters).
        """
        self.polygon = np.array(roi_polygon, dtype=np.int32)
        self.homography = homography_matrix

    def get_ground_contact_point(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculates the bottom-center point of the bounding box where vehicle wheels touch the ground.
        Using bottom-center avoids false positives caused by tall vehicles tilting across ROI edges.
        bbox format: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox[:4]
        contact_x = (x1 + x2) / 2.0
        contact_y = max(y1, y2)
        return contact_x, contact_y

    def is_inside_roi(self, contact_point: Tuple[float, float]) -> bool:
        """
        Ray-casting point-in-polygon test via OpenCV pointPolygonTest.
        Returns True if the contact point is strictly inside or on the boundary.
        """
        result = cv2.pointPolygonTest(self.polygon, contact_point, False)
        return result >= 0

    def project_to_ground(self, contact_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Projects image plane (u, v) coordinates to top-down metric ground plane (X, Y)
        using the planar homography matrix H.
        """
        if self.homography is None:
            # Fallback to normalized image coordinates if H is not calibrated
            return contact_point

        u, v = contact_point
        pt_homogeneous = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
        projected = np.matmul(self.homography, pt_homogeneous)
        
        # Normalize by scale factor W'
        w = projected[2, 0]
        if abs(w) < 1e-6:
            return 0.0, 0.0
        X = projected[0, 0] / w
        Y = projected[1, 0] / w
        return float(X), float(Y)

    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters raw YOLO detections, retaining only those whose ground contact point
        falls within this camera's ROI. Also attaches ground-plane metric coordinates.
        """
        valid_detections = []
        for det in detections:
            bbox = det["bbox"]
            contact_pt = self.get_ground_contact_point(bbox)
            if self.is_inside_roi(contact_pt):
                ground_x, ground_y = self.project_to_ground(contact_pt)
                valid_detections.append({
                    **det,
                    "contact_pt": contact_pt,
                    "ground_coords": (ground_x, ground_y)
                })
        return valid_detections
