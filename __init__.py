from .nodes import YOLODetectionNode, YOLOTrackingNode

NODE_CLASS_MAPPINGS = {
    "YOLODetectionNode": YOLODetectionNode,
    "YOLOTrackingNode": YOLOTrackingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLODetectionNode": "YOLO Detection & Selection",
    "YOLOTrackingNode": "YOLO Tracking & Crop"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']