from .nodes import YOLODetectionNode, YOLOTrackingNode
from .mask_crop_node import MaskCropExtractNode

NODE_CLASS_MAPPINGS = {
    "YOLODetectionNode": YOLODetectionNode,
    "YOLOTrackingNode": YOLOTrackingNode,
    "MaskCropExtractNode": MaskCropExtractNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLODetectionNode": "YOLO Detection & Selection",
    "YOLOTrackingNode": "YOLO Tracking & Crop",
    "MaskCropExtractNode": "Mask Crop & Extract"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']