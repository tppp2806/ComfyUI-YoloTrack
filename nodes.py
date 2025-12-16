import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
import os
import folder_paths
import sys

# Register 'yolo' folder in ComfyUI models directory
# This allows users to put their .pt files in ComfyUI/models/yolo/
if "yolo" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yolo"] = ([os.path.join(folder_paths.models_dir, "yolo")], folder_paths.supported_pt_extensions)

class YOLODetectionNode:
    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(s):
        # Get list of available models, fallback to standard ones if directory is empty
        available_models = folder_paths.get_filename_list("yolo")
        if not available_models:
            available_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (available_models, ),
                "select_criteria": (["Area Max", "Area Min", "Confidence Max", "Confidence Min", "Center Top", "Center Bottom", "Center Left", "Center Right"],),
                "select_count": ("INT", {"default": 1, "min": 1, "max": 100}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_mode": (["Merged Mask", "Individual Objects"],),
            },
            "optional": {
                "class_ids": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. 0, 2 (leave empty for all)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("preview_image", "mask", "x", "y", "width", "height", "x2", "y2")
    FUNCTION = "detect"
    CATEGORY = "YOLO"

    def detect(self, image, model_name, select_criteria, select_count, confidence_threshold, nms_threshold, output_mode, class_ids=""):
        # Load Model
        model_path = folder_paths.get_full_path("yolo", model_name)
        if model_path is None:
            # If not found in yolo folder, try using the name directly (ultralytics will download or find in current dir)
            model_path = model_name
        
        if self.model is None or self.current_model_name != model_path:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.current_model_name = model_path

        # Parse Class IDs
        target_classes = None
        if class_ids.strip():
            try:
                target_classes = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
            except ValueError:
                print("Invalid class_ids format. Using all classes.")

        # Prepare outputs
        out_images = []
        out_masks = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []
        out_x2 = []
        out_y2 = []

        # Process Batch
        for img_tensor in image:
            # Convert Tensor (B, H, W, C) to PIL Image
            # img_tensor is [H, W, C] (since we iterate)
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Inference
            results = self.model(pil_img, conf=confidence_threshold, iou=nms_threshold, verbose=False)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
                    
                    if target_classes is not None and cls not in target_classes:
                        continue
                        
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    detections.append({
                        "box": xyxy,
                        "conf": conf,
                        "area": area,
                        "center": (center_x, center_y),
                        "cls": cls
                    })

            # Sort/Select
            if not detections:
                # No detections: output empty/black
                selected = []
            else:
                if select_criteria == "Area Max":
                    detections.sort(key=lambda x: x["area"], reverse=True)
                elif select_criteria == "Area Min":
                    detections.sort(key=lambda x: x["area"], reverse=False)
                elif select_criteria == "Confidence Max":
                    detections.sort(key=lambda x: x["conf"], reverse=True)
                elif select_criteria == "Confidence Min":
                    detections.sort(key=lambda x: x["conf"], reverse=False)
                elif select_criteria == "Center Top":
                    detections.sort(key=lambda x: x["center"][1], reverse=False) # y small
                elif select_criteria == "Center Bottom":
                    detections.sort(key=lambda x: x["center"][1], reverse=True) # y large
                elif select_criteria == "Center Left":
                    detections.sort(key=lambda x: x["center"][0], reverse=False) # x small
                elif select_criteria == "Center Right":
                    detections.sort(key=lambda x: x["center"][0], reverse=True) # x large
                
                selected = detections[:select_count]

            # Generate Outputs for this image
            h_img, w_img = img_np.shape[:2]
            
            if output_mode == "Merged Mask":
                # 1. Preview Image (Original with boxes)
                preview_np = img_np.copy()
                # Draw boxes
                for det in selected:
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(preview_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                    cv2.putText(preview_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                out_images.append(torch.from_numpy(preview_np.astype(np.float32) / 255.0))

                # 2. Mask (Merged)
                mask = np.zeros((h_img, w_img), dtype=np.float32)
                union_box = [w_img, h_img, 0, 0] # x1, y1, x2, y2 (inverted init)
                has_valid_box = False

                for det in selected:
                    x1, y1, x2, y2 = map(int, det["box"])
                    # Ensure within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    mask[y1:y2, x1:x2] = 1.0
                    
                    # Update union box
                    union_box[0] = min(union_box[0], x1)
                    union_box[1] = min(union_box[1], y1)
                    union_box[2] = max(union_box[2], x2)
                    union_box[3] = max(union_box[3], y2)
                    has_valid_box = True
                
                out_masks.append(torch.from_numpy(mask))

                # 3. Box Coords (Union of all selected)
                if has_valid_box:
                    ux1, uy1, ux2, uy2 = union_box
                    out_x.append(ux1)
                    out_y.append(uy1)
                    out_w.append(ux2 - ux1)
                    out_h.append(uy2 - uy1)
                    out_x2.append(ux2)
                    out_y2.append(uy2)
                else:
                    out_x.append(0)
                    out_y.append(0)
                    out_w.append(0)
                    out_h.append(0)
                    out_x2.append(0)
                    out_y2.append(0)

            else: # Individual Objects
                # If no objects found, output one empty entry to keep flow?
                # Or skip? Skipping breaks batch alignment usually.
                # Let's output at least one empty if nothing found, or just nothing?
                # If nothing found, we probably should output a blank image/mask.
                
                if not selected:
                    # Fallback: Output original image and empty mask
                    out_images.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
                    out_masks.append(torch.zeros((h_img, w_img), dtype=np.float32))
                    out_x.append(0); out_y.append(0); out_w.append(0); out_h.append(0); out_x2.append(0); out_y2.append(0)
                else:
                    for det in selected:
                        # Preview: Image with THIS box
                        preview_np = img_np.copy()
                        x1, y1, x2, y2 = map(int, det["box"])
                        cv2.rectangle(preview_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                        cv2.putText(preview_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        out_images.append(torch.from_numpy(preview_np.astype(np.float32) / 255.0))
                        
                        # Mask: Only this object
                        mask = np.zeros((h_img, w_img), dtype=np.float32)
                        # Ensure bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)
                        mask[y1:y2, x1:x2] = 1.0
                        out_masks.append(torch.from_numpy(mask))
                        
                        # Box
                        out_x.append(x1)
                        out_y.append(y1)
                        out_w.append(x2 - x1)
                        out_h.append(y2 - y1)
                        out_x2.append(x2)
                        out_y2.append(y2)

        # Stack outputs
        if not out_images:
            # Should not happen if input batch > 0
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), 0, 0, 0, 0, 0, 0)

        result_images = torch.stack(out_images)
        result_masks = torch.stack(out_masks)
        
        # For INT outputs, ComfyUI expects list of ints or single int?
        # If we return a list, it might be interpreted as a batch of INTs if the node is set up right,
        # but standard nodes usually return single values.
        # However, custom nodes can return lists.
        # To be safe for batch processing, we usually return a list if the input was a batch.
        # But here we might have changed the batch size (Individual mode).
        # Let's return lists. ComfyUI handles lists in outputs often by auto-batching or passing list.
        
        return (result_images, result_masks, out_x, out_y, out_w, out_h, out_x2, out_y2)

class YOLOTrackingNode:
    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(s):
        available_models = folder_paths.get_filename_list("yolo")
        if not available_models:
            available_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (available_models, ),
                "select_criteria": (["Area Max", "Area Min", "Confidence Max", "Confidence Min", "Center Top", "Center Bottom", "Center Left", "Center Right"],),
                "select_count": ("INT", {"default": 1, "min": 1, "max": 100}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_ratio": (["Original", "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],),
                "crop_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "mask_type": (["Box", "Segmentation"],),
            },
            "optional": {
                "class_ids": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. 0, 2 (leave empty for all)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_image", "mask")
    FUNCTION = "track_and_crop"
    CATEGORY = "YOLO"

    def track_and_crop(self, image, model_name, select_criteria, select_count, confidence_threshold, nms_threshold, crop_ratio, crop_scale, offset_x, offset_y, smoothing, mask_type, class_ids=""):
        # Load Model
        model_path = folder_paths.get_full_path("yolo", model_name)
        if model_path is None:
            model_path = model_name
        
        if self.model is None or self.current_model_name != model_path:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.current_model_name = model_path

        # Parse Class IDs
        target_classes = None
        if class_ids.strip():
            try:
                target_classes = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
            except ValueError:
                print("Invalid class_ids format. Using all classes.")

        out_images = []
        out_masks = []
        
        current_boxes = [None] * select_count
        fixed_cw, fixed_ch = None, None

        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            h_img, w_img = img_np.shape[:2]
            
            # Inference
            results = self.model(pil_img, conf=confidence_threshold, iou=nms_threshold, verbose=False)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    if target_classes is not None and cls not in target_classes:
                        continue
                        
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    detections.append({
                        "box": xyxy,
                        "conf": conf,
                        "area": area,
                        "center": (center_x, center_y),
                        "cls": cls,
                        "index": i
                    })

            # Sort/Select
            if not detections:
                selected = []
            else:
                if select_criteria == "Area Max":
                    detections.sort(key=lambda x: x["area"], reverse=True)
                elif select_criteria == "Area Min":
                    detections.sort(key=lambda x: x["area"], reverse=False)
                elif select_criteria == "Confidence Max":
                    detections.sort(key=lambda x: x["conf"], reverse=True)
                elif select_criteria == "Confidence Min":
                    detections.sort(key=lambda x: x["conf"], reverse=False)
                elif select_criteria == "Center Top":
                    detections.sort(key=lambda x: x["center"][1], reverse=False)
                elif select_criteria == "Center Bottom":
                    detections.sort(key=lambda x: x["center"][1], reverse=True)
                elif select_criteria == "Center Left":
                    detections.sort(key=lambda x: x["center"][0], reverse=False)
                elif select_criteria == "Center Right":
                    detections.sort(key=lambda x: x["center"][0], reverse=True)
                
                selected = detections[:select_count]

            # Process each selected object
            for i in range(select_count):
                det = selected[i] if i < len(selected) else None
                
                # Determine target box
                if det is not None:
                    bx1, by1, bx2, by2 = det["box"]
                    center_x = (bx1 + bx2) / 2
                    center_y = (by1 + by2) / 2
                else:
                    if current_boxes[i] is not None:
                        center_x = (current_boxes[i][0] + current_boxes[i][2]) / 2
                        center_y = (current_boxes[i][1] + current_boxes[i][3]) / 2
                    else:
                        center_x = w_img / 2
                        center_y = h_img / 2

                # Calculate crop size (Target)
                if crop_ratio == "Original":
                    tw = w_img * crop_scale
                    th = h_img * crop_scale
                else:
                    rw, rh = map(int, crop_ratio.split(":"))
                    target_aspect = rw / rh
                    img_aspect = w_img / h_img
                    if target_aspect > img_aspect:
                        tw = w_img * crop_scale
                        th = tw / target_aspect
                    else:
                        th = h_img * crop_scale
                        tw = th * target_aspect

                # Target box coordinates
                tx1 = center_x - tw / 2
                ty1 = center_y - th / 2
                tx2 = center_x + tw / 2
                ty2 = center_y + th / 2
                target_box = np.array([tx1, ty1, tx2, ty2])

                # Smooth
                if current_boxes[i] is None:
                    current_boxes[i] = target_box
                else:
                    current_boxes[i] = current_boxes[i] * smoothing + target_box * (1.0 - smoothing)

                # Fix crop size for batch consistency
                if fixed_cw is None:
                    fixed_cw = int(tw)
                    fixed_ch = int(th)
                    if fixed_cw % 2 != 0: fixed_cw += 1
                    if fixed_ch % 2 != 0: fixed_ch += 1
                
                cw, ch = fixed_cw, fixed_ch
                
                # Recalculate box from center
                curr_cx = (current_boxes[i][0] + current_boxes[i][2]) / 2
                curr_cy = (current_boxes[i][1] + current_boxes[i][3]) / 2
                
                # Apply offset (relative to crop size)
                curr_cx += offset_x * cw
                curr_cy += offset_y * ch
                
                cx1 = int(curr_cx - cw / 2)
                cy1 = int(curr_cy - ch / 2)
                cx2 = cx1 + cw
                cy2 = cy1 + ch
                
                # Create canvas
                crop_img = np.zeros((ch, cw, 3), dtype=np.uint8)
                
                # Intersection
                ix1 = max(0, cx1)
                iy1 = max(0, cy1)
                ix2 = min(w_img, cx2)
                iy2 = min(h_img, cy2)
                
                if ix2 > ix1 and iy2 > iy1:
                    src_patch = img_np[iy1:iy2, ix1:ix2]
                    dest_x = ix1 - cx1
                    dest_y = iy1 - cy1
                    
                    dest_x = max(0, dest_x)
                    dest_y = max(0, dest_y)
                    
                    sh, sw = src_patch.shape[:2]
                    write_h = min(sh, ch - dest_y)
                    write_w = min(sw, cw - dest_x)
                    
                    crop_img[dest_y:dest_y+write_h, dest_x:dest_x+write_w] = src_patch[:write_h, :write_w]

                # Mask
                crop_mask = np.zeros((ch, cw), dtype=np.float32)
                
                if det is not None:
                    mask_found = False
                    
                    if mask_type == "Segmentation":
                        if hasattr(results[0], 'masks') and results[0].masks is not None:
                            orig_idx = det["index"]
                            if hasattr(results[0].masks, 'xy') and orig_idx < len(results[0].masks.xy):
                                polygon = results[0].masks.xy[orig_idx]
                                
                                full_mask = np.zeros((h_img, w_img), dtype=np.float32)
                                pts = polygon.astype(np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                cv2.fillPoly(full_mask, [pts], 1.0)
                                
                                if ix2 > ix1 and iy2 > iy1:
                                    src_mask_patch = full_mask[iy1:iy2, ix1:ix2]
                                    dest_x = ix1 - cx1
                                    dest_y = iy1 - cy1
                                    
                                    dest_x = max(0, dest_x)
                                    dest_y = max(0, dest_y)
                                    
                                    sh, sw = src_mask_patch.shape[:2]
                                    write_h = min(sh, ch - dest_y)
                                    write_w = min(sw, cw - dest_x)
                                    
                                    crop_mask[dest_y:dest_y+write_h, dest_x:dest_x+write_w] = src_mask_patch[:write_h, :write_w]
                                    mask_found = True

                    if not mask_found: # Box mask or fallback
                        bx1, by1, bx2, by2 = det["box"]
                        mx1 = int(bx1 - cx1)
                        my1 = int(by1 - cy1)
                        mx2 = int(bx2 - cx1)
                        my2 = int(by2 - cy1)
                        
                        mx1 = max(0, mx1); my1 = max(0, my1)
                        mx2 = min(cw, mx2); my2 = min(ch, my2)
                        
                        if mx2 > mx1 and my2 > my1:
                            crop_mask[my1:my2, mx1:mx2] = 1.0

                out_images.append(torch.from_numpy(crop_img.astype(np.float32) / 255.0))
                out_masks.append(torch.from_numpy(crop_mask))

        if not out_images:
             return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)))

        return (torch.stack(out_images), torch.stack(out_masks))
