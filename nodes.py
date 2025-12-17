import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import os
import folder_paths
from collections import deque

# Register 'yolo' folder in ComfyUI models directory
if "yolo" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yolo"] = ([os.path.join(folder_paths.models_dir, "yolo")], folder_paths.supported_pt_extensions)


class KalmanBoxTracker:
    """基于卡尔曼滤波的边界框追踪器"""
    
    count = 0
    
    def __init__(self, bbox):
        self.kf = self._create_kalman_filter()
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.kf['x'] = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
    def _create_kalman_filter(self):
        dt = 1.0
        
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt
        
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        
        # 更保守的参数设置
        Q = np.eye(8, dtype=np.float32)
        Q[0:4, 0:4] *= 1.0
        Q[4:8, 4:8] *= 0.01
        
        R = np.eye(4, dtype=np.float32) * 10.0
        
        P = np.eye(8, dtype=np.float32)
        P[0:4, 0:4] *= 10  # 降低位置不确定性
        P[4:8, 4:8] *= 100  # 降低速度不确定性
        
        return {
            'x': np.zeros(8, dtype=np.float32),
            'P': P, 'F': F, 'H': H, 'Q': Q, 'R': R,
        }
    
    def predict(self):
        kf = self.kf
        
        if kf['x'][2] + kf['x'][6] <= 0:
            kf['x'][6] = 0
        if kf['x'][3] + kf['x'][7] <= 0:
            kf['x'][7] = 0
            
        kf['x'] = kf['F'] @ kf['x']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self.get_state())
        return self.history[-1]
    
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float32)
        
        kf = self.kf
        
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)
        
        y = z - kf['H'] @ kf['x']
        kf['x'] = kf['x'] + K @ y
        
        I = np.eye(8, dtype=np.float32)
        kf['P'] = (I - K @ kf['H']) @ kf['P']
        
    def get_state(self):
        cx, cy, w, h = self.kf['x'][:4]
        w = max(w, 1)
        h = max(h, 1)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class AdvancedBBoxStabilizer:
    """高级边界框稳定器"""
    
    def __init__(self, window_size=10, max_age=30):
        self.window_size = window_size
        self.max_age = max_age
        
        self.history = deque(maxlen=window_size)
        self.kalman_tracker = None
        
        self.frames_since_detection = 0
        self.is_initialized = False
        self.last_valid_bbox = None
        self.outlier_threshold = 3.0
        self.frame_count = 0
        self.warmup_frames = 5  # 预热帧数
        
    def _is_outlier(self, bbox):
        if len(self.history) < 3:
            return False
            
        centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in self.history])
        new_center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        
        mean_center = np.mean(centers, axis=0)
        std_center = np.std(centers, axis=0) + 1e-6
        
        deviation = np.abs(new_center - mean_center) / std_center
        return np.any(deviation > self.outlier_threshold)
    
    def _predict_bbox(self):
        """预测丢失目标的位置"""
        # 优先使用简单的速度预测（更稳定）
        if len(self.history) >= 2:
            velocity = np.array(self.history[-1]) - np.array(self.history[-2])
            return np.array(self.history[-1]) + velocity
        elif self.kalman_tracker is not None and self.frame_count > self.warmup_frames:
            return self.kalman_tracker.predict()
        elif self.last_valid_bbox is not None:
            return self.last_valid_bbox
        return None
    
    def get_average_center(self, num_frames=None):
        """获取历史帧的平均中心位置"""
        if len(self.history) == 0:
            return None
        
        if num_frames is None or num_frames <= 0:
            num_frames = 1
        
        recent_history = list(self.history)[-num_frames:]
        centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in recent_history])
        return np.mean(centers, axis=0)
    
    def smooth_ema(self, current, alpha=0.3):
        """指数移动平均"""
        if not self.history:
            return current
        prev = np.array(self.history[-1])
        return alpha * current + (1 - alpha) * prev
    
    def smooth_kalman(self, current):
        """卡尔曼滤波平滑"""
        if self.kalman_tracker is None:
            self.kalman_tracker = KalmanBoxTracker(current)
            return current  # 首帧直接返回
        
        # 预热期间：只更新状态，返回混合结果
        if self.frame_count <= self.warmup_frames:
            self.kalman_tracker.predict()
            self.kalman_tracker.update(current)
            kalman_state = self.kalman_tracker.get_state()
            # 预热期间逐渐增加卡尔曼权重
            weight = self.frame_count / self.warmup_frames
            return weight * kalman_state + (1 - weight) * current
        
        self.kalman_tracker.predict()
        self.kalman_tracker.update(current)
        return self.kalman_tracker.get_state()
    
    def smooth_adaptive(self, current, base_alpha=0.3):
        """自适应平滑"""
        if not self.history:
            return current
            
        prev = np.array(self.history[-1])
        
        prev_center = np.array([(prev[0]+prev[2])/2, (prev[1]+prev[3])/2])
        curr_center = np.array([(current[0]+current[2])/2, (current[1]+current[3])/2])
        movement = np.linalg.norm(curr_center - prev_center)
        
        if len(self.history) >= 2:
            velocities = []
            for i in range(1, len(self.history)):
                h1, h2 = self.history[i-1], self.history[i]
                c1 = np.array([(h1[0]+h1[2])/2, (h1[1]+h1[3])/2])
                c2 = np.array([(h2[0]+h2[2])/2, (h2[1]+h2[3])/2])
                velocities.append(np.linalg.norm(c2 - c1))
            avg_velocity = np.mean(velocities) if velocities else 1.0
        else:
            avg_velocity = max(movement, 1.0)  # 避免除零
        
        velocity_ratio = movement / (avg_velocity + 1e-6)
        
        if velocity_ratio < 0.5:
            alpha = base_alpha * 0.3
        elif velocity_ratio < 1.5:
            alpha = base_alpha
        elif velocity_ratio < 3.0:
            alpha = min(base_alpha * 2, 0.8)
        else:
            alpha = 0.5
        
        return alpha * current + (1 - alpha) * prev
    
    def smooth_hybrid(self, current, kalman_weight=0.6):
        """混合平滑"""
        # 预热期间降低卡尔曼权重
        if self.frame_count <= self.warmup_frames:
            actual_weight = kalman_weight * (self.frame_count / self.warmup_frames)
        else:
            actual_weight = kalman_weight
            
        kalman_result = self.smooth_kalman(current.copy())
        adaptive_result = self.smooth_adaptive(current.copy())
        return actual_weight * kalman_result + (1 - actual_weight) * adaptive_result
    
    def update(self, bbox, method='kalman', strength=0.7):
        """更新稳定器"""
        # 目标丢失处理
        if bbox is None:
            self.frames_since_detection += 1
            
            if self.frames_since_detection <= self.max_age:
                predicted = self._predict_bbox()
                if predicted is not None:
                    return predicted
            
            return self.last_valid_bbox
        
        bbox = np.array(bbox, dtype=np.float32)
        
        # ===== 首帧：直接返回检测结果，不做任何平滑 =====
        if not self.is_initialized:
            self.is_initialized = True
            self.last_valid_bbox = bbox.copy()
            self.history.append(bbox.copy())
            self.frame_count = 1
            
            # 创建Kalman tracker（仅初始化，不进行预测）
            if method in ['kalman', 'hybrid']:
                self.kalman_tracker = KalmanBoxTracker(bbox)
            
            return bbox  # 首帧直接返回原始检测值
        
        self.frame_count += 1
        
        # 异常值处理
        if self._is_outlier(bbox) and len(self.history) >= 3:
            predicted = self._predict_bbox()
            if predicted is not None:
                bbox = 0.3 * bbox + 0.7 * predicted
        
        # 计算平滑强度（预热期间逐渐增加）
        if self.frame_count <= self.warmup_frames:
            effective_strength = strength * (self.frame_count / self.warmup_frames)
        else:
            effective_strength = strength
        
        alpha = 1.0 - effective_strength
        
        # 应用平滑方法
        if method == 'ema':
            smoothed = self.smooth_ema(bbox, alpha)
        elif method == 'kalman':
            smoothed = self.smooth_kalman(bbox)
        elif method == 'adaptive':
            smoothed = self.smooth_adaptive(bbox, base_alpha=alpha)
        elif method == 'hybrid':
            smoothed = self.smooth_hybrid(bbox, kalman_weight=0.5 + 0.3 * strength)
        else:
            smoothed = bbox
        
        self.history.append(smoothed.copy())
        self.last_valid_bbox = smoothed.copy()
        self.frames_since_detection = 0
        
        return smoothed
    
    def reset(self):
        """重置稳定器状态"""
        self.history.clear()
        self.kalman_tracker = None
        self.frames_since_detection = 0
        self.is_initialized = False
        self.last_valid_bbox = None
        self.frame_count = 0


class YOLODetectionNode:
    """YOLO检测节点"""
    
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
                "select_criteria": (["Area Max", "Area Min", "Confidence Max", "Confidence Min", 
                                     "Center Top", "Center Bottom", "Center Left", "Center Right"],),
                "select_count": ("INT", {"default": 1, "min": 1, "max": 100}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_mode": (["Merged Mask", "Individual Objects"],),
            },
            "optional": {
                "class_ids": ("STRING", {"default": "", "multiline": False, 
                                         "placeholder": "e.g. 0, 2 (leave empty for all)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "preview_with_boxes", "mask", "x", "y", "width", "height", "x2", "y2")
    FUNCTION = "detect"
    CATEGORY = "YOLO"

    def detect(self, image, model_name, select_criteria, select_count, confidence_threshold, 
               nms_threshold, output_mode, class_ids=""):
        model_path = folder_paths.get_full_path("yolo", model_name)
        if model_path is None:
            model_path = model_name
        
        if self.model is None or self.current_model_name != model_path:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.current_model_name = model_path

        target_classes = None
        if class_ids.strip():
            try:
                target_classes = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
            except ValueError:
                print("Invalid class_ids format. Using all classes.")

        out_images = []
        out_preview_images = []
        out_masks = []
        out_x, out_y, out_w, out_h, out_x2, out_y2 = [], [], [], [], [], []

        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            h_img, w_img = img_np.shape[:2]
            
            results = self.model(pil_img, conf=confidence_threshold, iou=nms_threshold, verbose=False)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    if target_classes is not None and cls not in target_classes:
                        continue
                        
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    
                    detections.append({
                        "box": xyxy, "conf": conf, "area": w * h,
                        "center": ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2),
                        "cls": cls
                    })

            # 排序选择
            if detections:
                sort_key = {
                    "Area Max": (lambda x: x["area"], True),
                    "Area Min": (lambda x: x["area"], False),
                    "Confidence Max": (lambda x: x["conf"], True),
                    "Confidence Min": (lambda x: x["conf"], False),
                    "Center Top": (lambda x: x["center"][1], False),
                    "Center Bottom": (lambda x: x["center"][1], True),
                    "Center Left": (lambda x: x["center"][0], False),
                    "Center Right": (lambda x: x["center"][0], True),
                }
                key_func, reverse = sort_key[select_criteria]
                detections.sort(key=key_func, reverse=reverse)
                selected = detections[:select_count]
            else:
                selected = []

            if output_mode == "Merged Mask":
                out_images.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
                
                # Preview - 始终显示检测框
                preview_np = img_np.copy()
                for det in selected:
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(preview_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                    cv2.putText(preview_np, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out_preview_images.append(torch.from_numpy(preview_np.astype(np.float32) / 255.0))

                # Merged Mask
                mask = np.zeros((h_img, w_img), dtype=np.float32)
                union_box = [w_img, h_img, 0, 0]
                has_valid_box = False

                for det in selected:
                    x1, y1, x2, y2 = map(int, det["box"])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    mask[y1:y2, x1:x2] = 1.0
                    union_box = [min(union_box[0], x1), min(union_box[1], y1),
                                 max(union_box[2], x2), max(union_box[3], y2)]
                    has_valid_box = True
                
                out_masks.append(torch.from_numpy(mask))

                if has_valid_box:
                    out_x.append(union_box[0])
                    out_y.append(union_box[1])
                    out_w.append(union_box[2] - union_box[0])
                    out_h.append(union_box[3] - union_box[1])
                    out_x2.append(union_box[2])
                    out_y2.append(union_box[3])
                else:
                    out_x.append(0); out_y.append(0); out_w.append(0)
                    out_h.append(0); out_x2.append(0); out_y2.append(0)

            else:  # Individual Objects
                if not selected:
                    out_images.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
                    out_preview_images.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
                    out_masks.append(torch.zeros((h_img, w_img), dtype=torch.float32))
                    out_x.append(0); out_y.append(0); out_w.append(0)
                    out_h.append(0); out_x2.append(0); out_y2.append(0)
                else:
                    for det in selected:
                        out_images.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
                        
                        preview_np = img_np.copy()
                        x1, y1, x2, y2 = map(int, det["box"])
                        cv2.rectangle(preview_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                        cv2.putText(preview_np, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        out_preview_images.append(torch.from_numpy(preview_np.astype(np.float32) / 255.0))
                        
                        mask = np.zeros((h_img, w_img), dtype=np.float32)
                        x1, y1, x2, y2 = map(int, det["box"])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)
                        mask[y1:y2, x1:x2] = 1.0
                        out_masks.append(torch.from_numpy(mask))
                        
                        out_x.append(x1); out_y.append(y1)
                        out_w.append(x2 - x1); out_h.append(y2 - y1)
                        out_x2.append(x2); out_y2.append(y2)

        if not out_images:
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512, 3)), 
                    torch.zeros((1, 512, 512)), 0, 0, 0, 0, 0, 0)

        return (torch.stack(out_images), torch.stack(out_preview_images), torch.stack(out_masks),
                out_x, out_y, out_w, out_h, out_x2, out_y2)


class YOLOTrackingNode:
    """YOLO追踪裁剪节点 - 支持稳定追踪"""
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.stabilizers = None  # 改为None，每次新批次时重新创建
        self.history_centers = None

    @classmethod
    def INPUT_TYPES(s):
        available_models = folder_paths.get_filename_list("yolo")
        if not available_models:
            available_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (available_models, ),
                "select_criteria": (["Area Max", "Area Min", "Confidence Max", "Confidence Min", 
                                     "Center Top", "Center Bottom", "Center Left", "Center Right",
                                     "Nearest to History", "Farthest from History"],),
                "select_count": ("INT", {"default": 1, "min": 1, "max": 100}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_ratio": (["Original", "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],),
                "crop_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "stabilization_method": (["None", "EMA", "Kalman", "Adaptive", "Hybrid"],),
                "stabilization_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "prediction_frames": ("INT", {"default": 10, "min": 0, "max": 60, "step": 1}),
                "mask_type": (["Box", "Segmentation"],),
            },
            "optional": {
                "class_ids": ("STRING", {"default": "", "multiline": False, 
                                         "placeholder": "e.g. 0, 2 (leave empty for all)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_image", "mask")
    FUNCTION = "track_and_crop"
    CATEGORY = "YOLO"

    def track_and_crop(self, image, model_name, select_criteria, select_count, 
                       confidence_threshold, nms_threshold, crop_ratio, crop_scale, 
                       offset_x, offset_y, stabilization_method, stabilization_strength,
                       prediction_frames, mask_type, class_ids=""):
        
        # Load Model
        model_path = folder_paths.get_full_path("yolo", model_name)
        if model_path is None:
            model_path = model_name
        
        if self.model is None or self.current_model_name != model_path:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.current_model_name = model_path

        # ===== 关键修复：每次执行时重新创建稳定器 =====
        # 这确保每个新视频批次从干净状态开始
        self.stabilizers = [
            AdvancedBBoxStabilizer(window_size=10, max_age=prediction_frames)
            for _ in range(select_count)
        ]
        self.history_centers = deque(maxlen=max(60, prediction_frames))

        # Parse Class IDs
        target_classes = None
        if class_ids.strip():
            try:
                target_classes = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
            except ValueError:
                print("Invalid class_ids format. Using all classes.")

        out_images = []
        out_masks = []
        
        fixed_cw, fixed_ch = None, None

        for frame_idx, img_tensor in enumerate(image):
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
                    center = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                    
                    detections.append({
                        "box": xyxy, "conf": conf, "area": w * h,
                        "center": center, "cls": cls, "index": i
                    })

            # 计算历史平均中心（用于Nearest/Farthest）
            history_avg_center = None
            if select_criteria in ["Nearest to History", "Farthest from History"]:
                num_history = max(1, prediction_frames) if prediction_frames > 0 else 1
                if len(self.history_centers) > 0:
                    recent_centers = list(self.history_centers)[-num_history:]
                    history_avg_center = np.mean(recent_centers, axis=0)

            # 排序选择
            if detections:
                if select_criteria == "Nearest to History":
                    if history_avg_center is not None:
                        for det in detections:
                            det["dist_to_history"] = np.linalg.norm(
                                np.array(det["center"]) - history_avg_center
                            )
                        detections.sort(key=lambda x: x["dist_to_history"], reverse=False)
                    else:
                        # 首帧没有历史，使用图像中心
                        img_center = np.array([w_img / 2, h_img / 2])
                        for det in detections:
                            det["dist_to_center"] = np.linalg.norm(
                                np.array(det["center"]) - img_center
                            )
                        detections.sort(key=lambda x: x["dist_to_center"], reverse=False)
                        
                elif select_criteria == "Farthest from History":
                    if history_avg_center is not None:
                        for det in detections:
                            det["dist_to_history"] = np.linalg.norm(
                                np.array(det["center"]) - history_avg_center
                            )
                        detections.sort(key=lambda x: x["dist_to_history"], reverse=True)
                    else:
                        img_center = np.array([w_img / 2, h_img / 2])
                        for det in detections:
                            det["dist_to_center"] = np.linalg.norm(
                                np.array(det["center"]) - img_center
                            )
                        detections.sort(key=lambda x: x["dist_to_center"], reverse=True)
                else:
                    sort_key = {
                        "Area Max": (lambda x: x["area"], True),
                        "Area Min": (lambda x: x["area"], False),
                        "Confidence Max": (lambda x: x["conf"], True),
                        "Confidence Min": (lambda x: x["conf"], False),
                        "Center Top": (lambda x: x["center"][1], False),
                        "Center Bottom": (lambda x: x["center"][1], True),
                        "Center Left": (lambda x: x["center"][0], False),
                        "Center Right": (lambda x: x["center"][0], True),
                    }
                    key_func, reverse = sort_key[select_criteria]
                    detections.sort(key=key_func, reverse=reverse)
                
                selected_info = detections[:select_count]
            else:
                selected_info = []

            # 更新历史中心
            if selected_info:
                self.history_centers.append(np.array(selected_info[0]["center"]))

            # ============ 处理追踪目标并创建融合Mask ============
            merged_mask = np.zeros((h_img, w_img), dtype=np.float32)
            all_smoothed_bboxes = []
            
            for i in range(select_count):
                det = selected_info[i] if i < len(selected_info) else None
                current_bbox = det["box"] if det is not None else None
                
                # 应用稳定化
                method_map = {
                    "None": None, "EMA": "ema", "Kalman": "kalman",
                    "Adaptive": "adaptive", "Hybrid": "hybrid"
                }
                method = method_map.get(stabilization_method)
                
                if method is not None:
                    smoothed_bbox = self.stabilizers[i].update(
                        current_bbox, method=method, strength=stabilization_strength
                    )
                else:
                    # 无平滑模式
                    if current_bbox is not None:
                        smoothed_bbox = current_bbox
                        self.stabilizers[i].last_valid_bbox = current_bbox.copy()
                        self.stabilizers[i].is_initialized = True
                    else:
                        smoothed_bbox = self.stabilizers[i].last_valid_bbox
                
                # 如果完全没有有效框，使用图像中心区域
                if smoothed_bbox is None:
                    smoothed_bbox = np.array([w_img * 0.25, h_img * 0.25, 
                                              w_img * 0.75, h_img * 0.75])
                
                all_smoothed_bboxes.append((smoothed_bbox, det))
                
                # 创建mask
                if det is not None:
                    mask_found = False
                    if mask_type == "Segmentation":
                        if hasattr(results[0], 'masks') and results[0].masks is not None:
                            orig_idx = det["index"]
                            if hasattr(results[0].masks, 'xy') and orig_idx < len(results[0].masks.xy):
                                polygon = results[0].masks.xy[orig_idx]
                                pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(merged_mask, [pts], 1.0)
                                mask_found = True
                    
                    if not mask_found:
                        x1, y1, x2, y2 = map(int, smoothed_bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)
                        if x2 > x1 and y2 > y1:
                            merged_mask[y1:y2, x1:x2] = 1.0

            # 计算裁剪区域（基于第一个目标）
            if all_smoothed_bboxes:
                primary_bbox = all_smoothed_bboxes[0][0]
            else:
                primary_bbox = np.array([w_img * 0.25, h_img * 0.25, 
                                         w_img * 0.75, h_img * 0.75])
            
            center_x = (primary_bbox[0] + primary_bbox[2]) / 2
            center_y = (primary_bbox[1] + primary_bbox[3]) / 2
            
            # 计算目标裁剪尺寸
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
            
            # 固定裁剪尺寸
            if fixed_cw is None:
                fixed_cw = int(tw) + (int(tw) % 2)
                fixed_ch = int(th) + (int(th) % 2)
            
            cw, ch = fixed_cw, fixed_ch
            
            # 应用偏移
            center_x += offset_x * cw
            center_y += offset_y * ch
            
            # 计算裁剪坐标
            cx1 = int(center_x - cw / 2)
            cy1 = int(center_y - ch / 2)
            cx2 = cx1 + cw
            cy2 = cy1 + ch
            
            # 创建裁剪图像
            crop_img = np.zeros((ch, cw, 3), dtype=np.uint8)
            
            ix1, iy1 = max(0, cx1), max(0, cy1)
            ix2, iy2 = min(w_img, cx2), min(h_img, cy2)
            
            if ix2 > ix1 and iy2 > iy1:
                src_patch = img_np[iy1:iy2, ix1:ix2]
                dest_x = max(0, ix1 - cx1)
                dest_y = max(0, iy1 - cy1)
                sh, sw = src_patch.shape[:2]
                write_h = min(sh, ch - dest_y)
                write_w = min(sw, cw - dest_x)
                crop_img[dest_y:dest_y+write_h, dest_x:dest_x+write_w] = src_patch[:write_h, :write_w]

            out_images.append(torch.from_numpy(crop_img.astype(np.float32) / 255.0))
            
            # 裁剪mask
            crop_mask = np.zeros((ch, cw), dtype=np.float32)
            if ix2 > ix1 and iy2 > iy1:
                src_mask_patch = merged_mask[iy1:iy2, ix1:ix2]
                dest_x = max(0, ix1 - cx1)
                dest_y = max(0, iy1 - cy1)
                sh, sw = src_mask_patch.shape[:2]
                write_h = min(sh, ch - dest_y)
                write_w = min(sw, cw - dest_x)
                crop_mask[dest_y:dest_y+write_h, dest_x:dest_x+write_w] = src_mask_patch[:write_h, :write_w]
            
            out_masks.append(torch.from_numpy(crop_mask))

        if not out_images:
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)))

        return (torch.stack(out_images), torch.stack(out_masks))


NODE_CLASS_MAPPINGS = {
    "YOLODetectionNode": YOLODetectionNode,
    "YOLOTrackingNode": YOLOTrackingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLODetectionNode": "YOLO Detection",
    "YOLOTrackingNode": "YOLO Tracking & Crop"
}
