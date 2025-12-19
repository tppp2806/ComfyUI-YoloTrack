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
    """重写的卡尔曼滤波追踪器 - 更强的平滑效果"""
    
    count = 0
    
    def __init__(self, bbox, smoothness=0.7):
        """
        smoothness: 0.0-1.0, 值越大越平滑
        """
        self.smoothness = smoothness
        self.kf = self._create_kalman_filter(smoothness)
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.kf['x'] = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.age = 0
        
    def _create_kalman_filter(self, smoothness):
        """根据平滑度创建卡尔曼滤波器参数"""
        dt = 1.0
        
        # 状态转移矩阵
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt
        
        # 观测矩阵
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        
        # 根据smoothness调整参数
        q_scale = 0.001 + (1 - smoothness) * 10
        r_scale = 1 + smoothness * 1000
        
        # 过程噪声
        Q = np.eye(8, dtype=np.float64)
        Q[0:4, 0:4] *= q_scale
        Q[4:8, 4:8] *= q_scale * 0.01
        
        # 测量噪声
        R = np.eye(4, dtype=np.float64) * r_scale
        
        # 初始协方差
        P = np.eye(8, dtype=np.float64) * 100
        
        return {
            'x': np.zeros(8, dtype=np.float64),
            'P': P, 'F': F, 'H': H, 'Q': Q, 'R': R,
        }
    
    def predict(self):
        """预测下一状态"""
        kf = self.kf
        
        if kf['x'][2] + kf['x'][6] <= 0:
            kf['x'][6] = 0
        if kf['x'][3] + kf['x'][7] <= 0:
            kf['x'][7] = 0
            
        kf['x'] = kf['F'] @ kf['x']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        
        self.age += 1
        self.time_since_update += 1
        
        return self.get_state()
    
    def update(self, bbox):
        """更新状态"""
        self.time_since_update = 0
        self.hits += 1
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float64)
        
        kf = self.kf
        
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)
        
        y = z - kf['H'] @ kf['x']
        kf['x'] = kf['x'] + K @ y
        
        I = np.eye(8, dtype=np.float64)
        kf['P'] = (I - K @ kf['H']) @ kf['P']
        
    def get_state(self):
        """获取当前边界框状态"""
        cx, cy, w, h = self.kf['x'][:4]
        w = max(w, 1)
        h = max(h, 1)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class BBoxStabilizer:
    """重写的边界框稳定器"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.kalman_tracker = None
        self.is_initialized = False
        self.last_valid_bbox = None
        self.frame_count = 0
        self.velocity_history = deque(maxlen=10)
        
    def _compute_velocity(self):
        """计算平均速度"""
        if len(self.history) < 2:
            return np.array([0, 0, 0, 0], dtype=np.float64)
        
        velocities = []
        hist_list = list(self.history)
        for i in range(1, min(len(hist_list), 5)):
            vel = np.array(hist_list[-i]) - np.array(hist_list[-i-1])
            velocities.append(vel)
        
        if velocities:
            return np.mean(velocities, axis=0)
        return np.array([0, 0, 0, 0], dtype=np.float64)
    
    def predict_position(self):
        """预测丢失目标的位置"""
        if self.kalman_tracker is not None:
            return self.kalman_tracker.predict()
        elif len(self.history) >= 2:
            velocity = self._compute_velocity()
            return np.array(self.history[-1]) + velocity
        elif self.last_valid_bbox is not None:
            return self.last_valid_bbox.copy()
        return None
    
    def smooth_ema(self, current, strength):
        """指数移动平均"""
        if len(self.history) == 0:
            return current
        
        alpha = 1.0 - strength
        prev = np.array(self.history[-1])
        return alpha * current + (1 - alpha) * prev
    
    def smooth_weighted_average(self, current, strength):
        """加权移动平均"""
        if len(self.history) == 0:
            return current
        
        hist_list = list(self.history)
        n = len(hist_list)
        
        decay = 1.0 - strength
        weights = np.array([decay ** i for i in range(n, 0, -1)])
        
        current_weight = decay ** 0
        
        total_weight = weights.sum() + current_weight
        weights = weights / total_weight
        current_weight = current_weight / total_weight
        
        result = current * current_weight
        for i, h in enumerate(hist_list):
            result += np.array(h) * weights[i]
        
        return result
    
    def smooth_kalman(self, current, strength):
        """卡尔曼滤波平滑"""
        if self.kalman_tracker is None:
            self.kalman_tracker = KalmanBoxTracker(current, smoothness=strength)
            return current
        
        self.kalman_tracker.predict()
        self.kalman_tracker.update(current)
        return self.kalman_tracker.get_state()
    
    def smooth_hybrid(self, current, strength):
        """混合平滑"""
        kalman_result = self.smooth_kalman(current.copy(), strength)
        weighted_result = self.smooth_weighted_average(current.copy(), strength)
        
        kalman_weight = 0.3 + 0.4 * strength
        return kalman_weight * kalman_result + (1 - kalman_weight) * weighted_result
    
    def update(self, bbox, method='kalman', strength=0.7, enable_prediction=True):
        """更新稳定器"""
        self.frame_count += 1
        
        # 目标丢失处理
        if bbox is None:
            if enable_prediction:
                predicted = self.predict_position()
                if predicted is not None:
                    self.history.append(predicted.copy())
                    self.last_valid_bbox = predicted.copy()
                    return predicted
            return self.last_valid_bbox
        
        bbox = np.array(bbox, dtype=np.float64)
        
        # 首帧处理
        if not self.is_initialized:
            self.is_initialized = True
            self.last_valid_bbox = bbox.copy()
            self.history.append(bbox.copy())
            
            if method in ['kalman', 'hybrid']:
                self.kalman_tracker = KalmanBoxTracker(bbox, smoothness=strength)
            
            return bbox
        
        # 应用平滑方法
        if method == 'ema':
            smoothed = self.smooth_ema(bbox, strength)
        elif method == 'weighted':
            smoothed = self.smooth_weighted_average(bbox, strength)
        elif method == 'kalman':
            smoothed = self.smooth_kalman(bbox, strength)
        elif method == 'hybrid':
            smoothed = self.smooth_hybrid(bbox, strength)
        else:
            smoothed = bbox
        
        self.history.append(smoothed.copy())
        self.last_valid_bbox = smoothed.copy()
        
        return smoothed
    
    def reset(self):
        """重置稳定器状态"""
        self.history.clear()
        self.kalman_tracker = None
        self.is_initialized = False
        self.last_valid_bbox = None
        self.frame_count = 0
        self.velocity_history.clear()


class YOLODetectionNode:
    """YOLO检测节点 - 支持历史追踪选择"""
    
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
                "select_criteria": ([
                    "Area Max", "Area Min", 
                    "Confidence Max", "Confidence Min", 
                    "Center Top", "Center Bottom", "Center Left", "Center Right",
                    "Center Nearest",  # 新增：距离图像中心最近
                    "Nearest to History", "Farthest from History"  # 新增：历史位置
                ],),
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
        
        # 所有历史中心点（用于 Nearest/Farthest to History）
        all_history_centers = []

        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            h_img, w_img = img_np.shape[:2]
            
            # 图像中心
            img_center = np.array([w_img / 2, h_img / 2])
            
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
                    center = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                    
                    # 计算距离图像中心的距离
                    dist_to_img_center = np.linalg.norm(np.array(center) - img_center)
                    
                    detections.append({
                        "box": xyxy, "conf": conf, "area": w * h,
                        "center": center, "cls": cls,
                        "dist_to_img_center": dist_to_img_center
                    })

            # ===== 计算所有历史的平均中心 =====
            history_avg_center = None
            if all_history_centers:
                history_avg_center = np.mean(all_history_centers, axis=0)

            # ===== 排序选择 =====
            if detections:
                if select_criteria == "Center Nearest":
                    # 距离图像中心最近
                    detections.sort(key=lambda x: x["dist_to_img_center"], reverse=False)
                    
                elif select_criteria == "Nearest to History":
                    if history_avg_center is not None:
                        for det in detections:
                            det["dist_to_history"] = np.linalg.norm(
                                np.array(det["center"]) - history_avg_center
                            )
                        detections.sort(key=lambda x: x["dist_to_history"], reverse=False)
                    else:
                        # 首帧：使用图像中心
                        detections.sort(key=lambda x: x["dist_to_img_center"], reverse=False)
                        
                elif select_criteria == "Farthest from History":
                    if history_avg_center is not None:
                        for det in detections:
                            det["dist_to_history"] = np.linalg.norm(
                                np.array(det["center"]) - history_avg_center
                            )
                        detections.sort(key=lambda x: x["dist_to_history"], reverse=True)
                    else:
                        # 首帧：使用图像中心
                        detections.sort(key=lambda x: x["dist_to_img_center"], reverse=True)
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
                
                selected = detections[:select_count]
            else:
                selected = []

            # ===== 更新历史中心 =====
            if selected:
                for det in selected:
                    all_history_centers.append(np.array(det["center"]))

            if output_mode == "Merged Mask":
                out_images.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
                
                preview_np = img_np.copy()
                for det in selected:
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(preview_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                    cv2.putText(preview_np, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out_preview_images.append(torch.from_numpy(preview_np.astype(np.float32) / 255.0))

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
    """YOLO追踪裁剪节点 - 重写版本"""
    
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
                "select_criteria": ([
                    "Area Max", "Area Min",
                    "Confidence Max", "Confidence Min",
                    "Center Top", "Center Bottom", "Center Left", "Center Right",
                    "Center Nearest",  # 新增：距离图像中心最近
                    "Nearest to History", "Farthest from History"
                ],),
                "select_count": ("INT", {"default": 1, "min": 1, "max": 100}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_ratio": (["Original", "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2"],),
                "crop_scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10000.0, "step": 0.001}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "stabilization_method": (["None", "EMA", "Weighted Average", "Kalman", "Hybrid"],),
                "stabilization_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "enable_prediction": ("BOOLEAN", {"default": True}),
                "mask_type": (["Box", "Segmentation"],),
                "limit_to_bounds": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "class_ids": ("STRING", {"default": "", "multiline": False,
                                         "placeholder": "e.g. 0, 2 (leave empty for all)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("cropped_image", "mask", "preview_with_marks")
    FUNCTION = "track_and_crop"
    CATEGORY = "YOLO"

    def track_and_crop(self, image, model_name, select_criteria, select_count,
                       confidence_threshold, nms_threshold, crop_ratio, crop_scale,
                       offset_x, offset_y, stabilization_method, stabilization_strength,
                       enable_prediction, mask_type, limit_to_bounds, class_ids=""):
        
        # Load Model
        model_path = folder_paths.get_full_path("yolo", model_name)
        if model_path is None:
            model_path = model_name
        
        if self.model is None or self.current_model_name != model_path:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.current_model_name = model_path

        # 每次执行时创建新的稳定器
        stabilizers = [BBoxStabilizer(window_size=100) for _ in range(select_count)]
        
        # 所有历史中心点（用于 Nearest/Farthest to History）
        all_history_centers = []

        # Parse Class IDs
        target_classes = None
        if class_ids.strip():
            try:
                target_classes = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
            except ValueError:
                print("Invalid class_ids format. Using all classes.")

        out_images = []
        out_masks = []
        out_preview_images = []
        
        fixed_cw, fixed_ch = None, None
        
        # 方法映射
        method_map = {
            "None": None, 
            "EMA": "ema", 
            "Weighted Average": "weighted",
            "Kalman": "kalman",
            "Hybrid": "hybrid"
        }

        for frame_idx, img_tensor in enumerate(image):
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            h_img, w_img = img_np.shape[:2]
            
            # 图像中心
            img_center = np.array([w_img / 2, h_img / 2])
            
            # YOLO推理
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
                    
                    # 计算距离图像中心的距离
                    dist_to_img_center = np.linalg.norm(np.array(center) - img_center)
                    
                    detections.append({
                        "box": xyxy, "conf": conf, "area": w * h,
                        "center": center, "cls": cls, "index": i,
                        "dist_to_img_center": dist_to_img_center
                    })

            # ===== 计算所有历史的平均中心 =====
            history_avg_center = None
            if all_history_centers:
                history_avg_center = np.mean(all_history_centers, axis=0)

            # ===== 排序选择 =====
            if detections:
                if select_criteria == "Center Nearest":
                    # 距离图像中心最近
                    detections.sort(key=lambda x: x["dist_to_img_center"], reverse=False)
                    
                elif select_criteria == "Nearest to History":
                    if history_avg_center is not None:
                        for det in detections:
                            det["dist_to_history"] = np.linalg.norm(
                                np.array(det["center"]) - history_avg_center
                            )
                        detections.sort(key=lambda x: x["dist_to_history"], reverse=False)
                    else:
                        # 首帧：使用图像中心
                        detections.sort(key=lambda x: x["dist_to_img_center"], reverse=False)
                        
                elif select_criteria == "Farthest from History":
                    if history_avg_center is not None:
                        for det in detections:
                            det["dist_to_history"] = np.linalg.norm(
                                np.array(det["center"]) - history_avg_center
                            )
                        detections.sort(key=lambda x: x["dist_to_history"], reverse=True)
                    else:
                        # 首帧：使用图像中心
                        detections.sort(key=lambda x: x["dist_to_img_center"], reverse=True)
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

            # ===== 更新历史中心（使用所有选中目标的中心） =====
            if selected_info:
                for det in selected_info:
                    all_history_centers.append(np.array(det["center"]))

            # ===== 处理追踪目标并创建融合Mask =====
            merged_mask = np.zeros((h_img, w_img), dtype=np.float32)
            all_smoothed_bboxes = []
            
            method = method_map.get(stabilization_method)
            
            for i in range(select_count):
                det = selected_info[i] if i < len(selected_info) else None
                current_bbox = det["box"] if det is not None else None
                
                # 应用稳定化
                smoothed_bbox = stabilizers[i].update(
                    current_bbox, 
                    method=method, 
                    strength=stabilization_strength,
                    enable_prediction=enable_prediction
                )
                
                # 如果完全没有有效框，使用图像中心区域
                if smoothed_bbox is None:
                    smoothed_bbox = np.array([w_img * 0.25, h_img * 0.25, 
                                              w_img * 0.75, h_img * 0.75])
                
                all_smoothed_bboxes.append((smoothed_bbox, det))
                
                # 创建mask
                if det is not None or (enable_prediction and smoothed_bbox is not None):
                    mask_found = False
                    if mask_type == "Segmentation" and det is not None:
                        if hasattr(results[0], 'masks') and results[0].masks is not None:
                            orig_idx = det["index"]
                            if hasattr(results[0].masks, 'xy') and orig_idx < len(results[0].masks.xy):
                                polygon = results[0].masks.xy[orig_idx]
                                pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(merged_mask, [pts], 1.0)
                                mask_found = True
                    
                    if not mask_found and smoothed_bbox is not None:
                        x1, y1, x2, y2 = map(int, smoothed_bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)
                        if x2 > x1 and y2 > y1:
                            merged_mask[y1:y2, x1:x2] = 1.0

            # ===== 计算裁剪区域（基于第一个目标）=====
            if all_smoothed_bboxes:
                primary_bbox = all_smoothed_bboxes[0][0]
            else:
                primary_bbox = np.array([w_img * 0.25, h_img * 0.25, 
                                         w_img * 0.75, h_img * 0.75])
            
            center_x = (primary_bbox[0] + primary_bbox[2]) / 2
            center_y = (primary_bbox[1] + primary_bbox[3]) / 2
            
            # 计算检测框的宽高（用于crop_scale基准）
            bbox_width = primary_bbox[2] - primary_bbox[0]
            bbox_height = primary_bbox[3] - primary_bbox[1]
            
            # 计算目标裁剪尺寸
            if crop_ratio == "Original":
                # 使用检测框大小作为基准，而不是整个画布
                tw = bbox_width * crop_scale
                th = bbox_height * crop_scale
            else:
                rw, rh = map(int, crop_ratio.split(":"))
                target_aspect = rw / rh
                bbox_aspect = bbox_width / bbox_height
                
                # 基于检测框大小计算裁剪尺寸
                if target_aspect > bbox_aspect:
                    tw = bbox_width * crop_scale
                    th = tw / target_aspect
                else:
                    th = bbox_height * crop_scale
                    tw = th * target_aspect
            
            # 固定裁剪尺寸（保证所有帧一致）
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
            
            # 限制尺寸模式：确保裁剪区域不超出原图边界
            if limit_to_bounds:
                # 调整裁剪区域使其不超出图像边界
                if cx1 < 0:
                    cx2 = min(cx2 - cx1, w_img)
                    cx1 = 0
                if cy1 < 0:
                    cy2 = min(cy2 - cy1, h_img)
                    cy1 = 0
                if cx2 > w_img:
                    cx1 = max(cx1 - (cx2 - w_img), 0)
                    cx2 = w_img
                if cy2 > h_img:
                    cy1 = max(cy1 - (cy2 - h_img), 0)
                    cy2 = h_img
                
                # 重新计算实际的裁剪尺寸
                cw = cx2 - cx1
                ch = cy2 - cy1
            
            # 创建裁剪图像（带边界处理）
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
            
            # ===== 创建标记预览图 =====
            preview_img = img_np.copy()
            
            # 绘制所有平滑后的检测框和标记
            for smoothed_bbox, det in all_smoothed_bboxes:
                if smoothed_bbox is not None:
                    x1, y1, x2, y2 = map(int, smoothed_bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    # 绘制检测框
                    cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 如果有检测信息，添加标签
                    if det is not None:
                        label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                        cv2.putText(preview_img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 如果是分割模式，绘制分割轮廓
                        if mask_type == "Segmentation":
                            if hasattr(results[0], 'masks') and results[0].masks is not None:
                                orig_idx = det["index"]
                                if hasattr(results[0].masks, 'xy') and orig_idx < len(results[0].masks.xy):
                                    polygon = results[0].masks.xy[orig_idx]
                                    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                                    # 绘制半透明分割区域
                                    overlay = preview_img.copy()
                                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                                    preview_img = cv2.addWeighted(preview_img, 0.7, overlay, 0.3, 0)
                                    # 绘制分割轮廓
                                    cv2.polylines(preview_img, [pts], True, (0, 255, 0), 2)
            
            # 绘制裁剪区域框
            crop_x1, crop_y1 = max(0, cx1), max(0, cy1)
            crop_x2, crop_y2 = min(w_img, cx2), min(h_img, cy2)
            cv2.rectangle(preview_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 2)
            
            out_preview_images.append(torch.from_numpy(preview_img.astype(np.float32) / 255.0))

        if not out_images:
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)), torch.zeros((1, 512, 512, 3)))

        return (torch.stack(out_images), torch.stack(out_masks), torch.stack(out_preview_images))


NODE_CLASS_MAPPINGS = {
    "YOLODetectionNode": YOLODetectionNode,
    "YOLOTrackingNode": YOLOTrackingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLODetectionNode": "YOLO Detection",
    "YOLOTrackingNode": "YOLO Tracking & Crop"
}
