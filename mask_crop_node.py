import torch
import numpy as np


class MaskCropExtractNode:
    """遮罩裁剪和提取节点
    
    根据遮罩的边界框裁剪遮罩和图片，并提取坐标信息
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_mask", "cropped_image", "x", "y", "width", "height", "x2", "y2")
    FUNCTION = "crop_and_extract"
    CATEGORY = "YOLO"
    
    def crop_and_extract(self, mask, image=None):
        """
        裁剪遮罩和图片，提取坐标信息
        
        参数:
            mask: 输入遮罩 (batch, height, width)
            image: 可选的输入图片 (batch, height, width, channels)
            
        返回:
            cropped_mask: 裁剪后的遮罩
            cropped_image: 裁剪后的图片（如果提供）
            x, y: 左上角坐标
            width, height: 宽度和高度
            x2, y2: 右下角坐标
        """
        out_masks = []
        out_images = []
        out_x, out_y, out_w, out_h, out_x2, out_y2 = [], [], [], [], [], []
        
        batch_size = mask.shape[0]
        
        for i in range(batch_size):
            # 获取当前遮罩
            current_mask = mask[i].cpu().numpy()
            
            # 查找遮罩的边界框
            rows = np.any(current_mask > 0, axis=1)
            cols = np.any(current_mask > 0, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                # 如果遮罩为空，返回零尺寸
                out_masks.append(torch.zeros((1, 1), dtype=torch.float32))
                if image is not None:
                    channels = image.shape[-1] if len(image.shape) == 4 else 3
                    out_images.append(torch.zeros((1, 1, channels), dtype=torch.float32))
                out_x.append(0)
                out_y.append(0)
                out_w.append(0)
                out_h.append(0)
                out_x2.append(0)
                out_y2.append(0)
                continue
            
            # 获取边界框坐标
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # 注意：y_max 和 x_max 是包含的最后一个索引，所以需要 +1
            y_max += 1
            x_max += 1
            
            # 计算宽度和高度
            width = x_max - x_min
            height = y_max - y_min
            
            # 裁剪遮罩
            cropped_mask = current_mask[y_min:y_max, x_min:x_max]
            out_masks.append(torch.from_numpy(cropped_mask))
            
            # 裁剪图片（如果提供）
            if image is not None:
                current_image = image[i].cpu().numpy()
                cropped_image = current_image[y_min:y_max, x_min:x_max]
                out_images.append(torch.from_numpy(cropped_image))
            
            # 存储坐标信息
            out_x.append(int(x_min))
            out_y.append(int(y_min))
            out_w.append(int(width))
            out_h.append(int(height))
            out_x2.append(int(x_max))
            out_y2.append(int(y_max))
        
        # 堆叠输出
        stacked_masks = torch.stack(out_masks) if out_masks else torch.zeros((1, 1, 1))
        
        if image is not None and out_images:
            stacked_images = torch.stack(out_images)
        else:
            # 如果没有提供图片，返回空白图片
            stacked_images = torch.zeros((batch_size, 1, 1, 3), dtype=torch.float32)
        
        return (stacked_masks, stacked_images, out_x, out_y, out_w, out_h, out_x2, out_y2)


NODE_CLASS_MAPPINGS = {
    "MaskCropExtractNode": MaskCropExtractNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCropExtractNode": "Mask Crop & Extract",
}