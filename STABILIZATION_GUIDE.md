# YOLO检测框抖动问题解决方案

## 问题描述

在视频处理中使用YOLO模型时,检测框会出现忽大忽小、抖动的问题。这是由于连续帧之间存在光照差异、像素偏移、噪声等因素导致的。

## 解决方案

本插件现在提供了多种稳定化算法来解决检测框抖动问题:

### 1. **稳定化方法 (Stabilization Method)**

在 `YOLOTrackingNode` 节点中,新增了以下参数:

- **stabilization_method**: 选择稳定化算法
  - `None`: 不使用额外稳定化
  - `EMA`: 指数移动平均 (推荐用于大多数场景)
  - `Moving Average`: 简单移动平均
  - `Dynamic`: 动态平滑 (根据移动幅度自动调整)

- **stabilization_strength**: 稳定化强度 (0.0-0.99)
  - 较低值 (0.3-0.5): 快速响应,适合快速移动的目标
  - 中等值 (0.5-0.7): 平衡响应速度和稳定性 (推荐)
  - 较高值 (0.7-0.9): 强稳定性,适合缓慢移动的目标

### 2. **原有平滑参数 (Smoothing)**

- **smoothing**: 0.0-0.99
  - 这是原有的平滑参数,与新的稳定化算法可以叠加使用
  - 建议: 使用新的稳定化算法时,可以将此参数设为较低值(0.0-0.3)

## 推荐配置

### 场景1: 普通视频跟踪
```
confidence_threshold: 0.5-0.6  (提高置信度减少误检)
stabilization_method: EMA
stabilization_strength: 0.7
smoothing: 0.2
```

### 场景2: 快速移动目标
```
confidence_threshold: 0.6
stabilization_method: Dynamic
stabilization_strength: 0.5
smoothing: 0.1
```

### 场景3: 静态或缓慢移动目标
```
confidence_threshold: 0.5
stabilization_method: EMA
stabilization_strength: 0.85
smoothing: 0.3
```

## 算法说明

### EMA (指数移动平均)
- **原理**: 给予最近的帧更高权重,历史帧权重逐渐衰减
- **优点**: 响应速度快,计算效率高
- **适用**: 大多数场景的首选

### Moving Average (移动平均)
- **原理**: 对最近N帧的检测框位置取平均值
- **优点**: 平滑效果稳定
- **适用**: 目标移动规律、速度均匀的场景

### Dynamic (动态平滑)
- **原理**: 根据检测框移动幅度自动调整平滑强度
  - 小幅移动时: 强平滑 (减少抖动)
  - 大幅移动时: 弱平滑 (快速响应)
- **优点**: 自适应性强,无需手动调参
- **适用**: 目标运动速度变化较大的场景

## 其他优化建议

1. **提高置信度阈值**: 将 `confidence_threshold` 从默认的 0.5 提高到 0.5-0.7,可以过滤掉不稳定的检测

2. **调整NMS阈值**: 适当提高 `nms_threshold` (如0.5-0.6) 可以减少重复检测

3. **使用合适的模型**: 
   - 较大的模型 (yolov8m/l/x) 通常检测更稳定
   - 较小的模型 (yolov8n/s) 速度快但可能不够稳定

4. **固定crop_scale**: 在视频处理中保持 `crop_scale` 恒定,避免裁剪区域大小变化

## 注意事项

1. **参数平衡**: `stabilization_strength` 和 `smoothing` 两个参数会叠加作用,总平滑强度不要过高,否则会导致响应迟钝

2. **首帧初始化**: 第一帧可能需要几帧才能稳定下来,这是正常现象

3. **场景切换**: 如果视频中有场景切换,建议重新初始化节点或降低平滑强度

4. **性能影响**: 稳定化算法对性能影响很小,可以放心使用

## 代码实现

新增的 `BBoxStabilizer` 类提供了独立的边界框稳定化功能,每个跟踪对象都有独立的稳定器实例,确保多目标跟踪时互不干扰。

```python
# 示例: 稳定器的使用
stabilizer = BBoxStabilizer(window_size=5)
smoothed_bbox = stabilizer.update(current_bbox, method='ema', alpha=0.7)
```

## 故障排除

**问题**: 检测框仍然抖动
- 解决: 提高 `stabilization_strength` 到 0.8-0.9
- 或者: 同时提高 `confidence_threshold` 到 0.6-0.7

**问题**: 检测框响应太慢
- 解决: 降低 `stabilization_strength` 到 0.4-0.6
- 或者: 切换到 `Dynamic` 方法

**问题**: 目标快速移动时跟丢
- 解决: 降低所有平滑参数
- 或者: 使用 `Dynamic` 方法,它会在大幅移动时自动降低平滑强度

## 更新日志

### v1.1
- 新增 `BBoxStabilizer` 类
- 新增三种稳定化算法: EMA, Moving Average, Dynamic
- 新增 `stabilization_method` 和 `stabilization_strength` 参数
- 为每个跟踪对象创建独立的稳定器实例
- 保持向后兼容,原有的 `smoothing` 参数仍然有效