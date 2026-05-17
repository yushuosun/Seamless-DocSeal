from ultralytics import YOLO

model = YOLO('yolov8s.pt') 

print("🚀 开始使用【单卡 GPU】高阶配置训练...")

results = model.train(
    data='/kaggle/working/seal_data.yaml',
    epochs=100,            
    patience=30,           
    imgsz=1024,            
    
    device=0,              # 改回单卡
    batch=16,              # 单卡用 16 比较稳妥
    
    # --- 数据增强参数 ---
    degrees=90,            
    perspective=0.001,     
    hsv_h=0.015,           
    hsv_s=0.5,             
    hsv_v=0.5,             
    
    project='/kaggle/working/runs', 
    name='seal_detector_pro', 
    workers=4              
)

print("✅ 模型训练完成！最强权重保存在: /kaggle/working/runs/seal_detector_pro/weights/best.pt")
