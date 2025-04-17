from ultralytics import YOLO

# 加载官方或自定义模型
model = YOLO('runs/best.pt')  # 加载一个官方的检测模型

# 使用模型进行追踪
# results = model.track(source=r'D:\YOLOv11\data\Single ROV\Target disappearing2.mp4', show=True, save=True, persist=True)  # 使用默认追踪器BoT-SORT进行追踪
results = model.track(source=r'D:\YOLOv11\data\video_FPS10.mp4', show=True, save=True, save_txt=True, save_conf=True, tracker="BotSort.yaml")  # 使用ByteTrack追踪器进行追踪
