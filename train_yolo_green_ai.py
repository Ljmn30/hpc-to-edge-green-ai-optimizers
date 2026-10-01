from ultralytics import YOLO

model = YOLO('yolov8n.pt')

 
results = model.train(
    data='/path/to/data.yaml',     
    epochs=500,                      # 500 for training with an early stop, or 200 for training without the early stop condition
    #patience=5,                      # This is commented out "#" when you don't want to use the early stop
    batch=16,                        
    imgsz=640,                       
    optimizer='SGD',                 # (SGD, Adam, NAdam, RAdam o Adadelta)
    lr0=0.01,                        
    device=0,                        # GPU ID (Tesla V100 or A30)
    plots=True,                      
    save=True,                       
    project='hpc_green_ai_study',
    name='yolov8n_sgd_run'
)