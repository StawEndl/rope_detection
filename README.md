## # 22.6.27更新简介

### 1、更新操作
1. 首次将shape从640扩大至1280，本代码由cascade_1.4修改而来。
 

启动命令：python train.py --data rope_dog.yaml --cfg yolov5s_edited.yaml --weights models/yolov5s.pt --batch-size 8 --epochs 150 --device 0,1 --imgsz 1280

