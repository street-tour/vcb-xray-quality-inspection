import os
from ultralytics import YOLO
import torch
import multiprocessing

def train_model():
    # # Windows DDP 에러 방지용 환경 변수 설정
    # os.environ["USE_LIBUV"] = "0"   # libuv 비활성화
    # os.environ["OTI_BACKEND"] = "gloo" # Windows는 gloo 백엔드 사용

    # # GPU 잘 잡히는지 확인
    # print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")

    model = YOLO('yolov8s.pt')

    results = model.train(
        # --- [기본 설정] ---
        data=r'C:\Users',
        epochs = 300,
        patience = 50,
        batch = 16,
        imgsz=1280,
        device='0',
        workers=4,

        amp=True,

        # --- [저장 설정] ---
        project='project directory name',
        name='y8s_1280_bellows_verticalflip',
        exist_ok=True,
        save=True,

        # --- [성능 최적화 옵션] ---
        optimizer='AdamW',
        cos_lr=True,
        lr0=0.001,

        # --- [X-Ray 특화 증강 옵션] ---
        hsv_h=0.01,
        hsv_s=0.0,
        hsv_v=0.4,

        # 기하학적 변환
        degrees=0.0,
        translate=0.1,
        scale=0.1,
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    train_model()