# ===============================
# 0. 기본 import
# ===============================
import os
import sys
import time
import warnings
import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# ===============================
# 1. Torch CUDA 강제 초기화
# ===============================
import torch
import cv2
import pymysql
from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

try:
    print("[DEBUG] Torch version:", torch.__version__)
    print("[DEBUG] CUDA available (build-time):", torch.cuda.is_available())
    if torch.cuda.is_available():
        _ = torch.zeros(1).cuda()
        print("[DEBUG] CUDA tensor allocation OK")
except Exception as e:
    print("[DEBUG] CUDA init failed:", e)

# ===============================
# 3. 경고 / 환경변수
# ===============================
warnings.filterwarnings("ignore", message="Error decoding JSON")

os.environ["ULTRALYTICS_CACHE_DIR"] = os.path.join(
    os.getcwd(), "ultralytics_cache"
)

# ===============================
# 4. BASE DIR (외부 리소스 우선)
# ===============================
def is_frozen():
    return getattr(sys, "frozen", False)

# EXE 실행 위치 (모델 교체 편의성)
BASE_DIR = Path(sys.executable).parent if is_frozen() else Path(__file__).resolve().parent

# 내부 리소스 경로 (PyInstaller 압축 해제 경로)
INTERNAL_DIR = Path(sys._MEIPASS) if is_frozen() else BASE_DIR

# ===============================
# 5. WATCH DIR (네트워크 경로 대응)
# ===============================

# X-Ray 장비가 저장하는 원격 폴더
NETWORK_DIR = Path(r"\\000.000.0.000\0. X-Ray-AI")

# 로컬에서 AI가 감시할 대기소
LOCAL_STAGE_DIR = BASE_DIR / "local_staging"
LOCAL_STAGE_DIR.mkdir(parents=True, exist_ok=True)

# 결과 저장 경로
ORIG_DIR = BASE_DIR / "_data" / "original"
RESULT_DIR = BASE_DIR / "_data" / "result"

ORIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# 7. 로그 설정
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ===============================
# 8. DB 설정
# ===============================
DB_HOST = "host"
DB_PORT = port
DB_USER = "username"
DB_PASS = "password"
DB_NAME = "table"

def get_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        charset="utf8mb4",
        autocommit=True
    )

# ===============================
# 9. SQL
# ===============================
SQL_INSERT_RAW = """
INSERT INTO WM_RAW_IMG
(DATIME, PRODMODELNAME, FILEPATH, REMARK)
VALUES (%s, %s, %s, %s)
"""

SQL_INSERT_DET = """
INSERT INTO WM_DETECT_IMG
(DETECTTIME, PRODMODELNAME, FILEPATH, DETECTFILENAME, ERRORTYPE, REMARK)
VALUES (%s, %s, %s, %s, %s, %s)
"""

# ===============================
# 10. resource_path (모델 로딩용)
# ===============================
def resource_path(rel_path: str) -> str:
    # EXE 옆에 있는 파일 우선 확인 (모델 교체 가능)
    external_path = BASE_DIR / rel_path
    if external_path.exists():
        return str(external_path)
    
    # 없으면 EXE 내부 파일 사용
    return str(INTERNAL_DIR / rel_path)

# ===============================
# 11. YOLO 모델 로드
# ===============================
MODEL_PATH = resource_path("models/best.pt")

if not os.path.exists(MODEL_PATH):
    print("[ERROR] 모델 파일 없음:", MODEL_PATH)
    input("엔터를 누르면 종료됩니다.")
    sys.exit(1)

print("[INFO] 모델 로드:", MODEL_PATH)
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.model.names if hasattr(model, "model") else {}

IOU_THRESH = 0.7
VALID_EXTS = {".jpg", ".jpeg", ".png"}


# 파일 잠금 해제 대기 함수
def wait_for_file_ready(file_path: Path, timeout: int = 5):
    """
    파일이 완전히 쓰여지고 잠금이 해제될 때까지 대기합니다.
    :param file_path: 파일 경로
    :param timeout: 최대 대기 시간
    :param check_interval: 확인 간격(초)
    :return: True(준비됨), False(시간 초과)
    """
    start = time.time()

    while time.time() - start < timeout:
        try:
            if not file_path.exists(): return False
            # 1. 파일 크기가 0이면 아직 데이터가 안 들어온 것이므로 대기
            if file_path.stat().st_size == 0:
                time.sleep(0.1)
                continue
            
            # 2. 배타적 모드 열기 시도
            with open(file_path, "ab"):
                pass
            return True # 접근 가능
        except PermissionError:
            # 다른 프로세서(X-Ray 장비)가 사용 중임
            time.sleep(0.2)
        except FileNotFoundError:
            # 파일이 처리 중 이동되었거나 삭제됨
            return False
        except Exception as e:
            logging.warning(f"파일 접근 대기 중 예외: {e}")
            time.sleep(0.2)
    return False

# ===============================
# 12. 이미지 처리 로직
# ===============================
def process_one(src: Path):
    """
    로컬 폴더에 들어온 파일을 처리
    처리가 끝나면 로컬 파일도 삭제
    """
    if src.suffix.lower() not in VALID_EXTS:
        return
    
    # 이미 로컬에 온 파일이므로 바로 읽기 시도
    img = None
    try:
        img_array = np.fromfile(str(src), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"이미지 디코딩 에러: {e}")
        
    if img is None:
        logging.warning(f"이미지 로드 실패 (파일 손상): {src.name}")
        # 손상된 파일도 처리 후 삭제해야 쌓이지 않음
        try: src.unlink()
        except: pass
        return
    
    now = datetime.now()

    # 원본 저장
    orig_path = ORIG_DIR / src.name
    is_success, im_buf = cv2.imencode(src.suffix, img)
    if is_success:
        im_buf.tofile(str(orig_path))

    # AI 추론
    preds = model.predict(
        source=img,
        iou=IOU_THRESH,
        verbose=False
    )

    result_img = img.copy()
    top_cls, top_conf = "Normal", 0.0

    if preds and preds[0].boxes:
        confs = preds[0].boxes.conf.cpu().numpy()
        if len(confs) > 0:
            idx = confs.argmax()
            cls_id = int(preds[0].boxes.cls[idx])

            # names 딕셔너리 안전하게 가져오기
            names = model.names
            top_cls = names[cls_id] if cls_id in names else str(cls_id)

            top_conf = float(confs[idx])
            result_img = preds[0].plot()

    # 결과 저장
    result_filename = f"{src.stem}_dt{src.suffix}"
    result_path = RESULT_DIR / result_filename

    is_success, im_buf = cv2.imencode(src.suffix, result_img)
    if is_success:
        im_buf.tofile(str(result_path))

    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(SQL_INSERT_RAW, (
                now, "진공차단기부품", str(orig_path.parent), ""
            ))
            cur.execute(SQL_INSERT_DET, (
                now, "진공차단기부품",
                str(result_path.parent),
                result_path.name,
                top_cls,
                ""
            ))
    except Exception as e:
        logging.exception(e)
    finally:
        try:
            conn.close()
        except:
            pass

    try:
        src.unlink()
    except Exception as e:
        logging.error(f"원본 삭제 실패: {e}")

    logging.info(f"처리 완료: {src.name} → {top_cls} ({top_conf:.3f})")

# ===============================
# 13. Watchdog
# ===============================
class Handler(PatternMatchingEventHandler):
    patterns = ["*.jpg", "*.jpeg", "*.png"]

    def on_created(self, event):
        if not event.is_directory:
            time.sleep(0.1)
            process_one(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith((".jpg", ".png")):
            time.sleep(0.1)
            process_one(Path(event.dest_path))

# ===============================
# 네트워크 -> 로컬 이동 로직
# ===============================
def move_network_to_local():
    """
    네트워크 폴더를 스캔하여 완료된 파일을 로컬로 '이동' 시킴.
    이동 = 복사 후 원본 삭제
    """
    if not NETWORK_DIR.exists():
        logging.error(f"네트워크 경로 없음: {NETWORK_DIR}")
        return
    
    # 네트워크 폴더의 이미지 파일을 확인
    for net_file in NETWORK_DIR.glob("*.*"):
        if net_file.suffix.lower() not in VALID_EXTS:
            continue

        # 파일이 완전히 쓰여졌는지 확인
        if wait_for_file_ready(net_file, timeout=2):
            try:
                dst_file = LOCAL_STAGE_DIR / net_file.name

                # shutil.move: 네트워크 파일 -> 로컬로 이동 (원본 자동 삭제)
                shutil.move(str(net_file), str(dst_file))
                logging.info(f"[파일 이동] {net_file.name} -> 로컬 대기소")

            except PermissionError:
                continue
            except Exception as e:
                logging.error(f"파일 이동 중 에러: {e}")   

# ===============================
# 14. Main
# ===============================
def main():
    print(f"--- 시스템 시작 ---")
    print(f"1. 네트워크 감시: {NETWORK_DIR}")
    print(f"2. 로컬 대기소: {LOCAL_STAGE_DIR}")

    # 1. 로컬 폴더 감시자 실행 (WATCHDOG)
    # 로컬로 파일이 이동되어 들어오는 순간 이벤트를 감지하여 AI 처리 시작
    observer = Observer()
    observer.schedule(LocalHandler(), str(LOCAL_STAGE_DIR), recursive=False)
    observer.start()
    logging.info("로컬 Watchdog 가동됨")

    try:
        while True:
            # 2. 주기적으로 네트워크 폴더 확인 및 파일 이동 실행
            move_network_to_local()

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("시스템 종료 중")
    
    observer.join()

if __name__ == "__main__":
    main()