# build_20k_from_mixed_train_val.py
# Python 3.10+
from pathlib import Path
import shutil, random, hashlib

# ====== 설정 ======
SRC_IMG_ROOT = Path("dataset/images")      # 기존 소스 루트 (train/val 둘 다 존재, 클래스 섞여있음)
SRC_LAB_ROOT = Path("dataset/labels")
OUT_IMG_ROOT = Path("datasets/images")  # 결과 루트 (새로 생성: train/ val)
OUT_LAB_ROOT = Path("datasets/labels")

DESIRED_TOTAL = 20_000
TRAIN_RATIO   = 0.8
RANDOM_SEED   = 42
CREATE_EMPTY_LABELS = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ====== 유틸 ======
def scan_images(img_root: Path):
    pool = []
    for sub in ["train", "val"]:
        d = img_root / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    # split 정보와 상대경로도 함께 저장 (라벨 매칭용)
                    rel = p.relative_to(img_root / sub)
                    pool.append((sub, p, rel))
    return pool

def unique_name(img_path: Path):
    # 원본 경로 + 크기 기반 해시로 고유 파일명 생성
    try:
        sz = img_path.stat().st_size
    except Exception:
        sz = 0
    h = hashlib.sha1(f"{img_path}|{sz}".encode("utf-8")).hexdigest()[:10]
    return f"{img_path.stem}_{h}{img_path.suffix.lower()}"

def label_for(img_split: str, img_rel: Path) -> Path:
    return SRC_LAB_ROOT / img_split / img_rel.with_suffix(".txt")

def copy_pair(items, dst_img_dir: Path, dst_lab_dir: Path):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    dup_cnt = 0
    has_label = 0
    no_label = 0

    for i, (split, src_img, rel) in enumerate(items, 1):
        # 새 고유 파일명
        new_img_name = unique_name(src_img)
        img_dst = dst_img_dir / new_img_name

        # 파일명 충돌 방지
        while img_dst.name in seen or img_dst.exists():
            dup_cnt += 1
            stem = Path(new_img_name).stem + f"__dup{dup_cnt:04d}"
            img_dst = dst_img_dir / (stem + Path(new_img_name).suffix)
        seen.add(img_dst.name)

        # 이미지 복사
        shutil.copy2(src_img, img_dst)

        # 라벨 경로 매핑(원본과 같은 상대경로에서 .txt 찾기)
        src_lab = label_for(split, rel)
        lab_dst = dst_lab_dir / (img_dst.stem + ".txt")

        if src_lab.exists() and src_lab.is_file():
            shutil.copy2(src_lab, lab_dst)
            has_label += 1
        else:
            if CREATE_EMPTY_LABELS:
                lab_dst.write_text("")
            no_label += 1
        if i % 500 == 0:
            print(f"[COPY] {i}/{len(items)} (labeled:{has_label}, no label:{no_label})")
    return has_label, no_label

def main():
    random.seed(RANDOM_SEED)

    # 1) 소스 스캔 (train+val 모두에서 이미지 수집)
    pool = scan_images(SRC_IMG_ROOT)
    if not pool:
        raise RuntimeError("소스에서 이미지를 찾지 못했습니다. 경로를 확인하세요: dataset/images/{train,val}")

    total_src = len(pool)
    print(f"[소스] images train+val 합계 = {total_src:,}")

    # 2) 20,000장으로 맞추기 (부족하면 중복 복제, 많으면 샘플링)
    if total_src >= DESIRED_TOTAL:
        picked = random.sample(pool, DESIRED_TOTAL)
        oversampled = False
    else:
        picked = list(pool)
        need = DESIRED_TOTAL - total_src
        while need > 0:
            picked.append(random.choice(pool))
            need -= 1
        oversampled = True
    print(f"[선정] 총 {len(picked):,}장 (oversampled={oversampled})")

    # 3) 8:2 분할
    random.shuffle(picked)
    k_train = round(len(picked) * TRAIN_RATIO)
    train_set = picked[:k_train]
    val_set   = picked[k_train:]
    print(f"[분할] train={len(train_set):,}  val={len(val_set):,}")

    # 4) 복사 (중복 방지 파일명)
    out_img_train = OUT_IMG_ROOT / "train"
    out_img_val   = OUT_IMG_ROOT / "val"
    out_lab_train = OUT_LAB_ROOT / "train"
    out_lab_val = OUT_LAB_ROOT / "val"

    has_tr, no_tr = copy_pair(train_set, out_img_train, out_lab_train)
    has_vl, no_vl = copy_pair(val_set, out_img_val, out_lab_val)

    print(f"✅ 완료: images(train={len(train_set):,}, val={len(val_set):,}) -> {OUT_IMG_ROOT}")
    print(f"✅ 완료: labels(train={len(train_set):,}, val={len(val_set):,}) -> {OUT_LAB_ROOT}")
    print(f"   라벨 통계  train: 존재 {has_tr:,} / 없음 {no_tr:,}")
    print(f"             val  : 존재 {has_vl:,} / 없음 {no_vl:,}")
    total_imgs = len(train_set) + len(val_set)
    total_labs = (has_tr+no_tr) + (has_vl+no_vl)
    print(f"총 이미지 파일 = {total_imgs:,}  |  총 라벨 파일 = {total_labs:,} (빈 라벨 포함)")
    print("TIP) 빈 라벨은 정상 이미지 등 객체가 없는 케이스를 의미합니다.")

if __name__ == "__main__":
    main()
