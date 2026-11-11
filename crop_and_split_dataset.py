"""
crop_and_split_dataset.py
-------------------------
1. Đối chiếu tập train/test từ `split_dataset/train_files.txt` và `split_dataset/test_files.txt`
   với thư mục `images/` và `labels/`.
2. Dùng bounding box (định dạng YOLO) để crop toàn bộ ảnh biển báo và gán nhãn theo `classes.txt`.
3. Tách riêng thành 2 nhóm theo QCVN 41:
   - `bien_chinh`: 49 lớp biển báo chính (Nhóm P, W, R, I, Camera)
   - `bien_phu`  : 3 lớp biển báo phụ (Nhóm S: S.505a*Xe máy, S.505a*Xe tải và công, S.505a*Xe tải)
"""

import os
import sys
import csv
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ARCHIVE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ARCHIVE_DIR / "images"
LABELS_DIR = ARCHIVE_DIR / "labels"
SPLIT_DIR = ARCHIVE_DIR / "split_dataset"
CLASSES_FILE = ARCHIVE_DIR / "classes.txt"
CLASSES_VIE_FILE = ARCHIVE_DIR / "classes_vie.txt"
CLASSES_EN_FILE = ARCHIVE_DIR / "classes_en.txt"

OUTPUT_DIR = ARCHIVE_DIR / "cropped_dataset"


def sanitize_windows_name(name: str) -> str:
    """Thay thế các ký tự không hợp lệ trong tên thư mục/file trên Windows (như dấu *)."""
    invalid_chars = '<>:"/\\|?*'
    res = name
    for ch in invalid_chars:
        res = res.replace(ch, "_")
    return res.strip()


def is_auxiliary_sign(label: str) -> bool:
    """Kiểm tra xem nhãn có thuộc nhóm Biển phụ (Nhóm S) hay không."""
    return label.startswith("S.")


def load_lines(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_label_files(base_dir: Path, prefix: str, records: list):
    fieldnames = [
        "crop_path", "sign_type", "crop_filename", "source_image", "box_index",
        "class_id", "group_class_id", "label", "safe_folder", "label_vie", "label_en",
        "x_min", "y_min", "x_max", "y_max", "crop_width", "crop_height",
        "yolo_xc", "yolo_yc", "yolo_w", "yolo_h",
    ]
    csv_path = base_dir / f"{prefix}_labels.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    txt_path = base_dir / f"{prefix}_labels.txt"
    with open(txt_path, "w", encoding="utf-8") as tf:
        for r in records:
            tf.write(
                f"{r['crop_path']}\t{r['sign_type']}\t{r['class_id']}\t"
                f"{r['group_class_id']}\t{r['label']}\t{r['label_vie']}\n"
            )


def main():
    # 1. Đọc danh sách class từ classes.txt
    classes = load_lines(CLASSES_FILE)
    classes_vie = load_lines(CLASSES_VIE_FILE)
    classes_en = load_lines(CLASSES_EN_FILE)

    main_classes = [(cid, lbl) for cid, lbl in enumerate(classes) if not is_auxiliary_sign(lbl)]
    aux_classes = [(cid, lbl) for cid, lbl in enumerate(classes) if is_auxiliary_sign(lbl)]

    cid_to_group_cid = {}
    for g_idx, (cid, _) in enumerate(main_classes):
        cid_to_group_cid[cid] = g_idx
    for g_idx, (cid, _) in enumerate(aux_classes):
        cid_to_group_cid[cid] = g_idx

    print(f"[*] Đã đọc {len(classes)} lớp biển báo từ: {CLASSES_FILE}")
    print(f"    - Biển chính (bien_chinh): {len(main_classes)} lớp")
    print(f"    - Biển phụ   (bien_phu)  : {len(aux_classes)} lớp ({', '.join(lbl for _, lbl in aux_classes)})")

    # 2. Đọc danh sách file train và test trong split_dataset
    train_list = load_lines(SPLIT_DIR / "train_files.txt")
    test_list = load_lines(SPLIT_DIR / "test_files.txt")

    train_set = set(train_list)
    test_set = set(test_list)

    all_images = sorted([p.name for p in IMAGES_DIR.glob("*.jpg")])
    all_images_set = set(all_images)

    overlap = train_set & test_set
    missing_train = train_set - all_images_set
    missing_test = test_set - all_images_set
    unassigned_images = sorted(all_images_set - train_set - test_set)

    unassigned_boxes = 0
    for img_name in unassigned_images:
        lbl_path = LABELS_DIR / f"{Path(img_name).stem}.txt"
        if lbl_path.exists():
            unassigned_boxes += len(load_lines(lbl_path))

    print("\n" + "=" * 75)
    print(" 1. KẾT QUẢ ĐỐI CHIẾU TẬP TRAIN & TEST (SPLIT_DATASET)")
    print("=" * 75)
    print(f"  - Tổng số ảnh gốc trong images/          : {len(all_images):,}")
    print(f"  - Số ảnh trong split_dataset/train_files : {len(train_list):,} (Unique: {len(train_set):,}, Thiếu: {len(missing_train)})")
    print(f"  - Số ảnh trong split_dataset/test_files  : {len(test_list):,} (Unique: {len(test_set):,}, Thiếu: {len(missing_test)})")
    print(f"  - Tổng số ảnh được chia (Train + Test)   : {len(train_set) + len(test_set):,} ({len(train_set)/(len(train_set)+len(test_set))*100:.2f}% Train / {len(test_set)/(len(train_set)+len(test_set))*100:.2f}% Test)")
    print(f"  - Số ảnh trùng giữa Train và Test        : {len(overlap)}")
    print(f"  - Số ảnh ngoài danh sách (không có bbox) : {len(unassigned_images)} ảnh (tổng số bbox = {unassigned_boxes})")

    # 3. Chuẩn bị thư mục đầu ra
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_sp in ["train", "test"]:
        old_sp_dir = OUTPUT_DIR / old_sp
        if old_sp_dir.exists():
            shutil.rmtree(old_sp_dir)

    for grp in ["bien_chinh", "bien_phu"]:
        for sp in ["train", "test"]:
            target_sp_dir = OUTPUT_DIR / grp / sp
            if target_sp_dir.exists():
                shutil.rmtree(target_sp_dir)
            target_sp_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_list,
        "test": test_list,
    }

    class_counts = {"train": Counter(), "test": Counter()}
    class_img_counts = {"train": defaultdict(set), "test": defaultdict(set)}

    summary_stats = {
        "total_images_in_archive": len(all_images),
        "train_images": len(train_list),
        "test_images": len(test_list),
        "unassigned_empty_images": unassigned_images,
        "num_classes_total": len(classes),
        "num_classes_bien_chinh": len(main_classes),
        "num_classes_bien_phu": len(aux_classes),
        "splits": {},
        "groups": {
            "bien_chinh": {"train_crops": 0, "test_crops": 0, "total_crops": 0, "num_classes": len(main_classes)},
            "bien_phu": {"train_crops": 0, "test_crops": 0, "total_crops": 0, "num_classes": len(aux_classes)},
        },
        "per_class_counts": [],
    }

    for split_name, file_list in splits.items():
        all_records = []
        group_records = {"bien_chinh": [], "bien_phu": []}
        skipped_boxes = 0
        images_with_boxes = 0

        for img_name in file_list:
            img_path = IMAGES_DIR / img_name
            stem = Path(img_name).stem
            lbl_path = LABELS_DIR / f"{stem}.txt"

            if not img_path.exists() or not lbl_path.exists():
                continue

            lines = load_lines(lbl_path)
            if not lines:
                continue

            images_with_boxes += 1
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                img_w, img_h = im_rgb.size

                for box_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        skipped_boxes += 1
                        continue

                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])

                    if cls_id < 0 or cls_id >= len(classes):
                        skipped_boxes += 1
                        continue

                    raw_label = classes[cls_id]
                    vie_label = classes_vie[cls_id] if cls_id < len(classes_vie) else ""
                    en_label = classes_en[cls_id] if cls_id < len(classes_en) else ""
                    safe_folder_name = sanitize_windows_name(raw_label)
                    sign_type = "bien_phu" if is_auxiliary_sign(raw_label) else "bien_chinh"
                    group_cls_id = cid_to_group_cid[cls_id]

                    x_min = max(0, int(round((xc - bw / 2.0) * img_w)))
                    y_min = max(0, int(round((yc - bh / 2.0) * img_h)))
                    x_max = min(img_w, int(round((xc + bw / 2.0) * img_w)))
                    y_max = min(img_h, int(round((yc + bh / 2.0) * img_h)))

                    if x_max <= x_min or y_max <= y_min:
                        skipped_boxes += 1
                        continue

                    crop_img = im_rgb.crop((x_min, y_min, x_max, y_max))

                    cls_dir = OUTPUT_DIR / sign_type / split_name / safe_folder_name
                    cls_dir.mkdir(parents=True, exist_ok=True)

                    crop_filename = f"{stem}_box{box_idx:02d}.jpg"
                    crop_rel_path = f"{sign_type}/{split_name}/{safe_folder_name}/{crop_filename}"
                    crop_full_path = cls_dir / crop_filename

                    crop_img.save(crop_full_path, quality=95)

                    class_counts[split_name][cls_id] += 1
                    class_img_counts[split_name][cls_id].add(img_name)

                    rec = {
                        "crop_path": crop_rel_path,
                        "sign_type": sign_type,
                        "crop_filename": crop_filename,
                        "source_image": img_name,
                        "box_index": box_idx,
                        "class_id": cls_id,
                        "group_class_id": group_cls_id,
                        "label": raw_label,
                        "safe_folder": safe_folder_name,
                        "label_vie": vie_label,
                        "label_en": en_label,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "crop_width": x_max - x_min,
                        "crop_height": y_max - y_min,
                        "yolo_xc": xc,
                        "yolo_yc": yc,
                        "yolo_w": bw,
                        "yolo_h": bh,
                    }
                    all_records.append(rec)

                    grp_rec = dict(rec)
                    grp_rec["crop_path"] = f"{split_name}/{safe_folder_name}/{crop_filename}"
                    group_records[sign_type].append(grp_rec)

        write_label_files(OUTPUT_DIR, split_name, all_records)
        for sign_type in ["bien_chinh", "bien_phu"]:
            write_label_files(OUTPUT_DIR / sign_type, split_name, group_records[sign_type])
            summary_stats["groups"][sign_type][f"{split_name}_crops"] = len(group_records[sign_type])
            summary_stats["groups"][sign_type]["total_crops"] += len(group_records[sign_type])

        summary_stats["splits"][split_name] = {
            "num_source_images": len(file_list),
            "images_with_boxes": images_with_boxes,
            "total_cropped_signs": len(all_records),
            "bien_chinh_crops": len(group_records["bien_chinh"]),
            "bien_phu_crops": len(group_records["bien_phu"]),
            "skipped_boxes": skipped_boxes,
            "classes_present": len(class_counts[split_name]),
        }

    # Lưu danh sách classes cho từng nhóm
    with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as cf:
        cf.write("\n".join(classes) + "\n")
    with open(OUTPUT_DIR / "bien_chinh" / "classes_chinh.txt", "w", encoding="utf-8") as cf:
        cf.write("\n".join(lbl for _, lbl in main_classes) + "\n")
    with open(OUTPUT_DIR / "bien_phu" / "classes_phu.txt", "w", encoding="utf-8") as cf:
        cf.write("\n".join(lbl for _, lbl in aux_classes) + "\n")

    for cid, raw_label in enumerate(classes):
        safe_folder = sanitize_windows_name(raw_label)
        vie_name = classes_vie[cid] if cid < len(classes_vie) else ""
        en_name = classes_en[cid] if cid < len(classes_en) else ""
        sign_type = "bien_phu" if is_auxiliary_sign(raw_label) else "bien_chinh"
        tr_cnt = class_counts["train"][cid]
        te_cnt = class_counts["test"][cid]
        tot_cnt = tr_cnt + te_cnt

        summary_stats["per_class_counts"].append({
            "class_id": cid,
            "group_class_id": cid_to_group_cid[cid],
            "sign_type": sign_type,
            "label": raw_label,
            "safe_folder": safe_folder,
            "label_vie": vie_name,
            "label_en": en_name,
            "train_crops": tr_cnt,
            "test_crops": te_cnt,
            "total_crops": tot_cnt,
            "train_source_images": len(class_img_counts["train"][cid]),
            "test_source_images": len(class_img_counts["test"][cid]),
        })

    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as jf:
        json.dump(summary_stats, jf, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(" 2. THỐNG KÊ TÁCH RIÊNG BIỂN CHÍNH & BIỂN PHỤ")
    print("=" * 75)
    for grp_key, grp_title in [("bien_chinh", "BIỂN CHÍNH (49 lớp: P, W, R, I, Camera)"), ("bien_phu", "BIỂN PHỤ (3 lớp: Nhóm S)")]:
        g_st = summary_stats["groups"][grp_key]
        print(f"  * {grp_title}:")
        print(f"      - Train crops : {g_st['train_crops']:,} ảnh")
        print(f"      - Test crops  : {g_st['test_crops']:,} ảnh")
        print(f"      - Tổng cộng   : {g_st['total_crops']:,} ảnh (Lưu tại: cropped_dataset/{grp_key}/)")

    print(f"\n[✓] Hoàn tất! Dữ liệu đã được tách riêng vào '{OUTPUT_DIR / 'bien_chinh'}' và '{OUTPUT_DIR / 'bien_phu'}'.")


if __name__ == "__main__":
    main()
