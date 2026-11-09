r"""
build_freeform_vqa_split.py
---------------------------
Xây dựng bộ dữ liệu Free-form VQA phân chia chuẩn Train / Test (không rò rỉ dữ liệu)
từ `cropped_dataset/` (`bien_chinh` & `bien_phu`) và `qa_52_signs_detailed.json` + `law_database.json`
để phục vụ huấn luyện Proposed Model (Fine-tuned Qwen VLM) và đánh giá 3 Baselines:
  - Baseline 1: Pre-trained VLM – Zero-shot
  - Baseline 2: Pre-trained VLM – Few-shot
  - Baseline 3: Frozen VLM + Legal Retrieval (BGE Embedding)
  - Proposed  : Fine-tuned VLM trên Free-form VQA

Bộ dữ liệu bao gồm đầy đủ 4 nhóm câu hỏi mục tiêu:
  1. Dữ liệu biển đơn (`single_sign`): Nhận diện tên và mã biển báo chính.
  2. Dữ liệu biển chính + biển phụ (`main_and_aux`): Nhận diện nội dung biển phụ đi kèm biển chính.
  3. Câu hỏi về quan hệ giữa các biển (`sign_relationship`): Biển phụ bổ sung ý nghĩa gì cho biển chính?
  4. Câu hỏi giải thích quy định giao thông & tuân thủ Yes/No (`regulation_explanation` & `yes_no_compliance`).
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ARCHIVE_DIR = Path(__file__).resolve().parent
CROPPED_DIR = ARCHIVE_DIR / "cropped_dataset"
QA_DETAILED_PATH = ARCHIVE_DIR / "qa_52_signs_detailed.json"
LAW_DB_PATH = ARCHIVE_DIR / "law_database.json"
VQA_OUT_DIR = ARCHIVE_DIR / "vqa_dataset"

# Phân loại 6 câu hỏi q1..q6 vào đúng 4 nhóm nhiệm vụ của bài toán Free-form VQA
QUESTION_CATEGORY_MAP = {
    "q1": "single_sign",               # Nhận diện biển đơn / biển chính
    "q2": "main_and_aux",              # Nội dung biển phụ trong cụm biển chính + phụ
    "q3": "sign_relationship",         # Quan hệ ngữ nghĩa: biển phụ bổ sung gì cho biển chính
    "q4": "regulation_explanation",    # Giải thích quy định giao thông tổng thể
    "q5": "yes_no_compliance",         # Câu hỏi giải thích quy định dạng Yes/No (Không)
    "q6": "yes_no_compliance",         # Câu hỏi giải thích quy định dạng Yes/No (Có)
}


def main():
    VQA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QA_DETAILED_PATH, "r", encoding="utf-8") as f:
        qa_detailed = json.load(f)

    with open(LAW_DB_PATH, "r", encoding="utf-8") as f:
        law_db = json.load(f)

    # Gom 6 câu hỏi q1..q6 theo từng class_id (0..51)
    qa_by_class = {}
    for item in qa_detailed:
        cid = item["class_id"]
        q_suffix = item["id"].split("_")[-1]  # q1..q6
        qa_by_class.setdefault(cid, {})[q_suffix] = item

    sign_classes_info = {c["class_id"]: c for c in law_db["sign_classes_52"]}
    articles_dict = law_db["articles"]

    # Chọn tối đa N ảnh đại diện chất lượng cao (kích thước rõ nhất) mỗi lớp cho tập Train và Test VQA
    # để vừa bao phủ đủ 52/52 lớp ở cả Train và Test, vừa đảm bảo thời gian huấn luyện/đánh giá VLM hợp lý
    MAX_IMAGES_PER_CLASS_TRAIN = 3
    MAX_IMAGES_PER_CLASS_TEST = 1

    splits_data = {"train": [], "test": []}
    stats = {"train": Counter(), "test": Counter()}

    for split_name, max_imgs in [("train", MAX_IMAGES_PER_CLASS_TRAIN), ("test", MAX_IMAGES_PER_CLASS_TEST)]:
        sample_counter = 0
        for cid in range(52):
            cls_meta = sign_classes_info[cid]
            sign_type = cls_meta["sign_type"]  # 'bien_chinh' | 'bien_phu'
            raw_label = cls_meta["dataset_label"]
            safe_folder = raw_label
            for ch in '<>:"/\\|?*':
                safe_folder = safe_folder.replace(ch, "_")

            cls_folder = CROPPED_DIR / sign_type / split_name / safe_folder
            if not cls_folder.exists():
                continue

            # Ưu tiên lấy các ảnh có dung lượng file lớn nhất (ảnh rõ nét nhất trong tập đó)
            jpg_files = sorted(cls_folder.glob("*.jpg"), key=lambda p: (-p.stat().st_size, p.name))
            selected_files = jpg_files[:max_imgs]

            cls_qas = qa_by_class.get(cid, {})
            primary_aid = cls_meta["primary_article_id"]
            primary_art = articles_dict.get(primary_aid, {})

            for img_idx, img_path in enumerate(selected_files):
                rel_img_path = f"cropped_dataset/{sign_type}/{split_name}/{safe_folder}/{img_path.name}"
                for q_key in ["q1", "q2", "q3", "q4", "q5", "q6"]:
                    if q_key not in cls_qas:
                        continue
                    qa_obj = cls_qas[q_key]
                    sample_counter += 1
                    cat = QUESTION_CATEGORY_MAP[q_key]
                    stats[split_name][cat] += 1

                    splits_data[split_name].append({
                        "vqa_id": f"{split_name}_{cid:02d}_{img_idx+1:02d}_{q_key}",
                        "split": split_name,
                        "question_key": q_key,
                        "task_category": cat,
                        "question_type": qa_obj["question_type"],
                        "short_answer": qa_obj["short_answer"],
                        "class_id": cid,
                        "dataset_label": raw_label,
                        "canonical_sign_code": cls_meta["canonical_sign_code"],
                        "sign_type": sign_type,
                        "name_vie": cls_meta["name_vie"],
                        "image_path": rel_img_path,
                        "question": qa_obj["question"],
                        "answer": qa_obj["answer"],
                        "primary_article_id": primary_aid,
                        "relevant_article_ids": cls_meta["relevant_article_ids"],
                        "gold_legal_title": primary_art.get("title", ""),
                        "gold_legal_summary": primary_art.get("summary", ""),
                    })

    # Chọn 4 mẫu Few-shot đại diện từ tập Train (không trùng tập Test) cho Baseline 2:
    # 1. Biển đơn (P.102 - q1)
    # 2. Biển chính + biển phụ (S.505a*Xe tải - q2)
    # 3. Quan hệ giữa biển phụ và biển chính (P.130 - q3)
    # 4. Giải thích quy định giao thông & Yes/No (P.103a - q4 & q5)
    fewshot_examples = [
        {
            "category": "single_sign (Biển đơn)",
            "sign_code": "P.102",
            "question": "Biển chính trong ảnh là biển gì?",
            "answer": "Biển báo cấm đi ngược chiều (mã biển P.102).",
        },
        {
            "category": "main_and_aux (Biển chính + biển phụ)",
            "sign_code": "S.505a*Xe tải",
            "question": "Biển phụ có nội dung gì?",
            "answer": "Biển phụ là biển số S.505a có hình biểu tượng xe ô tô tải, chỉ định đối tượng áp dụng là xe tải.",
        },
        {
            "category": "sign_relationship (Quan hệ giữa các biển)",
            "sign_code": "P.130",
            "question": "Biển phụ bổ sung ý nghĩa gì cho biển chính?",
            "answer": "Biển phụ bổ sung thông tin về phạm vi không gian, khung thời gian hoặc loại phương tiện chịu hiệu lực của biển cấm dừng và đỗ xe.",
        },
        {
            "category": "regulation_explanation (Giải thích quy định giao thông)",
            "sign_code": "P.107",
            "question": "Hãy giải thích ý nghĩa tổng thể của cụm biển.",
            "answer": "Cụm biển quy định cấm cả xe ô tô chở khách và các loại xe ô tô tải (kể cả máy kéo và xe máy chuyên dùng, trừ xe ưu tiên) đi vào đoạn đường theo đúng phạm vi và điều kiện chỉ dẫn trên biển phụ.",
        },
        {
            "category": "yes_no_compliance (Tuân thủ quy định Có/Không)",
            "sign_code": "P.103a",
            "question": "Khi gặp biển cấm xe ô tô (P.103a), người điều khiển xe mô tô hai bánh (xe máy) và xe gắn máy có được phép đi vào đoạn đường đó không?",
            "answer": "Có. Theo quy định tại mục B.3 Phụ lục B QCVN 41:2024/BGTVT, biển P.103a không cấm xe máy hai bánh, xe gắn máy và xe thô sơ.",
        },
    ]

    with open(VQA_OUT_DIR / "vqa_train.json", "w", encoding="utf-8") as f:
        json.dump(splits_data["train"], f, ensure_ascii=False, indent=2)

    with open(VQA_OUT_DIR / "vqa_test.json", "w", encoding="utf-8") as f:
        json.dump(splits_data["test"], f, ensure_ascii=False, indent=2)

    with open(VQA_OUT_DIR / "fewshot_exemplars.json", "w", encoding="utf-8") as f:
        json.dump(fewshot_examples, f, ensure_ascii=False, indent=2)

    summary = {
        "train_total_samples": len(splits_data["train"]),
        "test_total_samples": len(splits_data["test"]),
        "train_by_category": dict(stats["train"]),
        "test_by_category": dict(stats["test"]),
        "fewshot_exemplars_count": len(fewshot_examples),
    }
    with open(VQA_OUT_DIR / "vqa_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[✓] Đã tạo tập VQA Train ({len(splits_data['train'])} mẫu) tại: {VQA_OUT_DIR / 'vqa_train.json'}")
    print(f"[✓] Đã tạo tập VQA Test  ({len(splits_data['test'])} mẫu) tại: {VQA_OUT_DIR / 'vqa_test.json'}")
    print(f"[✓] Đã tạo bộ Few-shot Exemplars ({len(fewshot_examples)} ví dụ) tại: {VQA_OUT_DIR / 'fewshot_exemplars.json'}")
    print("    Thống kê theo nhóm câu hỏi:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

