r"""
build_law_database.py
---------------------
Từ các file ảnh biển báo đã cắt trong `cropped_dataset/` (`bien_chinh` & `bien_phu`)
và danh sách nhãn `classes.txt`, trích xuất toàn bộ:
1. Mã biển chuẩn QCVN 41 (canonical sign codes) & thông tin phân nhóm (Biển chính / Biển phụ, Nhóm P, W, R, I, S).
2. Toàn bộ Điều luật (Điều 1 -> Điều 86) và các Mục quy chuẩn chi tiết tại Phụ lục B, C, D, E, F
   từ văn bản gốc `C:\ki8\DSP\datadsp\archive\51-bgtvt-kem.pdf`.
3. Ánh xạ đầy đủ từng lớp biển báo đã cắt với các Điều khoản chung & Mục Phụ lục tương ứng,
   xuất ra file `law_database.json`.
"""

import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict
import pymupdf

sys.stdout.reconfigure(encoding="utf-8")

ARCHIVE_DIR = Path(__file__).resolve().parent
BASE_DIR = ARCHIVE_DIR.parent
PDF_PATH = ARCHIVE_DIR / "51-bgtvt-kem.pdf"
CROPPED_DIR = ARCHIVE_DIR / "cropped_dataset"
SUMMARY_PATH = CROPPED_DIR / "dataset_summary.json"


def clean_pdf_text(text: str) -> str:
    """Làm sạch triệt để các ký tự điều khiển (\x01, \x02, \x03, \x04, \x08, \xa0) trong file 51-bgtvt-kem.pdf."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\xa0]+", " ", text)
    text = re.sub(r"B\s*i\s*ể\s*n", "Biển", text, flags=re.IGNORECASE)
    text = re.sub(r"s\s*ố", "số", text, flags=re.IGNORECASE)
    text = re.sub(r"H\s*ì\s*n\s*h", "Hình", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([PWRIS]|DP|IE)\s*\.\s*([0-9]{1,3})", r"\1.\2", text)
    text = re.sub(r"R\s*[\.,]\s*E\s*[\.,]\s*([0-9]{1,2})\s*([a-d]?)\b", r"R.E.\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"S\s*\.\s*([HG])\s*,\s*([0-9]{1,2})\s*([a-c]?)\b", r"S.\1,\2\3", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    return text


# Bảng ánh xạ từ nhãn trong classes.txt sang mã biển chuẩn QCVN 41:2024/BGTVT và mục Phụ lục
DATASET_TO_QCVN_INFO = {
    "W.224": {"code": "W.224", "section": "C.24", "qcvn_name": "Đường người đi bộ cắt ngang", "variant": None},
    "W.205c": {"code": "W.205c", "section": "C.5", "qcvn_name": "Đường giao nhau cùng mức của các đường đồng cấp (ngã ba bên trái)", "variant": "Ngã ba"},
    "P.102": {"code": "P.102", "section": "B.2", "qcvn_name": "Cấm đi ngược chiều", "variant": None},
    "R.302a": {"code": "R.302a", "section": "D.2", "qcvn_name": "Hướng phải đi vòng chướng ngại vật (sang bên phải)", "variant": "Vòng bên phải"},
    "W.205a": {"code": "W.205a", "section": "C.5", "qcvn_name": "Đường giao nhau cùng mức của các đường đồng cấp (ngã tư)", "variant": "Ngã tư"},
    "W.207": {"code": "W.207a", "section": "C.7", "qcvn_name": "Giao nhau với đường không ưu tiên", "variant": None},
    "W.201a": {"code": "W.201a", "section": "C.1", "qcvn_name": "Chỗ ngoặt nguy hiểm vòng bên trái", "variant": "Vòng bên trái"},
    "P.123a": {"code": "P.123a", "section": "B.21", "qcvn_name": "Cấm rẽ trái", "variant": None},
    "I.434a": {"code": "I.434a", "section": "E.34", "qcvn_name": "Bến xe buýt", "variant": None},
    "R.303": {"code": "R.303", "section": "D.3", "qcvn_name": "Nơi giao nhau chạy theo vòng xuyến", "variant": None},
    "P.130": {"code": "P.130", "section": "B.28", "qcvn_name": "Cấm dừng xe và đỗ xe", "variant": None},
    "I.409": {"code": "I.409", "section": "E.9", "qcvn_name": "Chỗ quay xe", "variant": None},
    "R.415a": {"code": "R.415a", "section": "D.17", "qcvn_name": "Biển gộp làn đường theo phương tiện", "variant": None},
    "W.245a": {"code": "W.245a", "section": "C.45", "qcvn_name": "Đi chậm", "variant": None},
    "P.106a*Xe tải": {"code": "P.106a", "section": "B.6", "qcvn_name": "Cấm xe ô tô tải", "variant": "Xe tải"},
    "W.203c": {"code": "W.203c", "section": "C.3", "qcvn_name": "Đường bị thu hẹp về phía phải", "variant": "Phía phải"},
    "P.117*": {"code": "P.117", "section": "B.17", "qcvn_name": "Hạn chế chiều cao", "variant": None},
    "P.124a*": {"code": "P.124a", "section": "B.22", "qcvn_name": "Cấm quay đầu xe", "variant": None},
    "P.107": {"code": "P.107", "section": "B.7", "qcvn_name": "Cấm xe ô tô khách và xe ô tô tải", "variant": None},
    "P.124d": {"code": "P.124d", "section": "B.22", "qcvn_name": "Cấm rẽ phải và quay đầu xe", "variant": None},
    "P.103a": {"code": "P.103a", "section": "B.3", "qcvn_name": "Cấm xe ô tô", "variant": None},
    "W.203b": {"code": "W.203b", "section": "C.3", "qcvn_name": "Đường bị thu hẹp về phía trái", "variant": "Phía trái"},
    "W.221b": {"code": "W.221b", "section": "C.21", "qcvn_name": "Đường có sóng mấp mô nhân tạo (gồ giảm tốc)", "variant": None},
    "P.111": {"code": "P.111a", "section": "B.11", "qcvn_name": "Cấm xe gắn máy / xe hai và ba bánh", "variant": None},
    "P.129": {"code": "P.129", "section": "B.27", "qcvn_name": "Kiểm tra", "variant": None},
    "S.505a*Xe máy": {"code": "S.505a", "section": "F.5", "qcvn_name": "Loại xe (Biển phụ chỉ loại xe chịu hiệu lực: Xe máy)", "variant": "Xe máy"},
    "W.246a": {"code": "W.246a", "section": "C.46", "qcvn_name": "Chú ý chướng ngại vật (vòng tránh sang hai bên)", "variant": "Hai bên"},
    "W.225": {"code": "W.225", "section": "C.25", "qcvn_name": "Trẻ em", "variant": None},
    "S.505a*Xe tải và công": {"code": "S.505a", "section": "F.5", "qcvn_name": "Loại xe (Biển phụ chỉ loại xe chịu hiệu lực: Xe tải và xe container)", "variant": "Xe tải và công"},
    "P.104": {"code": "P.104", "section": "B.4", "qcvn_name": "Cấm xe máy", "variant": None},
    "S.505a*Xe tải": {"code": "S.505a", "section": "F.5", "qcvn_name": "Loại xe (Biển phụ chỉ loại xe chịu hiệu lực: Xe tải)", "variant": "Xe tải"},
    "Camera": {"code": "Camera", "section": "CAMERA", "qcvn_name": "Biển chỉ dẫn đường có camera giám sát giao thông", "variant": None},
    "P.123b": {"code": "P.123b", "section": "B.21", "qcvn_name": "Cấm rẽ phải", "variant": None},
    "W.202b": {"code": "W.202b", "section": "C.2", "qcvn_name": "Nhiều chỗ ngoặt nguy hiểm liên tiếp (chỗ đầu tiên sang phải)", "variant": "Sang phải"},
    "P.108a": {"code": "P.108a", "section": "B.8A", "qcvn_name": "Cấm xe sơ-mi rơ-moóc", "variant": None},
    "B.8a": {"code": "P.108a", "section": "B.8A", "qcvn_name": "Cấm xe sơ-mi rơ-moóc", "variant": None},
    "P.137": {"code": "P.137", "section": "B.35", "qcvn_name": "Cấm rẽ trái, rẽ phải", "variant": None},
    "P.139": {"code": "P.139", "section": "B.37", "qcvn_name": "Cấm đi thẳng, rẽ phải", "variant": None},
    "W.205b": {"code": "W.205b", "section": "C.5", "qcvn_name": "Đường giao nhau cùng mức của các đường đồng cấp (ngã ba)", "variant": "Ngã ba"},
    "P.127*50": {"code": "P.127", "section": "B.25", "qcvn_name": "Tốc độ tối đa cho phép (50 km/h)", "variant": "50 km/h"},
    "P.127*60": {"code": "P.127", "section": "B.25", "qcvn_name": "Tốc độ tối đa cho phép (60 km/h)", "variant": "60 km/h"},
    "P.127*80": {"code": "P.127", "section": "B.25", "qcvn_name": "Tốc độ tối đa cho phép (80 km/h)", "variant": "80 km/h"},
    "P.127*40": {"code": "P.127", "section": "B.25", "qcvn_name": "Tốc độ tối đa cho phép (40 km/h)", "variant": "40 km/h"},
    "R.301e": {"code": "R.301e", "section": "D.1", "qcvn_name": "Các xe chỉ được rẽ trái (đặt sau ngã ba, ngã tư)", "variant": "Rẽ trái sau nút giao"},
    "W.239b*": {"code": "W.239b", "section": "C.39", "qcvn_name": "Chiều cao tĩnh không thực tế", "variant": None},
    "W.233": {"code": "W.233", "section": "C.33", "qcvn_name": "Nguy hiểm khác", "variant": None},
    "I.407a": {"code": "I.407a", "section": "E.7", "qcvn_name": "Đường một chiều (đi thẳng)", "variant": None},
    "P.131a": {"code": "P.131a", "section": "B.29", "qcvn_name": "Cấm đỗ xe", "variant": None},
    "P.124b1": {"code": "P.124b", "section": "B.22", "qcvn_name": "Cấm ô tô quay đầu xe (được rẽ trái)", "variant": None},
    "W.210": {"code": "W.210", "section": "C.10", "qcvn_name": "Giao nhau với đường sắt có rào chắn", "variant": None},
    "P.124c": {"code": "P.124c", "section": "B.22", "qcvn_name": "Cấm rẽ trái và quay đầu xe", "variant": None},
    "W.201b": {"code": "W.201b", "section": "C.1", "qcvn_name": "Chỗ ngoặt nguy hiểm vòng bên phải", "variant": "Vòng bên phải"},
    "W.246c": {"code": "W.246c", "section": "C.46", "qcvn_name": "Chú ý chướng ngại vật (vòng tránh sang bên trái/phải)", "variant": "Một bên"},
}

GROUP_GENERAL_ARTICLES = {
    "P": ["QCVN41_Điều_11", "QCVN41_Điều_15", "QCVN41_Điều_21", "QCVN41_Điều_22", "QCVN41_Điều_26"],
    "W": ["QCVN41_Điều_11", "QCVN41_Điều_15", "QCVN41_Điều_27", "QCVN41_Điều_28", "QCVN41_Điều_30"],
    "R": ["QCVN41_Điều_11", "QCVN41_Điều_15", "QCVN41_Điều_31", "QCVN41_Điều_32", "QCVN41_Điều_34"],
    "I": ["QCVN41_Điều_11", "QCVN41_Điều_15", "QCVN41_Điều_35", "QCVN41_Điều_36", "QCVN41_Điều_40"],
    "S": ["QCVN41_Điều_11", "QCVN41_Điều_15", "QCVN41_Điều_41", "QCVN41_Điều_42", "QCVN41_Điều_44"],
}

APPENDIX_TITLES = {
    "B": "Phụ lục B - Ý nghĩa và sử dụng Biển báo cấm (Nhóm P)",
    "C": "Phụ lục C - Ý nghĩa và sử dụng Biển báo nguy hiểm và cảnh báo (Nhóm W)",
    "D": "Phụ lục D - Ý nghĩa và sử dụng Biển hiệu lệnh (Nhóm R)",
    "E": "Phụ lục E - Ý nghĩa và sử dụng Biển chỉ dẫn (Nhóm I)",
    "F": "Phụ lục F - Ý nghĩa và sử dụng Các biển phụ (Nhóm S)",
}


def get_sign_group_info(code: str):
    if code.startswith("P.") or code.startswith("DP."):
        return "P", "Biển báo cấm (Nhóm P)", "bien_chinh"
    if code.startswith("W."):
        return "W", "Biển báo nguy hiểm và cảnh báo (Nhóm W)", "bien_chinh"
    if code.startswith("R."):
        return "R", "Biển hiệu lệnh (Nhóm R)", "bien_chinh"
    if code.startswith("S."):
        return "S", "Biển phụ (Nhóm S)", "bien_phu"
    return "I", "Biển chỉ dẫn (Nhóm I)", "bien_chinh"


def parse_pdf_regulations(doc: pymupdf.Document):
    """
    Trích xuất:
    1. Các Điều luật từ trang 4 đến trang 79 (Điều 1 -> Điều 86).
    2. Các Mục Phụ lục B, C, D, E, F từ trang 80 đến trang 157.
    """
    articles = {}

    # --- 1. Trích xuất Điều 1 -> Điều 86 (Trang 4 - 79) ---
    dieu_pat = re.compile(r"^Điều\s+(\d+)\.\s*(.*)", re.IGNORECASE)
    chuong_pat = re.compile(r"^Chương\s+([IVXLC]+)\b\s*(.*)", re.IGNORECASE)

    current_chapter = "Chương I - Quy định chung"
    current_dieu_num = None
    current_dieu_title = ""
    current_dieu_lines = []
    current_dieu_pages = set()

    def save_current_dieu():
        if current_dieu_num is None:
            return
        art_id = f"QCVN41_Điều_{current_dieu_num}"
        full_txt = " ".join(current_dieu_lines).strip()
        sentences = re.split(r"(?<=[\.\;])\s+", full_txt)
        summary = " ".join(sentences[:2]).strip() if sentences else full_txt[:300]
        articles[art_id] = {
            "article_id": art_id,
            "article_type": "dieu_luat",
            "law_id": "QCVN 41:2024/BGTVT",
            "source_pdf": "archive/51-bgtvt-kem.pdf",
            "section": f"Điều {current_dieu_num}",
            "chapter_or_appendix": current_chapter,
            "title": f"Điều {current_dieu_num}. {current_dieu_title}",
            "pages": sorted(list(current_dieu_pages)),
            "summary": summary[:400],
            "full_text": full_txt,
        }

    for p_idx in range(3, 79):
        page_num = p_idx + 1
        raw_text = doc[p_idx].get_text()
        for raw_line in raw_text.splitlines():
            line = clean_pdf_text(raw_line).strip()
            if not line or line == str(page_num):
                continue
            m_ch = chuong_pat.match(line)
            if m_ch:
                current_chapter = line
                continue
            m_d = dieu_pat.match(line)
            if m_d:
                save_current_dieu()
                current_dieu_num = int(m_d.group(1))
                current_dieu_title = m_d.group(2).strip()
                current_dieu_lines = [line]
                current_dieu_pages = {page_num}
            elif current_dieu_num is not None:
                current_dieu_lines.append(line)
                current_dieu_pages.add(page_num)

    save_current_dieu()

    # --- 2. Trích xuất Phụ lục B, C, D, E, F (Trang 80 - 157) ---
    sec_pat = re.compile(r"^([BCDEF]\.[0-9]+[a-z]?)\b[\.\s]*(.*)", re.IGNORECASE)
    current_sec = None
    current_sec_lines = []
    current_sec_pages = set()

    def save_current_sec():
        if not current_sec:
            return
        app_letter = current_sec[0].upper()
        art_id = f"QCVN41_{current_sec}"
        lines = [l.strip() for l in current_sec_lines if l.strip()]
        header_title = lines[1] if len(lines) > 1 else lines[0]
        body_text = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
        sentences = re.split(r"(?<=[\.\;])\s+", body_text)
        summary = " ".join(sentences[:2]).strip() if sentences else body_text[:350]
        mentioned_codes = sorted(
            list(
                set(
                    re.findall(
                        r"\b(?:[PWRIS]|DP|IE)\.[0-9]{1,3}[a-pA-P]?\b|\bS\.[HG],[0-9]{1,2}[a-c]?\b",
                        body_text,
                    )
                )
            )
        )
        articles[art_id] = {
            "article_id": art_id,
            "article_type": "phu_luc",
            "law_id": "QCVN 41:2024/BGTVT",
            "source_pdf": "archive/51-bgtvt-kem.pdf",
            "section": current_sec,
            "chapter_or_appendix": APPENDIX_TITLES.get(app_letter, f"Phụ lục {app_letter}"),
            "title": f"Phụ lục {app_letter} - Mục {current_sec}: {header_title}",
            "pages": sorted(list(current_sec_pages)),
            "sign_codes_in_section": mentioned_codes,
            "summary": summary[:400],
            "full_text": body_text,
        }

    for p_idx in range(79, 157):
        page_num = p_idx + 1
        raw_text = doc[p_idx].get_text()
        for raw_line in raw_text.splitlines():
            line = clean_pdf_text(raw_line).strip()
            if not line or line == str(page_num):
                continue
            m_s = sec_pat.match(line)
            if m_s:
                save_current_sec()
                current_sec = m_s.group(1).upper()
                rest = m_s.group(2).strip()
                current_sec_lines = [current_sec] + ([rest] if rest else [])
                current_sec_pages = {page_num}
            elif current_sec:
                current_sec_lines.append(line)
                current_sec_pages.add(page_num)

    save_current_sec()

    # Bổ sung mục Camera giám sát
    articles["QCVN41_CAMERA"] = {
        "article_id": "QCVN41_CAMERA",
        "article_type": "phu_luc",
        "law_id": "QCVN 41:2024/BGTVT",
        "source_pdf": "archive/51-bgtvt-kem.pdf",
        "section": "CAMERA",
        "chapter_or_appendix": APPENDIX_TITLES["I"] if "I" in APPENDIX_TITLES else APPENDIX_TITLES["E"],
        "title": "Quy chuẩn QCVN 41:2024/BGTVT - Biển chỉ dẫn: Đường có camera giám sát giao thông",
        "pages": [31, 130],
        "sign_codes_in_section": ["Camera"],
        "summary": "Biển chỉ dẫn đoạn đường có hệ thống camera giám sát giao thông, nhắc nhở người điều khiển phương tiện chấp hành nghiêm quy tắc giao thông đường bộ.",
        "full_text": "Biển báo 'Đường có camera giám sát' thuộc nhóm biển chỉ dẫn trên đường bộ theo QCVN 41:2024/BGTVT (Chương 7 - Điều 35), thông báo cho người tham gia giao thông biết khu vực hoặc tuyến đường phía trước có lắp đặt hệ thống camera giám sát giao thông để chủ động tuân thủ tốc độ, làn đường và hiệu lệnh báo hiệu đường bộ.",
    }

    return articles


def main():
    print(f"[*] Đang đọc tài liệu PDF: {PDF_PATH}")
    doc = pymupdf.open(str(PDF_PATH))
    articles_dict = parse_pdf_regulations(doc)

    dieu_cnt = sum(1 for a in articles_dict.values() if a["article_type"] == "dieu_luat")
    phuluc_cnt = sum(1 for a in articles_dict.values() if a["article_type"] == "phu_luc")
    print(f"[✓] Đã trích xuất {len(articles_dict)} điều khoản từ PDF ({dieu_cnt} Điều luật + {phuluc_cnt} Mục Phụ lục B, C, D, E, F).")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        crop_summary = json.load(f)

    # Kiểm tra ảnh mẫu PDF trong data/pdf_signs (nếu có)
    pdf_signs_dir = BASE_DIR / "data" / "pdf_signs"

    extracted_classes = []
    canonical_signs_map = {}

    for cls_info in crop_summary["per_class_counts"]:
        cid = cls_info["class_id"]
        group_cid = cls_info["group_class_id"]
        sign_type = cls_info["sign_type"]  # 'bien_chinh' | 'bien_phu'
        raw_label = cls_info["label"]
        safe_folder = cls_info["safe_folder"]
        name_vie = cls_info["label_vie"]
        name_en = cls_info["label_en"]

        qcvn_meta = DATASET_TO_QCVN_INFO.get(raw_label, {
            "code": raw_label.split("*")[0],
            "section": "B.1",
            "qcvn_name": name_vie,
            "variant": None,
        })
        canonical_code = qcvn_meta["code"]
        sec_code = qcvn_meta["section"]
        primary_art_id = f"QCVN41_{sec_code}"
        if primary_art_id not in articles_dict:
            # Tìm mục phụ lục chứa mã biển
            for aid, aobj in articles_dict.items():
                if aobj["article_type"] == "phu_luc" and canonical_code in aobj.get("sign_codes_in_section", []):
                    primary_art_id = aid
                    sec_code = aobj["section"]
                    break

        grp_code, grp_name, _ = get_sign_group_info(canonical_code)
        general_art_ids = [aid for aid in GROUP_GENERAL_ARTICLES.get(grp_code, []) if aid in articles_dict]
        relevant_art_ids = [primary_art_id] + [aid for aid in general_art_ids if aid != primary_art_id]

        primary_art = articles_dict.get(primary_art_id, {})

        # Lấy danh sách mẫu đường dẫn ảnh đã crop trong train và test
        train_folder_rel = f"cropped_dataset/{sign_type}/train/{safe_folder}"
        test_folder_rel = f"cropped_dataset/{sign_type}/test/{safe_folder}"
        train_samples = sorted([
            f"{train_folder_rel}/{p.name}"
            for p in (CROPPED_DIR / sign_type / "train" / safe_folder).glob("*.jpg")
        ])[:5]
        test_samples = sorted([
            f"{test_folder_rel}/{p.name}"
            for p in (CROPPED_DIR / sign_type / "test" / safe_folder).glob("*.jpg")
        ])[:5]

        pdf_templates = []
        if (pdf_signs_dir / canonical_code).exists():
            pdf_templates = sorted([
                f"data/pdf_signs/{canonical_code}/{p.name}"
                for p in (pdf_signs_dir / canonical_code).glob("*.png")
            ])

        cls_entry = {
            "class_id": cid,
            "group_class_id": group_cid,
            "sign_type": sign_type,
            "sign_type_name": "Biển phụ (Nhóm S)" if sign_type == "bien_phu" else "Biển chính (Nhóm P, W, R, I)",
            "dataset_label": raw_label,
            "canonical_sign_code": canonical_code,
            "variant_detail": qcvn_meta["variant"],
            "qcvn_group_code": grp_code,
            "qcvn_group_name": grp_name,
            "name_vie": name_vie,
            "official_qcvn_name": qcvn_meta["qcvn_name"],
            "name_en": name_en,
            "cropped_folders": {
                "train_dir": train_folder_rel,
                "test_dir": test_folder_rel,
                "train_crops_count": cls_info["train_crops"],
                "test_crops_count": cls_info["test_crops"],
                "total_crops_count": cls_info["total_crops"],
                "sample_train_crops": train_samples,
                "sample_test_crops": test_samples,
            },
            "pdf_template_images": pdf_templates,
            "primary_article_id": primary_art_id,
            "relevant_article_ids": relevant_art_ids,
            "legal_regulation": {
                "article_id": primary_art_id,
                "section": primary_art.get("section", sec_code),
                "chapter_or_appendix": primary_art.get("chapter_or_appendix", ""),
                "title": primary_art.get("title", ""),
                "pages_in_pdf": primary_art.get("pages", []),
                "summary": primary_art.get("summary", ""),
                "full_text": primary_art.get("full_text", ""),
            },
        }
        extracted_classes.append(cls_entry)

        # Gom theo mã biển chuẩn QCVN (canonical_sign_code)
        if canonical_code not in canonical_signs_map:
            canonical_signs_map[canonical_code] = {
                "canonical_sign_code": canonical_code,
                "sign_type": sign_type,
                "qcvn_group_code": grp_code,
                "qcvn_group_name": grp_name,
                "official_qcvn_name": qcvn_meta["qcvn_name"],
                "dataset_classes": [],
                "total_train_crops": 0,
                "total_test_crops": 0,
                "total_crops": 0,
                "primary_article_id": primary_art_id,
                "relevant_article_ids": relevant_art_ids,
                "legal_regulation": cls_entry["legal_regulation"],
            }
        canonical_signs_map[canonical_code]["dataset_classes"].append({
            "class_id": cid,
            "dataset_label": raw_label,
            "variant_detail": qcvn_meta["variant"],
            "train_crops": cls_info["train_crops"],
            "test_crops": cls_info["test_crops"],
            "total_crops": cls_info["total_crops"],
        })
        canonical_signs_map[canonical_code]["total_train_crops"] += cls_info["train_crops"]
        canonical_signs_map[canonical_code]["total_test_crops"] += cls_info["test_crops"]
        canonical_signs_map[canonical_code]["total_crops"] += cls_info["total_crops"]

    law_database = {
        "metadata": {
            "law_id": "QCVN 41:2024/BGTVT",
            "law_name": "Quy chuẩn kỹ thuật Quốc gia về báo hiệu đường bộ (National Technical Regulation on Traffic Signs and Signals)",
            "source_pdf": "C:/ki8/DSP/datadsp/archive/51-bgtvt-kem.pdf",
            "total_pdf_pages": len(doc),
            "total_cropped_images": crop_summary["groups"]["bien_chinh"]["total_crops"] + crop_summary["groups"]["bien_phu"]["total_crops"],
            "total_dataset_classes": len(extracted_classes),
            "bien_chinh_classes": crop_summary["num_classes_bien_chinh"],
            "bien_phu_classes": crop_summary["num_classes_bien_phu"],
            "unique_canonical_sign_codes": len(canonical_signs_map),
            "total_legal_articles_extracted": len(articles_dict),
            "general_dieu_articles_count": dieu_cnt,
            "appendix_sections_count": phuluc_cnt,
        },
        "sign_classes_52": extracted_classes,
        "canonical_signs": list(canonical_signs_map.values()),
        "articles": articles_dict,
    }

    output_paths = [
        ARCHIVE_DIR / "law_database.json",
        CROPPED_DIR / "law_database.json",
        BASE_DIR / "law_database.json",
        BASE_DIR / "data" / "law_database.json",
    ]
    for out_p in output_paths:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(law_database, f, ensure_ascii=False, indent=2)
        print(f"[✓] Đã lưu law_database.json tại: {out_p}")


if __name__ == "__main__":
    main()
