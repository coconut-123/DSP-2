# -*- coding: utf-8 -*-
"""
server.py
---------
FastAPI Backend Server dùng DUY NHẤT mô hình vừa Fine-tune:
- Base Model : Qwen/Qwen2-VL-2B-Instruct
- LoRA Adapter: qwen_vlm_freeform_lora_adapter (đã huấn luyện trên bộ Free-form & Yes/No VQA 52 mã biển)
- Cơ sở dữ liệu luật: law_database.json (trích xuất từ QCVN 41:2024/BGTVT - 51-bgtvt-kem.pdf)
- Bộ câu hỏi gợi ý: qa_52_signs_detailed.json (6 câu hỏi q1..q6 cho mỗi biển)
"""

import os
import io
import re
import time
import json
import base64
import threading
import webbrowser
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

CURRENT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = CURRENT_DIR if (CURRENT_DIR / "qwen_vlm_freeform_lora_adapter").exists() else (CURRENT_DIR / "archive")
WEB_UI_DIR = CURRENT_DIR if (CURRENT_DIR / "index.html").exists() else CURRENT_DIR.parent

ADAPTER_DIR = ARCHIVE_DIR / "qwen_vlm_freeform_lora_adapter"
LAW_DB_PATH = ARCHIVE_DIR / "law_database.json"
QA_DETAILED_PATH = ARCHIVE_DIR / "qa_52_signs_detailed.json"
BASE_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

vlm_model = None
vlm_processor = None
vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
model_loaded = False
adapter_loaded = False
_vlm_lock = threading.Lock()

law_db: Dict[str, Any] = {}
articles_by_id: Dict[str, Dict[str, Any]] = {}
signs_catalog: Dict[str, Dict[str, Any]] = {}
qa_by_sign: Dict[str, List[Dict[str, Any]]] = {}
last_active_sign_code: str = "P.103a"


def get_category_by_code(code: str) -> str:
    if code.startswith("P.") or code.startswith("DP."):
        return "Biển báo cấm (Nhóm P)"
    elif code.startswith("W."):
        return "Biển báo nguy hiểm và cảnh báo (Nhóm W)"
    elif code.startswith("R."):
        return "Biển hiệu lệnh (Nhóm R)"
    elif code.startswith("I.") or code.startswith("IE.") or code == "Camera":
        return "Biển chỉ dẫn (Nhóm I)"
    elif code.startswith("S."):
        return "Biển phụ (Nhóm S)"
    return "Biển báo hiệu đường bộ"


def image_to_data_url(img: Image.Image, max_size: int = 160) -> str:
    """Chuyển ảnh PIL sang chuỗi base64 data URL để hiển thị thumbnail không bao giờ bị lỗi vỡ ảnh."""
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_archive_knowledge_base():
    """Nạp law_database.json và qa_52_signs_detailed.json của 52 mã biển."""
    global law_db, articles_by_id, signs_catalog, qa_by_sign
    if LAW_DB_PATH.exists():
        with open(LAW_DB_PATH, "r", encoding="utf-8") as f:
            law_db = json.load(f)
        raw_arts = law_db.get("articles", {})
        if isinstance(raw_arts, dict):
            articles_by_id.update(raw_arts)
        elif isinstance(raw_arts, list):
            for art in raw_arts:
                if isinstance(art, dict) and art.get("article_id"):
                    articles_by_id[art["article_id"]] = art

        for cls_item in law_db.get("sign_classes_52", []):
            ds_label = cls_item.get("dataset_label", "")
            can_code = cls_item.get("canonical_sign_code", ds_label)
            cf = cls_item.get("cropped_folders", {})
            test_imgs = cf.get("sample_test_crops", []) or cf.get("sample_train_crops", [])
            img_rel = test_imgs[0] if test_imgs else f"images/{can_code}/{can_code}_01.png"
            entry = {
                "sign_code": can_code,
                "dataset_label": ds_label,
                "canonical_sign_code": can_code,
                "sign_name": cls_item.get("official_qcvn_name") or cls_item.get("name_vie", ds_label),
                "name_vie": cls_item.get("name_vie", ds_label),
                "category": get_category_by_code(can_code),
                "sign_type": cls_item.get("sign_type", "bien_chinh"),
                "image": img_rel,
                "relevant_articles": cls_item.get("relevant_article_ids", []),
            }
            signs_catalog[ds_label] = entry
            if can_code not in signs_catalog:
                signs_catalog[can_code] = entry

    if QA_DETAILED_PATH.exists():
        with open(QA_DETAILED_PATH, "r", encoding="utf-8") as f:
            qa_list = json.load(f)
        for q_item in qa_list:
            lbl = q_item.get("dataset_label", "")
            can = q_item.get("canonical_sign_code", "")
            if lbl:
                qa_by_sign.setdefault(lbl, []).append(q_item)
            if can and can != lbl:
                qa_by_sign.setdefault(can, []).append(q_item)

    print(
        f"[✓] Đã nạp {len(law_db.get('sign_classes_52', []))} mã biển và {len(articles_by_id)} điều khoản từ {LAW_DB_PATH.name}."
    )


load_archive_knowledge_base()


# Bảng ánh xạ từ cụm từ tên biển tiếng Việt mà VLM nhận diện được sang đúng mã chuẩn QCVN 41
VIET_NAME_TO_CODE_RULES: List[Tuple[List[str], List[str], str]] = [
    (["cấm ô tô khách và ô tô tải", "cấm xe ô tô khách và xe ô tô tải"], [], "P.107"),
    (["cấm xe ô tô tải", "cấm ô tô tải", "cấm xe tải"], ["khách", "công"], "P.106a*Xe tải"),
    (["cấm sơ-mi rơ-moóc", "cấm xe sơ-mi rơ-moóc", "rơ-moóc"], [], "P.108a"),
    (["cấm ô tô quay đầu", "cấm xe ô tô quay đầu"], [], "P.124b1"),
    (["cấm xe ô tô", "cấm ô tô"], ["tải", "khách", "quay đầu", "rẽ", "máy"], "P.103a"),
    (["cấm mô tô và xe máy", "cấm xe máy", "cấm xe mô tô"], ["hai và ba bánh"], "P.104"),
    (["cấm xe hai và ba bánh", "hai và ba bánh"], [], "P.111"),
    (["cấm đi ngược chiều", "ngược chiều"], [], "P.102"),
    (["cấm dừng và đỗ xe", "cấm dừng xe và đỗ xe"], [], "P.130"),
    (["cấm đỗ xe"], ["dừng và đỗ"], "P.131a"),
    (["cấm rẽ trái và quay đầu"], [], "P.124c"),
    (["cấm rẽ phải và quay đầu"], [], "P.124d"),
    (["cấm rẽ trái và phải", "cấm rẽ trái và rẽ phải"], [], "P.137"),
    (["cấm đi thẳng và rẽ phải"], [], "P.139"),
    (["cấm quay đầu"], ["rẽ", "ô tô quay đầu"], "P.124a*"),
    (["cấm rẽ trái"], ["quay đầu", "phải"], "P.123a"),
    (["cấm rẽ phải"], ["quay đầu", "trái", "đi thẳng"], "P.123b"),
    (["hạn chế chiều cao", "giới hạn chiều cao"], [], "P.117*"),
    (["tốc độ tối đa cho phép (40", "tối đa 40", "40 km/h"], [], "P.127*40"),
    (["tốc độ tối đa cho phép (50", "tối đa 50", "50 km/h"], [], "P.127*50"),
    (["tốc độ tối đa cho phép (60", "tối đa 60", "60 km/h"], [], "P.127*60"),
    (["tốc độ tối đa cho phép (80", "tối đa 80", "80 km/h"], [], "P.127*80"),
    (["kiểm tra"], [], "P.129"),
    (["người đi bộ cắt ngang", "nhường đường cho người đi bộ"], [], "W.224"),
    (["trẻ em"], [], "W.225"),
    (["đi chậm"], [], "W.245a"),
    (["gồ giảm tốc"], [], "W.221b"),
    (["đường sắt có rào chắn"], [], "W.210"),
    (["thu hẹp về phía trái"], [], "W.203b"),
    (["thu hẹp về phía phải"], [], "W.203c"),
    (["ngoặt nguy hiểm vòng bên trái"], [], "W.201a"),
    (["ngoặt nguy hiểm vòng bên phải"], [], "W.201b"),
    (["nhiều chỗ ngoặt nguy hiểm"], [], "W.202b"),
    (["vòng xuyến"], [], "R.303"),
    (["vòng sang bên phải"], ["chướng ngại vật"], "R.302a"),
    (["chỉ được rẽ trái"], [], "R.301e"),
    (["gộp làn đường theo phương tiện"], [], "R.415a"),
    (["bến xe buýt"], [], "I.434a"),
    (["chỗ quay xe"], [], "I.409"),
    (["đường một chiều"], [], "I.407a"),
    (["camera giám sát"], [], "Camera"),
    (["ngã ba bên phải"], [], "W.205c"),
    (["ngã ba bên trái"], [], "W.205b"),
    (["đường đồng cấp dạng ngã tư", "giao nhau với đường đồng cấp"], ["ngã ba"], "W.205a"),
    (["giao nhau với đường không ưu tiên"], [], "W.207"),
    (["vòng tránh sang hai bên"], [], "W.246a"),
    (["vòng tránh sang một bên"], [], "W.246c"),
    (["chiều cao tĩnh không"], [], "W.239b*"),
    (["nguy hiểm khác"], [], "W.233"),
    (["s.505a*xe tải và công", "xe ô tô tải và xe sơ-mi rơ-moóc"], [], "S.505a*Xe tải và công"),
    (["s.505a*xe máy", "biểu tượng xe mô tô (xe máy)"], [], "S.505a*Xe máy"),
    (["s.505a*xe tải", "biểu tượng xe ô tô tải, chỉ định"], ["công", "rơ-moóc"], "S.505a*Xe tải"),
]


def match_sign_code_by_vietnamese_name(text: str) -> Optional[str]:
    """Khớp tên biển tiếng Việt trong câu trả lời của VLM với đúng mã biển trong 52 lớp."""
    t = text.lower()
    for must_have_any, must_not_have, target_code in VIET_NAME_TO_CODE_RULES:
        if any(phrase in t for phrase in must_have_any):
            if not any(bad in t for bad in must_not_have):
                return target_code
    return None


def extract_sign_code_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b((?:P|DP|W|R|I|IE|S)\.[0-9]{3}[a-e]?(?:\*[^\s\)\.,]+)?)\b", text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        parts = raw.split(".")
        prefix = parts[0].upper()
        suffix = parts[1]
        return f"{prefix}.{suffix}"
    if "camera" in text.lower():
        return "Camera"
    return None


def resolve_valid_sign_and_fix_answer(vlm_text: str, q1_text: str = "") -> Tuple[Optional[str], str]:
    """
    Kiểm tra chéo giữa tên biển tiếng Việt mà VLM nhận diện được và mã biển trong ngoặc:
    - Nếu VLM đọc đúng tên biển ("Biển báo cấm xe ô tô") nhưng ghi nhầm mã không tồn tại ("P.105a"),
      hàm tự động sửa về đúng mã chuẩn trong 52 mã biển ("P.103a").
    """
    combined_for_name = f"{q1_text} {vlm_text}"
    code_by_name = match_sign_code_by_vietnamese_name(combined_for_name)
    raw_code = extract_sign_code_from_text(vlm_text) or extract_sign_code_from_text(q1_text)

    valid_code = None
    if code_by_name and code_by_name in signs_catalog:
        valid_code = code_by_name
    elif raw_code and raw_code in signs_catalog:
        valid_code = raw_code

    fixed_text = vlm_text
    if valid_code:
        canonical = signs_catalog[valid_code]["canonical_sign_code"]
        if raw_code and raw_code != canonical and raw_code in fixed_text:
            fixed_text = fixed_text.replace(raw_code, canonical)

    return valid_code, fixed_text


def build_proposed_finetuned_prompt(question: str) -> str:
    """Prompt chuẩn khớp 100% với lúc Fine-tune model qwen_vlm_freeform_lora_adapter."""
    return (
        "Hãy quan sát kỹ biển báo giao thông trong ảnh và trả lời ngắn gọn, chính xác "
        "câu hỏi sau bằng tiếng Việt theo QCVN 41:2024/BGTVT.\n"
        f"Câu hỏi: {question}\n"
        "Trả lời:"
    )


def init_vlm_model():
    """Khởi tạo duy nhất mô hình Qwen2-VL-2B-Instruct + qwen_vlm_freeform_lora_adapter."""
    global vlm_model, vlm_processor, vlm_device, model_loaded, adapter_loaded
    if model_loaded:
        return

    vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from peft import PeftModel

        proc_src = str(ADAPTER_DIR) if (ADAPTER_DIR / "processor_config.json").exists() else BASE_MODEL_NAME
        vlm_processor = AutoProcessor.from_pretrained(
            proc_src,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        safetensors_path = ADAPTER_DIR / "adapter_model.safetensors"
        if safetensors_path.exists():
            print(f"[*] Đang nạp trọng số LoRA vừa train từ: {ADAPTER_DIR}")
            peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
            merged_model = peft_model.merge_and_unload()
            del peft_model, base_model
            vlm_model = merged_model.to(vlm_device)
            adapter_loaded = True
        else:
            raise FileNotFoundError(f"Không tìm thấy {safetensors_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vlm_model.eval()
        model_loaded = True
        print(f"[✓] Model mới train (qwen_vlm_freeform_lora_adapter) đã sẵn sàng trên {vlm_device}!")
    except Exception as e:
        print(f"[!] Lỗi khi nạp model: {e}")
        model_loaded = False
        adapter_loaded = False


def vlm_generate(image: Image.Image, question: str, max_new_tokens: int = 160) -> Tuple[str, float]:
    """Suy luận trực tiếp từ ảnh + câu hỏi bằng model Qwen2-VL đã fine-tune."""
    t0 = time.perf_counter()
    prompt_text = build_proposed_finetuned_prompt(question)
    rgb_img = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    with _vlm_lock, torch.inference_mode():
        chat_text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[chat_text],
            images=[rgb_img],
            padding=True,
            return_tensors="pt",
        ).to(vlm_model.device)

        out = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        answer = vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    elapsed = round(time.perf_counter() - t0, 3)
    return answer, elapsed


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vlm_model()
    print("\n" + "=" * 70)
    print(" SERVER MODEL MỚI TRAIN ĐÃ SẴN SÀNG! MỞ TRÌNH DUYỆT TẠI:")
    print(" 👉 http://localhost:8000  (hoặc http://127.0.0.1:8000)")
    print(f" 📦 LoRA Adapter: {ADAPTER_DIR}")
    print("=" * 70 + "\n")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8000")).start()
    yield


app = FastAPI(
    title="Vietnamese Traffic Sign Free-form VQA Server (Fine-tuned Qwen2-VL)",
    description="Server chạy trực tiếp model Qwen2-VL-2B-Instruct + qwen_vlm_freeform_lora_adapter vừa train",
    version="5.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.endswith((".js", ".html", ".css")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "backend": "Fine-tuned Qwen2-VL-2B-Instruct (qwen_vlm_freeform_lora_adapter)",
        "vlm_model_loaded": model_loaded,
        "adapter_loaded": adapter_loaded,
        "base_model": BASE_MODEL_NAME,
        "adapter_dir": str(ADAPTER_DIR),
        "device": vlm_device,
        "total_signs": len(law_db.get("sign_classes_52", [])),
        "total_articles": len(articles_by_id),
    }


@app.get("/api/legal_ir")
async def query_legal_ir(
    query: str = Query(..., description="Câu hỏi pháp lý giao thông"),
    sign_code: Optional[str] = Query(None, description="Mã biển báo"),
    top_k: int = Query(3, description="Số điều luật trả về"),
):
    results = []
    if sign_code and sign_code in signs_catalog:
        for aid in signs_catalog[sign_code].get("relevant_articles", [])[:top_k]:
            if aid in articles_by_id:
                results.append(articles_by_id[aid])
    if not results:
        q_lower = query.lower()
        for art in articles_by_id.values():
            if any(w in art.get("title", "").lower() or w in art.get("summary", "").lower() for w in q_lower.split()[:4]):
                results.append(art)
                if len(results) >= top_k:
                    break
    return {"query": query, "visual_sign_code": sign_code, "results": results}


@app.post("/api/chat")
async def chat_multimodal(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None),
    sign_code: Optional[str] = Form(None),
    query: str = Form("Biển chính trong ảnh là biển gì?"),
):
    global last_active_sign_code
    t_start = time.perf_counter()
    image = None
    uploaded_new_image = False

    # 1. Đọc hình ảnh người dùng gửi lên
    try:
        if file and file.filename:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            uploaded_new_image = True
        elif image_base64:
            img_str = image_base64.strip()
            if img_str.startswith("data:image") or len(img_str) > 260:
                if "," in img_str:
                    img_str = img_str.split(",")[1]
                img_bytes = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                uploaded_new_image = True
            else:
                rel_clean = img_str.lstrip("/\\")
                for base_candidate in (ARCHIVE_DIR, WEB_UI_DIR):
                    local_path = base_candidate / rel_clean
                    if local_path.exists():
                        image = Image.open(local_path).convert("RGB")
                        uploaded_new_image = True
                        break
    except Exception as e:
        print(f"[!] Lỗi đọc ảnh đầu vào: {e}")
        image = None

    if image is None:
        active_info = signs_catalog.get(sign_code or last_active_sign_code, {})
        rel_img = active_info.get("image", "")
        for base_candidate in (ARCHIVE_DIR, WEB_UI_DIR):
            if rel_img and (base_candidate / rel_img).exists():
                image = Image.open(base_candidate / rel_img).convert("RGB")
                break
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

    q_clean = query.strip() or "Biển chính trong ảnh là biển gì?"
    q_lower = q_clean.lower()

    # 2. Nhận diện tên biển bằng câu hỏi chuẩn q1 ("Biển chính trong ảnh là biển gì?")
    q1_ans, t_q1 = await run_in_threadpool(vlm_generate, image, "Biển chính trong ảnh là biển gì?", 96)

    # Nếu câu hỏi của người dùng hỏi về "ý nghĩa" chung (như "Biển báo này có ý nghĩa gì"),
    # kết hợp câu nhận diện biển (q1) + câu giải thích ý nghĩa tổng thể (q4) mà model đã được học trong vqa_train.json
    is_general_meaning_query = (
        ("ý nghĩa" in q_lower or "là gì" in q_lower or "như thế nào" in q_lower)
        and "biển phụ" not in q_lower
    )

    if q_clean == "Biển chính trong ảnh là biển gì?":
        ai_response_text = q1_ans
        gen_time = t_q1
    elif is_general_meaning_query:
        q4_ans, t_q4 = await run_in_threadpool(
            vlm_generate, image, "Hãy giải thích ý nghĩa tổng thể của cụm biển.", 160
        )
        ai_response_text = f"{q1_ans}\n\n**Giải thích ý nghĩa:** {q4_ans}"
        gen_time = round(t_q1 + t_q4, 3)
    else:
        ans, t_ans = await run_in_threadpool(vlm_generate, image, q_clean, 180)
        ai_response_text = ans
        gen_time = round(t_q1 + t_ans, 3)

    # 3. Chuẩn hóa mã biển từ tên biển tiếng Việt & sửa mã biển nếu VLM ghi nhầm (ví dụ P.105a -> P.103a)
    detected_code, ai_response_text = resolve_valid_sign_and_fix_answer(ai_response_text, q1_ans)

    if not detected_code:
        detected_code = extract_sign_code_from_text(q_clean)
    if not detected_code and sign_code:
        detected_code = extract_sign_code_from_text(sign_code) or sign_code.strip()
    if not detected_code or detected_code not in signs_catalog:
        detected_code = last_active_sign_code

    last_active_sign_code = detected_code
    sign_info = signs_catalog.get(
        detected_code,
        {
            "sign_code": detected_code,
            "canonical_sign_code": detected_code,
            "sign_name": "Biển báo giao thông",
            "category": get_category_by_code(detected_code),
            "image": "",
            "relevant_articles": [],
        },
    )

    # Đảm bảo ảnh thumbnail trong thẻ detected_sign luôn hiển thị được (ưu tiên ảnh vừa gửi lên dưới dạng base64)
    thumb_url = sign_info.get("image", "")
    if uploaded_new_image or not thumb_url or not ((ARCHIVE_DIR / thumb_url).exists() or (WEB_UI_DIR / thumb_url).exists()):
        thumb_url = image_to_data_url(image)

    # 4. Lấy các điều luật liên quan từ law_database.json
    rel_article_ids = sign_info.get("relevant_articles", [])[:3]
    retrieved_articles = [articles_by_id[aid] for aid in rel_article_ids if aid in articles_by_id]

    # 5. Gợi ý 6 câu hỏi chuẩn (q1..q6) từ bộ QA 52 biển
    sign_qas = qa_by_sign.get(detected_code, [])
    suggestions = [item["question"] for item in sign_qas[:6]] if sign_qas else [
        "Biển chính trong ảnh là biển gì?",
        "Biển phụ có nội dung gì?",
        "Biển phụ bổ sung ý nghĩa gì cho biển chính?",
        "Hãy giải thích ý nghĩa tổng thể của cụm biển.",
    ]

    total_latency = round(time.perf_counter() - t_start, 3)

    return JSONResponse(
        content={
            "status": "success",
            "ai_engine": "Qwen2-VL-2B-Instruct + qwen_vlm_freeform_lora_adapter",
            "latency_sec": total_latency,
            "detected_sign": {
                "sign_code": sign_info.get("canonical_sign_code", detected_code),
                "sign_name": sign_info.get("sign_name", "Biển báo giao thông"),
                "category": sign_info.get("category", get_category_by_code(detected_code)),
                "confidence": 1.0,
                "image": thumb_url,
            },
            "vlm": {
                "active": model_loaded,
                "adapter_loaded": adapter_loaded,
                "adapter_dir": str(ADAPTER_DIR),
                "generation_time_sec": gen_time,
            },
            "answer": ai_response_text,
            "relevant_articles": rel_article_ids,
            "retrieved_legal_passages": retrieved_articles,
            "prompt_suggestions": suggestions,
        }
    )


if (ARCHIVE_DIR / "cropped_dataset").exists():
    app.mount(
        "/cropped_dataset",
        StaticFiles(directory=str(ARCHIVE_DIR / "cropped_dataset")),
        name="cropped_dataset",
    )

app.mount("/", StaticFiles(directory=str(WEB_UI_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
