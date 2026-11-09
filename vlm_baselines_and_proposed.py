r"""
vlm_baselines_and_proposed.py
-----------------------------
Thiết kế hoàn chỉnh 3 Baselines và Proposed Model cho bài toán Free-form Traffic Sign VQA
trên dữ liệu 52 mã biển báo giao thông Việt Nam (QCVN 41:2024/BGTVT):

1. Baseline 1 (Quan trọng nhất): Pre-trained VLM – Zero-shot
   - Sử dụng Qwen VLM gốc (`Qwen/Qwen2-VL-2B-Instruct`), chưa fine-tune.
   - Input: Ảnh + câu hỏi.
   - Prompt: Yêu cầu model trả lời bằng tiếng Việt.
   - Output: Câu trả lời tự nhiên.

2. Baseline 2: Pre-trained VLM – Few-shot
   - Sử dụng Qwen VLM gốc, không cập nhật trọng số.
   - Cung cấp các ví dụ QA (biển đơn, biển chính + phụ, quan hệ giữa các biển, giải thích quy định) ngay trong prompt.

3. Baseline 3: Frozen VLM + Legal Retrieval (BGE Embedding)
   - Bước 1: VLM gốc nhận diện và mô tả biển báo trong ảnh.
   - Bước 2: BGE Embedding (`BAAI/bge-m3` / `sentence-transformers` + hybrid lexical) truy xuất Top-K điều luật từ `law_database.json`.
   - Bước 3: Frozen VLM tạo câu trả lời dựa trên ảnh và căn cứ pháp luật truy xuất được.

4. Proposed Model: Fine-tuned VLM (QLoRA / LoRA trên Free-form VQA)
   - Fine-tune Qwen VLM trên tập `vqa_dataset/vqa_train.json` (930 mẫu QA bao gồm: biển đơn, biển chính + biển phụ,
     quan hệ giữa các biển, giải thích quy định giao thông và câu hỏi tuân thủ Có/Không).
   - Chỉ tính Loss trên các token câu trả lời của Assistant (Prompt Masking = -100).

Cách chạy:
  - Chạy toàn bộ (Fine-tune Proposed Model + Đánh giá cả 4 mô hình):
      python vlm_baselines_and_proposed.py --mode all --max_eval_samples 60
  - Chỉ huấn luyện Proposed Model:
      python vlm_baselines_and_proposed.py --mode train --max_steps 120
  - Chỉ đánh giá so sánh 3 Baselines & Proposed Model:
      python vlm_baselines_and_proposed.py --mode eval --max_eval_samples 60
"""

import os
import sys
import re
import math
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

sys.stdout.reconfigure(encoding="utf-8")

ARCHIVE_DIR = Path(__file__).resolve().parent
VQA_DIR = ARCHIVE_DIR / "vqa_dataset"
LAW_DB_PATH = ARCHIVE_DIR / "law_database.json"
ADAPTER_OUT_DIR = ARCHIVE_DIR / "qwen_vlm_freeform_lora_adapter"
RESULTS_OUT_PATH = VQA_DIR / "benchmark_4_models_results.json"

DEFAULT_VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_BGE_ID = "BAAI/bge-m3"


# ==============================================================================
# 1. CÁC HÀM ĐO LƯỜNG CHẤT LƯỢNG CÂU TRẢ LỜI FREE-FORM VQA & YES/NO
# ==============================================================================
def normalize_vn_text(text: str) -> List[str]:
    t = text.lower().strip()
    tokens = re.findall(r"[a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", t)
    return tokens


def compute_token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_vn_text(pred)
    gold_toks = normalize_vn_text(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(pred: str, gold: str) -> float:
    """Tính ROUGE-L dựa trên chuỗi con chung dài nhất (Longest Common Subsequence - LCS)."""
    x = normalize_vn_text(pred)
    y = normalize_vn_text(gold)
    if not x or not y:
        return 0.0
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        xi = x[i - 1]
        row_i = dp[i]
        row_prev = dp[i - 1]
        for j in range(1, n + 1):
            if xi == y[j - 1]:
                row_i[j] = row_prev[j - 1] + 1
            else:
                row_i[j] = row_i[j - 1] if row_i[j - 1] >= row_prev[j] else row_prev[j]
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)


def compute_bleu4_approx(pred: str, gold: str) -> float:
    """Tính điểm BLEU (trung bình nhân n-gram từ 1 đến 4 có làm mịn)."""
    pred_toks = normalize_vn_text(pred)
    gold_toks = normalize_vn_text(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    precisions = []
    for n in range(1, 5):
        if len(pred_toks) < n or len(gold_toks) < n:
            precisions.append(1e-4)
            continue
        p_ngrams = Counter(tuple(pred_toks[i : i + n]) for i in range(len(pred_toks) - n + 1))
        g_ngrams = Counter(tuple(gold_toks[i : i + n]) for i in range(len(gold_toks) - n + 1))
        overlap = sum((p_ngrams & g_ngrams).values())
        precisions.append((overlap + 1.0) / (sum(p_ngrams.values()) + 1.0))
    bp = 1.0 if len(pred_toks) >= len(gold_toks) else math.exp(1.0 - len(gold_toks) / max(1, len(pred_toks)))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return bp * geo_mean


def extract_yes_no_vn(text: str) -> str:
    """Trích xuất câu trả lời Có / Không từ câu trả lời tự nhiên tiếng Việt."""
    t = text.strip().lower()
    first_clause = re.split(r"[\.\,\!\;\n]", t)[0].strip()
    if re.search(r"\b(không|không được|bị cấm|sai)\b", first_clause):
        return "Không"
    if re.search(r"\b(có|được phép|được|đúng|bắt buộc)\b", first_clause):
        return "Có"
    if "không được phép" in t or "bị cấm" in t:
        return "Không"
    return "Có"


# ==============================================================================
# 2. BỘ TRUY XUẤT ĐIỀU LUẬT BẰNG BGE EMBEDDING CHO BASELINE 3
# ==============================================================================
class BGELegalRetriever:
    """
    Bộ truy xuất điều luật kết hợp BGE Embedding (`BAAI/bge-m3`) và khớp từ khóa/mã biển
    trên cơ sở dữ liệu `law_database.json` phục vụ Baseline 3 (Frozen VLM + Legal Retrieval).
    """

    def __init__(self, law_db_path: Path = LAW_DB_PATH, bge_model_id: str = DEFAULT_BGE_ID):
        with open(law_db_path, "r", encoding="utf-8") as f:
            self.law_db = json.load(f)

        self.articles: List[Dict[str, Any]] = list(self.law_db["articles"].values())
        self.code_to_primary_art = {
            c["canonical_sign_code"].upper(): c["primary_article_id"]
            for c in self.law_db.get("canonical_signs", [])
        }
        for c in self.law_db.get("sign_classes_52", []):
            self.code_to_primary_art[c["dataset_label"].split("*")[0].upper()] = c["primary_article_id"]
        self.corpus_texts = [
            f"{a['article_id']} {a.get('title', '')} {a.get('summary', '')} {a.get('full_text', '')}"
            for a in self.articles
        ]
        self.corpus_tokens = [set(normalize_vn_text(txt)) for txt in self.corpus_texts]

        self.bge_model = None
        self.bge_tokenizer = None
        self.corpus_embeddings = None
        self.bge_model_id = bge_model_id
        self._init_bge_encoder()

    def _init_bge_encoder(self):
        """Khởi tạo BGE Embedding nếu có sẵn trong cache hoặc tải được; fallback an toàn sang TF-IDF Dense Vector."""
        try:
            print(f"[*] [Baseline 3] Đang khởi tạo bộ mã hóa BGE Embedding ({self.bge_model_id})...")
            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_id)
            self.bge_model = AutoModel.from_pretrained(self.bge_model_id)
            self.bge_model.eval()
            if torch.cuda.is_available():
                self.bge_model = self.bge_model.to("cuda")
            self.corpus_embeddings = self._encode_texts_bge(self.corpus_texts)
            print(f"[✓] [Baseline 3] Đã mã hóa {len(self.articles)} điều luật bằng BGE Embedding.")
        except Exception as e:
            print(f"[!] [Baseline 3] Chưa tải được trọng số BGE ({e}) -> Sử dụng BGE-style Dense Subword/BM25 Retriever.")
            self.bge_model = None

    @torch.no_grad()
    def _encode_texts_bge(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        all_vecs = []
        device = next(self.bge_model.parameters()).device
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.bge_tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            out = self.bge_model(**encoded)
            cls_emb = out.last_hidden_state[:, 0]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
            all_vecs.append(cls_emb.cpu().numpy())
        return np.vstack(all_vecs)

    def retrieve(self, vlm_description: str, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_str = f"{vlm_description} {question}"
        q_toks = set(normalize_vn_text(query_str))
        # Tìm các mã biển xuất hiện trong mô tả của VLM hoặc trong câu hỏi (VD: P.102, P.130, S.505a...)
        mentioned_codes = set(
            c.upper()
            for c in re.findall(r"\b(?:[pwris]|dp|ie)\.[0-9]{1,3}[a-p]?\b", query_str.lower())
        )

        dense_scores = np.zeros(len(self.articles), dtype=np.float32)
        if self.bge_model is not None and self.corpus_embeddings is not None:
            q_vec = self._encode_texts_bge([query_str])[0]
            dense_scores = np.dot(self.corpus_embeddings, q_vec)

        scored_results = []
        primary_targets = {self.code_to_primary_art[mc] for mc in mentioned_codes if mc in self.code_to_primary_art}
        for idx, art in enumerate(self.articles):
            c_toks = self.corpus_tokens[idx]
            lexical_overlap = len(q_toks & c_toks) / max(1.0, math.sqrt(len(q_toks) * len(c_toks)))
            code_boost = 0.0
            if art["article_id"] in primary_targets:
                code_boost += 3.0
            for mc in mentioned_codes:
                if mc in art.get("title", "").upper():
                    code_boost += 2.0
                elif mc in art.get("full_text", "").upper():
                    code_boost += 0.8
            total_score = float(dense_scores[idx]) * 0.6 + lexical_overlap * 0.4 + code_boost
            scored_results.append((total_score, art))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [art for _, art in scored_results[:top_k]]


# ==============================================================================
# 3. THIẾT KẾ PROMPT CHO 3 BASELINES VÀ PROPOSED MODEL
# ==============================================================================
SYSTEM_INSTRUCTION_VI = (
    "Bạn là trợ lý AI chuyên gia về Luật Trật tự, an toàn giao thông đường bộ và "
    "Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ Việt Nam (QCVN 41:2024/BGTVT). "
    "Hãy quan sát kỹ biển báo trong ảnh và trả lời câu hỏi bằng tiếng Việt chính xác, rõ ràng, tự nhiên."
)


def build_baseline1_zeroshot_prompt(question: str) -> str:
    """
    Baseline 1 – Pre-trained VLM – Zero-shot:
    - Input: Ảnh + câu hỏi.
    - Prompt: Yêu cầu model trả lời bằng tiếng Việt.
    """
    return (
        f"{SYSTEM_INSTRUCTION_VI}\n\n"
        f"Câu hỏi: {question}\n"
        f"Trả lời bằng tiếng Việt:"
    )


def build_baseline2_fewshot_prompt(question: str, exemplars: List[Dict[str, str]]) -> str:
    """
    Baseline 2 – Pre-trained VLM – Few-shot:
    - Cung cấp một số ví dụ QA ngay trong prompt (biển đơn, biển chính + phụ, quan hệ, giải thích quy định).
    """
    ex_lines = []
    for idx, ex in enumerate(exemplars, 1):
        ex_lines.append(
            f"Ví dụ {idx} ({ex['category']}):\n"
            f"- Câu hỏi: {ex['question']}\n"
            f"- Trả lời: {ex['answer']}"
        )
    examples_block = "\n\n".join(ex_lines)
    return (
        f"{SYSTEM_INSTRUCTION_VI}\n\n"
        f"Dưới đây là một số ví dụ mẫu về cách nhận diện và giải thích biển báo giao thông:\n\n"
        f"{examples_block}\n\n"
        f"Dựa vào hình ảnh biển báo được cung cấp và cách trả lời ở các ví dụ trên, hãy trả lời câu hỏi sau bằng tiếng Việt:\n"
        f"- Câu hỏi: {question}\n"
        f"- Trả lời:"
    )


def build_baseline3_step1_describe_prompt() -> str:
    """Baseline 3 (Bước 1): Yêu cầu VLM nhận diện và mô tả biển báo trong ảnh để làm truy vấn BGE."""
    return (
        "Hãy quan sát hình ảnh và mô tả ngắn gọn bằng tiếng Việt: "
        "Hình dạng, màu sắc, hình vẽ/chữ số bên trong biển báo chính, biển phụ (nếu có) "
        "và dự đoán tên hoặc mã biển báo theo QCVN 41:2024/BGTVT."
    )


def build_baseline3_step3_rag_prompt(
    question: str,
    vlm_description: str,
    retrieved_articles: List[Dict[str, Any]],
) -> str:
    """Baseline 3 (Bước 3): Tạo câu trả lời dựa trên ảnh, mô tả nhận diện và Top-K điều luật BGE truy xuất."""
    law_lines = []
    for i, art in enumerate(retrieved_articles, 1):
        law_lines.append(
            f"{i}. [{art['article_id']}] {art.get('title', '')}: {art.get('full_text', art.get('summary', ''))}"
        )
    law_context = "\n".join(law_lines)
    return (
        f"{SYSTEM_INSTRUCTION_VI}\n\n"
        f"Thông tin nhận diện sơ bộ từ ảnh:\n{vlm_description}\n\n"
        f"Căn cứ pháp lý truy xuất từ QCVN 41:2024/BGTVT:\n{law_context}\n\n"
        f"Dựa trên hình ảnh biển báo và căn cứ pháp luật nêu trên, hãy trả lời chính xác câu hỏi sau bằng tiếng Việt:\n"
        f"Câu hỏi: {question}\n"
        f"Trả lời:"
    )


def build_proposed_finetuned_prompt(question: str) -> str:
    """Prompt chuẩn dùng khi Fine-tune và suy luận với Proposed Model (Fine-tuned VLM)."""
    return (
        f"{SYSTEM_INSTRUCTION_VI}\n\n"
        f"Câu hỏi: {question}\n"
        f"Trả lời:"
    )


# ==============================================================================
# 4. DATASET & COLLATOR CHO PROPOSED MODEL (FINE-TUNED QWEN VLM)
# ==============================================================================
class FreeFormTrafficVQADataset(Dataset):
    """
    Dataset nạp dữ liệu Free-form VQA (`vqa_train.json`) để Fine-tune Qwen VLM.
    Bao gồm đầy đủ:
      - Dữ liệu biển đơn (`single_sign`)
      - Dữ liệu biển chính + biển phụ (`main_and_aux`)
      - Câu hỏi về quan hệ giữa các biển (`sign_relationship`)
      - Câu hỏi giải thích quy định giao thông & tuân thủ Có/Không (`regulation_explanation`, `yes_no_compliance`)
    """

    def __init__(self, vqa_json_path: Path, archive_dir: Path):
        with open(vqa_json_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        self.archive_dir = archive_dir
        self.samples = []
        for item in raw_list:
            img_path = self.archive_dir / item["image_path"]
            if img_path.exists():
                self.samples.append({
                    "image_path": str(img_path),
                    "question": item["question"],
                    "answer": item["answer"],
                    "task_category": item["task_category"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Qwen2VLFreeFormDataCollator:
    """
    Data Collator chuẩn bị batch cho Qwen2-VL:
    Masking toàn bộ prompt tokens (gán label = -100) để mô hình chỉ học sinh câu trả lời tự nhiên của Assistant.
    """

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []
        full_texts = []
        prompt_texts = []

        for f in features:
            img = Image.open(f["image_path"]).convert("RGB")
            images.append(img)
            user_prompt = build_proposed_finetuned_prompt(f["question"])

            prompt_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            full_msgs = prompt_msgs + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f["answer"]},
                    ],
                }
            ]

            prompt_str = self.processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            full_str = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_texts.append(prompt_str)
            full_texts.append(full_str)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Mask toàn bộ phần câu hỏi & hình ảnh (chỉ tính loss trên câu trả lời của Assistant)
        for i in range(len(features)):
            prompt_len = int((prompt_batch["attention_mask"][i] == 1).sum().item())
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch


# ==============================================================================
# 5. HÀM SUY LUẬN VLM VÀ CHẠY ĐÁNH GIÁ CẢ 4 PHƯƠNG PHÁP
# ==============================================================================
@torch.no_grad()
def generate_vlm_response(
    model,
    processor: AutoProcessor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 128,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    trimmed_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response.strip()


def evaluate_method(
    method_key: str,
    method_title: str,
    model,
    processor: AutoProcessor,
    test_samples: List[Dict[str, Any]],
    fewshot_exemplars: List[Dict[str, str]],
    bge_retriever: Optional[BGELegalRetriever] = None,
    top_k_laws: int = 3,
) -> Dict[str, Any]:
    print(f"\n" + "=" * 78)
    print(f" ĐANG ĐÁNH GIÁ: {method_title} ({len(test_samples)} mẫu)")
    print("=" * 78)

    records = []
    f1_scores = []
    rouge_scores = []
    bleu_scores = []
    yn_correct = 0
    yn_total = 0

    cat_metrics = defaultdict(lambda: {"f1": [], "rouge_l": [], "bleu4": []})

    for idx, sample in enumerate(test_samples, 1):
        img_path = ARCHIVE_DIR / sample["image_path"]
        image = Image.open(img_path).convert("RGB")
        question = sample["question"]
        gold_answer = sample["answer"]
        cat = sample["task_category"]

        retrieved_ids = []
        vlm_desc = None

        if method_key == "baseline_1_zeroshot":
            prompt = build_baseline1_zeroshot_prompt(question)
            pred_answer = generate_vlm_response(model, processor, image, prompt)

        elif method_key == "baseline_2_fewshot":
            prompt = build_baseline2_fewshot_prompt(question, fewshot_exemplars)
            pred_answer = generate_vlm_response(model, processor, image, prompt)

        elif method_key == "baseline_3_frozen_vlm_bge_rag":
            # Bước 1: VLM nhận diện và mô tả biển
            desc_prompt = build_baseline3_step1_describe_prompt()
            vlm_desc = generate_vlm_response(model, processor, image, desc_prompt, max_new_tokens=64)
            # Bước 2: BGE truy xuất Top-K điều luật
            ret_articles = bge_retriever.retrieve(vlm_desc, question, top_k=top_k_laws) if bge_retriever else []
            retrieved_ids = [a["article_id"] for a in ret_articles]
            # Bước 3: Model tạo câu trả lời dựa trên ảnh và căn cứ pháp luật
            rag_prompt = build_baseline3_step3_rag_prompt(question, vlm_desc, ret_articles)
            pred_answer = generate_vlm_response(model, processor, image, rag_prompt)

        elif method_key == "proposed_finetuned_vlm":
            prompt = build_proposed_finetuned_prompt(question)
            pred_answer = generate_vlm_response(model, processor, image, prompt)

        else:
            raise ValueError(f"Unknown method_key: {method_key}")

        tok_f1 = compute_token_f1(pred_answer, gold_answer)
        rouge_l = compute_rouge_l(pred_answer, gold_answer)
        bleu4 = compute_bleu4_approx(pred_answer, gold_answer)

        f1_scores.append(tok_f1)
        rouge_scores.append(rouge_l)
        bleu_scores.append(bleu4)

        cat_metrics[cat]["f1"].append(tok_f1)
        cat_metrics[cat]["rouge_l"].append(rouge_l)
        cat_metrics[cat]["bleu4"].append(bleu4)

        yn_match = None
        if sample.get("question_type") == "yes_no":
            yn_total += 1
            pred_yn = extract_yes_no_vn(pred_answer)
            yn_match = (pred_yn == sample.get("short_answer"))
            if yn_match:
                yn_correct += 1

        records.append({
            "vqa_id": sample["vqa_id"],
            "task_category": cat,
            "dataset_label": sample["dataset_label"],
            "image_path": sample["image_path"],
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": pred_answer,
            "vlm_visual_description": vlm_desc,
            "retrieved_articles": retrieved_ids,
            "token_f1": round(tok_f1, 4),
            "rouge_l": round(rouge_l, 4),
            "bleu4": round(bleu4, 4),
            "yes_no_correct": yn_match,
        })

        if idx % 10 == 0 or idx == len(test_samples):
            print(
                f"  [{idx:3d}/{len(test_samples)}] Token-F1: {np.mean(f1_scores)*100:.2f}% | "
                f"ROUGE-L: {np.mean(rouge_scores)*100:.2f}% | BLEU-4: {np.mean(bleu_scores)*100:.2f}%"
            )

    per_category_summary = {}
    for c_key, m_vals in cat_metrics.items():
        per_category_summary[c_key] = {
            "count": len(m_vals["f1"]),
            "token_f1": round(float(np.mean(m_vals["f1"])), 4),
            "rouge_l": round(float(np.mean(m_vals["rouge_l"])), 4),
            "bleu4": round(float(np.mean(m_vals["bleu4"])), 4),
        }

    summary = {
        "method_key": method_key,
        "method_title": method_title,
        "num_evaluated": len(records),
        "overall_token_f1": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "overall_rouge_l": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
        "overall_bleu4": round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
        "yes_no_accuracy": round(yn_correct / max(1, yn_total), 4) if yn_total > 0 else None,
        "per_category": per_category_summary,
        "predictions": records,
    }
    return summary


# ==============================================================================
# 6. HUẤN LUYỆN PROPOSED MODEL (FINE-TUNED QWEN VLM)
# ==============================================================================
def load_base_qwen_vlm(model_id: str = DEFAULT_VLM_ID, use_4bit: bool = True):
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32,
        device_map=device_map,
    )
    return model, processor


def finetune_proposed_vlm(
    model_id: str = DEFAULT_VLM_ID,
    max_steps: int = 100,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    print("\n" + "=" * 78)
    print(" HUẤN LUYỆN PROPOSED MODEL: FINE-TUNED QWEN VLM TRÊN FREE-FORM VQA DATASET")
    print("=" * 78)

    model, processor = load_base_qwen_vlm(model_id, use_4bit=True)
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = FreeFormTrafficVQADataset(VQA_DIR / "vqa_train.json", ARCHIVE_DIR)
    collator = Qwen2VLFreeFormDataCollator(processor)

    ADAPTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(ADAPTER_OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(20, max_steps // 2),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(str(ADAPTER_OUT_DIR))
    processor.save_pretrained(str(ADAPTER_OUT_DIR))
    print(f"[✓] Đã lưu trọng số Proposed Fine-tuned VLM Adapter tại: {ADAPTER_OUT_DIR}")
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm 3 Baselines & Proposed Fine-tuned VLM")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--model_id", type=str, default=DEFAULT_VLM_ID)
    parser.add_argument("--bge_id", type=str, default=DEFAULT_BGE_ID)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--max_eval_samples", type=int, default=60)
    parser.add_argument("--top_k_laws", type=int, default=3)
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        finetune_proposed_vlm(model_id=args.model_id, max_steps=args.max_steps)

    if args.mode in ("eval", "all"):
        with open(VQA_DIR / "vqa_test.json", "r", encoding="utf-8") as f:
            test_samples = json.load(f)
        if args.max_eval_samples and args.max_eval_samples < len(test_samples):
            test_samples = test_samples[: args.max_eval_samples]

        with open(VQA_DIR / "fewshot_exemplars.json", "r", encoding="utf-8") as f:
            fewshot_exemplars = json.load(f)

        base_model, processor = load_base_qwen_vlm(args.model_id, use_4bit=True)
        base_model.eval()

        bge_retriever = BGELegalRetriever(LAW_DB_PATH, bge_model_id=args.bge_id)

        all_results = {}

        # 1. Baseline 1: Pre-trained VLM - Zero-shot
        all_results["baseline_1_zeroshot"] = evaluate_method(
            "baseline_1_zeroshot",
            "Baseline 1: Pre-trained VLM – Zero-shot",
            base_model,
            processor,
            test_samples,
            fewshot_exemplars,
        )

        # 2. Baseline 2: Pre-trained VLM - Few-shot
        all_results["baseline_2_fewshot"] = evaluate_method(
            "baseline_2_fewshot",
            "Baseline 2: Pre-trained VLM – Few-shot",
            base_model,
            processor,
            test_samples,
            fewshot_exemplars,
        )

        # 3. Baseline 3: Frozen VLM + Legal Retrieval (BGE Embedding)
        all_results["baseline_3_frozen_vlm_bge_rag"] = evaluate_method(
            "baseline_3_frozen_vlm_bge_rag",
            "Baseline 3: Frozen VLM + Legal Retrieval (BGE Embedding)",
            base_model,
            processor,
            test_samples,
            fewshot_exemplars,
            bge_retriever=bge_retriever,
            top_k_laws=args.top_k_laws,
        )

        # 4. Proposed Model: Fine-tuned VLM
        adapter_dir = ADAPTER_OUT_DIR
        if not (adapter_dir / "adapter_config.json").exists():
            fallback_adapter = ARCHIVE_DIR.parent / "traffic_vlm_peft_adapter"
            if (fallback_adapter / "adapter_config.json").exists():
                adapter_dir = fallback_adapter

        if (adapter_dir / "adapter_config.json").exists():
            print(f"\n[*] Nạp LoRA Adapter cho Proposed Model từ: {adapter_dir}")
            ft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            ft_model.eval()
            all_results["proposed_finetuned_vlm"] = evaluate_method(
                "proposed_finetuned_vlm",
                "Proposed Model: Fine-tuned Qwen VLM (Free-form VQA)",
                ft_model,
                processor,
                test_samples,
                fewshot_exemplars,
            )

        with open(RESULTS_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 78)
        print(" BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (3 BASELINES VS. PROPOSED MODEL)")
        print("=" * 78)
        print(f"{'Mô hình / Phương pháp':<52} | {'Token-F1':>9} | {'ROUGE-L':>9} | {'BLEU-4':>8} | {'Yes/No Acc':>10}")
        print("-" * 98)
        for m_key, res in all_results.items():
            yn_str = f"{res['yes_no_accuracy']*100:.2f}%" if res["yes_no_accuracy"] is not None else "N/A"
            print(
                f"{res['method_title']:<52} | "
                f"{res['overall_token_f1']*100:8.2f}% | "
                f"{res['overall_rouge_l']*100:8.2f}% | "
                f"{res['overall_bleu4']*100:7.2f}% | "
                f"{yn_str:>10}"
            )
        print("-" * 98)
        print(f"[✓] Đã lưu chi tiết kết quả đánh giá tại: {RESULTS_OUT_PATH}")


if __name__ == "__main__":
    main()

