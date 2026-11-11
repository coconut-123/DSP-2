# -*- coding: utf-8 -*-
"""
Script kiểm thử (Test & Evaluate) mô hình Qwen2-VL-2B-Instruct đã Fine-tune LoRA
trên bộ dữ liệu Free-form VQA biển báo giao thông Việt Nam (vqa_dataset/vqa_test.json).
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from vlm_baselines_and_proposed import (
    ADAPTER_OUT_DIR,
    ARCHIVE_DIR,
    DEFAULT_VLM_ID,
    VQA_DIR,
    evaluate_method,
    load_base_qwen_vlm,
)


def main():
    parser = argparse.ArgumentParser(description="Test model Qwen2-VL đã fine-tune trên bộ vqa_test.json")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=str(ADAPTER_OUT_DIR),
        help="Đường dẫn tới thư mục chứa LoRA adapter (mặc định: qwen_vlm_freeform_lora_adapter)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=18,
        help="Số lượng mẫu câu hỏi trong vqa_test.json dùng để test nhanh (mặc định: 18 mẫu = 3 biển x 6 câu hỏi q1..q6; đặt 0 để chạy toàn bộ 312 mẫu)",
    )
    parser.add_argument(
        "--compare_zero_shot",
        action="store_true",
        help="Chạy thêm Baseline 1 (Zero-shot) trên cùng tập mẫu để đối chiếu trực tiếp",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    safetensors_file = adapter_path / "adapter_model.safetensors"
    bin_file = adapter_path / "adapter_model.bin"

    if not safetensors_file.exists() and not bin_file.exists():
        print("\n[!] CẢNH BÁO: Không tìm thấy file trọng số 'adapter_model.safetensors' trong:")
        print(f"    -> {adapter_path}")
        return

    with open(VQA_DIR / "vqa_test.json", "r", encoding="utf-8") as f:
        all_test_samples = json.load(f)

    # Chọn tập mẫu đại diện đa dạng (gồm cả biển cảnh báo, biển cấm xe máy/ô tô, biển hiệu lệnh, biển phụ)
    if args.num_samples and 0 < args.num_samples < len(all_test_samples):
        preferred_labels = ["W.224", "P.104", "P.130", "R.301d", "S.505a*Xe máy"]
        selected = []
        seen_labels = set()
        for item in all_test_samples:
            lbl = item["dataset_label"]
            if lbl in preferred_labels:
                selected.append(item)
                seen_labels.add(lbl)
            if len(selected) >= args.num_samples:
                break
        if len(selected) < args.num_samples:
            for item in all_test_samples:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= args.num_samples:
                    break
        test_samples = selected[: args.num_samples]
    else:
        test_samples = all_test_samples

    with open(VQA_DIR / "fewshot_exemplars.json", "r", encoding="utf-8") as f:
        fewshot_exemplars = json.load(f)

    print(f"[*] Đang tải mô hình gốc: {DEFAULT_VLM_ID} (CUDA available: {torch.cuda.is_available()})")
    base_model, processor = load_base_qwen_vlm(DEFAULT_VLM_ID, use_4bit=True)
    base_model.eval()

    results = {}

    if args.compare_zero_shot:
        print(f"\n[*] Đang chạy Baseline 1 (Zero-shot) trên {len(test_samples)} mẫu...")
        results["baseline_1_zeroshot"] = evaluate_method(
            "baseline_1_zeroshot",
            "Baseline 1: Pre-trained VLM – Zero-shot",
            base_model,
            processor,
            test_samples,
            fewshot_exemplars,
        )

    print(f"\n[*] Đang gắn LoRA Adapter từ: {adapter_path}")
    ft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    ft_model.eval()

    print(f"[*] Đang test Proposed Fine-tuned VLM trên {len(test_samples)} mẫu...")
    ft_summary = evaluate_method(
        "proposed_finetuned_vlm",
        "Proposed Model: Fine-tuned Qwen VLM",
        ft_model,
        processor,
        test_samples,
        fewshot_exemplars,
    )
    results["proposed_finetuned_vlm"] = ft_summary

    out_file = ARCHIVE_DIR / "test_model_predictions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print(" KẾT QUẢ DỰ ĐOÁN CHI TIẾT CỦA MODEL ĐÃ FINE-TUNE TRÊN TẬP TEST")
    print("=" * 92)
    zs_preds = {}
    if "baseline_1_zeroshot" in results:
        zs_preds = {r["vqa_id"]: r["predicted_answer"] for r in results["baseline_1_zeroshot"]["predictions"]}

    for item in ft_summary["predictions"]:
        print(f"- [{item['vqa_id']}] Mã biển: {item['dataset_label']} ({item['task_category']}) | Ảnh: {item['image_path']}")
        print(f"  + Câu hỏi        : {item['question']}")
        print(f"  + Đáp án chuẩn   : {item['gold_answer']}")
        if item["vqa_id"] in zs_preds:
            print(f"  + Zero-shot (B1) : {zs_preds[item['vqa_id']]}")
        print(f"  + Fine-tuned VLM : {item['predicted_answer']}")
        yn_info = f" | Yes/No Đúng: {item['yes_no_correct']}" if item["yes_no_correct"] is not None else ""
        print(f"  + Điểm số        : Token-F1 = {item['token_f1']*100:.1f}% | ROUGE-L = {item['rouge_l']*100:.1f}%{yn_info}")
        print("-" * 92)

    print("\n" + "=" * 92)
    print(" BẢNG TỔNG HỢP ĐIỂM SỐ ĐÁNH GIÁ")
    print("=" * 92)
    for k, res in results.items():
        yn_str = f"{res['yes_no_accuracy']*100:.2f}%" if res["yes_no_accuracy"] is not None else "N/A"
        print(
            f"{res['method_title']:<42} | "
            f"Token-F1: {res['overall_token_f1']*100:6.2f}% | "
            f"ROUGE-L: {res['overall_rouge_l']*100:6.2f}% | "
            f"BLEU-4: {res['overall_bleu4']*100:6.2f}% | "
            f"Yes/No Acc: {yn_str}"
        )
    print("=" * 92)
    print(f"[✓] Đã lưu toàn bộ kết quả dự đoán tại: {out_file}")


if __name__ == "__main__":
    main()
