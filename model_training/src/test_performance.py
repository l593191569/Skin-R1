"""Model performance evaluation script over standardized datasets."""

import logging
import os
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import csv

# Qwen2.5-VL is the model used in the paper. Qwen3-VL / LLaVA classes are
# optional (only needed for those backbones) and may be unavailable depending on
# the installed transformers version, so import them defensively.
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None
try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:  # pragma: no cover
    LlavaNextForConditionalGeneration = None

logger = logging.getLogger(__name__)

def load_standardized_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load a standardized dataset."""
    logger.info(f"Loading standardized dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    # samples = samples[:20]
    logger.info(f"Loaded {len(samples)} samples from {data.get('dataset_name', 'unknown')}")
    
    return samples

def load_model_and_processor(model_name_or_path: str, checkpoint_path: Optional[str] = None):
    """Load the model and processor."""
    logger.info(f"Loading model from: {model_name_or_path}")
    
    if "qwen3" in model_name_or_path.lower():
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available in your transformers "
                "version. Upgrade transformers to use a Qwen3-VL model."
            )
        logger.info("Using Qwen3-VL model")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    elif "llava" in model_name_or_path.lower():
        if LlavaNextForConditionalGeneration is None:
            raise ImportError(
                "LlavaNextForConditionalGeneration is not available in your transformers version."
            )
        logger.info("Using LLaVA model")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif "qwen2.5" in model_name_or_path.lower():
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen2_5_VLForConditionalGeneration is not available in your transformers version."
            )
        logger.info("Using Qwen2.5-VL model")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        logger.info("Using standard Vision2Seq model")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Load weights if a checkpoint is provided
    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        if checkpoint_path.endswith(".pt"):
            checkpoint_state = torch.load(checkpoint_path, map_location="cuda")
            model.load_state_dict(checkpoint_state, strict=False)
        else:
            # Assume it is a LoRA checkpoint
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path)
    
    model.eval()
    logger.info("Model loaded successfully")
    if "qwen2.5" in model_name_or_path.lower():
        max_pixels = 448*448
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, max_pixels=max_pixels)
    else:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    
    if (
        "qwen3" in model_name_or_path.lower()
    ) and hasattr(processor, "image_processor"):
        processor.image_processor.do_resize = True
        processor.image_processor.size = {"shortest_edge": 448, "longest_edge": 768}
        logger.info(f"✅ Resize enabled: {processor.image_processor.do_resize}")
        logger.info(f"✅ Resize size: {processor.image_processor.size}")
        logger.info(f"✅ Resize interpolation: {processor}")


        



    return model, processor

def extract_final_diagnosis(response: str) -> str:
    """Extract the final diagnosis from the model response."""
    pattern = r'<final diagnosis>(.*?)</final diagnosis>'
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    
    if match:
        diagnosis = match.group(1).strip()
        # Extract the option letter (A, B, C, D, ...)
        option_match = re.search(r'[:\(\s]*([ABCDEFGHIJK])[:\)\s]*', diagnosis, re.I)
        if option_match:
            return option_match.group(1).upper()
    
    return "Z"

def test_performance(
    model_name_or_path: str,
    dataset_path: str,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    comment: str = ""
):
    """Evaluate the model's performance on a standardized dataset."""
    
    # Load the model
    model, processor = load_model_and_processor(model_name_or_path, checkpoint_path)
    
    # Load the dataset
    dataset = load_standardized_dataset(dataset_path)
    
    logger.info(f"Starting performance test with {len(dataset)} samples")
    
    # Evaluate performance
    correct_predictions = 0
    total_predictions = 0
    results = []
    debug_print = True

    with torch.no_grad():
        # Process data in batches
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_items = dataset[batch_start:batch_end]
            
            # Prepare batch data
            batch_texts = []
            batch_images = []
            batch_ground_truths = []
            batch_question_ids = []
            
            for item in batch_items:
                # Check the image path
                image_path = item["image_path"]
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Load the image
                image = Image.open(image_path).convert("RGB")
                
                # Prepare the text
                question_completed = item["question_completed"]
                
                # Build a different input format depending on the model type
                if "llava" in model_name_or_path.lower():
                    # LLaVA format: use the conversation format
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": question_completed}
                            ]
                        }
                    ]
                    text = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    batch_texts.append(text)
                    batch_images.append(image)
                    batch_ground_truths.append(item["answer_letter"])
                    batch_question_ids.append(item["question_id"])
                else:
                    # Format for Qwen3 and other models
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": question_completed}
                            ]
                        }
                    ]
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_texts.append(text)
                    batch_images.append(image)
                    batch_ground_truths.append(item["answer_letter"])
                    batch_question_ids.append(item["question_id"])
            
            if not batch_texts:  # skip if the batch has no valid data
                continue
            
            # Preprocess - pick a processing path based on the model type
            if "llava" in model_name_or_path.lower():
                # LLaVA path: apply the chat template first, then batch
                inputs = processor(
                    images=batch_images,
                    text=batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)
            else:
                # Path for Qwen3 and other models
                inputs = processor(
                    text=batch_texts,
                    images=batch_images,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)
            
            # Generation kwargs
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": processor.tokenizer.eos_token_id
            }
            
            # # If temperature > 0, sample; otherwise use greedy decoding
            # if temperature > 0:
            #     generate_kwargs["temperature"] = temperature
            #     generate_kwargs["do_sample"] = True
            # else:
            #     generate_kwargs["do_sample"] = False
            generate_kwargs["do_sample"] = False
            
            # Generate
            generated_ids = model.generate(**generate_kwargs)
            
            # Decode the outputs
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = [
                out_ids[prompt_len:] for out_ids in generated_ids
            ]
            
            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Evaluate the results
            for i, (output, ground_truth, question_id) in enumerate(zip(outputs, batch_ground_truths, batch_question_ids)):
                prediction = extract_final_diagnosis(output)
                is_correct = (prediction == ground_truth)
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                results.append({
                    "question_id": question_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "correct": is_correct,
                    "response": output
                })
                
                if debug_print:
                    print(f"question_id: {question_id}, ground_truth: {ground_truth}, prediction: {prediction}, correct: {is_correct}, response: {output}")
                    print(f"--------------------------------")
                    debug_print = False

            if batch_end % (batch_size * 10) == 0 or batch_end == len(dataset):
                if outputs:
                    preview_idx = 0
                    prev_qid = batch_question_ids[preview_idx]
                    prev_gt = batch_ground_truths[preview_idx]
                    prev_pred = extract_final_diagnosis(outputs[preview_idx])
                    prev_correct = (prev_pred == prev_gt)
                    prev_resp = outputs[preview_idx]
                    print(f"preview | qid: {prev_qid} | gt: {prev_gt} | pred: {prev_pred} | correct: {prev_correct} | resp: {prev_resp}")
                print(f"--------------------------------")
                current_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
                logger.info(f"Processed {batch_end}/{len(dataset)} samples, current accuracy: {current_acc:.4f}")
    
    # Compute the final performance
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    logger.info(f"Final accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    # Save the results
    save_results(results, accuracy, model_name_or_path, dataset_path, checkpoint_path, comment)
    
    return accuracy, results

def save_results(results: List[Dict], accuracy: float, model_name: str, dataset_path: str, checkpoint_path: Optional[str], comment: str):
    """Save the evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "performance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Derive the dataset name
    dataset_name = os.path.basename(dataset_path).replace("_standardized.json", "")
    safe_model_name = model_name.replace("/", "_")
    # Save the detailed results
    results_file = os.path.join(results_dir, f"test_results_{safe_model_name}_{dataset_name}_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "comment": comment,
            "accuracy": accuracy,
            "total_samples": len(results),
            "timestamp": timestamp,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Save the CSV summary
    csv_file = os.path.join(results_dir, f"performance_summary_{safe_model_name}_{dataset_name}_{timestamp}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "ground_truth", "prediction", "correct"])
        for result in results:
            writer.writerow([
                result["question_id"],
                result["ground_truth"],
                result["prediction"],
                result["correct"]
            ])
    
    logger.info(f"Summary saved to: {csv_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model performance on standardized datasets")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to the base model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the standardized dataset JSON file")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to the checkpoint (optional)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--comment", type=str, default="",
                       help="Comment for the test")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting performance test:")
    logger.info(f"  Model: {args.model_name_or_path}")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Checkpoint: {args.checkpoint_path}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    accuracy, results = test_performance(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        comment=args.comment
    )
    
    print(f"Test completed with accuracy: {accuracy:.4f}")
