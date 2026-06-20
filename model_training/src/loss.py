import re
import torch
import logging

# Use the same logger pattern as train_sft.py for consistency
logger = logging.getLogger(__name__)

from .prompts import get_message


def extract_concepts_from_text(text: str) -> set:
    """Extract concepts from text like 'Presence of concept1, concept2, concept3.'"""
    # Remove the rule tags and "Presence of" prefix
    text = re.sub(r'<rule>|</rule>', '', text)
    text = re.sub(r'Presence of\s*', '', text, flags=re.IGNORECASE)
    
    # Handle "no concept" case
    if 'no concept' in text.lower():
        return set()
    
    # Extract concepts by splitting on commas and cleaning
    concepts = []
    for concept in text.split(','):
        concept = concept.strip().strip('.')
        if concept and concept.lower() != 'no concept':
            concepts.append(concept)
    
    return set(concepts)

def evaluate_concept_accuracy(model, eval_dataset, processor, device, max_samples=None):
    """Evaluate model based on concept matching (order-independent)."""
    model.eval()
    total_samples = 0
    correct_concepts = 0
    total_concepts = 0
    
    sample_count = 0
    for batch in eval_dataset:
        if max_samples and sample_count >= max_samples:
            break
            
        try:
            # Generate response using multimodal message format (same as training)
            prompt = "List the observed clinical concepts in a single sentence starting with 'Presence of', and separate the concepts with commas."
            
            # Create multimodal message format
            messages = get_message(prompt)
            
            # Apply chat template and process with images
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[batch["image"]], return_tensors="pt").to(device)
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=128)
                generated_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            # Extract ground truth concepts
            ground_truth = batch["response"]
            gt_concepts = extract_concepts_from_text(ground_truth)
            
            # Extract predicted concepts
            pred_concepts = extract_concepts_from_text(generated_text)
            
            # Calculate concept-level accuracy
            if gt_concepts:
                # Case 1: Ground truth has concepts
                correct = len(gt_concepts.intersection(pred_concepts))
                precision = correct / len(pred_concepts) if pred_concepts else 0
                recall = correct / len(gt_concepts) if gt_concepts else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                correct_concepts += correct
                total_concepts += len(gt_concepts)
            else:
                # Case 2: Ground truth is "no concept"
                if not pred_concepts:
                    # Both GT and prediction are "no concept" - correct
                    precision = 1.0
                    recall = 1.0
                    f1 = 1.0
                    correct_concepts += 0  # No concepts to count as correct
                    total_concepts += 0    # No concepts in ground truth
                else:
                    # GT is "no concept" but prediction has concepts - wrong
                    precision = 0.0
                    recall = 1.0  # We found all 0 concepts correctly
                    f1 = 0.0
                    correct_concepts += 0
                    total_concepts += 0
            
            total_samples += 1
            sample_count += 1
            
            # Collect sample results for summary (not logging each sample)
            pass
            
        except Exception as e:
            logger.warning(f"Evaluation failed on sample {sample_count}: {e}")
            continue
    
    # Calculate overall metrics
    overall_precision = correct_concepts / total_concepts if total_concepts > 0 else 0
    overall_recall = correct_concepts / total_concepts if total_concepts > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        "concept_precision": overall_precision,
        "concept_recall": overall_recall,
        "concept_f1": overall_f1,
        "total_samples": total_samples,
        "total_concepts": total_concepts,
        "correct_concepts": correct_concepts
    }


def evaluate_trajectory_accuracy(model, eval_dataset, processor, device, max_samples=None):
    """Evaluate trajectory model based on diagnosis accuracy and response cross-entropy."""
    model.eval()
    total_samples = 0
    correct_diagnoses = 0
    total_loss = 0.0
    
    sample_count = 0
    for batch in eval_dataset:
        if max_samples and sample_count >= max_samples:
            break
            
        try:
            # Get ground truth response and diagnosis
            ground_truth_response = batch["response"]
            ground_truth_diagnosis = batch.get("diagnosis", "")
            
            # If diagnosis field is empty, try to extract from response text
            if not ground_truth_diagnosis:
                diagnosis_match = re.search(r'<diagnosis>(.*?)</diagnosis>', ground_truth_response, re.IGNORECASE) or re.search(r'<final diagnosis>(.*?)</final diagnosis>', ground_truth_response, re.IGNORECASE)
                if diagnosis_match:
                    ground_truth_diagnosis = diagnosis_match.group(1).strip()
                else:
                    ground_truth_diagnosis = ""
            
            # Generate response using the same prompt as training
            prompt = batch["prompt"]
            
            # Create multimodal message format
            messages = get_message(prompt)
            
            # Apply chat template and process with images
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[batch["image"]], return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Calculate cross-entropy loss
                outputs = model(**inputs, labels=inputs.get("input_ids"))
                loss = outputs.loss.item() if hasattr(outputs, "loss") else 0.0
                total_loss += loss
                
                # Generate response
                gen_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            # Extract diagnosis from generated response
            pred_diagnosis_match = re.search(r'<diagnosis>(.*?)</diagnosis>', generated_text, re.IGNORECASE) or re.search(r'<final diagnosis>(.*?)</final diagnosis>', generated_text, re.IGNORECASE)
            predicted_diagnosis = ""
            if pred_diagnosis_match:
                predicted_diagnosis = pred_diagnosis_match.group(1).strip()
            
            # Check diagnosis accuracy (case-insensitive)
            diagnosis_correct = False
            if ground_truth_diagnosis and predicted_diagnosis:
                diagnosis_correct = ground_truth_diagnosis.lower() == predicted_diagnosis.lower()
                if diagnosis_correct:
                    correct_diagnoses += 1
            
            total_samples += 1
            sample_count += 1
            
            # Log sample details for debugging (optional)
            if sample_count <= 3:  # Log first 3 samples
                logger.info(f"Sample {sample_count}:")
                logger.info(f"  GT Diagnosis: {ground_truth_diagnosis}")
                logger.info(f"  Pred Diagnosis: {predicted_diagnosis}")
                logger.info(f"  Correct: {diagnosis_correct}")
                logger.info(f"  Loss: {loss:.4f}")
                logger.info(f"  GT Response: {ground_truth_response[:100]}...")
                logger.info(f"  Pred Response: {generated_text[:100]}...")
            
        except Exception as e:
            logger.warning(f"Evaluation failed on sample {sample_count}: {e}")
            continue
    
    # Calculate overall metrics
    diagnosis_accuracy = correct_diagnoses / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    return {
        "diagnosis_accuracy": diagnosis_accuracy,
        "avg_cross_entropy_loss": avg_loss,
        "total_samples": total_samples,
        "correct_diagnoses": correct_diagnoses
    }

