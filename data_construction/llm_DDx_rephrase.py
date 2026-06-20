import os
import argparse
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import re

# Initialize the OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(text):
        prompt = f"""
                You are a dermatology expert. The following text is extracted from a dermatology textbook and contains information about differential diagnosis (DDx).

                Your task:
                - Identify and extract the **differential diagnosis (DDx)** list for a clearly defined subject disease.
                - Only include DDx **based on symptoms or visual/clinical signs**.
                - **Do NOT include** DDx based on histologic, cytologic, or biopsy/pathology findings.
                - The DDx list must be **explicit**, with a **clearly defined subject disease** and at least one listed differential diagnosis.
                - Do not infer or add any diagnoses that are not explicitly mentioned in the text.
                - **Ensure that the subject disease name and all DDx names are written in their full, singular form without abbreviations**.
                - If a disease name appears as an abbreviation (e.g., "BCC"), replace it with its full form only if the full form is explicitly given in the text. 

      
                Output formatting rules:
                - If valid DDx information is found, return in the format:
                `<DDx>{{Subject Disease}} : {{DDx 1}}, {{DDx 2}}, {{DDx 3}}, ...</DDx>`
                - Replace `{{Subject Disease}}` with the name of the disease being differentiated.
                - Replace `{{DDx 1}}, {{DDx 2}}, ...` with the listed differential diagnoses.
                - If the subject disease is missing or unclear, or if the DDx list is ambiguous or incomplete, return:
                `<DDx> NOT A DDX </DDx>`

                Original text:
                {text}
                """
        return prompt


def call_gpt(prompt, model="gpt-4.1", max_tokens=256, temp=0.0, retries=3, sleep_time=0):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                messages=[
                    {"role": "system", "content": "You are a dermatologist and your output is strictly in accordance with the requirements."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            print(answer)
            
            # Extract the DDx information
            m = re.search(r"<DDx>(.*?)</DDx>", answer, re.DOTALL)
            if m:
                ddx_content = m.group(1).strip()
                if ddx_content == "NOT A DDX":
                    return {
                        "subject": None, 
                        "ddx_list": [], 
                        "is_valid": False,
                        "llm_answer": answer,
                        "reason": "explicit_not_a_ddx"
                    }
                else:
                    # Parse the DDx content
                    parts = ddx_content.split(":", 1)
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        ddx_list_str = parts[1].strip()
                        ddx_list = [ddx.strip() for ddx in ddx_list_str.split(",") if ddx.strip()]
                        
                        # Check whether the DDx list contains "NOT A DDX"
                        if "NOT A DDX" in ddx_list:
                            return {
                                "subject": subject,
                                "ddx_list": ddx_list,
                                "is_valid": False,
                                "llm_answer": answer,
                                "reason": "ddx_list_contains_not_a_ddx"
                            }
                        
                        # Lowercase all disease names
                        subject_lower = subject.lower() if subject else ""
                        ddx_list_lower = [ddx.lower() for ddx in ddx_list]
                        
                        return {
                            "subject": subject_lower, 
                            "ddx_list": ddx_list_lower, 
                            "is_valid": True,
                            "llm_answer": answer
                        }
                    else:
                        return {
                            "subject": None, 
                            "ddx_list": [], 
                            "is_valid": False,
                            "llm_answer": answer,
                            "reason": "invalid_format"
                        }
            else:
                return {
                    "subject": None, 
                    "ddx_list": [], 
                    "is_valid": False,
                    "llm_answer": answer,
                    "reason": "no_ddx_tag_found"
                }
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] LLM call failed: {e}")
                return {
                    "subject": None, 
                    "ddx_list": [], 
                    "is_valid": False, 
                    "error": str(e),
                    "llm_answer": None
                }

def process_json_file(json_file_path, model="gpt-4o", max_items=None):
    """Read text blocks from a JSON file and extract DDx information."""
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    total_items = len(data) if isinstance(data, list) else 1
    
    max_items = max_items if max_items is not None and max_items < total_items else total_items
    if isinstance(data, list):
        items_to_process = data[:max_items] if max_items else data
    else:
        items_to_process = [data] if max_items is None or max_items > 0 else []
    
    print(f"Processing {len(items_to_process)} text blocks...")
    
    for idx, item in enumerate(items_to_process):
        # Extract the text
        text = str(item.get("text", "")).strip()
       
        if not text:
            print(f"[{idx+1}/{len(items_to_process)}] Skipping empty text record")
            continue
        
        # Build the prompt and call the LLM
        prompt = build_prompt(text=text)
        ddx_result = call_gpt(prompt, model=model)
       
        # Keep the original info in the result (including invalid records)
        result_item = {
            "index": idx,
            "text": text,
            "ddx_result": ddx_result
        }
        
        results.append(result_item)
        
        # For invalid records, print a skip message
        if not ddx_result.get("is_valid", False):
            reason = ddx_result.get("reason", "unknown")
            print(f"[{idx+1}/{len(items_to_process)}] Invalid DDx record (reason: {reason})")
            continue
        
        # Print progress
        status = "OK" if ddx_result.get("is_valid", False) else "FAIL"
        subject = ddx_result.get("subject", "N/A")
        ddx_count = len(ddx_result.get("ddx_list", []))
        print(f"[{idx+1}/{len(items_to_process)}] {status} Subject: {subject}, DDx count: {ddx_count}")
        
        # Optional delay to avoid API rate limits
        time.sleep(0)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract DDx information from a JSON file")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum number of text blocks to process")
    
    args = parser.parse_args()
    
    # Check the input file
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    print("=== DDx extraction ===")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Max items: {args.max_items if args.max_items else 'all'}")
    print()
    
    try:
        # Process the JSON file
        results = process_json_file(args.input, args.model, args.max_items)
        
        # Statistics
        valid_count = sum(1 for r in results if r["ddx_result"].get("is_valid", False))
        invalid_count = sum(1 for r in results if not r["ddx_result"].get("is_valid", False))
        total_count = len(results)
        
        # Save the results
        output_data = {
            "metadata": {
                "input_file": args.input,
                "model": args.model,
                "total_processed": total_count,
                "valid_ddx_count": valid_count,
                "invalid_ddx_count": invalid_count,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print()
        print("Done.")
        print(f"Total processed: {total_count}")
        print(f"Valid DDx: {valid_count}")
        print(f"Invalid DDx: {invalid_count}")
        print(f"Success rate: {valid_count/total_count*100:.1f}%")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return


if __name__ == "__main__":
    main()
