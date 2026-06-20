import argparse
import json
import os
from pathlib import Path

def is_title_block(text):
    t = text.strip().lower()
    return t == 'differential diagnosis' or (t.startswith('differential diagnosis') and len(t) < 30)

def extract_differential_diagnosis_from_dir(input_dir, output_path):
    input_dir = Path(input_dir)
    results = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == 'text.json':
                text_path = Path(root) / file
                with open(text_path, 'r', encoding='utf-8') as f:
                    try:
                        text_blocks = json.load(f)
                    except Exception:
                        continue
                    for block in text_blocks:
                        text = block.get('text', '')
                        if 'differential diagnosis' in text.lower() and not is_title_block(text):
                            block_info = {k: block[k] for k in block if k in ['text', 'bbox', 'page', 'size', 'font', 'color', 'origin', 'flags', 'ascender', 'descender']}
                            block_info['json_path'] = str(text_path)
                            results.append(block_info)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract differential diagnosis text blocks from text.json files.')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing text.json files')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    args = parser.parse_args()
    extract_differential_diagnosis_from_dir(args.input_dir, args.output) 