import os
import argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time
import re

# Initialize the OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(text, context=None, label=None, sublabel=None):
    if isinstance(context, list):
        context_str = "\n".join([str(c).strip() for c in context if str(c).strip()])
    elif isinstance(context, str):
        context_str = context.strip()
    else:
        context_str = ""

    if label or sublabel:
        name = "Image " + (str(label) if label else "") + (str(sublabel) if sublabel else "")
        prompt = f"""
        You are a dermatology expert. The following text is extracted from a dermatology textbook and is associated with the image {name}, though it may also contain unrelated information.

                Your task:
                - Extract and rewrite only the **diagnostic rule(s)** for the condition shown in image {name}, based on symptoms or visual signs described.
                - First, identify which parts of the context are relevant only to this image — ignore unrelated content even if medically relevant.
                - Focus on **how a physician would reason from visual symptoms to diagnosis**, using specific features or patterns.
                - Rewrite the rule(s) in a **concise, declarative format** (not explanatory or descriptive).
                - Do not include any information that is not contained in the original text.
                - **Avoid vague phrases** such as "aids diagnosis", "these features are characteristic", or general comments about appearance.

                Example:
                <Diagnosis rule>Presence of a rough surface and “stuck-on” appearance indicates seborrheic keratosis.</Diagnosis rule>
                <Diagnosis result>seborrheic keratosis</Diagnosis result>

                Formatting rules:
                - If there is a valid diagnostic rule: return it in  
                `<Diagnosis rule>...</Diagnosis rule>`  
                and the corresponding diagnosis in  
                `<Diagnosis result>...</Diagnosis result>`
                - If no diagnostic rule is present, return:  
                `<Diagnosis rule> NOT A RULE </Diagnosis rule>`  
                `<Diagnosis result></Diagnosis result>`
                

              
                Original text:
                Caption:
                {text}
                """
        if context_str:
            prompt += f"\nContext:\n{context_str}\n"

        prompt += "\nOutput format:\n <Diagnosis rule>...</Diagnosis rule>\n<Diagnosis result>...</Diagnosis result>\n<Other hint>...</Other hint>  (optional)"
    else:
        prompt = f"""
                You are a dermatology expert. The following text may or may not describe a diagnostic rule. 
                Your task:
                - Extract and rewrite only the **diagnostic rule(s)** for the condition described in the text, based on symptoms or visual signs.
                - Focus on **how a physician would reason from visual symptoms to diagnosis**, using specific features or patterns.
                - Rewrite the rule(s) in a **concise, declarative format** (not explanatory or descriptive).
                - Do not include any information that is not contained in the original text.
                - **Avoid vague phrases** such as "aids diagnosis", "these features are characteristic", or general comments about appearance.

                Example:
                <Diagnosis rule>Presence of a rough surface and “stuck-on” appearance indicates seborrheic keratosis.</Diagnosis rule>
                <Diagnosis result>seborrheic keratosis</Diagnosis result>

                Formatting rules:
                - If there is a valid diagnostic rule: return it in  
                `<Diagnosis rule>...</Diagnosis rule>`  
                and the corresponding diagnosis in  
                `<Diagnosis result>...</Diagnosis result>`
                - If no diagnostic rule is present, return:  
                `<Diagnosis rule> NOT A RULE </Diagnosis rule>`  
                `<Diagnosis result></Diagnosis result>`
                
                Original text:
                {text}
                """
        
        prompt += "\nOutput format:\n<Diagnosis rule>...</Diagnosis rule>\n<Diagnosis result>...</Diagnosis result>"
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
            # Extract the diagnosis rule
            m = re.search(r"<Diagnosis rule>(.*?)</Diagnosis rule>", answer, re.DOTALL)
            if m:
                diagnosis_rule = m.group(1).strip()
            else:
                diagnosis_rule = ""
            
            # Extract the diagnosis result
            m = re.search(r"<Diagnosis result>(.*?)</Diagnosis result>", answer, re.DOTALL)
            if m:
                diagnosis_result = m.group(1).strip()
            else:
                diagnosis_result = ""
            
            # Extract any other hint
            m = re.search(r"<Other hint>(.*?)</Other hint>", answer, re.DOTALL)
            if m:
                other_hint = m.group(1).strip()
            else:
                other_hint = ""
            
            # Combine the results
            result_parts = []
            if diagnosis_rule:
                result_parts.append(f"Rule: {diagnosis_rule}")
            if diagnosis_result:
                result_parts.append(f"Result: {diagnosis_result}")
            if other_hint:
                result_parts.append(f"Hint: {other_hint}")
            
            return " | ".join(result_parts) if result_parts else ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] LLM call failed: {e}")
                return ""

def main():
    parser = argparse.ArgumentParser(description="Use an LLM to extract image-related diagnostic rules")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="OpenAI model name")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to process")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_rows:
        df = df.head(args.max_rows)

    results = []
    for idx, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        sublabel = str(row.get("sublabel", "")).strip()
        text = str(row.get("text", "")).strip()
        context = str(row.get("context", "")).strip()

        label = label if label.lower() != "nan" and label != "" else None
        sublabel = sublabel if sublabel.lower() != "nan" and sublabel != "" else None

        prompt = build_prompt(text=text, context=context, label=label, sublabel=sublabel)
        rephrase = call_gpt(prompt, model=args.model)
        results.append(rephrase)
        print(f"[Row {idx}] Done. LLM_rephrase: {rephrase[:50]}")

    df["LLM_rephrase"] = results
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
