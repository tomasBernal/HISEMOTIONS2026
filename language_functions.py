import os
import re
import json

import pandas as pd

from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope"]


def filter_response(response, candidate_labels):

    response = str(response).strip().lower()

    response = " ".join(response.split())

    if response in candidate_labels:
        return response

    matches = [label for label in candidate_labels if label in response]

    if len(matches) == 1:
        return matches[0]

    return "invalid-response"

def load_language_model(model_id):
    if model_id == 0:
        model_name = "google/gemma-4-E2B-it"
    elif model_id == 1:
        model_name = "google/gemma-4-E4B-it"

    elif model_id == 2:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif model_id == 3:
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_id == 4:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

    elif model_id == 5:
        model_name = "Qwen/Qwen3.5-2B"
    elif model_id == 6:
        model_name = "Qwen/Qwen3.5-4B"
    elif model_id == 7:
        model_name = "Qwen/Qwen3.5-9B"
    elif model_id == 8:
        model_name = "Qwen/Qwen3.5-27B"
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )

    raw_processor = None

    if model_id in [0, 1]:

        raw_processor = AutoProcessor.from_pretrained(model_name)
        text_processor = raw_processor.tokenizer if hasattr(raw_processor, "tokenizer") else raw_processor
        
    else:
        text_processor = AutoTokenizer.from_pretrained(model_name)

    if model_id in [2, 3, 4]:
        text_processor.pad_token = text_processor.eos_token

    terminators = [text_processor.eos_token_id]

    # If specific models have different terminators, adjust as needed
    if model_id in [2, 3, 4]:
        terminators.append(text_processor.convert_tokens_to_ids("<|eot_id|>"))

    return model, text_processor, terminators

def labels_to_json(row):
    return {
        label: int(row[label]) if pd.notna(row[label]) else 0
        for label in LABEL_COLS
    }

def build_fewshot_examples(train_df, n_shots=3):
    train_df = train_df.copy()
    train_df[LABEL_COLS] = train_df[LABEL_COLS].fillna(0).astype(int)

    positive_df = train_df[train_df[LABEL_COLS].sum(axis=1) > 0]

    if len(positive_df) >= n_shots:
        examples_df = positive_df.sample(n=n_shots, random_state=42)
    else:
        examples_df = train_df.sample(n=n_shots, random_state=42)

    examples = []

    for _, row in examples_df.iterrows():
        examples.append({
            "text": str(row["text"]),
            "labels": labels_to_json(row),
        })

    return examples


def build_prompt(text, fewshot_examples):
    examples_text = ""

    for i, ex in enumerate(fewshot_examples, start=1):
        labels_json = json.dumps(ex["labels"], ensure_ascii=False, indent=2)

        examples_text += f"""Example {i}

Text:
{ex["text"]}

Output:
{labels_json}

"""

    prompt = f"""For the following text, detect whether each emotion is present.

Emotions:
- anger
- fear
- joy
- sadness
- surprise
- hope

Return ONLY a JSON object with 0 or 1 for each emotion.
Do not explain your answer.

Few-shot examples:

{examples_text}
Now classify this text:

Text:
{text}

Output:
"""

    return prompt

def parse_json_response(response):
    default = {label: 0 for label in LABEL_COLS}

    response = response.strip()

    match = re.search(r"\{.*?\}", response, flags=re.DOTALL)

    if not match:
        return default

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except Exception:
        return default

    parsed = {}

    for label in LABEL_COLS:
        value = data.get(label, 0)

        try:
            value = int(value)
        except Exception:
            value = 0

        parsed[label] = 1 if value == 1 else 0

    return parsed