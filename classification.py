import argparse
import torch
import gc
import os

import pandas as pd

from threading import Thread
from transformers import TextIteratorStreamer
from language_functions import LABEL_COLS, load_language_model, parse_json_response, build_fewshot_examples, build_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser()

parser.add_argument('-l', type=int, help='Language model to use', required=True)

args = parser.parse_args()

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

OUTPUT_PATH = "predictions_llm_" + str(args.l) + ".csv"

print("Inicio de modelo " + str(args.l))

if __name__ == '__main__':

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    fewshot_examples = build_fewshot_examples(train_df, n_shots=5)

    model, text_processor, terminators = load_language_model(args.l)

    results = []

    for idx, row in test_df.iterrows():
    
        raw_prompt = build_prompt(
            text=str(row["text"]),
            fewshot_examples=fewshot_examples,
        )

        conversation = [
            {
                "role": "system",
                "content": "You are an expert in emotion analysis of Early Modern Spanish letters."
            },
            {
                "role": "user",
                "content": raw_prompt,
            },
        ]

        prompt = text_processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False)

        # Tokenizing
        inputs = text_processor(text=prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

        streamer = TextIteratorStreamer(text_processor,
                                    skip_prompt=True,
                                    skip_special_tokens=True)

        # Adjusting generation parameters
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=64,
            eos_token_id=terminators, 
            pad_token_id=text_processor.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Generating response
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        response = "".join(text for text in streamer)

        start = response.find("{")
        end = response.find("}")

        if start != -1 and end != -1 and end > start:
            response = response[start:end + 1]

        parsed = parse_json_response(response)


        row_result = {}

        for label in LABEL_COLS:
            row_result[label] = parsed[label]

        results.append(row_result)

        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

print("Fin de modelo " + str(args.l))