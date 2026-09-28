#!/usr/bin/env python3
"""
Minimal inference example for Llama-3-RGDU-DKPO-8B.

This script loads:
  1. Meta-Llama-3-8B-Instruct
  2. the released RGDU-DKPO LoRA adapter

Important:
This example demonstrates standard PEFT inference only.
It does NOT reproduce the paper's inference-time activation suppression.
To reproduce suppression with `act_inhibit_ratio` and the selected FFN layers,
use the modified model/evaluation code from the accompanying code repository.
"""

import argparse
import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal inference example for the RGDU-DKPO LoRA adapter."
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model name or local path.",
    )

    parser.add_argument(
        "--adapter",
        type=str,
        default=".",
        help=(
            "Adapter repository ID or local path. "
            "Default: current directory."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Answer the following question using the provided context.\n\n"
            "Context: The retrieved passage states that Lake Example "
            "has a maximum depth of 120 meters.\n\n"
            "Question: What is the maximum depth of Lake Example?"
        ),
        help="User prompt.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of newly generated tokens.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading RGDU-DKPO LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
    )

    model.eval()

    messages = [
        {
            "role": "user",
            "content": args.prompt,
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    )

    # Put the input tensors on the device of the first model parameter.
    input_device = next(model.parameters()).device
    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    print("\n========== PROMPT ==========")
    print(args.prompt)

    print("\n========== RESPONSE ==========")
    print(response)

    print("\n========== NOTE ==========")
    print(
        "This example loads the released LoRA adapter only. "
        "For inference-time activation suppression, use the "
        "modified model/evaluation implementation from the code release."
    )


if __name__ == "__main__":
    main()
