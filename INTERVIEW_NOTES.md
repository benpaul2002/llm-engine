# P10 — talking notes

Verbal revision sheet. Assume 2-3 minutes total unless they dig.

## Opener

> I wrote an inference engine for Qwen2.5 in PyTorch. The model code is mine — RMSNorm,
> RoPE, grouped-query attention, SwiGLU — I only borrow the pretrained weights and the
> tokenizer. On top of that sits a paged KV cache with prefix caching, a continuous-batching
> scheduler, and speculative decoding.

## 1. Forward pass

- Config-driven: reads `config.json`, so the same code runs the 0.5B and the 1.5B.
- **Prefill vs decode is the split that matters.** Prefill = whole prompt at once, compute-bound.
  Decode = one token per sequence, memory-bandwidth-bound (reload ~3 GB of weights per step).
  This is why batching is nearly free, and it justifies everything downstream.
- GQA is a KV-cache optimization: 12 Q heads share 2 KV heads → cache 6x smaller than MHA.
- Verified by comparing **logits** against HuggingFace with a tolerance, not generated text —
  bf16 reductions aren't order-stable, so token output legitimately varies with batch size.

## 2. Paged KV cache

- Contiguous cache forces reserving max sequence length per request; most is never used.
  Paging fixes it like virtual memory does.
- Fixed-size blocks (16 tokens) from one global pool. Each sequence owns a **block table**;
  logical position → `pos // 16`, `pos % 16`. Fragmentation capped at <1 block per sequence.
- **Tradeoff I took:** attention can't read a contiguous slice, so each step gathers blocks into
  a dense tensor before SDPA. Real copy, per layer, per step. A PagedAttention kernel would walk
  the block table in place — out of scope, and the first thing I'd fix.
- It's the foundation: prefix sharing = two block tables pointing at one block; spec rollback =
  truncating a block table.

## 3. Prefix caching

- Hash each **block-aligned chunk**, chained with the previous block's hash, so a block's
  identity is "these tokens, after exactly this history". Registry maps hash → block.
- Refcounts; block freed only when the last user drops it.
- **Only fully-filled blocks are shareable.** The tail block is about to be written, so it can't
  be shared — which means shared blocks are never written, so no copy-on-write needed.
- Blocks at refcount 0 keep their registry entry until reallocated, so they can be **revived**.
- Stored: `registry: hash → (block_id, chunk)` (chunk compared on hit, so no silent collisions),
  `block_hashes: block_id → hash` (to evict stale entries on reallocation), `seq.prefix_hash`
  (running chain value). The block itself stores only K/V.
- Numbers: 4 requests sharing a 54-token system prompt, 20 blocks → 10. TTFT 84 ms → 23 ms.

## 4. Continuous batching

- Static batching makes everyone wait for the longest request. Continuous batching works per
  decode step: finished sequences leave, new ones take the slot.
- Loop: retire finished → admit (which means **prefill first**, a new request has no KV) →
  one decode step over the running set.
- **Scan-past admission**, capped at 3 skips so long prompts can't starve.
  213 decode steps FIFO vs 184 scan-past; mean batch 2.7 → 3.1.
- **Optimistic admission** — don't reserve prompt + max_new_tokens, almost nothing uses its full
  budget. On exhaustion, **preempt the newest** running request (least work invested), free its
  blocks, requeue at front, rebuild KV on readmission.
  5 requests / 9-block pool: peak batch 2 → 4, 69 → 49 decode steps, same outputs.
- Why recompute not CPU swap: trades PCIe bandwidth for recompute; at this size recompute is
  fast and much simpler.

## 5. Speculative decoding

- Decode is memory-bound, so a forward pass costs ~the same for 1 token or 5. Draft (0.5B)
  proposes k=4; target (1.5B) verifies all k in **one pass**; accept longest valid prefix.
- **Rollback is the hard part.** Reject at position j → discard target KV past j. Cheap with a
  block table. Get it wrong and there's no crash, just subtly wrong text.
  Draft has its own cache, lags by the rejected tokens, needs a catch-up prefill.
- Stochastic acceptance: accept w.p. `min(1, p/q)`, resample from normalized residual `(p-q)+`.
  Output distribution provably matches the target alone. Collapses to exact match at T=0.
- **Lead with the honest result:** correct — 56 tokens from 19 target passes (~3x fewer),
  acceptance 0.52, token-identical output — but **0.8x wall clock**. At batch 1 a decode step is
  dominated by launch overhead, not FLOPs, so the saved target pass is cheaper than 4 draft
  passes. Needs a bigger target; 8 GB wouldn't fit one. Mechanism right, model ratio wrong.
- Not batched: sequences accept different token counts and desync the batch. Deliberate scope call.

## Numbers worth memorizing

| | |
|---|---|
| Batching | batch-8 step = 1.4x batch-1 cost, serves 8. 47 → 156 tok/s |
| Prefix caching | 20 → 10 blocks; TTFT 84 → 23 ms |
| Preemption | 2x concurrency, 29% fewer decode steps |
| Speculation | ~3x fewer target passes, 0.52 acceptance, 0.8x wall clock |

The batching one is load-bearing — it justifies the whole scheduler.

## Limitations (volunteer these)

- KV gather is a copy, not a kernel. First thing I'd fix.
- Speculation is batch-1 only, doesn't compose with the scheduler.
- 8 GB capped model choice, which is why speculation shows no speedup.
- No chunked prefill — a long prompt blocks the decode loop.
- No custom CUDA, no quantization — deliberate; quantization was the previous project.

## Framing

Learning project on the **serving** side — memory management, scheduling, and how they
interact — after a from-scratch C++ transformer that covered the compute side. One concrete
thing learned: the prefill/decode asymmetry drives every design decision in the engine.
