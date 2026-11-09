# GLM-5.3 (744B): decode and prefill speed recommendations

A code review of Maya's GLM engine as it serves **GLM-5.3 (744B)** (GGUF architecture `glm-dsa`), with every
recommendation I could find for faster **decode** (writing the answer) and faster **prefill** (reading the prompt).
Written against v1.1.57 (`0065548`). Back to the [model's page](GLM53.md).

## 1. Summary

**Read this first: nothing here was measured.** The review machine has no GPU and no copy of the model. Every
expected gain is an estimate from the code, from the model's shapes, or from a measurement the project wrote down
for another model or PC. Section 8 says how to check each one.

**The three findings that shape everything else.**

1. **A decoded token reads about 21 GB of weights, and two thirds of that is dense weights - mostly the
   attention's projections - not experts.** GLM-5.3 has a full attention on each of its 78 layers; Flash has one on
   11 of 45. So GLM-5.3 is bound
   by memory bandwidth and by how many experts sit in VRAM - and on a prompt, by how often the experts must be
   streamed (once per chunk, about 10 s on one V100).
2. **Much of what would make it faster is already in the tree but off, gated to Flash's sizes, or not wired.**
   Context-cache streaming is implemented end to end and disabled by one product flag. Decode graphs, the
   tensor-core decode attention and native Q8_0 attention weights wait "until validated". Six pieces of the
   speculative decode are not reached by its loop. Section 4 lists all of it.
3. **Several defaults and one data file do not fit GLM-5.3.** The shipped expert profile ranks 8 of the last
   layer's experts among its first 4,800 places, where every other layer has 49 - 78; the tier's ageing, the split's
   chunk cap and the short-prompt threshold were measured on Flash.

**The recommendations with the most value for the effort.**

| | Recommendation | Expected effect (estimate) | Effort | Changes the answers? | See |
| --- | --- | --- | --- | --- | --- |
| **Decode** | | | | | |
| 1 | Turn on the cache streaming the engine already has | Frees 4.9 GiB of VRAM at a 128K context, 11 GiB at 256K: about 450 / 1,100 more experts resident, and larger prompt chunks | Validation, one flag | No | M1 |
| 2 | Keep `attn_k_b` / `attn_v_b` in native Q8_0 | -1.1 GB of VRAM and -5% of the bytes every token reads | A parity run, a pack rebuild | Within rounding (closer to the GGUF) | M3 |
| 3 | FP16 index keys by default | -9% cache; half the key bytes the indexer reads (saves 0.7 GB a token at 128K) | A flag, a KL check | Slightly | M2 |
| 4 | Validate and enable decode graphs | About 1,500 kernel launches a token become one replay a GPU; Flash gained 8 - 9% from a smaller version | Validation | No | G1 |
| 5 | Tensor cores for the decode attention | The attention is 11 G multiply-adds a token on CUDA cores; Flash's equivalent kernels gained about 4x | Settings first, then one kernel | Within rounding | G2 |
| 6 | CPU lane: pick the kernel per expert type by timing, huge pages, per-class calibration | 10 - 30 ms a token on a RAM-bound PC | Small to medium | No | C1 - C3 |
| 7 | Finish the speculative decode's wiring; measure acceptance; let the tuner choose "no drafting" | Removes cases where drafting is slower than not drafting (non-Latin answers on 3+ GPUs; RAM-bound splits) | Small | No | S1 - S3 |
| 8 | Expert tiers: rebuild the profile, free unused spare slots, RAM-resident mode from the setup, GLM-5.3 constants | A few percent more experts in VRAM; no churn where RAM holds the model | Small | No | T1 - T5 |
| 9 | A GLM-5.3 Turbo preset (skip the least important missing experts) | The largest single lever on a RAM-bound PC; Flash measured +7% to +62% | A KL sweep over 75 layers | **Yes** (opt-in) | A1 |
| **Prefill** | | | | | |
| 10 | On a split, pipeline reads that fit in one chunk | -10% to -18% on 4K - 16K reads (fit to the 3x V100 timings) | A setting to try today, then a small change | No | P1 |
| 11 | Size the prompt's buffers for GLM-5.3's one residual stream, not Flash's four (and row 1) | About 25% more tokens a chunk on one GPU, each avoided chunk about 10 s; 1.2 GB of pinned RAM back per GPU of a split | Small, exact | No | P2 |
| 12 | Start the CPU's share at the plan; hand the routes over before the shared expert | Seconds per long read where a second lane exists | Small | No | P3, P4 |
| 13 | A faster prompt attention kernel, on every architecture | The attention is about 30% of a long read; the kernel reaches about 25 TFLOPS on a V100 (cuBLAS: 53), and Turing and non-RDNA AMD run F32 | A new kernel | Within rounding | K1 |
| 14 | Continue from the decoded answer instead of reading it again | Agent turns read only the new tool result | Small to medium | No | R1 |

**Where to start** depends on the PC (section 7.4): on the usual RAM-bound PC, rows 1 - 3, 6 and 8 come first; on a
PC that holds the model in VRAM, rows 4 and 5.

**One caveat.** The project's own long-context quality gate does not pass on this model yet at 32K (section 3.4).
Settle that before spending effort on long-context speed: the fix may change what the indexer costs.

## 2. Scope, method and how far to trust each claim

**What was read.** The whole GLM-5.3 path: the decode token path (`src/core/glm_fast_path.cu`,
`glm_residual.cu`, `glm_dsa_mixer.{hpp,cu}`, `glm_dsa_fast*.{cu,cpp,hpp}`), its kernels (`src/kernels/cuda/glmd_*.cu`
and `.cuh`, the GLM-5.3 parts of `glm_fast.cu`), the expert tiers and the CPU lane (`glm_tier_*`, `glm_heat_evict`,
`glm_warm_seq`, `src/kernels/cpu/*`), drafting (`glm_mtp*.cu`, `glm_window.cu`, `glm_dsa_spec.cpp`,
`glm_ngram_draft.cpp`), the prompt path (`glm_prefill.cu`, `glm_prefill_dsa.cu`, `glmd_batch.cu`,
`glm_dsa_prompt_plan.cpp`, `glm_dsa_prefill_stream*.cpp`, `src/prefill/*`), the state / cache / split code
(`glm_state_layout.cpp`, `glm_model.cu`, `glm_dsa_split.cpp`, `src/kernels/cuda/glm_kv_stream.cu`), the pack tools
(`tools/iq_pack.py`, `tools/strata/glm_dsa_pack.py`) and what the setup writes for it (`maya.py`,
`serve/glm_dsa_product.py`, `serve/glm_dsa_tune.py`).

**What was not done.** Nothing was run. The review machine has no GPU and no copy of the 239 GB model, so there is
**no new measurement in this report** (the one file that was processed is the shipped expert profile, T1). The
review was done in several parallel reading passes; the findings kept here were then re-checked against the lines
they cite. Every number is one of three kinds, and each is labelled:

| Label | Meaning |
| --- | --- |
| **[code]** | Read directly in the source; the `file:line` is given. |
| **[project]** | A measurement the project itself wrote down (a code comment, `docs/SETTINGS.md`, the changelog). Quoted, not re-measured. Most are for GLM-5.3-Flash or for a 3x V100 PC, and are named as such. |
| **[estimate]** | My arithmetic from the model's shapes and the pack's tensor types. Expect ±20%. |

A recommendation's *expected effect* is therefore a reasoned estimate, never a promise. Section 8 says how to measure
each one with the tools the repository already has.

**Four kinds of PC.** Which recommendation matters depends on where a token's time goes, so the report keeps
naming these. The first three are **the targets** (the owner's decision, section 1); the fourth is not optimised at
this stage.

- **RAM-bound** *(target)* - the usual GLM-5.3 PC today ("64+ GB of GPUs + 192 GB RAM", `docs/MEMORY.md:55`): most
  experts sit in the pinned RAM tier, and a token waits on the CPU lane and the PCIe link.
- **Disk-bound** *(target)* - VRAM and RAM together do not hold the model: part of every token comes from the NVMe.
- **Unified memory** *(target)* - the Gorgon Halo with 160 GB of GPU memory and 32 GB of RAM, running a GGUF of
  about 170 GB. Nearly every expert is in GPU memory, and a token is bound by the memory's bandwidth. Section 7.
- **VRAM-rich** *(not a target now)* - several discrete GPUs that hold every expert. The items that serve only this
  kind are marked "deferred". Most of what makes it fast (graphs, the attention kernels, the verify window) is
  needed anyway: by the Gorgon Halo, and by Turbo, which turns a RAM-bound PC into a GPU-bound one (section 5.7).

## 3. Where the time goes

### 3.1 A decoded token reads about 21 GB of weights

GLM-5.3 differs from GLM-5.3-Flash in one way that changes everything about its speed: **every one of its 78 layers
is a full attention layer with large dense projections**, where Flash has 11 such layers among 45. The routed
experts are no longer most of what a token reads.

Per layer, with the tensor types of unsloth's UD-IQ2_M as `tests/glm_dsa/glmd_dec_bench.cu:162-187` lists them
**[code]**, in megabytes **[estimate]**:

| Read by every token | Type in the pack | MB a layer | Layers | GB a token |
| --- | --- | ---: | ---: | ---: |
| `attn_output` (16384 -> 6144) | Q5_K | 69.2 | 78 | 5.40 |
| `attn_q_b` (2048 -> 16384) | Q8_0 | 35.7 | 78 | 2.78 |
| `attn_v_b` (64 x 256 x 512) | BF16 (upcast from Q8_0) | 16.8 | 78 | 1.31 |
| `attn_k_b` (64 x 512 x 192) | BF16 (upcast from Q8_0) | 12.6 | 78 | 0.98 |
| `attn_q_a` (6144 -> 2048) | Q5_K | 8.7 | 78 | 0.67 |
| `attn_kv_a_mqa` (6144 -> 576) | Q8_0 | 3.8 | 78 | 0.29 |
| indexer projections (full layers only) | BF16 | 18.7 | 21 | 0.39 |
| shared expert (gate, up, down) | Q5_K, Q5_K, Q6_K | 27.6 | 75 | 2.07 |
| router `ffn_gate_inp` | F32 | 6.3 | 75 | 0.47 |
| dense FFN (layers 0-2) | Q5_K, Q5_K, Q6_K | 165.7 | 3 | 0.50 |
| **dense weights, without the head** | | | | **≈ 14.9** |
| output head (154,880 x 6144) | native | | 1 | 0.5 - 1.0 |
| **routed experts, 8 a layer** | i-quants, ≈ 9.7 - 14.5 MB each | ≈ 92 | 75 | **≈ 6.9** |
| **all of it** | | | | **≈ 21 - 22** |

Two cross-checks against the project's own figures: it measured 14.5 GB of dense weights in VRAM on 3x V100
(`serve/glm_dsa_product.py:38`) **[project]**, and the file size (238.6 GB) less the dense weights, over the 19,456
experts of the trunk and the NextN block, gives 11.5 MB an expert on average **[estimate]**.

What follows from the table:

1. **Two thirds of a token's bytes are dense weights, and three quarters of those are the attention's
   projections.** `attn_output` and `attn_q_b` alone are 8.2 GB a token - more than all eight routed experts of all
   75 layers. On Flash the experts dominated; on GLM-5.3 they do not.
2. **A token has a bandwidth floor even with every expert in VRAM.** The project's own dense GEMV reaches about
   575 GB/s on a V100 (a Q6_K 8192 x 4096 GEMV in 47.8 us, `src/kernels/cuda/glm_fast.cu:1510-1511`) **[project]**.
   At that rate 21 GB is about 37 ms: a token-by-token decode cannot pass roughly 27 tok/s on one V100-class card
   whatever else is tuned **[estimate]**. On a Strix / Gorgon Halo, whose GEMVs are bound by LPDDR5X at 220-240 GB/s
   (`src/kernels/cuda/glmd_mv_ext.cuh:375-376`) **[project]**, the floor is about 90 ms a token.
3. **So the only ways to go faster are** (a) read fewer bytes a token, (b) read the bytes once for several tokens
   (drafting with a verify window - the dense 15 GB are shared by every row of a window), (c) read them on several
   memory systems at once (GPUs working on consecutive tokens; both CPU sockets; every PCIe link), and (d) put the
   bytes where the bandwidth is highest (more experts in VRAM). Every decode recommendation below is one of these
   four, plus the smaller matter of launch overhead and kernel efficiency.

### 3.2 What else a token costs

| Cost | Size | Source |
| --- | --- | --- |
| Kernel launches | about 19 a layer that reuses a selection, 21 a full-indexer one, 8 of them the MoE's: roughly **1,500 a token** | **[code]** counted from `glm_residual.cu:110-160`, `glm_dsa_mixer.hpp:167-328`, `glm_fast_path.cu:6169-6355`, `glmd_fast.cu:404-454` |
| Attention arithmetic | 78 layers x 2,048 cells x 64 heads x (576 + 512) = **11.1 G multiply-adds a token** once the context passes 2,048 | **[estimate]** |
| Indexer scores | context x 32 heads x 128 on each of the 22 full layers (21 + the NextN block): 3.0 G multiply-adds a token at 32K, 11.8 G at 128K | **[estimate]** |
| Index keys read | 22 x context x 512 B (F32 keys): 0.37 GB a token at 32K, **1.5 GB at 128K, 11.8 GB at 1M** | **[estimate]** from `serve/glm_dsa_product.py:110-124` |
| A missing expert | about 1 ms over PCIe 3.0 (the fetch kernel pulls 11.5 GB/s, `glm_fast.cu:4639`), or its share of the CPU lane | **[project]** |

### 3.3 A prompt chunk streams most of the model

For a chunk of T tokens **[estimate]**:

| Work | Per 8,192-token chunk |
| --- | --- |
| Dense GEMMs (FP16, cuBLAS) | 262 TFLOP |
| Routed experts (MMQ, int8) | 371 TFLOP-equivalent |
| Attention (2,048 cells a query) | 182 TFLOP |
| Indexer scores | T x context x 4096 x 2 x 21: 92 TFLOP for a whole 32K prompt, 1,480 for 128K, 94,600 for 1M |
| **Expert bytes that must reach a GPU or the CPU lane** | **nearly all 19,200 experts, ≈ 220 GB, once a chunk** |

The last row is the one that decides GLM-5.3's prompt speed on a RAM-bound PC. A chunk routes 8 x T times into 256
experts a layer, so 100 tokens already touch about 96% of a layer's experts and 30 tokens about 60% (uniform routing;
real routing is skewed, so a little fewer) **[estimate]**. The code says the same from a measurement: "the card holds
~18 of a layer's 256 experts, so every chunk streams nearly the whole model (~110 GB on a V100, a fifth of it from
the disk)" (`src/core/glm_prefill.cu:1258-1260`) **[project]**. The only GLM-5.3 prompt timings in the repository
show that fixed cost plainly **[project]**:

| PC | 2K tokens | 4K | 8K | 16K | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| 1x V100, UD-IQ2_M | 15.9 s | | 33.6 s | 68.0 s | `glm_prefill.cu:1261-1262` |
| 3x V100, UD-IQ2_M | 13.3 s | 20.5 s | 33.0 s | 45.7 s | `glm_prefill.cu:2428-2429` |

Read as a line, each PC shows a fixed cost of very roughly 6-10 s a chunk and then about 3 ms a token inside it. So
a short read is mostly the fixed cost - how many chunks it is cut into and how fast expert bytes move (PCIe links,
the CPU lane, the SSD) - while a long chunk is mostly the per-token part, which is the kernels: at 8K tokens it is
70 - 80% of the read on both PCs **[estimate]**. Section 6.1 has the fit.

### 3.4 One caveat before tuning long contexts

The project's own long-context gate does not pass yet on this model: "Measured 2026-11-07 (UD-IQ2_M, 3x V100): 8K
docs -0.06 (PASS), 16K docs+code +0.14, 32K docs+code +1.38 (FAIL; GLM-5.3-Flash on the same file +0.22)", and "the
same 8K rows read with the attention sink + the last 2,044 cells instead of the indexer's choice ... score 2.45
against the indexer's 3.35" (`tools/strata/glm53_long_gate.py:18-21`) **[project]**. That is a quality question, not
a speed one, and this report does not try to answer it. It matters here for two reasons: every recommendation that
touches the indexer or its keys (M2, G3) must be judged against a baseline that is itself still moving, and whatever
resolves it may change what the indexer costs. Settle it before spending effort on long-context speed.

## 4. What GLM-5.3-Flash has that GLM-5.3 does not reach yet

GLM-5.3 was brought up on Flash's engine, and much of what makes Flash fast is switched off, gated to Flash's
shapes, or not wired for it. This table is the shortest summary of the report: most recommendations close one of
these gaps. Everything in it is **[code]**.

| What | State for GLM-5.3 today | Evidence | See |
| --- | --- | --- | --- |
| Context-cache (KV) streaming | The engine implements it; the setup writes `--kv-streaming off` | `serve/glm_dsa_product.py:25-29`, `maya.py:3487-3497` | M1 |
| Decode graphs (a part's layer loop as one CUDA graph) | Off "until validated on the test PC" | `include/strata/kernels/glmd_graph.hpp:25` | G1 |
| Tensor cores for the decode attention | Flash: on for every NVIDIA generation. GLM-5.3: an F32 CUDA-core kernel; the tensor-core one is opt-in | `include/strata/kernels/glmd_decode_policy.hpp:73`, `:87` | G2 |
| The newer decode attention on AMD | Only gfx1151; other Radeons run the older kernels | `src/kernels/cuda/glmd_dec.cu:265-279` | G2 |
| Tensor-core indexer scorer in decode | Above 16,384 cells only; never on AMD | `glmd_decode_policy.hpp:100-120` | G3 |
| Native Q8_0 `attn_k_b` / `attn_v_b` | Upcast to BF16 "until the hardware parity run" | `tools/strata/glm_dsa_pack.py:115-125` | M3 |
| Drafting on one GPU | Off (the NextN block is not loaded) | `src/core/glm_mtp.cu:85-102` | S5 |
| A pipeline deeper than two stages | The pipelined window calls Flash-only code | `src/core/glm_window.cu:1365`, `:1399` | S6 |
| Draft features: whole-head fallback, n-gram drafts, early draft, cancel, statistics | Present in the tree, not reached by the decode loop | section 5.3, S1 | S1 |
| A drafter of its own (DFlash2) | Refused for this architecture | `src/core/glm_maya_sync.cpp:17-22` | - |
| Batched `--parallel` rows | Refused: one conversation's token at a time | `glm_maya_sync.cpp:24` | S7 |
| Turbo | Hidden until "its own values are measured on it" | `serve/models.py:837-840` | A1 |
| Deferred CPU-lane add | Only in Flash's layer loop | `src/core/glm_fast_path.cu:6376-6381` | A2 |
| Tuned AMD expert kernels (decode), fused AMD prompt experts | Require `n_embd == 4096` | `src/kernels/cuda/glm_fast.cu:6913`, `src/core/glm_prefill.cu:313-316` | T7, K5 |
| `mma.sync` prompt attention (Ampere+), Turing tensor-core attention | Flash has both; GLM-5.3 has one WMMA kernel, and F32 on Turing | `src/core/glm_prefill_dsa.cu:540-583` | K1 |
| Huge pages / node-split RAM tier | The port's own code for it is never called | `src/core/glm_tier_numa.cu:236-241` | C2, C4 |
| A helper GPU for experts | Refuses `n_embd > 4096` | `glm_fast_path.cu:2249` | T8 |
| The waiting GPU helping a one-chunk prompt | EXL3 only, or NVLink only, and two GPUs only | `glm_prefill.cu:2842`, `:4673-4676` | P1, P6 |
| Tier constants, the split's chunk cap, the token-path threshold, the expert profile | Flash's values or Flash-era data | T1, T5, P1, P6 | T1, T5, P1 |
| Setup: per-PC tuning | `--calibrate` sweeps three knobs; the setup writes no GLM-5.3 switch on NVIDIA | `tools/calibrate_glm.py:84-88`, `maya.py:3495-3502` | 7.1 |

Two of these are bigger than they look. The decode attention runs on 79 layers a token here, against 11 on Flash,
so its kernel choice matters seven times as much. And the context cache is five times Flash's per token, so cache
streaming frees five times the VRAM at the same context.

## 5. Decode recommendations

Each one: what to do, the evidence, what to expect, the effort and risk, how to check it. The order inside each
group is by expected value for the effort.

### 5.1 Memory: put more experts where the bandwidth is

On a RAM-bound PC the fastest thing to do for a token is to stop it missing. Each ~10-11.5 MB freed in VRAM is one
more expert that no longer costs a PCIe pull or CPU-lane time. Three changes free gigabytes each, and none of them
needs a new kernel.

#### M1. Turn on the context-cache streaming the engine already has for GLM-5.3

**What.** The setup writes `--kv-streaming off` into every GLM-5.3 config and the docs say streaming is "not served
on GLM-5.3 yet". The engine code says otherwise: GLM-5.3's streaming is implemented end to end (its comments call it
"L1-6"). Validate it, then flip the product flag.

**Evidence [code].**
- The state layout carves a resident window for glm-dsa: "L1-6 glm-dsa: the latent AND k_pe hold lat_rows slots (one
  residency map for both), the index keys stay whole" - only the slow F32 reference path is refused
  (`src/core/glm_state_layout.cpp:105-112`).
- The host copies and residency maps are made for glm-dsa, k_pe included (`src/core/glm_model.cu:887-959`, k_pe at
  `:905-906`); the decode mixer resolves a selection into VRAM slots before the attention
  (`src/core/glm_dsa_mixer.cu:393-443`, `glm_dsa_mixer.hpp:304-308`); `dsa_prep` writes the host copy
  (`src/kernels/cuda/glmd_fast.cu:143-176`); the prompt path stages a streamed layer
  (`src/core/glm_prefill_dsa.cu:468-503`, `:530-531`, `:802`); the saved slots and the prompt cache know about it
  (`src/core/glm_fast_path.cu:8201`, `glm_prompt_cache_model.cu:92-106`).
- There is even an automatic rule for it: with `STRATA_GLM_KV_AUTO=1`, "`--kv-streaming on` ... streams a glm-dsa
  model's caches only when the whole cache does not fit" (`src/core/glm_model.cu:855-860`, `:2014-2029`).
- The product layer still says no: `KV_STREAMING = False`, with the comment "glm_state_carve refuses a resident
  window smaller than the context on glm-dsa" (`serve/glm_dsa_product.py:25-29`) - which the carve above no longer
  does. `maya.py:3487-3497` therefore writes `--kv-streaming off`, `maya.py:3582-3584` drops an older config's
  streaming flags, and `maya.py:1902-1924` offers only the contexts whose whole cache fits.
- A gap to close first: nothing under `tests/` runs a streamed GLM-5.3 window. The test hook exists
  (`STRATA_GLM_KV_RESIDENT_TEST_MIN`, `glm_model.cu:867-870`) but no test in this repository sets it.

**Why it matters for GLM-5.3 [estimate].** The cache is 64,352 bytes a context token, all in VRAM. Only the index
keys (11,264 bytes a token) must stay resident - the indexer scores every position - so 53,088 bytes a token can
stream:

| Context | Whole cache in VRAM | With a 32,768-position window | Freed | ≈ expert slots |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 1.96 GiB | 1.96 GiB (not streamed) | 0 | 0 |
| 128K | 7.86 GiB | 3.00 GiB | 4.86 GiB | ~450-530 |
| 256K | 15.71 GiB | 4.37 GiB | 11.34 GiB | ~1,050-1,250 |
| 1M | 62.84 GiB | 12.62 GiB | 50.22 GiB | ~4,700-5,500 |

**Expect.** At 32K nothing. From 128K up: more experts resident (fewer misses a token), a bigger prompt chunk (the
chunk's buffers are borrowed from the same pool, section 6), and contexts that today "would not start" on 24-32 GB
cards. For Flash the project measured, at 512K on an RTX 3090, "1,176 expert slots instead of 714, and prompts 4-9x
faster ..., decode +5%" (`docs/SETTINGS.md:163`) **[project]**. GLM-5.3's cache is 5x Flash's per token, so the
effect at long context should be larger.

**Cost to watch.** A streamed layer adds a resolve (and up to two small copy launches) before its attention, on 79
layers a token; a miss is a PCIe read. Measure decode at 128K with the window resident against streamed.

**A follow-up that is specific to GLM-5.3 [estimate].** 57 of the 78 layers reuse the selection of the full layer
before them (`glm_dsa_mixer.hpp:281-285`), but each streamed layer still resolves that same list through its own map
(`glm_dsa_mixer.cu:420-442`). One residency map per *selection group* (a full layer and the shared layers behind it:
the same cells, so the same blocks) would cut the resolves from 79 to 22 a token.

**Effort / risk.** Small in code (a flag, the docs, the offered contexts); the work is validation. Decode should give
the same tokens as the resident cache (it does on Flash, `docs/SETTINGS.md:163`); a prompt can differ slightly, as
with any change in which experts fit.

**Check.** A streamed variant of `glm_dsa_prefill_decode_test` (using `STRATA_GLM_KV_RESIDENT_TEST_MIN` so a short
context evicts), `tools/strata/glm53_gpu_checks.sh long` at 128K, then token identity and `bench_http.py` at 32K /
128K with and without.

#### M2. Store the index keys in FP16 by default

**What.** Make `MAYA_GLM_IDX_FP16=1` the default layout for GLM-5.3.

**Evidence [code].** The keys are F32 unless the switch is set (`src/core/glm_state_layout.cpp:154-161`, `:239-241`).
They are 22 x 128 x 4 = 11,264 of the 64,352 bytes a token (`serve/glm_dsa_product.py:110-124`). The tensor-core
scorer - the prompt path's default and the decode's above 16,384 cells - **already rounds F32 keys to FP16 before it
multiplies**: `d[0] = __halves2half2(half_sat(v.x), half_sat(v.y))` (`src/kernels/cuda/glmd_volta_idx.cu:116-126`;
the prompt path's scorer likewise works "from FP16 queries and keys with F32 accumulation",
`src/kernels/cuda/glmd_batch.cu:269`). So wherever that scorer runs, F32 storage buys no precision: the scores are
FP16-key scores either way. Only the
CUDA-core scorer (decode below 16,384 cells, and AMD) sees the difference.

**Expect [estimate].** 8.75% less cache (5,632 bytes a token: 0.69 GiB at 128K, 5.5 GiB at 1M - about 65 and 520
expert slots) and half the key bytes the scorer reads a token (0.74 instead of 1.48 GB at 128K; 5.9 instead of 11.8 GB
at 1M, where the keys are otherwise half as much traffic as the weights).

**Effort / risk.** Small. The docs note that the answers can differ (the selection comes from FP16 scores); the
prompt path's own check found the F32 and tensor-core selections share 99.97% of their cells
(`src/core/glm_prefill_dsa.cu:13-14`) **[project]**. A saved slot of one layout is refused by the other
(the fingerprint, `glm_state_layout.hpp:123-131`), so a flip should bump nothing else. Needs a KL check against F32
keys at long context (`glm53_gpu_checks.sh kld`, `long`).

**Later, if long contexts matter:** INT8 keys with one FP16 scale a key (136 bytes instead of 512) for a first pass,
then an exact re-score of the cells near the cut-off from the full-precision keys. The top-2,048 would stay exact
(a cell far from the 2,048th score cannot change sides), at about a quarter of the key traffic. This is a research
item, not a quick win.

#### M3. Keep `attn_k_b` / `attn_v_b` in their native Q8_0 (`STRATA_PACK_NATIVE32`)

**What.** Build GLM-5.3's pack with `STRATA_PACK_NATIVE32=1` once its parity run is done, and make that the default.

**Evidence [code].** The default pack dequantizes the absorbed MLA projections from the GGUF's Q8_0 to BF16
(`tools/strata/glm_dsa_pack.py:55-60`, `:140-144`). Behind `STRATA_PACK_NATIVE32=1` they stay native: "Default off
until the hardware parity run (BUILD-PLAN B10.7: greedy 64 tokens identical or KLD < 1e-4 vs the BF16-upcast build)"
(`glm_dsa_pack.py:115-125`). The engine side is there: the binder serves a native Q8_0 slot as it is
(`src/core/glm_dsa_fast.cu:197-205`, `glm_dsa_fast_plan.cpp:46-52`), the decode's headwise GEMV has a Q8_0 dot
(`src/kernels/cuda/glmd_volta_mla.cu:94-109`), and the prompt path dequantizes it to FP16
(`src/core/glm_prefill_dsa.cu:238-245`).

**Expect [estimate].** BF16 doubles these two tensors: 29.4 MB a layer instead of 15.6. Native Q8_0 saves 13.8 MB
on each of 79 layers - **about 1.1 GB of VRAM (≈ 100 more expert slots) and 1.1 GB fewer bytes read by every
token** (5% of a token's traffic, 7% of its dense traffic).

It is also the more faithful copy: BF16 keeps 8 bits of mantissa, while a Q8_0 value is a 7-bit integer times an
FP16 scale, so the upcast rounds ("the default pack's rounding: dequantize, then BF16 RN-even",
`src/core/glm_dsa_mixer.cu:179-180`). Native Q8_0 is what llama.cpp itself multiplies.

**Effort / risk.** The parity run the code already asks for, then a pack rebuild for existing installs (the setup
rebuilds a pack without downloading again). The same idea would serve the indexer's three projections (BF16 in the
pack, Q8_0 / F32 in the GGUF: 18.7 -> 10.5 MB on 22 layers), but they feed the selection, so treat that as a separate,
smaller step.

#### M4. Offer a smaller dense quant at load (optional, changes the answers)

**What.** The DeepSeek engine has `MAYA_DS4_DENSE_REQ=q5_k`: the attention's largest Q8_0 matrices re-quantized at
load, cached in the pack. GLM-5.3 would gain more from the same switch.

**Evidence [project].** On DeepSeek-V4.1-Flash it took the dense weights from 8.46 to 6.70 GB, added 184 experts to
VRAM and 6% to decode, at KL 0.027 against 0.019 (`docs/SETTINGS.md:1330-1349`); "the gain comes from the experts the
freed VRAM holds, so it is largest where the CPU lane runs many experts".

**Expect [estimate].** GLM-5.3's `attn_q_b` is Q8_0 on all 78 layers: 2.78 GB. At Q5_K it is 1.80 GB - about 1 GB of
VRAM and of per-token reads. Taking `attn_output` from Q5_K to Q4_K would save another 0.98 GB. Unlike M1-M3 this
changes the model's output, so it belongs with the opt-in approximations (section 5.6) and needs its own KL table
before anyone is pointed at it.

### 5.2 The GPU path: fewer launches, tensor cores for the attention

#### G1. Validate and enable the decode CUDA graphs (`MAYA_GLM_DSA_GRAPHS`)

**What.** GLM-5.3 can replay a GPU part's whole layer loop as one captured graph a token. It is off: "default off
until validated on the test PC" (`include/strata/kernels/glmd_graph.hpp:25`).

**Evidence [code].** The switch reads `false` by default (`src/kernels/cuda/glmd_graph.cu:27-32`). The capture
covers every layer of the part plus the residual close (`src/core/glm_residual.cu:94-167`), keyed on everything that
could change a kernel choice (`glmd_graph.hpp:93-110`); positions and counts are device data, so a replay is safe
(`glmd_graph.hpp:4-24`). A token issues about 1,500 launches without it (section 3.2).

**Expect.** The closest measurements are Flash's **[project]**: its DSA layers' feed-forward in graphs gave "about
8-9% faster" decode (CHANGELOG v1.1.52), its KDA layers +2.35% (`docs/GLM-KDA-GRAPHS.md`). GLM-5.3 captures far more
per token (78 attention layers, not a feed-forward on 11), so the same or better is plausible on a VRAM-rich PC
**[estimate]**. On a RAM-bound PC the gain is smaller in percent, but there is a second effect: the thread that
issues the launches competes with the CPU lane for cores (`docs/SETTINGS.md:39-47`), and a replay needs almost none
of it.

**Limits today [code].** Graphs stay off for a request with profiling, seam dumps, expert prefetch or lookahead
(`src/core/glm_dsa_fast.hpp:141-147`), and a part re-captures when the context crosses a power of two or a kernel
regime (`glmd_graph.hpp:67-80`) - a handful of times a session.

**Effort / risk.** Validation, not code. The tests exist: `glmd_graph_test`, `glmd_graph_gen_gpu_test`, and
`glm_dsa_prefill_decode_test_volta`, which already runs with `STRATA_GLM_DSA_GRAPHS=1`
(`cmake/glm_dsa.d/55-glm-dsa-wire.cmake:556-557`).

**Check.** Greedy token identity graphs on / off (a request can switch them: `"strata_tune": {"graphs": false}`,
`docs/SETTINGS.md:2221`), then `bench_http.py` decode at 4K and 32K, on one GPU and on a split, with the pipelined
speculative decode on.

#### G2. Put GLM-5.3's decode attention on the tensor cores by default

**What.** Flash's decode attention runs on tensor cores by default on every NVIDIA generation. GLM-5.3's does not -
and GLM-5.3 runs it on 79 layers a token instead of 11.

**Evidence [code].**
- GLM-5.3's default kernel is `mla_dec`: "Arithmetic f32 throughout" on the CUDA cores
  (`include/strata/kernels/glmd_dec.hpp:13`, `src/kernels/cuda/glmd_dec.cu:156-204`). One 256-thread block an SM
  (`__launch_bounds__(kThreads, 1)`, 74 KB of shared memory, `glmd_dec.cu:73`, `:255-259`).
- A tensor-core kernel exists for it - `glmd::mla_fused`, FP16 WMMA with F32 sums - but only behind
  `STRATA_GLMD_FUSED_MLA=1` (`include/strata/kernels/glmd_decode_policy.hpp:73`, `:87`;
  `src/kernels/cuda/glmd_volta_mla.cu:173-345`).
- Flash, by contrast: `mla_split_mma_kernel` (`mma.sync`) on by default on sm_80+
  (`src/kernels/cuda/glm_fast.cu:6450-6469`) and the Volta tensor-core kernel on by default on sm_70
  (`glm_fast.cu:6533-6540`). Neither takes GLM-5.3's rows: `MLA_KV` is fixed at 512 with no rope tail
  (`glm_fast.cu:3496-3504`), and GLM-5.3's cell is 512 + 64.
- On AMD only gfx1151 gets even `mla_dec`; every other Radeon runs the older `mla2` kernels "until someone measures
  it there (it builds and fits on all of them)" (`glmd_dec.cu:265-279`).

**How much is at stake [project].** Before `mla_dec`, the attention "took ~85 us a layer on a V100 at a short context
and ~240 us at the 2,048 selected cells ...: 6.8 ms of every token's GPU time at a 150-token context, ~19 ms at 2K+"
(`glmd_dec.hpp:3-5`). That is still what a Radeon other than gfx1151 runs. For Flash's kernels of the same shape,
tensor cores were worth about 4x: "RTX 5060 Ti 49 -> 11 us a token and DSA layer", "A V100: the attention
192 + 43 -> 49 + 6 us a layer" (`docs/SETTINGS.md:166-167`).

**Do, in this order.**
1. *AMD, no code:* run `glmd_dec_bench` on gfx1100 (RX 7900) and gfx12 (RX 9070 / R9700) with
   `STRATA_GLMD_MLA_DEC=1`, and widen `auto_device` (`glmd_dec.cu:270-279`) where it wins. Possibly several
   milliseconds a token for free.
2. *NVIDIA, no new kernel:* time `STRATA_GLMD_FUSED_MLA=1` against `mla_dec` on sm_70, sm_80/86, sm_89 and sm_120,
   and let the decode policy choose per device, as `dev_cfg` already does for `mla_dec`. Its sums differ from
   `mla_dec`'s in rounding, so it needs the KL gate, not bit identity.
3. *NVIDIA sm_80+, new kernel:* extend Flash's `mla_split_mma_kernel` to a 576-wide row (stage `k_pe` beside the
   latent; 36 k-steps instead of 32). It keeps F32 precision by feeding `q` and the softmax weights as FP16 high +
   low parts (`glm_fast.cu:3505-3511`), uses 128-thread blocks with a small shared-memory footprint (several blocks an SM), and
   already takes several tokens in one launch (`MlaToks`) - which the verify window wants (S-group below).
4. *AMD, new kernel:* the prompt path already has an RDNA matrix-core attention for GLM-5.3 (`mla_prompt_rdna`,
   `src/core/glm_prefill_dsa.cu:573-578`); a decode version of it is the AMD counterpart of step 3.

**Expect [estimate].** On a VRAM-rich NVIDIA PC, a few milliseconds a token (the attention is 11 G multiply-adds
a token). On a RAM-bound PC it is hidden behind the misses - do M1-M3 and the C-group first there.

#### G3. The indexer's scorer below 16K cells, and on AMD

**What.** Between 2,049 and 16,384 visible cells the decode scores on the CUDA cores with a kernel that spends most
of its time reducing, not multiplying. AMD runs that kernel at every context length.

**Evidence [code].** `idx_score_kernel` gives a warp to each cell and, for each of the 32 heads, a 128-wide dot
followed by a five-step shuffle reduction: 160 shuffle rounds a cell around 4 loads a lane
(`src/kernels/cuda/glmd_mixer.cu:64-75`). The tensor-core scorer takes over only above
`kIdxWmmaMinDecode = 16384` cells (`include/strata/kernels/glmd_volta.hpp:128`,
`glmd_decode_policy.hpp:112-120`), a crossover set on a V100. A HIP build never has it:
`decode_policy_built` forces it off and `decode_score` calls `idx_score` (`glmd_decode_policy.hpp:100-106`,
`src/kernels/cuda/glmd_mixer.cu:151-156`). This runs on 22 full layers a token.

**Do.**
1. Re-measure the crossover on each architecture (`STRATA_GLMD_IDX_WMMA=1` forces the tensor-core scorer from 2,049
   cells; `glmd_volta_parity` and the bench in `glmd_dec_bench` cover it) and make `kIdxWmmaMinDecode` per device.
2. Give the CUDA-core scorer `mla_dec`'s shape: one cell a lane, the head loop inside, no shuffle chain - the change
   that took the attention from `mla2` to `mla_dec` (`glmd_dec.hpp:6-9`). The scores keep their F32 arithmetic.
3. AMD: a matrix-core scorer. DeepSeek's indexer already has one on RDNA 3 (`DS4_INDEX_TC`, CHANGELOG v1.1.54:
   "+3% at 8K tokens, +6% at 32K, +10% at 64K" of prompt speed) **[project]**.

**Expect [estimate].** Small below 8K context, growing with it: at 128K a token's 22 scorer passes are 11.8 G
multiply-adds and 1.5 GB of keys (0.74 GB with M2). On AMD at long context this is likely one of the larger GPU-side
costs of a token.

**Also here.** The single-block top-k (`glmd_topk.cuh:263-273`: one 1,024-thread block, six passes over the row)
serves up to 32,768 cells before the multi-block one takes over (`glmd_volta.hpp:158`). The bucket selector
(`STRATA_GLMD_TOPK_HIST`, same cells, off by default, `glmd_decode_policy.hpp:79`, `:93`) has no measurement in the
tree: time it with the crossover above.

#### G4. A tiled multi-column GEMV for Q5_K

**What.** When a GEMV has several columns (a verify window's rows, the NextN block's cache rows), Q8_0, Q4_K, Q6_K
and BF16 weights go to register-blocked kernels that decode each weight once for all columns. Q5_K does not - and
Q5_K is the type of GLM-5.3's biggest matrices.

**Evidence [code].** The tiled launch loops over `{8, 12, 14, BF16}` only
(`src/kernels/cuda/glm_fast.cu:5751-5779`); type 13 falls through to the column kernel
`mv_kernel_t_cols<13>` (`glm_fast.cu:5782-5801`), which calls the dot once a column (`row_dot_cols`,
`glm_fast.cu:1397-1413`). In UD-IQ2_M, `attn_output`, `attn_q_a`, the shared expert's gate / up and the dense FFN's
gate / up are all Q5_K (`tests/glm_dsa/glmd_dec_bench.cu:169-187`) - 7.7 GB of the 14.9 GB of dense weights
**[estimate]**. The missing piece is small: `mvc_kernel_t<T, NC>` needs a `DecW<T>`, and `DecW<13>` already exists
for the one-column path (`src/kernels/cuda/glmd_mv_ext.cuh:175-216`); it is only defined after `mvc_kernel_t`
(the include is at `glm_fast.cu:2146`, the kernel at `:1860`).

**Expect.** Nothing for a token-by-token decode. For every multi-row pass - the one-GPU MTP decode, the pipelined
verify window, any future wider window - the Q5_K GEMVs stop growing with the row count the way the comment at
`glm_fast.cu:1664-1669` describes for the dp4a column loop ("8 columns ~3x one") **[project]**. This is a
prerequisite for the S-group paying off on GLM-5.3.

**Effort / risk.** Small: move the specialization (or the include), add `mvc_launch<13>`, extend `glmd_mv_parity`.
Bit-identical to the one-column sums by construction, like the other tiled types.

#### G5. Smaller items on the token path

- **The head.** Every token runs the whole 154,880 x 6144 head and an argmax, then a sampled request launches the
  sampler after a host round trip (`src/core/glm_fast_path.cu:6629-6639`, `:6793-6800`). On a VRAM-rich PC, put the
  sampler in the same stream as the head for sampled requests (the three-GPU speculative path already picks on the
  device, `glm_fast_path.cu:7656-7664`).
- **Host wait.** `fast_token` waits for the token in a `cudaEventQuery` + `yield` loop
  (`glm_fast_path.cu:6644-6656`): one more core kept busy beside the CPU lane and the service threads. With G1 the
  issuing thread has nothing else to do, so a blocking wait there costs nothing.
- **The embedding row** is dequantized on the host and copied up each token (`glm_fast_path.cu:6585-6593`); the
  on-device `embed_tok` already exists for the draft chain (`glm_fast.cu:5266`). Negligible alone, but it is the last
  host buffer between a token and the next, which matters once a part is one graph.

### 5.3 Drafting: read the weights once for more than one token

What runs today **[code]**:

| PC | Decode mode for GLM-5.3 | Why |
| --- | --- | --- |
| One GPU (an APU included) | **token by token, no drafting**; the NextN block is not even loaded | `mtp_decode_wanted` returns false for glm-dsa unless `STRATA_GLM_MTP_DECODE=1` (`src/core/glm_mtp.cu:85-102`); the block loads only on a split or when asked (`src/core/glm_model.cu:3504-3510`) |
| Two GPUs | `decode_spec`: GPU 0 runs the next position on a one-token draft while GPU 1 finishes the current one | `src/core/glm_fast_path.cu:7561-7896`; the two-GPU flow is pinned to v1.1.51's (`:7656-7662`) |
| Three GPUs or more | the same two-stage pipeline with a head *group* and a tail *group* | `include/strata/core/glm_dsa_spec.hpp:3-6` |

Two consequences. First, the pipeline is two stages deep whatever the number of GPUs, so with N cards at most two
work at once and the best case is 2x (Flash, two V100s, greedy, 97% of drafts accepted: 26.2 -> 40.0 tok/s,
`bench/results/2026-10-06-glm-spec/README.md`) **[project]**. Second, no default mode ever puts two tokens through
one read of the weights - the one thing section 3.1 says would lift the bandwidth floor.

The one-GPU window is off for a measured reason, which stands: where experts stream from RAM or the SSD "a verify
window reads the union of its rows' experts: on the 8065S a 4-row round read 5x a token's experts from the disk and
the decode ran at x0.62 of the token path with 3 drafts" (`glm_mtp.cu:25-28`) **[project]**. So the window is a
VRAM-rich tool; the cheap wins below are for everyone on two or more GPUs.

#### S1. Finish the wiring of the speculative decode (several small, verified gaps)

Each of these is a feature the tree already carries for GLM-5.3 that the decode loop does not reach. **[code]**

| Gap | Evidence | Effect of closing it |
| --- | --- | --- |
| **The draft vocabulary has no way back to the whole head on 3+ GPUs.** `fast_mtp` drafts over the Latin / code block plus digits and control tokens whenever the split is not two GPUs. | `sub = (sub \|\| (!full_head && !sample_draft_on && n_parts_ != 2)) && F->mtp_dv > 0` (`glm_fast_path.cu:6712`); the rule that returns to the whole head when the answer leaves that vocabulary lives only in the one-GPU decode (`glm_mtp.cu:606-611`, `:657`) | An answer in Chinese, Japanese, Korean, Cyrillic or Arabic cannot be drafted at all on a 3+ GPU split, so nearly every step is a miss and costs a redo: slower than no drafting **[estimate]**. Port the `mtp_out_` rule into `decode_spec`. Until then `MAYA_GLM_MTP_DRAFT_VOCAB=0` is the workaround. |
| **N-gram drafts are picked once per request.** | `ng.pick` runs before the loop (`glm_fast_path.cu:7725`); inside it only `ng.push` runs (`:7778`) and `verdict` is never called. The intended calls are in `tests/glm_dsa/glm_ngram_draft_test.cpp`. | `MAYA_GLM_NGRAM_DRAFT=1` does almost nothing today. Wired, it raises acceptance where the answer repeats the context (edits, quoted code, tool-call arguments). |
| **`SPEC_DRAFT_EARLY` changes the price, not the order.** | `spec_tail_front` / `spec_tail_back` are defined (`glm_fast_path.cu:7479`, `:7527`) and never called; the switch only alters the head pricing and profiler labels (`:7743`, `glm_dsa_spec.hpp:73-76`) | Today it can change which head group is chosen without changing what runs. Wire it (the draft comes off the tail's critical path) or remove its pricing effect. |
| **`SPEC_CANCEL` cannot cancel on GLM-5.3.** | The gate is launched only in the mHC layer loop (`glm_fast_path.cu:6397`); the plain-residual loop GLM-5.3 runs has none (`src/core/glm_residual.cu:110-160`) | A rejected draft's head pass always runs to its end. Add the gate to `fast_layers_plain` and a skip check to the `glmd` kernels. Worth it only where the head is the slower group. |
| **The acceptance statistics are never fed.** | `glmspec::note` has no caller (`include/strata/core/glm_dsa_spec.hpp:151-156`); only two counters are kept (`glm_fast_path.cu:7797-7799`, `:7846-7847`) | `MAYA_GLM_SPEC_LOG` / `_TRACE` print nothing. There is **no GLM-5.3 acceptance number anywhere in the repository**; the pricing assumes 0.8 (`glm_dsa_spec.hpp:78`). Feed it: every decision below needs that number. |
| **`"spec": 1` does not turn drafting off on a split.** | The DFlash and one-GPU branches test `req_spec != 1` (`src/program/generate.cpp:2394`, `:2411`); the `decode_spec` branch does not (`:2432`), although `serve/glm_dsa_tune.py:22` documents it | A/B runs of drafting through `bench_http.py --tune '{"spec": 1}'` measure nothing on a split; today they need `MAYA_GLM_NO_SPEC=1` and a restart. |

**Effort / risk.** Each is a few lines to a few dozen; none changes a token (drafts are checked by the model).

#### S2. Measure acceptance at GLM-5.3's own sampling, then decide on speculative sampling

**What.** GLM-5.3's default is temperature 1.0, top-p 0.95 (`docs/GLM53.md:51`). The draft is a greedy argmax, so
every default request accepts a draft only as often as the model's own draw lands on the draft block's top token.
`MAYA_GLM_SPEC_SAMPLE=1` draws the draft from the block's distribution and keeps it with probability min(1, p/q): the
tokens keep the plain draw's distribution exactly (`glm_fast_path.cu:7635-7642`).

**Expect.** On Flash: "drafts kept 68.6 -> 74.7 %, decode +1.6-2.7 % alone, +5.3 % with `MAYA_GLM_SPEC_CANCEL=1`"
(`docs/SETTINGS.md:140`) **[project]**. With a two-stage pipeline a step costs about max(H, T) on a hit and H + T on
a miss, so six points of acceptance are worth about 4-5% **[estimate]**.

**Risk.** A fixed seed gives a different (equally valid) answer than without it; penalties are not taken on this
path. Do S1's statistics first: the right default depends on a number nobody has measured.

#### S3. Let the engine conclude that drafting loses, and stop

**What.** The head tuner tries every head size from 1 to n - 1 and never "no drafting"
(`src/core/glm_dsa_spec.cpp:154-166`). When the price model itself finds the best head under 1.05x it only prints
"STRATA_GLM_NO_SPEC=1 may be faster" (`glm_dsa_spec.cpp:120-122`).

**Why it matters on a RAM-bound PC.** With three or more GPUs the parts share one CPU-lane pool, and the model's own
hit price is then the head's lane work plus the tail's (`glm_dsa_spec.hpp:51-53`): the two groups do not overlap
where the lane is the bottleneck, and every rejected draft spends scarce lane, PCIe and SSD time on a position that
is thrown away. **[code]** for the mechanism, **[estimate]** for the size.

**Do.** Add "plain" as a candidate to `HeadTuner` (two GPUs included), measured the same way (128 tokens a
candidate), and honour S1's `"spec": 1`.

#### S4. Plan the split for the decode that actually runs (3+ GPUs)

**Evidence [code].** The split search minimises the slowest part only for exactly two GPUs:
`pipelined = n == 2 && mtp && ...`, `obj = pipelined ? slow : ... : sum` (`src/core/glm_model.cu:1673`, `:1889`).
With three or four cards it minimises the *sum* - the token-by-token time - and only afterwards prices head groups
for the split it already chose (`glm_model.cu:2010`).

**Expect [estimate].** Three equal parts give a hit of about 0.70 of a token-time whichever head is taken (one group
always holds two thirds of the layers). A split of one half and two quarters, head = GPU 0, gives about 0.55 - roughly
+18% tokens/s at 80% acceptance when the GPUs set the pace. The search already knows each placement's expert room and
miss cost, so the change is its objective: the best `spec_price(...).step_ms` over the head sizes.

#### S5. Make a verify window cheap on GLM-5.3 (the prerequisite for any multi-token pass)

A verify window is the one mechanism that reads the 15 GB of dense weights once for several tokens. It is the
**Gorgon Halo's main decode lever** (section 7), and the next step after Turbo on a RAM-bound PC. The window code
runs on GLM-5.3, but row by row where it could share. **[code]**

- **On NVIDIA the dense weights are read twice a layer.** With the default stagger the rows go as two groups, each
  with its own GEMV pass (`src/core/glm_residual.cu:259-273`), whether or not the CPU lane has anything to do - the
  stagger exists to overlap the lane. Stagger only when a layer has lane work (`MAYA_GLM_WIN_STAGGER=0` turns it
  off for an A/B). On AMD the batched window and its stagger are off by default (`glm_residual.cu:263`), so the
  rows already go a layer at a time.
- **`attn_k_b` / `attn_v_b` are read once per row.** `mixer_rows` batches the three GEMV groups but calls the
  attention row by row (`src/core/glm_dsa_mixer.hpp:421-507`), and `mla2` runs both headwise products inside each call
  (`src/kernels/cuda/glmd_fast.cu:419-452`): 29 MB a layer and row at BF16. Flash's window reads them once
  (`mla_q_abs` / `mla_out` with `ncol`, `src/kernels/cuda/glm_fast.cu:6410-6429`).
- **Prep, rope, score, top-k and attention launch once per row**, although the prompt path has multi-row kernels for
  each (`dsa_prep_rows` is "byte-identical to glmd::dsa_prep" at T = 1, `include/strata/kernels/glmd_batch.hpp:132-134`;
  `index_score_tile`, `topk_merge_tile`). The attention is the exception: the prompt kernels are FP16 tensor-core
  ones, not bit-equal to `mla_dec`, so sharing them would loosen "the window's tokens are the token path's". G2's
  step 3 (a multi-token `mma.sync` kernel used by both paths) removes that conflict.
- **Q5_K GEMVs take the column kernel** (G4).
- **The NextN cache rows run a whole mixer** whose "attention output ... is not used"
  (`src/core/glm_mtp.cu:422-427`): only the projections and `dsa_prep` are needed there.

**Expect.** Nothing by default (no default mode uses a window on GLM-5.3). It decides whether the one-GPU drafting
can pay - on the Gorgon Halo an estimated 1.4 - 1.7x (section 7) - and S6 after it.

#### S6. A deeper pipeline for 3+ GPUs: make the pipelined window work on GLM-5.3

*Deferred: it pays only where several GPUs hold the experts, which is not a target at this stage. The safety fix
at the end of the evidence (refuse the switch on GLM-5.3) is not deferred.*

**What.** `MAYA_GLM_SPEC_WIN=1` (`decode_spec_pipe`) is the mode that would use more than two GPUs at once: the NextN
block drafts a short chain and every GPU works on a consecutive position (`src/core/glm_mtp.cu:817-828`).

**Evidence [code].** It is not GLM-5.3-safe today. Its per-part step calls the mHC-only `gf::embed_streams` (which
writes four streams of `n_embd`) and `gf::head_prep` (which refuses `n_embd` above 4096)
(`src/core/glm_window.cu:1365`, `:1399`; `src/kernels/cuda/glm_fast.cu:6085-6088`), where the batched window in the
same file has the plain-residual branches (`glm_window.cu:1118-1120`, `:1152-1155`). The lint that guards against
exactly this skips `glm_window.cu` on the grounds that DFlash is refused on glm-dsa
(`tests/glm_dsa/glm_residual_hooks_lint.cmake`), but this path needs no DFlash drafter - the window's buffers are
reserved for any model with a NextN block when the switch is set (`src/core/glm_fast_path.cu:1173-1181`). Setting
`MAYA_GLM_SPEC_WIN=1` on GLM-5.3 should therefore be refused until it is fixed.

**Expect [estimate].** With three GPUs and a two-draft chain at 80% / 75% acceptance, about 0.61 of a token-time a
token against about 0.75 for today's two-stage pipeline. For scale, Flash's pipelined window with the DFlash2 drafter
measured "+44% on nine GPUs and +12% on two against the MTP block" (`maya.py:1749`) **[project]**. Only for PCs
whose experts are mostly resident: a chain's extra positions read their experts' union (the x0.62 above).

**Effort.** Medium: the `glmres` hooks in `win_pipe_part`, skip flags in the `glmd` kernels, S5, then measurement.

#### S7. Several requests at once

*Deferred: total throughput for several users, not one answer's speed.*

`--parallel` answers GLM-5.3 requests one token of one conversation at a time (`docs/GLM53.md:59-60`;
`BATCHED_ROWS = False`, `serve/glm_dsa_product.py:30-31`; `glm_rows_ok` is false for glm-dsa,
`src/core/glm_maya_sync.cpp:24`). N conversations therefore read the 15 GB of dense weights N times a round. Batched
rows are the same mechanism as S5 with a cache per row; they raise total throughput, not one answer's speed, so they
come after S5.

### 5.4 The CPU lane (the RAM-bound PC's decode)

On a RAM-bound PC roughly half of a token's 600 expert evaluations do not find their expert in VRAM
**[estimate]**. Each goes to the CPU lane or over PCIe, whichever the start-up timing prefers, and the GPU waits for
both. What the lane does for one layer **[code]**: the service thread quantizes the layer's input, the pool computes
the experts' gate / up rows in contiguous pieces, the service thread quantizes each hidden vector, the pool computes
the down rows, the service thread sums (`src/core/glm_fast_path.cu:2808-2963`).

Two facts frame the recommendations:
- **Below about a dozen cores the lane is bound by its kernels, above that by memory.** The i-quant dots run at
  "~5 GB/s per core" (`src/kernels/cpu/native_expert.cpp:248-249`) **[project]**; a two-socket Xeon's lane reached
  0.154 ms for a 10.8 MB expert on 40 threads, about 70 GB/s of the ~92 GB/s a socket reads
  (`glm_fast_path.cu:2817-2820`) **[project]**.
- **The only GLM-5.3 lane timing in the tree** is 0.350 ms an expert on 36 threads of the 2x Xeon Gold 6138 test PC,
  on an IQ1_S layer of the UD-IQ1_S quant (`src/core/glm_cpu_bench.cpp:16-17`) **[project]** - about 28 GB/s, well
  under what that host's memory delivers. On that PC, with those kernels, the lane is compute-bound.

#### C1. Pick the one-token CPU kernel per quant type by timing, and fill the gaps

**Evidence [code].** A decode step's rows go to the first of: the engine's own exact one-token rows, the 4-row
kernel if `MAYA_GLM_LANE_ROW4=1`, ggml's `vec_dot` (`src/kernels/cpu/native_expert.cpp:285-329`). Coverage by type:

| Expert type | One token, default | Available but not used by default | Several tokens (window, prompt) |
| --- | --- | --- | --- |
| IQ2_XXS | own AVX-512 only with VNNI + VBMI + BMI2 (`iq_avx512_one.cpp:152-155`); else ggml | row4; own AVX2 (`MAYA_IQ2XXS_1T`) | own (AVX-512 or AVX2) |
| IQ2_XS, IQ3_XXS, IQ3_S | own AVX2 (`iq_avx2.cpp:803-808`) | row4 for IQ2_XS / IQ3_XXS exists but is never reached (the exact rows win first) | own |
| **IQ2_S, IQ4_XS** | **ggml** | row4 | own |
| **Q2_K, Q3_K** | **ggml** | row4 | own AVX-512 kernel in prompts only (`MAYA_GLM_KQ`), else ggml |
| **IQ1_S** | **ggml** | row4 | ggml |
| Q4_K, Q5_K, Q6_K | ggml | - | own for Q4_K / Q5_K; none for Q6_K |

The 4-row kernel covers Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ2_S and IQ4_XS
(`src/kernels/cpu/glm_lane_row4_avx2.cpp:591-605`), is bit-identical to ggml's dot by test, is off by default
(`glm_lane_row4.cpp:164-171`), and **has no speed number anywhere in the repository** (`docs/SETTINGS.md:2199` gives
none). The tuning tool sweeps `pcie_frac`, `cpu_threads` and `read_chunks`, not `lane_row4`
(`tools/calibrate_glm.py:17`). Meanwhile the DeepSeek engine's lane has rows of its own that the GLM lane does not
borrow **[project]**: AVX-512 rows that need no VBMI (a Xeon Gold 6152, one thread, IQ2_XS at one token 2.32 -> 3.62
GB/s, `docs/SETTINGS.md:1212-1223`), and vector IQ1_S rows that are "timed against ggml-cpu's dot at the start and
taken where 10 % faster" (a Ryzen 9 5900X, one token 1.43 -> 1.14 ms an expert, `docs/SETTINGS.md:1176-1182`).

**Why it matters for GLM-5.3.** UD-IQ2_M routes IQ2_XXS, IQ2_S, IQ3_XXS and Q2_K experts
(`tests/glm_dsa/glmd_width_cpu_test.cpp:206-209`). Of those, IQ2_S and Q2_K run ggml's dot on every CPU, and
IQ2_XXS does on every CPU without AVX-512 VBMI - Zen 2 / Zen 3, Intel 12th-14th generation desktops, Skylake and
Cascade Lake Xeons, the test PC among them.

**Do.**
1. At start, time ggml's dot, the own rows and the 4-row kernel on each type the pack routes and keep the fastest.
   All three give the same bits, so this needs no quality gate - only the check the lane already runs at first use.
2. Let the 4-row kernel compete for IQ2_XS and IQ3_XXS too.
3. Bring DeepSeek's exact rows (IQ1_S, IQ2_S, IQ2_XXS on AVX2; IQ2_XS on AVX-512 without VBMI) to the GLM lane.
4. Add `lane_row4` and `cpu_split` to `--calibrate`'s sweep.

**Expect [estimate].** On a compute-bound lane, time falls in proportion to the kernel's GB/s. If a RAM-bound token
spends 60-100 ms in the lane, a 1.2-1.5x kernel is 10-30 ms a token. Unmeasured: run
`glm_cpu_bench <pack> <layer> --row4 both` on a layer of each type first (`src/core/glm_cpu_bench.cpp:3-27`).

#### C2. Huge pages for the RAM tier on one-socket PCs

**Evidence [code].** The RAM tier gets `MADV_HUGEPAGE` only on the path taken with two or more NUMA nodes
(`numa_pinned` returns at once otherwise, `src/core/glm_fast_path.cu:441-443`, `:497-524`); a one-socket PC - most
desktops and workstations - pins it with `cudaHostAlloc` (`glm_fast_path.cu:1958-1962`, `:2005-2010`). The project's
own measurement of the difference: "4 KB pages ... (the CPU lane reads them 6-8% slower than huge ones)"
(`glm_fast_path.cu:500-501`, `:525-526`) **[project]**.

**Do.** Take the same `mmap` + `MADV_HUGEPAGE` + `cudaHostRegister` path on one node (it needs no `mbind`), with the
existing free-2-MB-blocks test, or offer a reserved hugetlb pool as the DeepSeek engine does
(`docs/SETTINGS.md:1756-1777`). A GLM-5.3 tier is 100-200 GB of ~11 MB experts read at random: 2,800 4-KB pages an
expert against 6 huge ones.

**Expect.** The 6-8% above on the lane's read speed, where the lane is memory-bound **[project]**; unverified whether
`cudaHostAlloc` memory is already huge-page backed on a given driver - check `AnonHugePages` in `/proc/<pid>/smaps`
first.

#### C3. Calibrate per expert class, keep re-timing, and count the GPU's own work

**Evidence [code].** One layer (`il_cal`) is timed at start for both lanes, and one plan word then serves every
layer (`glm_fast_path.cu:3540-3581`, `:6232`). The plan minimises `max(p * (f - k), 0.04 + c * k)`
(`glm_fast_path.cu:3425-3447`): it leaves out the GPU's own gate / up and down kernels, which run before it collects
the lane's answer (`:6337-6346`), so it leans towards PCIe. The decode re-times the CPU side for at most eight
windows of 2,000 experts and three changes, and never the PCIe side (`:4583-4636`).

**Why it matters for GLM-5.3.** Its layers do not share one expert format: the pack has several size classes
(9.7 to 14.5 MB, different kernels per type - C1's table), so a cost measured on one layer is wrong for others by
tens of percent. The expert classes already exist in the tier (`glm_fast_path.cu:1986-1990`).

**Do.** Time `c` and `p` per class, plan per class, and keep a moving average for the whole session instead of a
budget that ends after about fifty tokens.

#### C4. Smaller lane items

- **Two GPUs halve the cores per lane** (`threads /= n_parts`, `glm_fast_path.cu:3496`), each part with a pool of
  its own. For a split of nine, one shared pool measured 0.32 ms an expert against 0.74 (`:3463-3466`)
  **[project]**. On two GPUs try `MAYA_GLM_CPU_SHARED=1` - a setting, no code - and let `--calibrate` try it.
- **Serial work on the service thread**: the activation quantize (ggml's scalar `from_float`), one hidden-vector
  quantize per expert and the weighted sum (`glm_fast_path.cu:2870`, `:2941-2942`, `:2956-2960`). The plan itself
  budgets 0.04 ms a call for it: up to 3 ms a token at 75 calls. The device could hand over the quantized activation
  (7 KB instead of 24.6 KB of floats over the mapped buffer), and the quantize of `h` belongs in the down pieces.
- **Windows spin loops have no `pause`**: `cpu_relax()` is empty unless `__x86_64__` and not `_WIN32`
  (`src/core/glm_fast_state.hpp:80-84`), so every idle lane worker spins flat out beside its SMT sibling.
- **Hybrid CPUs**: the pieces are equal and there is no P-core / E-core distinction (`glm_fast_path.cu:2862-2863`);
  the slowest core's piece ends the call. Size the pieces by bytes with a short tail, or claim smaller pieces last.
- **Two NUMA nodes**: `MAYA_GLM_ROW_HALVES=1` measured 42.5 -> ~46 tok/s on a two-socket Xeon
  (`docs/SETTINGS.md:69`) **[project]** and is off; it also forces 4 KB pages (`glm_fast_path.cu:475`), which C2 says
  costs 6-8%. The GLM-5.3 port's own answer to both - `pin_class`, node-split segments with huge pages per node - is
  in the tree but **never called**: `T.split_req = false` with "pin_class's node-split segments are not wired in"
  (`src/core/glm_tier_numa.cu:236-241`), so `MAYA_GLM_NUMA_SPLIT`, `STRATA_GLM_THP`, `STRATA_GLM_RAM_SEG_GB` and
  `STRATA_GLM_PIN_THREADS` do nothing today. Wiring it gives a two-socket GLM-5.3 PC node-local rows *and* huge pages.
- **A route's disk reads finish before its lane experts start** (`glm_fast_path.cu:5161-5214`). On a disk-bound PC
  the two could overlap.

### 5.5 The expert tiers and the launches around a route

A token routes 600 of the model's 19,200 experts. Which of them sit in VRAM is decided by tier code GLM-5.3 shares
with Flash. One data file and several constants in it do not fit the larger model.

#### T1. Rebuild the shipped expert profile: it starves the last layer and does not rank the NextN block

**Evidence [code, and the file itself].** `data/expert-profile-glm53.bin` is what the setup passes when a pack has no
profile of its own (`maya.py:3575-3577`). It ranks all 79 x 256 (layer, expert) pairs. Parsed for this review:

| Layer | Median rank of its 256 experts | Its experts among the first 4,800 pairs | among the first 9,600 |
| --- | ---: | ---: | ---: |
| layers 3 - 76 | 7,830 - 13,375 | 49 - 78 | 92 - 175 |
| **layer 77** (the last trunk layer) | **17,149** | **8** | **21** |
| **layer 78** (the NextN block) | 19,322: all 256 at the very end, in id order | 0 | 0 |

The engine hands out VRAM slots in this order - the next slot goes to the layer whose next expert ranks highest
(`src/core/glm_fast_path.cu:1506-1513`, `:1573-1593`) - and warms each layer's slots in it.

**Consequence.** Layer 77 runs on every decoded token like any other layer, yet it gets little more than the floor
of slots. The NextN block is protected by a rule of its own (it keeps the equal share, `glm_fast_path.cu:1564-1571`),
but its slots are warmed with experts 0, 1, 2, ... - arbitrary ones. The CPU lane does not serve that block
(`if (!Ly.moe || Ly.mtp) continue`, `glm_fast_path.cu:3543`), so each of its misses is a PCIe fetch on the draft's
path until the tier has adapted.

**Likely cause [estimate].** A one-GPU prompt read leaves out the last layer's feed-forward ("The terminal layer's
outputs have no reader unless NextN is loaded", `src/core/glm_prefill.cu:2189-2195`) and does not load the NextN
block. Counts recorded from such reads never see layer 77's routes or the block's.

**Do.** Rebuild the profile from decode-side counts taken on a split with the NextN block loaded:
`tools/strata/glm_dsa_profile.py` already takes usage files (`MAYA_GLM_USAGE`), `STRATA_GLM_TRACE` records and
`--expert-profile-save` output. It is a data fix. Add a check to the tool's test that no MoE layer's share of the
first N pairs is under half the average, so it cannot come back.

**Expect [estimate].** A few fewer off-card experts a token (one layer of 75: about 1% of decode on a RAM-bound PC)
and better drafts at the start of each conversation on a split. Small, and free.

#### T2. Do not keep three empty spare slots a layer where no route can use them

**Evidence [code].** Every MoE layer keeps three slots empty as "spares" (`src/core/glm_fast_state.hpp:580`). The
warm-up fills `slots - spares` (`src/core/glm_memory.hpp:137-141`), and every boundary between tokens refills a used
spare - from a free slot, else by evicting a resident (`glm_fast_path.cu:5687-5793`). A spare is taken only by an
expert that a route fetches over PCIe or waits for from the disk (`src/kernels/cuda/glm_fast.cu:4287-4300`). Where
the plan sends every RAM-tier expert to the CPU lane, "no route fetches one into VRAM any more"
(`glm_fast_path.cu:5814-5817`, `:5884-5885`), and the background moves that keep VRAM following the conversation use
*free* slots, never spares (`:5896-5919`).

So with an all-CPU plan and nothing on disk, 3 x 75 = 225 slots hold nothing for the life of the engine - 2.2 to
3.3 GB of VRAM **[estimate]**.

**Do.** In that case keep one spare a trunk layer (or none) and give the others to the background refill. Under a
mixed plan, and on the NextN block, the spares do their job and should stay.

**Expect [estimate].** 150 - 225 more resident experts where it applies: about +3% of the residents on a 96 GB rig,
+10% on 24 GB. Check with the `promotions` and `scratch_uses` counters the engine already keeps.

#### T3. Turn the RAM-resident tier on where RAM holds every off-card expert

**Evidence [project].** `MAYA_GLM_RAM_RESIDENT=1` (`--glm-ram-resident`): "the RAM tier holds every expert not in
VRAM and nothing is evicted to disk ... (a 4090 D with 128 GB: decode 1.9x as fast with `PROMOTE_MIN=6`)"
(`docs/SETTINGS.md:63`); `MAYA_GLM_PROMOTE_MIN=6` "stopped that churn on one 4090 D ...; with little RAM it costs
(2x V100, 30 GB: 29.6 -> 28.3 tokens/s)" (`docs/SETTINGS.md:126`). Both are Flash measurements; both are off, and
the GLM-5.3 setup writes neither (`maya.py:3495-3502`).

**Do.** Have the setup choose them from the machine: RAM-resident when free RAM covers the experts that VRAM does
not (with GLM-5.3 that is roughly 224 GB less the VRAM pool: a 256 GB PC qualifies, a 192 GB one only with about
60 GB of expert room in VRAM), `PROMOTE_MIN` by measurement on GLM-5.3 (T5 explains why Flash's 6 is not the same bar here). Let
`--calibrate` sweep `promote_min`; it is already a per-request knob (`serve/glm_dsa_tune.py:27-35`).

**A step further [estimate].** The tiers are exclusive: an expert promoted to VRAM gives up its RAM slot
(`glm_fast_path.cu:4974-4976`), so each later eviction copies 10 - 13 MB back over PCIe (`:5774`). Where RAM has
room for all of it, keeping the RAM copy would make an eviction a table edit.

#### T4. Skip the empty launches on a VRAM-rich NVIDIA PC, and fuse them elsewhere

*Step 1 is deferred (it needs discrete GPUs that hold every expert). Steps 2 and 3 are not. On AMD - the Gorgon
Halo - the skip is already on by default, but it only acts while every expert is in GPU memory, which a 170 GB
model on 160 GB will not quite reach (section 7).*

**Evidence [code].** Every MoE layer of every token launches `moe_wait` and `moe_fetch`, and `moe_cpu_wait` when a
CPU lane exists - unless every expert is in VRAM *and* the skip is on (`glm_fast_path.cu:6282-6298`, `:6344-6348`).
The skip is on by default on AMD only (`glm_memory.hpp:13-23`). Measured there: "an empty launch and its gap were
~2.5 ms of a 54 ms Maya-S token on the 8065S" (`glm_fast_path.cu:6284-6285`) **[project]**. A fused wait + fetch
launch exists but only batched rows use it (`:6289-6292`).

**Do.**
1. A/B `MAYA_GLM_RESIDENT_SKIP=1` on an NVIDIA rig that holds the whole model and make it the default there:
   GLM-5.3 has 75 MoE layers, so it is 150 - 225 launches a token.
2. Before combining it with the decode graphs (G1), close a gap: GLM-5.3's graph key has no `all_resident` field
   (`include/strata/kernels/glmd_graph.hpp:94-110`), while Flash's layer-graph key has one for exactly this reason
   ("fast_moe leaves its wait / fetch launches out then: another graph", `src/core/glm_layer_graphs.hpp:19-23`).
   A graph captured while everything was resident could be replayed after a prompt's lending moved experts out -
   without the fetch. The same goes for per-request route arguments: `set_miss_skip` changes them without a new
   graph generation (`glm_fast_path.cu:4678-4684`). Both need a key field or a generation bump **[estimate: read
   from the key's fields, not reproduced]**.
3. Where the pool is not complete, use the fused `moe_wait_fetch` on the token path too.

#### T5. Scale the tier's ageing and promotion constants to GLM-5.3

**Evidence [code].** The usage counts halve every 32,768 routed experts on the host and every 4,096 routes on the
device (`glm_fast_path.cu:4981-4985`, `glm_fast.cu:3996-3998`). Flash routes 42 layers a token, GLM-5.3 routes 75:
the counts' half-life is about 97 tokens on Flash and **55 on GLM-5.3 on one GPU**, and about 160 on each part of a
three-GPU split **[estimate]**. Only one constant was rescaled for the port (the RAM tier's protection window,
`glm_fast_path.cu:5591`). The others still assume Flash: `PROMOTE_MIN` compares against those counts, the
background refill moves 24 experts a boundary ("about a token's worth of the x8 link's copies", `:5869-5877`) and
at most 8 swaps (`:5865-5868`), for a model with 19,200 experts instead of 12,096.

**Do.** Express the half-life in tokens (routes per token of the part) and size the refill by the part's layers.
Replay route logs through `tools/glm_tier_replay.py` before and after. The heat-based eviction
(`MAYA_GLM_HEAT_EVICT`, off; to be adopted only if a hit-rate curve shows topic drift costing two points or more,
`include/strata/core/glm_heat_evict.hpp:20-22`)
belongs in the same measurement.

#### T6. Let the CPU lane start before the shared expert, and overlap its wait

**Evidence [code].** A layer's router and shared-expert products go out as one batch, and the batch launches its
Q5_K jobs first and the F32 router last (`glm_fast_path.cu:6177-6197`, `glm_fast.cu:5820-5880`). The route kernel
that publishes the request to the host therefore runs after the shared expert's gate / up GEMV, although routing
needs only the router's output. Later the layer ends in `moe_cpu_wait`, where the device spins until the lane
answers and nothing else is queued behind it (`glm_fast_path.cu:6344-6348`).

**Do.**
1. *Exact:* launch the router first and split the route kernel's publish from its shared-expert half. The lane then
   starts one GEMV launch earlier on each of 75 layers: 2 - 3 ms a token on a RAM-bound PC **[estimate]** (a Q5_K
   2048 x 6144 GEMV is 19 us on a V100, `src/kernels/cuda/glmd_mv_ext.cuh:400` **[project]**, twice, plus the gap).
2. *Changes the output slightly:* the deferred add of A2 below, which lets the next layer's attention run during
   the wait. It is the larger of the two.

#### T7. Measure and re-shape the expert kernels at GLM-5.3's width

**Evidence [code].** The token path's expert kernels were tuned on 4096-wide experts: the block shape quotes "8 per
warp measured 90 vs 65 us a layer: occupancy" (`glm_fast.cu:4746-4747`), the down kernel uses 4 output rows a block
- 1,536 blocks at 6144, each copying its codebook into shared memory first (`glm_fast.cu:4866-4955`, `:767-784`) -
and the weights are loaded in 16-bit pieces because the blocks are only 2-byte aligned (`:179-184`). On one card
the project found that load pattern to be the limit: a P100's Q6_K GEMV was "bound by its loads ... the
instructions' arithmetic does not matter" and gained 1.2 - 1.5x from aligned word loads (`glm_fast.cu:1955-1965`)
**[project]**. Whether the expert dots are load-bound on newer cards is not known: there is no CUDA
micro-benchmark of an expert at 6144 x 2048 in the tree.

On AMD the tuned expert kernels are closed to GLM-5.3 altogether: they require `n_embd == 4096 && n_ff == 2048`
(`glm_fast.cu:6913`, `:7036`, `:7077`, `:7099`), so every Radeon runs the generic ones.

**Do.** Add the micro-benchmark first (the `glmd_dec_bench` pattern). Then try more down rows a block, aligned
loads, and the RDNA kernels at 6144. Experts are a third of a token's bytes, so each 10% here is about 3% of a
VRAM-rich token **[estimate]**.

#### T8. Smaller tier items

- **A helper GPU cannot serve GLM-5.3.** `MAYA_GLM_EXPERT_HELPER` refuses `n_embd > 4096`
  (`glm_fast_path.cu:2249`). A spare card that is not part of the split could hold and compute experts.
- **`moe_fetch` sizes its grid from the first GPU** - a process-wide `static int sms`
  (`glm_fast.cu:6793-6801`) - so a split over unlike cards launches the wrong number of blocks on the others.
- **Every part's boundary runs before part 0 starts a token** (`glm_fast_path.cu:6570-6575`), each with a stream
  sync and linear scans of the RAM slots (`:5536-5539`, `:5740-5745`). Later parts could run theirs while earlier
  parts compute.
- **The `vram_evict` request knob does nothing**: it is parsed (`include/strata/core/glm_req_knobs.hpp:136-138`)
  but the eviction rule reads only the environment (`glm_fast_path.cu:5721-5722`). An A/B through
  `bench_http.py --tune` silently measures nothing.
- **Undocumented switches** that shape GLM-5.3's tiers: `STRATA_GLM_RAM_EVICT`, `_RAM_PROTECT`, `_RAM_FREE`,
  `_TIER_GC`, `_AHEAD` / `_AHEAD_READ`. Document or remove them.

### 5.6 Turbo: the fastest GLM-5.3 that still answers coherently

**The owner's decision (2026-11-09).** GLM-5.3 gets a Turbo mode. In Turbo, speed comes first: quality may drop, and
the only thing that must hold is that answers stay **coherent**. KL against the exact model is recorded for
information; it is not a gate. Section 9.6 has the sweep and the coherence check to run.

Turbo is hidden for this model today ("until its own values are measured on it", `serve/models.py:837-840`), and
every number in the settings page is GLM-5.3-Flash's. What follows is what to measure and what to build.

#### A1. The Turbo levers that already work on GLM-5.3

All of these act in the route kernel that GLM-5.3 shares with Flash (`src/core/glm_fast_path.cu:6206-6240`,
`src/kernels/cuda/glm_fast.cu:3921-3951`), so they work today. **[code]** for what they do, **[project]** for the
Flash numbers, which are the only ones there are.

| Switch | What it does | Measured on Flash | For which PC |
| --- | --- | --- | --- |
| **`MAYA_GLM_ROUTE_AFFINITY=<bonus>`** | Adds the bonus to the selection score of every expert that is in VRAM, so a route prefers experts it can reach for free (`glm_fast.cu:4002-4003`). Eight experts still run and their weights stay their own. | Nothing: undocumented, never measured | RAM- and disk-bound: **the big one** (below) |
| `MAYA_GLM_MISS_SKIP_W`, `_FROM` | Leaves out an expert that is not in VRAM and carries less than this share of the route's weight, on layers from `_FROM` on. Also a per-request knob (`"strata_tune": {"miss_skip_w": ..}`) | 27.4 -> 30.9 tok/s at 0.16 (KL 0.075), `docs/SETTINGS.md:135` | RAM-bound |
| `MAYA_GLM_TRUNK_KEEP`, `_MAP` | Leaves out experts missing from VRAM from route rank n on; `_MAP` is a digit per layer | +23% and +25% with a per-layer map (KL 0.018), `docs/SETTINGS.md:133` | RAM-bound |
| `MAYA_GLM_DISK_SKIP_W`, `_FROM` | Never waits for the SSD for a light or low-ranked expert | 13.3 -> 21.6 tok/s at 0.12 (KL 0.12), `docs/SETTINGS.md:134` | Disk-bound |
| `MAYA_GLM_RESIDENT_TOPK=<k>`, `MAYA_GLM_RESIDENT_SKIP_W` | Computes only the k heaviest routed experts, **wherever they sit**; the kept weights are renormalised (`glm_fast.cu:4206-4225`) | Strix Halo, all experts in VRAM: 33.2 -> 38.8 tok/s at k = 4; k = 3 broke the output; `_SKIP_W=0.20` "garbage", `docs/SETTINGS.md:136` | **Gorgon Halo**, and everything else |
| `MAYA_GLM_MTP_KEEP=0` | The draft block never fetches a missing expert | default 2 (`glm_fast_path.cu:6134-6145`) | splits |
| Greedy decoding, thinking off or at its lowest level | Request level: fewer tokens, and more drafts accepted | part of Flash's Turbo | all |

**Why route affinity is the big one [estimate].** On a RAM-bound PC a token's time is its missing experts: about
half of the 600 it routes go to the CPU lane or cross PCIe, and some come from the SSD. With a bonus larger than any
score difference (scores are a probability plus the model's selection bias; 2 is above that), a layer's top 8 are
taken from the experts in VRAM alone - and every layer holds at least 8 (the pool's floor,
`src/core/glm_model.cu:1800-1801`). Then **no expert is fetched at all**: no CPU lane, no PCIe, no SSD. The token
costs what it would on a PC that holds the whole model: roughly 40 - 50 ms on V100-class cards instead of 150 - 250.
The price is that each layer chooses among the experts it holds - perhaps 40 of 256 on a 64 GB rig - instead of
all of them. Smaller bonuses (0.05 - 0.3) only tip close calls; the sweep in section 9.6 finds the largest bonus
that stays coherent.

Two things follow from it:
- With a saturated bonus the resident set never changes (nothing is fetched, so nothing is promoted). Which experts
  a layer holds is then decided by the expert profile alone - T1 (the profile fix) matters twice in Turbo.
- The `maya serve: decode expert cache hit rate` line (`src/program/generate.cpp:8657`) shows when it has
  saturated: 100% hits.

#### A2. New code for Turbo, by expected value

1. **Give the prompt path the same route bias and miss skip.** None of the switches above reach a prompt read
   (the route kernels of `src/core/glm_prefill.cu` take none of them; the settings page says "prefill and the MTP
   layer stay exact"). With the affinity bonus in the prompt's route, a chunk routes only into experts that are in
   VRAM: nothing is staged, prestaged, sent to the CPU or read from the SSD. The fixed cost of a chunk - about 10 s
   on one V100, section 6.1 - disappears, and so does most of the chunk's borrowed memory (no staging ring, no
   landing ring). **This is the largest prefill lever there is on a RAM-bound PC [estimate]**, and it is small:
   the bonus is one add per expert before the top-8 (`route_rows` in `src/kernels/cuda/glmd_batch.cu`), reading the
   same residency table the decode's route reads.
2. **A per-layer map for `RESIDENT_TOPK`**, as `TRUNK_KEEP_MAP` has. Flash's layers differ by 10 - 20x in how much
   they notice a left-out expert (KL 0.040 for layers 3-7 against 0.002-0.003 for 30-44,
   `docs/SETTINGS.md:133`), so a single k for all 75 layers wastes most of what the late layers could give.
3. **Lenient draft acceptance on a split.** A draft is kept only when it equals the model's own pick. Keeping it
   when the model gives it at least a set probability (or ranks it in its top few) raises acceptance, and each
   avoided miss saves a whole pass of one group of GPUs (section 5.3). There is no such switch today
   (`STRATA_GLM_SPEC_ACCEPT` only sets the acceptance the price model assumes, `src/core/glm_dsa_spec.cpp:76`).
4. **The deferred CPU add on GLM-5.3's layer loop** (`MAYA_GLM_DEFER_CPU`): a layer's CPU-lane experts are added
   after the next layer's attention, so the lane overlaps it. Flash: 13.0 -> 14.0 tok/s at KL 0.034 - 0.038
   (`docs/SETTINGS.md:137`). It is read only in Flash's loop (`glm_fast_path.cu:6376-6381`); GLM-5.3's always waits
   (`src/core/glm_residual.cu:151`). Only worth it where misses remain after A1.
5. **A smaller dense quant at load** (M4): `attn_q_b` from Q8_0 to Q5_K or Q4_K, `attn_output` from Q5_K to Q4_K.
   1 - 2 GB fewer bytes a token. It matters most where a token is bound by memory bandwidth alone: the Gorgon Halo.
6. **Attend to fewer cells.** The attention reads 2,048 selected cells on every layer. A Turbo cap (1,024 or 512),
   or the attention sink plus the most recent cells with no indexer at all, cuts the attention's arithmetic and, at
   long context, the whole key scan. The second was tried outside the tree and scored *better* than the indexer on
   an 8K test (`tools/strata/glm53_long_gate.py:18-21`: 2.45 against 3.35) - which is also the open quality
   question of section 3.4. Settle that first; it may turn out to be the right default, not a Turbo trick.

#### A3. What Turbo should look like for each target PC

| PC | Start the sweep from | Then add |
| --- | --- | --- |
| VRAM + RAM (+ NVMe) | `MAYA_GLM_ROUTE_AFFINITY=2`, greedy, lowest thinking level, `MAYA_GLM_MTP_KEEP=0` | `MAYA_GLM_RESIDENT_TOPK` 6 -> 5 -> 4; if the bonus must come down for coherence: `MAYA_GLM_MISS_SKIP_W` and `MAYA_GLM_DISK_SKIP_W` for what still misses |
| Gorgon Halo, most experts in VRAM | `MAYA_GLM_RESIDENT_TOPK=4` (Flash's value there), greedy, lowest thinking level | `MAYA_GLM_ROUTE_AFFINITY` for the experts that do not fit; A2.5 once it exists |

Section 9.6 turns this into a sweep with a pass / fail rule, and says where the winning values go in
`serve/models.py`.

## 6. Prefill recommendations

### 6.1 What a prompt read does, and what it costs

**The path [code].** A prompt is cut into chunks. One GPU reads full chunks and then the remainder (at most 32,768
tokens a chunk); a split reads equal chunks (at most 8,192) and its parts pipeline them: part k reads chunk c while
part k - 1 reads chunk c + 1 (`src/core/glm_prefill.cu:1139-1153`, `:4373-4384`, `:4801-4803`). Inside a chunk each
layer runs:

1. The attention in sub-batches of 2,048 rows (`src/core/glm_prefill_dsa.cu:587`): FP16 cuBLAS projections whose
   weights are dequantized again for every sub-batch, the indexer's scores and top-2,048 on the 21 full layers,
   GLM-5.3's own attention kernel, then `attn_output`.
2. The shared expert and the router, then **one host sync** for the 256 route counts (`glm_prefill.cu:2882-2902`).
3. The host's plan (`glm_prefill.cu:2946-3196`): experts already in VRAM run where they sit; the others are copied
   from RAM in groups of four through a three-group ring, were prestaged during the attention, are computed by the
   CPU lane, or come from the SSD.

**The cost [project numbers, my fit].** The only GLM-5.3 timings in the tree (section 3.3) fit two simple lines:

| PC | Time for a read | What it says |
| --- | --- | --- |
| 1x V100 | ≈ 10 s x chunks + 2.95 ms x tokens | each chunk pays about 10 s before its first token: nearly every expert crosses PCIe once a chunk |
| 3x V100 | ≈ (chunks + 2) x (2.2 - 2.7 s + 1.05 ms x tokens in a chunk) | a chunk passes through three stages, and only overlapping stages run at the same time |

The per-token part (about 3 ms across the 78 layers) is of the size the kernels' own rates predict for a V100:
roughly 0.9 ms of attention, 0.5 ms of projections and 0.9 - 1.8 ms of routed experts **[estimate]**. So a read has
three levers:

- **How many chunks** (6.2). It decides short reads, and multi-chunk reads on one GPU.
- **How the idle time is used** (6.2): on a split only one GPU works on a one-chunk read; the CPU's share starts
  late.
- **The kernels** (6.3). They decide long chunks: the per-token part is 70 - 80% of an 8K read.

And one before all of them: not reading what is already known (6.4).

### 6.2 Chunks and scheduling

#### P1. On a split, pipeline the reads that fit in one chunk

**Evidence [code].** The pipeline needs two chunks: `pipelined = np > 1 && nc > 1` (`glm_prefill.cu:4664`). A read
of up to 8,192 tokens - most turns of a long conversation - therefore runs on one GPU after the other
(`:4738-4799`), and the waiting parts only prestage their first MoE layer (`:4745`,
`src/core/glm_dsa_prefill_stream_plan.cpp:25-30`). On three GPUs each card is idle two thirds of the read.
`MAYA_GLM_PREFILL_MIN_CHUNKS=<n>[,<min tokens>]` cuts such a read into n chunks so the parts overlap; it is off and
has no GLM-5.3 measurement (`docs/SETTINGS.md:123`).

**Expect [estimate, from the fit above].** Time in c chunks ≈ (c + parts - 1) x (fixed + per-token x tokens / c):

| Read on 3x V100 | Today | Best c | Predicted | Change |
| --- | ---: | ---: | ---: | ---: |
| 2K | 13.3 s (1 chunk) | 1 - 2 | ≈ 13 s | none |
| 4K | 20.5 s (1 chunk) | 2 | ≈ 18 s | -12% |
| 8K | 33.0 s (1 chunk) | 3 | ≈ 27 s | -18% |
| 16K | 45.7 s (2 chunks) | 4 | ≈ 41 s | -10% |

The best count is about sqrt((parts - 1) x per-token cost x tokens / fixed cost). The fit ignores that parts
working at once share the CPU lane and the SSD, so treat the numbers as an upper bound.

**Do.**
1. *A setting, today:* measure `MAYA_GLM_PREFILL_MIN_CHUNKS=3,2048` on a GLM-5.3 split.
2. *Then:* let the engine choose the count per read from the per-part stage times it already collects
   (`MAYA_GLM_PREFILL_STAGES`), instead of a fixed number.
3. *Also:* the 8,192 cap on a split rests on a Flash measurement ("two V100s 709 at 8192 against 551 at 32768",
   `glm_prefill.cu:1139-1142`). Re-measure it on GLM-5.3, whose per-chunk and per-token costs are both different.

#### P2. One GPU: fit more tokens in a chunk

On one GPU every extra chunk costs about 10 s (the fit above; "the fewest chunks win", `glm_prefill.cu:1258-1262`).
A chunk's size is what the expert pool can lend, so every byte the chunk's layout wastes is a smaller chunk.

**Evidence [code].**

| What | Where | Size |
| --- | --- | --- |
| The residual buffer is carved for Flash's four streams. GLM-5.3 has one: it writes and reads T x n_embd of it. | `glm_prefill.cu:1494`, `:1580` against `:4658`, `glm_prefill_dsa.cu:427-445` | 73.7 KB of the 192.7 KB a chunk token keeps: 0.9 GB at a 12,288-token chunk |
| The same factor of four in the hand-over between parts: pinned buffer and both copies. | `glm_prefill.cu:1316`, `:4758-4777`, `:4812-4859` | 1.6 GB pinned for each part but the last at 8,192 tokens, 0.4 GB of it used; it comes out of the RAM tier (`glm_fast_path.cu:1777-1780`, `:1822`): about 100 experts a part |
| The MoE output window never goes below 21,824 rows - sized for 682 lightly routed experts, where a GLM-5.3 layer has 256. | `glm_prefill.cu:118`, `:278-286` | 1.2 GB even for a 3K-token chunk |
| The attention scratch is fixed at 2,048-row sub-batches; the planner's own byte budget is never passed. | `glm_prefill.cu:364`, `glm_prefill_dsa.cu:145-149`, `src/core/glm_dsa_prompt_plan.cpp:45-58` | 1.3 GB, plus a mask that grows with the context |
| The NextN fill reserves its buffers for the whole chunk but works in 2,048-row sub-batches. | `glm_prefill.cu:1513-1514`, `glm_prefill_dsa.cu:830-845` | 147 KB a token; binds only above about 8.7K-token chunks on the part with the block |
| The whole context cache sits in the same VRAM. | M1, M2 | the largest term at long context |

**Expect [estimate].** With the residual at its real size, the loan that gives 12,288-token chunks on one V100 gives
about 15,600; a 16K prompt becomes one chunk instead of two - about 10 s of 68 s. The context cache matters more:
on one 32 GB card the lendable pool shrinks as the context grows, to roughly 5 - 10K-token chunks at a 32K context,
about 1K at 64K, and from about 90K "can lend no prompt chunk - token by token" (`glm_prefill.cu:1284-1288`).
(These three are extrapolated from the one measured point; the start-up line prints the real chunk.) That is the
strongest argument for M1: on one card, streaming the cache is what makes a long prompt readable at all.

**Effort / risk.** The residual and hop sizes are an exact change (`residual_floats()`-style sizing by
architecture); the outputs do not change. The window floor and the sub-batch budget are small planner changes.

#### P3. Start the CPU's share when the plan is made

**Evidence [code].** For EXL3 experts the CPU share runs on a thread of its own from the plan on, for a stated
reason: the loop that enqueues the staged experts "blocks once the device's launch queue is full, and the host's
experts, queued after it, started near the end of the copies they were planned beside (an 8K prompt on one 3090: at
plan + ~290 of a layer's ~400 ms, ~110 ms of work)" (`glm_prefill.cu:3197-3200`, `:3319-3326`). GGUF experts - all
of GLM-5.3 - still run their share on the calling thread *after* that loop (`:3809-3832`), and the loop also
busy-waits for every disk expert that has not landed (`:3741-3745`). The plan then reads the late finish as a slow
CPU and gives it less.

**Do.** The same thread for GGUF experts. **Expect [estimate]:** most where a disk tier or a full launch queue
delays the loop; on short turns the CPU could take about half of the staging. Check with
`STRATA_GLM_PREFILL_TRACE=1`: the gap between the plan and the CPU share's start.

#### P4. Hand the routes to the host before the shared expert

**Evidence [code].** The plan needs only the 256 counts, but the sync that delivers them also waits for the shared
expert's three GEMMs, which are queued first (`glm_prefill.cu:2882-2902`). They use different buffers.

**Do.** Queue the router and the count first, wait on an event, and plan while the shared expert runs: the copies
and the CPU share start one shared-expert time earlier on each of 75 layers - about 1 s of an 8K chunk on a V100
**[estimate]**.

#### P5. One expert pass a prompt instead of one a chunk (one GPU, structural)

**Idea [estimate].** The loop is chunk by chunk, then layer by layer, so a 64K read on one V100 streams the experts
about six times (6 x 10 s). Layer by layer over the whole prompt - keeping only the residual of every chunk
(24.6 KB a token) - would stream them once. It is a large change (the attention still runs chunk by chunk inside a
layer, the residuals of the whole prompt must live in VRAM or stream), and it gives up the pipeline on a split. Worth
a design note only if P2 and M1 leave multi-chunk reads common on one-GPU PCs.

#### P6. Smaller scheduling items

- **The CPU gate.** A prompt gives the CPU lane experts only when a PCIe copy costs more than twice the CPU's time
  for one (`MAYA_GLM_PREFILL_CPU_GATE`, `glm_prefill.cu:2462-2473`). On a PC where it gave the CPU nothing, `0`
  measured 309 / 323 -> 404 / 428 tokens/s on Flash (`docs/SETTINGS.md:149`) **[project]**. GLM-5.3 moves nearly
  every expert every chunk, so it has more to gain from a second lane: measure the gate at 0 and 1, and add it to
  `--calibrate`.
- **Two GPUs, one-chunk reads.** The waiting GPU's link can be put to work (`MAYA_GLM_DUAL_PREFILL`, +10 to +28% on
  Flash with NVLink; `MAYA_GLM_PREFILL_CALC`, +25 to +48%, `docs/SETTINGS.md:84-85`) **[project]** - but the second
  is EXL3-only (`glm_prefill.cu:2842`) and both need exactly two parts. P1 covers most of the same idle time; a
  GGUF version of `PREFILL_CALC` would cover the rest on two-GPU PCs.
- **The prestage buffer is sized once, for the largest chunk** (`glm_prefill.cu:1300-1301`, `:2421`). On one GPU a
  mid-size read leaves the link idle during each layer's attention with only 17 slots to fill; sizing the buffer to
  the read (as the split's rule does, `:2425-2433`) could hide more copies. Unmeasured.
- **Short turns.** Below a token count the prompt goes token by token: 8 on one GPU - measured on Flash - and 32
  where the NextN block is loaded (`glm_prefill.cu:4366-4372`, `glm_dsa_prefill_stream_plan.cpp:54-56`). A GLM-5.3
  token costs several times a Flash token and a chunk's fixed cost is also larger: measure 8 / 16 / 32 / 64 with
  `MAYA_GLM_PREFILL_MIN`.
- **The first prompt of a session.** `MAYA_GLM_WARM_PROMPT` (off) exists because "the first prompt of every session
  read 95-133 ms/token against 23-44 warm" (`docs/SETTINGS.md:156`) **[project]**. Have the setup turn it on.
- **The stage line's "disk wait" is always 0**: the wait is added to another counter (`glm_prefill.cu:3744`) than
  the one the line prints. Fix it before using that line to judge any of the above.

### 6.3 Prompt kernels

#### K1. A faster attention kernel, and one for every architecture

**Evidence.** GLM-5.3's prompt attention on tensor cores (`mla_prompt_h64`) takes "20.3 / 23.4 / 23.7 ms" for
2,048 queries on a V100 (`include/strata/kernels/glmd_batch.hpp:79-83`) **[project]** - about 25 TFLOPS, where
cuBLAS reaches "~53 TFLOPS" on the same card (`glm_prefill.cu:3510`) **[project]**. It runs one 8-warp block an SM
(97 KB of shared memory), and the same WMMA code serves every NVIDIA generation: there is no `mma.sync` version for
Ampere and newer **[code]**. Flash has one, and it was worth 13 - 16% of a whole prompt there: "RTX 3090 ... 674 /
832 tok/s with the 91 KB kernel, 759 / 966 with this" (`src/kernels/cuda/glm_batch.cu:1447-1455`) **[project]** -
on a model whose attention is on a quarter of the layers, not all of them.

Coverage **[code]** (`glm_prefill_dsa.cu:540-583`, `:763-766`):

| GPU | Prompt attention | Indexer scorer |
| --- | --- | --- |
| Volta, Ampere and newer (shared memory fits) | 64-head WMMA | WMMA |
| Turing (64 KB of shared memory) | **F32 on the CUDA cores** | F32 |
| RDNA 3 / 3.5 / 4 | RDNA matrix-core kernel | RDNA kernel |
| other AMD | **F32** | F32 |

For scale, on a Radeon 8065S the F32 kernel takes 497 - 581 ms where the RDNA kernel takes 66 - 79
(`glm_prefill_dsa.cu:574-577`) **[project]**.

**Do.**
1. *Ampere and newer:* the `mma.sync` + `ldmatrix` kernel at GLM-5.3's row width (512 + 64). It can be the same
   kernel G2's step 3 asks for, used by decode and prompt alike.
2. *Turing:* a register-accumulator tensor-core kernel as Flash's 58 KB one (`glm_batch.cu:1310-1314`); Quadro RTX
   8000 and Titan RTX cards are plausible GLM-5.3 hosts.
3. *Cheap check first:* the F32 and 16-head kernels still walk every cell up to four times the selection size
   (`gather_ratio(... 4.0)`, `glm_prefill_dsa.cu:583`), which the same file measured up to 9x slower on HIP. Try
   `STRATA_GLMD_GATHER_RATIO=1` on a Turing card.

**Expect [estimate].** The attention is about 30% of a compute-bound read on a V100 (0.9 of 3 ms a token). A kernel
at cuBLAS's rate would take 10 - 15% off a long read on a VRAM-rich PC.

#### K2. The copies and checks around the attention

**Evidence [code].** In every sub-batch of every layer (`glm_prefill_dsa.cu:762-791`):
- the attention writes its context as F32 (268 MB for 2,048 rows), `pack_f16` reads it again to make FP16, and the
  FP16 range guard reads the F32 a third time (`:770-771`); the same three passes run for the queries and the
  attention output (`:725-726`, `:788-789`);
- each guard is a separate launch of one block a row, and each guarded GEMM is followed by an un-scale launch even
  when every scale is 1 (`src/kernels/cuda/glmd_guards.cu:49-71`, `:240-264`);
- `pack_f16` does 64-bit divisions per element for layouts that are already contiguous
  (`src/kernels/cuda/glmd_batch.cu:1570-1580`).

**Do.** Let the attention kernel write FP16 context directly (it is an average of FP16 latents: it cannot leave
FP16's range); fold the range check into the kernel that produces each operand; use 32-bit index math.

**Expect [estimate].** About 1 GB less memory traffic per sub-batch and layer, 5 - 10% of the attention phase, and
270 MB of scratch back to the expert pool.

#### K3. Dequantize the projections once a layer, not once a sub-batch

**Evidence [code].** `gemm()` dequantizes its weight to FP16 on every call (`glm_prefill_dsa.cu:262-305`), and it is
called inside the sub-batch loop (`:617-621`, `:657-658`, `:790`), as is the FP16 copy of `attn_k_b` / `attn_v_b`
(`:724`, `:769`). A layer's projections are 165 M weights - 330 MB of FP16 written per sub-batch, 1.3 GB a layer at
an 8K chunk **[estimate]**. Before these dequantizers were vectorised the project measured them at "13-19% of
prefill" on Flash (`src/kernels/cuda/dequant_bf16.cu:186-187`) **[project]**.

**Do.** Run the two input projections (`attn_q_a`, `attn_kv_a`: their outputs are 10 KB a token) over the whole
chunk in one call; for the others keep the FP16 copy across a layer's sub-batches when the loan has room for it.
On AMD the shared expert still runs in 256-row sub-batches (`glm_prefill.cu:345-356`) - eight times the
dequantizations of the CUDA build.

**Expect [estimate].** 2 - 5% of the attention phase.

#### K4. The indexer at long context

**Evidence [estimate from the code's shapes].** Each full layer scores every query of a sub-batch against every
earlier position: T x context x 4,096 multiply-adds, 21 times. At about 129K context that equals the attention of
all 78 layers, and it grows with the square of the prompt.

**Do (all exact).**
- Skip key blocks that lie wholly after a sub-batch's last query (today the tile is computed and masked).
- Convert a sub-batch's queries to FP16 once, not once per key block (`src/kernels/cuda/glmd_batch.cu:309-336`).
- Drop tile cells below the running selection's minimum before the top-k's four radix passes
  (`glmd_batch.cu:489-542`).
- M2 (FP16 keys) halves the bytes the scorer reads.

Only worth the work for contexts above about 64K; remember section 3.4 first.

#### K5. The routed experts

- **Volta.** An expert with 128 or more rows in the chunk is dequantized whole to FP16 and multiplied by cuBLAS
  (`glm_prefill.cu:3508-3539`). The threshold and its gain are Flash's (a 27K prompt 39.7 -> 31.7 s). At 6144 x 2048
  a GLM-5.3 expert is 37.7 M weights - 75 MB of FP16 written per expert, about 19 GB a layer at an 8K chunk
  **[estimate]**. The kernel that avoids that write (`MAYA_PF_WMMA`, +5.8 to +11.6% on Flash,
  `docs/SETTINGS.md:99`) is off, covers only some quant types, and with the FP16 path on sees only the experts
  under 128 rows (`glm_prefill.cu:3576-3597`). Measure `MAYA_GLM_PREFILL_BLAS` at 0 / 128 / 512 with
  `MAYA_PF_WMMA` at 0 / 1 on GLM-5.3.
- **Light kernels** take resident experts with at most 32 rows, but only for IQ2_XXS, IQ3_XXS and IQ1_S layers
  (`glm_prefill.cu:2958-2966`). UD-IQ2_M's IQ2_S and Q2_K layers go through MMQ whatever their row count.
- **AMD.** The fused expert kernel requires `n_embd == 4096` (`glm_prefill.cu:313-316`): GLM-5.3 runs MMQ on every
  Radeon.
- **IQ1_M** experts have no MMQ kernel, so a pack that uses them reads its whole prompt token by token
  (`glm_prefill.cu:1102-1116`). Not UD-IQ2_M, but it rules out the smallest quants.

#### K6. Small kernel-side items

- On a full layer the indexer's query projection and weights are computed even when the sub-batch sees at most
  2,048 keys and nothing reads them (`glm_prefill_dsa.cu:621`, `:658` against `:717`). Exact and trivial to skip;
  it helps short prompts a little.
- With the 64-head kernel the selection mask is carved and never written (`glm_prefill_dsa.cu:613-614`, `:721`):
  33 MB at a 128K context, 268 MB at 1M.
- The chunk's embedding rows are dequantized on one host thread, twice a chunk, while the GPU waits
  (`glm_prefill.cu:4650-4655`).
- The whole attention phase is one profiler mark. Marks inside `dsa_layer` (projections, score, top-k, attention,
  output) would settle K1 - K4 with one run.

### 6.4 Not reading at all: reuse

GLM-5.3's cache is positional - no recurrent state - so a checkpoint or a snapshot is just a position and costs
nothing (`glm_fast_path.cu:7934-7966`, `:8082-8090`). That makes reuse cheaper than on Flash, and three paths do not
use it yet.

#### R1. Continue from the decoded answer

**Evidence [code].** After a request the saved position is the *prompt's* end, one token before it
(`src/program/generate.cpp:1946`, `:2154-2157`). The answer's rows are in the cache too, but only a paused request
moves the snapshot onto them (`:2191-2223`). So the next turn - which starts with the previous prompt and answer -
restores the prompt and reads the answer again through the prompt path.

**Do.** Record prompt + answer as the conversation's tokens and restore at the longest common prefix with the next
prompt (the rule the NVMe prompt cache already uses, `include/strata/core/glm_prompt_cache.hpp:22-27`).

**Expect [estimate].** It saves the re-read whenever the client sends the answer back token for token: tool-using
agents that return the reasoning, and any request with thinking off. An agent turn then reads only the new tool
result. For the web chat, which drops the reasoning, it saves nothing. On a RAM-bound PC each avoided read is at
least a chunk's fixed cost (2 - 10 s), so measure the common-prefix length on real agent traces first.

#### R2. Smaller reuse items

- **A new conversation's read is cut at the end of the system prompt** when slots are on, so the state there can be
  saved (`generate.cpp:1988-1997`). On GLM-5.3 that costs one more chunk (about 10 s on one V100, a pipeline fill on
  a split) per new system prompt, and positions are free: save the slot after an uncut read instead.
- **With `MAYA_GLM_PROMPT_CACHE_DIR`, a checkpoint hit hides the cache.** Both cache calls run only when nothing
  else matched (`generate.cpp:1895`, `:1910`), so a prompt that shares only a system-prompt checkpoint with the last
  one never looks for a longer entry on disk and never stores the outgoing conversation.
- **Saving a conversation is slow at GLM-5.3's size.** A slot or cache entry is every cache row: 64 MB per 1,000
  tokens, 8.4 GB at 128K. It goes through a pageable 64 MB buffer with synchronous copies
  (`glm_fast_path.cu:8272-8290`). Pinned buffers and a writer thread would shorten every switch between
  conversations; M2 makes each entry 9% smaller.
- **Thinking.** GLM-5.3 defaults to its largest thinking level (`docs/GLM53.md:51`). It is the right default for
  quality; it is also most of the wall time of a short answer. Nothing to change in the engine - but any user-facing
  speed comparison should say which level it used.

## 7. Suggested order of work

### 7.1 First: settings and data (measure, then flip - no new code)

One GPU session with the model covers most of these. Each needs its check from section 8 and nothing else.

| # | Do | Who gains | See |
| ---: | --- | --- | --- |
| 1 | Validate the cache streaming the engine already has; set the product flag | 64K contexts and up; long prompts on one card | M1 |
| 2 | Run the parity check of native Q8_0 `attn_k_b` / `attn_v_b`; rebuild the pack with it | everyone (1.1 GB) | M3 |
| 3 | KL-check FP16 index keys; make them the default layout | long contexts | M2 |
| 4 | Validate the decode graphs; turn them on | everyone, most on VRAM-rich PCs | G1 |
| 5 | Rebuild the shipped expert profile from decode counts on a split | everyone | T1 |
| 6 | A/B `MAYA_GLM_PREFILL_MIN_CHUNKS=3,2048` | every split | P1 |
| 7 | Time the decode attention: `STRATA_GLMD_MLA_DEC=1` on the other Radeons, `STRATA_GLMD_FUSED_MLA=1` on each NVIDIA generation | VRAM-rich, AMD | G2 |
| 8 | Time the CPU lane's kernels per expert type (`glm_cpu_bench --row4 both`); try `MAYA_GLM_LANE_ROW4=1`, and `MAYA_GLM_CPU_SHARED=1` on two GPUs | RAM-bound | C1, C4 |
| 9 | Measure draft acceptance at the default sampling; A/B drafting off (`MAYA_GLM_NO_SPEC=1`) and `MAYA_GLM_SPEC_SAMPLE=1` | every split | S2, S3 |
| 10 | A/B `--glm-ram-resident` with `MAYA_GLM_PROMOTE_MIN` where RAM holds all off-card experts; `MAYA_GLM_RESIDENT_SKIP=1` where VRAM holds them all | 256 GB RAM PCs; VRAM-rich NVIDIA | T3, T4 |
| 11 | Turn on `MAYA_GLM_WARM_PROMPT`; A/B `MAYA_GLM_PREFILL_CPU_GATE=0`; on Turing `STRATA_GLMD_GATHER_RATIO=1` | first prompt; PCIe 4 / 5 PCs; Turing | P6, K1 |

Then let the setup write what won, per PC, and let `--calibrate` sweep the per-request knobs it does not touch
today (`lane_row4`, `cpu_split`, `promote_min`, `mtp_keep`, `pcie_frac_head`, `graphs` - all listed in
`serve/glm_dsa_tune.py:9-34`; the tool sweeps three, `tools/calibrate_glm.py:84-86`).

### 7.2 Then: small, exact code changes

Each is days, not weeks, and none changes the model's output.

1. **P2** - the prompt's residual and hand-over buffers at GLM-5.3's real size (a quarter); the window floor.
2. **S1** - the six unwired pieces of the speculative decode, starting with the statistics and the whole-head
   fallback.
3. **C1, C2, C3** - choose the CPU kernel per expert type by timing; huge pages on one-socket PCs; calibrate per
   expert class.
4. **T2, T4 (graph key), T5, T6 (router first)** - the tier's spares, the missing graph-key field, the constants.
5. **P3, P4** - start the prompt's CPU share at the plan; hand the routes over before the shared expert.
6. **G4** - the tiled multi-column GEMV for Q5_K (needed before any multi-token pass pays).
7. **K2, K3, K6** - the copies, checks and repeated dequantization around the prompt attention.
8. **R1, R2** - continue from the decoded answer; the reuse corner cases.

### 7.3 Then: the larger pieces

| Piece | Why it is worth the effort | See |
| --- | --- | --- |
| One `mma.sync` attention kernel at GLM-5.3's row width, used by decode and prompt (Ampere and newer); its Turing and RDNA counterparts | the attention is 11 G multiply-adds a token and about 30% of a compute-bound prompt | G2, K1 |
| A GLM-5.3 Turbo preset, and the deferred CPU add | on a RAM-bound PC the largest single lever; needs a KL sweep of 75 layers | A1, A2 |
| A cheap verify window, then the pipelined window on 3+ GPUs, then batched `--parallel` rows | the only way to read the dense 15 GB once for several tokens | S5, S6, S7 |
| The split planned for the speculative decode | up to about 18% on three equal cards | S4 |
| Expert kernels measured and shaped at 6144, AMD's included | a third of a token's bytes | T7, K5 |
| The indexer's scorer at mid and long context | grows with the context; all of AMD | G3, K4 |
| One expert pass a prompt on one GPU | 10 s a chunk on a V100 | P5 |
| A smaller dense quant at load | 1 - 2 GB a token; changes the answers | M4 |

### 7.4 By kind of PC

| PC | A token mostly waits on | Start with |
| --- | --- | --- |
| **RAM-bound** (the usual one) | the CPU lane and PCIe | M1 - M3 and T1 - T3 (more experts in VRAM), C1 - C3 (a faster lane), S3 (check that drafting pays), then A1. Prompts: P1 or P2, P3 |
| **VRAM-rich** | the GPUs' memory bandwidth and launch overhead | G1, G2, T4, then G4 + S5 -> S6. Prompts: K1 - K3 |
| **Disk-bound** | the SSD | M1 - M3 first (every GB freed is a GB not read from disk), then `MAYA_GLM_DISK_SKIP_W` (A1) and `MAYA_GLM_DISK_READERS` for prompts (`docs/SETTINGS.md:151`: +50% on one SSD, -15% on another - measure) |
| **AMD** | generic kernels | G2 step 1, G3 step 3, T7, K5 |

## 8. How to measure

### 8.1 Ground rules

- **Decode and prompt separately**, same quant, same context length, same thinking level. The bench and the
  calibration both run with thinking off.
- **Three runs.** `bench_http.py` flags a spread above 5%.
- **"Same answers" means greedy token identity with the CPU lane off** (`STRATA_GLM_CPU_LANE=0`,
  `STRATA_GLM_PREFILL_CPU=0`). The project's own A/B found that with the lane on "even base vs base differs"
  (`tools/strata/glm53_gpu_checks.sh:20-23`), because the lane's plan comes from timings taken at load.
- **Anything that may change answers** gets the KL check and the long-context gate (below), not just a speed number.
- **Cold prompt numbers** need `STRATA_GLM_NO_REUSE=1` (`tools/strata/bench_http.py:17`).
- **Drafting A/Bs on a split** need `MAYA_GLM_NO_SPEC=1` and a restart until S1 is fixed; the per-request
  `"spec": 1` is ignored there. `MAYA_GLM_SPEC_FOLLOW` makes two runs decode the same text
  (`src/program/generate.cpp:2224-2228`).

### 8.2 Tools already in the repository

| Question | Tool |
| --- | --- |
| Decode and prompt tokens/s, end to end | `tools/strata/bench_http.py --corpus data/glm-dsa/bench-corpus --ctx 4096,32768 --prompt-lens 8192,32768 --runs 3`; `--tune '{...}'` sets per-request knobs without a restart |
| Where a token's GPU time goes | `STRATA_GLM_PROF=1` (a mark after each kernel; graphs are off while it is on) |
| The CPU lane | `STRATA_GLM_LANE_PROF=1`; `glm_cpu_bench <pack> <layer> --row4 both` for one expert type |
| Drafting | `STRATA_GLM_SPEC_PROF=1`; `MAYA_GLM_SPEC_LOG` once S1 feeds it |
| A prompt's stages | `MAYA_GLM_PREFILL_STAGES` (one line a chunk and part), `STRATA_GLM_PREFILL_PROF`, `STRATA_GLM_PREFILL_TRACE=1` |
| Decode kernels alone | `glmd_dec_bench` (`tests/glm_dsa/glmd_dec_bench.cu`) |
| Prompt kernels alone | `glmd_batch_parity` with `GLMD_BATCH_BENCH=1` |
| Tier policies | `STRATA_GLM_ROUTE_LOG` and `tools/glm_tier_replay.py`; the slots-per-layer line under `STRATA_GLM_TIMING` |
| A split's price | `STRATA_GLM_SPLIT_LOG=1`; `STRATA_GLM_SPLIT_CARDS` prices cards the PC does not have |
| The worst case for the tiers | `STRATA_GLM_RANDOM_ROUTING` |
| Output quality | `tools/strata/glm53_gpu_checks.sh kld` and `long`; `tools/strata/glm53_long_gate.py` |

### 8.3 Numbers the repository does not have yet

A first measuring session should produce these. They replace this report's estimates and decide its order.

1. A per-kernel decode profile of GLM-5.3 on one VRAM-rich and one RAM-bound PC (replaces section 3).
2. Draft acceptance at temperature 1.0, top-p 0.95 (S2 - S4 depend on it; the code assumes 0.8).
3. The CPU lane's speed per expert type, for ggml's dot, the engine's own rows and the 4-row kernel (C1).
4. An expert kernel's time at 6144 x 2048 on each GPU architecture (T7).
5. A prompt's time split inside the attention phase: projections, scores, top-k, attention, output (K1 - K4).
6. `MAYA_GLM_PREFILL_MIN_CHUNKS` on a GLM-5.3 split (P1), and the split's chunk cap.
7. The length of the common prefix between consecutive requests of real agent sessions (R1).

## Appendix A. Reviewed and found sound

Listed so that nobody audits them again without a reason.

- **The route kernel**: selection, plan, tier lookups and the request to the host in one launch; the shared
  expert's activation and down product ride launches that exist anyway (`glm_fast.cu:4749-4750`).
- **Selection reuse**: 57 of 78 layers reuse the previous full layer's top-2,048 in decode and in prompts. Only the
  list can be shared - each layer has its own latents.
- **The hop between GPUs** in decode: one residual (24.6 KB), device to pinned host to device, ordered by an event,
  with no host wait (`glm_fast_path.cu:7142-7154`).
- **Disk reads in decode**: the answer is posted before the read and each chunk is copied as it lands.
- **Positional state**: checkpoints and snapshots cost nothing on GLM-5.3.
- **Prompt memory**: the score tile is bounded (16,384 keys), never T x context; only 256 route counts cross to the
  host per layer; the SSD is read ahead a layer.
- **The 64-head prompt attention's INT8 decode**: once a tile for all 64 heads.
- **The NextN cache fill** in prompts: only the projections and the cache rows - under 0.3% of a chunk.
- **MMQ's fused SwiGLU** on Turing and newer.
- **The one-chunk rule on one GPU** ("the fewest chunks win") and the split's prestage in proportion to the chunk:
  both already measured on GLM-5.3 by the project.

## Appendix B. Stale comments and docs noticed on the way

- `serve/glm_dsa_product.py:25-29` says "glm_state_carve refuses a resident window smaller than the context on
  glm-dsa". It no longer does (`src/core/glm_state_layout.cpp:105-112`). This comment is what keeps M1 off.
- `docs/SETTINGS.md:150`: `MAYA_GLM_PRESTAGE` "160 on one GPU, 0 on more" - a GLM-5.3 split's parts get 160
  (`glm_prefill.cu:1181`).
- `src/core/glm_model.cu:1376` prints "one 64 KB host hop per boundary per token"; GLM-5.3's hop is 24.6 KB.
- `src/core/glm_prefill_dsa.hpp:77`: "False on HIP: the glm-dsa prompt kernels are CUDA-only" - `built()` returns
  true on HIP (`glm_prefill_dsa.cu:196`), and
  `cmake/glm_dsa.d/50-glm-dsa-prefill.cmake:7-8` still says HIP has only the F32 prompt attention.
- `docs/SETTINGS.md` documents none of the `MAYA_GLMD_*` switches this report names (`_FUSED_MLA`, `_MLA_DEC`,
  `_IDX_WMMA`, `_TOPK_HIST`, `_GATHER_RATIO`, `_PF_*`).

## Appendix C. Code map

| Area | Files |
| --- | --- |
| Decode layer loop, graphs | `src/core/glm_residual.cu`, `glm_dsa_mixer.{hpp,cu}`, `glm_dsa_fast*.{cu,cpp,hpp}`, `src/kernels/cuda/glmd_graph.cu` |
| Decode kernels | `src/kernels/cuda/glmd_dec.cu`, `glmd_fast.cu`, `glmd_mixer.cu`, `glmd_volta_*.cu`, `glmd_topk.cuh`, `glmd_mv_ext.cuh`, `include/strata/kernels/glmd_decode_policy.hpp` |
| MoE on the token path, tiers | `src/core/glm_fast_path.cu` (`fast_moe`, `fast_boundary`, `fast_setup`), `src/kernels/cuda/glm_fast.cu`, `src/core/glm_memory.hpp`, `glm_tier_*.{cu,cpp}`, `glm_heat_evict.cpp` |
| CPU lane | `glm_fast_path.cu` (`lane_experts`, `lane_plan`), `src/kernels/cpu/*` |
| Drafting | `glm_fast_path.cu` (`decode_spec`, `fast_mtp`), `src/core/glm_mtp.cu`, `glm_window.cu`, `glm_dsa_spec.cpp`, `glm_ngram_draft.cpp` |
| Prompt path | `src/core/glm_prefill.cu`, `glm_prefill_dsa.cu`, `glm_dsa_prompt_plan.cpp`, `glm_dsa_prefill_stream*.cpp`, `src/kernels/cuda/glmd_batch.cu`, `glmd_guards.cu`, `src/prefill/*` |
| State, cache, split | `src/core/glm_state_layout.cpp`, `glm_model.cu`, `glm_dsa_split.cpp`, `src/kernels/cuda/glm_kv_stream.cu` |
| Requests, reuse | `src/program/generate.cpp`, `src/core/glm_prompt_cache*.{cu,cpp}` |
| Pack, setup, tuning | `tools/strata/glm_dsa_pack.py`, `glm_dsa_profile.py`, `maya.py`, `serve/glm_dsa_product.py`, `serve/glm_dsa_tune.py`, `tools/calibrate_glm.py` |
