# GLM-5.3 (744B): decode and prefill speed recommendations

A code review of Maya's GLM engine as it serves **GLM-5.3 (744B)** (GGUF architecture `glm-dsa`), with every
recommendation I could find for faster **decode** (writing the answer) and faster **prefill** (reading the prompt).
Written against v1.1.57 (`0065548`). Back to the [model's page](GLM53.md).

## 1. Summary

**Read this first: nothing here was measured.** The review machine has no GPU and no copy of the model. Every
expected gain is an estimate from the code, from the model's shapes, or from a measurement the project wrote down
for another model or PC. Section 9 is a runbook: the model that carries this report out runs each check there and
decides from the result.

**The owner's decisions (2026-11-09).**

1. **The targets** are PCs that run the model from VRAM + RAM, with the NVMe when needed, and the **Gorgon Halo**
   (160 GB of GPU memory, 32 GB of RAM) running a GGUF of about 170 GB. Rigs that hold the whole model in VRAM are
   not optimised at this stage; items that serve only them are marked "deferred".
2. **Validation is not part of this report.** The steps are (section 9); whoever reads it runs them.
3. **Turbo is wanted for GLM-5.3:** speed first, and coherent answers are the only requirement (section 5.7).

**The five findings that shape everything else.**

1. **A decoded token reads about 21 GB of weights, and two thirds of that is dense weights - mostly the
   attention's projections - not experts.** GLM-5.3 has a full attention on each of its 78 layers; Flash has one on
   11 of 45. So GLM-5.3 is bound by memory bandwidth and by how many experts sit in VRAM - and on a prompt, by how
   often the experts must be streamed (once per chunk, about 10 s on one V100).
2. **Much of what would make it faster is already in the tree but off, gated to Flash's sizes, or not wired.**
   Context-cache streaming is implemented end to end and disabled by one product flag. Decode graphs, the
   tensor-core decode attention and native Q8_0 attention weights wait "until validated". Six pieces of the
   speculative decode are not reached by its loop. Section 4 lists all of it.
3. **Several defaults and one data file do not fit GLM-5.3.** The shipped expert profile ranks 8 of the last
   layer's experts among its first 4,800 places, where every other layer has 49 - 78; the tier's ageing, the split's
   chunk cap and the short-prompt threshold were measured on Flash.
4. **Turbo's biggest lever already exists and was never measured.** One switch makes routing prefer experts that
   are already in VRAM. Turned up, no expert is fetched at all - no CPU lane, no PCIe, no NVMe - and a RAM-bound PC
   decodes like one that holds the model. The same idea in the prompt path would remove a chunk's fixed cost.
5. **On the Gorgon Halo a 170 GB GGUF almost fits under Windows and fits completely under Linux** - by the
   engine's own sizing rules; Linux was never run on that machine. Either way a token is bound by the memory's
   bandwidth, about 11 tokens/s. The lever past that is one-GPU drafting, which is off for GLM-5.3 for a reason
   that belonged to the 239 GB quant. And the GGUF itself matters: with IQ1_M experts the engine has no batched
   prompt path at all (section 7.5).

**The recommendations with the most value for the effort.** All effects are estimates.

| | Recommendation | Expected effect | Effort | Changes the answers? | See |
| --- | --- | --- | --- | --- | --- |
| **VRAM + RAM (+ NVMe), decode** | | | | | |
| 1 | Turn on the cache streaming the engine already has | Frees 4.9 GiB of VRAM at a 128K context, 11 GiB at 256K: about 450 / 1,100 more experts resident, and larger prompt chunks | Validation, one flag | No | M1 |
| 2 | Keep `attn_k_b` / `attn_v_b` in native Q8_0 | -1.1 GB of VRAM and -5% of the bytes every token reads | A parity run, a pack rebuild | Within rounding | M3 |
| 3 | Expert tiers: rebuild the profile, free unused spare slots, RAM-resident mode from the setup, stop sending evicted VRAM experts to disk, count usage in tokens | More experts in VRAM, fewer NVMe reads; no churn where RAM holds the model | Small | No | T1 - T5, D1, D2 |
| 4 | CPU lane: pick the kernel per expert type by timing, huge pages, per-class calibration | 10 - 30 ms a token on a RAM-bound PC | Small to medium | No | C1 - C3 |
| 5 | Finish the speculative decode's wiring; measure acceptance; let the tuner choose "no drafting" | Removes cases where drafting is slower than not drafting | Small | No | S1 - S3 |
| 6 | FP16 index keys; decode graphs | -9% cache and half the key reads; 1,500 launches a token become one replay a GPU | Validation | Slightly; no | M2, G1 |
| **VRAM + RAM (+ NVMe), prompts** | | | | | |
| 7 | On a split, pipeline reads that fit in one chunk | -10% to -18% on 4K - 16K reads (fit to the 3x V100 timings) | A setting to try, then a small change | No | P1 |
| 8 | Size the prompt's buffers for GLM-5.3's one residual stream, not Flash's four | About 25% more tokens a chunk on one GPU, each avoided chunk about 10 s; 1.2 GB of pinned RAM back per GPU of a split | Small, exact | No | P2 |
| 9 | Start the CPU's share at the plan; hand the routes over before the shared expert; continue from the decoded answer | Seconds per long read; agent turns read only the new tool result | Small | No | P3, P4, R1 |
| **Turbo** | | | | | |
| 10 | Sweep the route bias (`MAYA_GLM_ROUTE_AFFINITY`) against the coherence gate | No expert fetched: roughly 3 - 5x decode on a RAM- or disk-bound PC | A sweep, no code | **Yes** | A1, 9.6 |
| 11 | The same bias and skips in the prompt path | A chunk stops streaming experts: its fixed cost (about 10 s on one V100) goes | Small code | **Yes** | A2 |
| 12 | Fewer experts a token (`RESIDENT_TOPK`), per layer; lenient draft acceptance; a smaller dense quant | Fewer bytes a token everywhere; fewer redone drafts on a split | Sweep, then small code | **Yes** | A1, A2 |
| **Gorgon Halo** | | | | | |
| 13 | Run it on Linux with a small carve-out - or tighten the fit on Windows | Every expert resident on Linux; on Windows 3 - 7% stay outside GPU memory | A measurement | No | H1, H3 |
| 14 | Switch on one-GPU drafting for GLM-5.3 once the model fits | 1.2 - 1.4x decode as is, up to about 1.6x with two small code changes | A setting, then small code | No | H2 |
| 15 | In the GGUF: no IQ1_M experts; the dense types the kernels serve best | A prompt path at all; each GB of dense weights is about 4.3 ms a token | The quant's design | - | 7.5 |
| 16 | Prompts: a hipBLASLt table for the installed version; the fused AMD expert kernel at this model's shape | Flash precedents: 128 -> 168 and 233 -> 271 tokens/s | Small; medium | Within rounding | H5 |

**Where to start:** section 8 has the order, section 8.6 by kind of PC, section 9 the procedures.

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
`serve/glm_dsa_product.py`, `serve/glm_dsa_tune.py`). For the later sections: the unified-memory and Windows APU
handling (`include/strata/platform/*.hpp`, `docs/AMD_MAYA.md`), the disk reads, Turbo's product side
(`serve/models.py`), and the repository's own test and bench tools (`tools/strata/glm53_gpu_checks.sh`,
`bench_http.py`, `tools/maya_quant/loop_test.py`).

**What was not done.** Nothing was run. The review machine has no GPU and no copy of the 239 GB model, so there is
**no new measurement in this report** (the one file that was processed is the shipped expert profile, T1). The
review was done in several parallel reading passes; the findings kept here were then re-checked against the lines
they cite. Every number is one of three kinds, and each is labelled:

| Label | Meaning |
| --- | --- |
| **[code]** | Read directly in the source; the `file:line` is given. |
| **[project]** | A measurement the project itself wrote down (a code comment, `docs/SETTINGS.md`, the changelog). Quoted, not re-measured. Most are for GLM-5.3-Flash or for a 3x V100 PC, and are named as such. |
| **[estimate]** | My arithmetic from the model's shapes and the pack's tensor types. Expect ±20%. |

A recommendation's *expected effect* is therefore a reasoned estimate, never a promise. Section 9 says how to measure
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
| Setup: per-PC tuning | `--calibrate` sweeps three knobs; the setup writes no GLM-5.3 switch on NVIDIA | `tools/calibrate_glm.py:84-88`, `maya.py:3495-3502` | 8.1 |

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
changes the model's output, so it belongs with Turbo (section 5.7) and needs its own check
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

**Expect [estimate].** Where the GPUs set the pace, a few milliseconds a token (the attention is 11 G multiply-adds
a token). On a RAM-bound PC it is hidden behind the misses - do M1-M3 and the C-group first there - until Turbo
removes the misses (section 5.7): then this is what a token's time is made of.

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
can pay - on the Gorgon Halo an estimated 1.2 - 1.6x (section 7.4, H2) - and S6 after it.

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

### 5.6 The NVMe tier (PCs whose VRAM and RAM do not hold the model)

**What a disk read is [code].** A routed expert that is in neither VRAM nor the RAM tier parks the GPU
(`moe_wait`) while the host reads it:

- An expert is not one piece on disk. A layer is three tensors of 256 experts, so its gate, up and down parts lie at
  three file positions (`native_experts.txt` gives a layer's three offsets, `src/core/glm_model.cu:3036-3052`).
- Each part is read in 8 chunks: 24 reads an expert, on 8 worker threads plus the service thread
  (`src/core/glm_fast_path.cu:2112-2113`, `:5104-5111`, `:5161-5163`).
- Linux reads with `O_DIRECT` into a per-thread bounce buffer and then copies (`glm_fast_path.cu:691-724`); a
  refused `O_DIRECT` silently becomes a buffered read (`:725-733`). Windows uses an unbuffered overlapped handle.
- The bytes land in a free RAM slot, else in pinned staging, and the GPU pulls each chunk as it lands
  (`:5088-5102`, `:5122-5172`).

**What it costs.** For Flash-size experts the project measured "3.6 -> 3.1 ms a disk wait" on NVMe
(`glm_fast_path.cu:5104-5106`), and on a SATA SSD "~13 reads, 155 of 190 ms" of a token (`:2116-2117`)
**[project]**. GLM-5.3's experts are larger: about 4 - 5.5 ms each at 2 - 3 GB/s **[estimate]**. How many a token:
with the engine's own coverage rule (`1 - (1 - f)^3`, `glm_model.cu:1730-1737`), holding 80% of the experts in VRAM
and RAM gives about 5 reads a token, 70% about 16, 60% about 38 **[estimate]**. At 70% that is 70 ms of every token
spent waiting for the SSD. So the first rule on such a PC is section 5.1: every GB freed in VRAM or RAM is a GB not
read from disk. The rest of this section is what remains.

#### D1. Do not send an evicted VRAM expert to disk while the RAM tier holds colder ones

**Evidence [code].** When a spare slot must be refilled, the layer's least recently used resident leaves VRAM. It
goes to a free RAM slot if there is one. If there is none, it should replace the RAM tier's coldest expert when it
is hotter - but under the default recency rule its count is taken as zero (`const uint32_t c = vram_lru ? 0u : ..`,
`glm_fast_path.cu:5728`), so the test `rc_min < bc` (`:5762`) can never pass, and the expert is dropped to disk
(`:5783-5791`). Only four RAM slots a class are kept free (`:5797-5798`).

**Consequence [estimate].** On a disk-bound PC, every promotion beyond those four a token turns an expert that was
used recently enough to be in VRAM into a future disk read, while colder experts keep their RAM slots.

**Do.** Under the recency rule compare by last use (the tick both tiers already keep), or simply swap with the RAM
tier's victim. **Try first, no code:** `MAYA_GLM_VRAM_EVICT=least` (the count rule) and `MAYA_GLM_RAM_FREE=32`.
**Measure:** `STRATA_GLM_TIER_DIAG=1` (`glm_fast_path.cu:5995`) prints the disk reads of experts that were dropped
from VRAM; the `glm stat` line gives reads a token (`src/program/generate.cpp:2487`).

#### D2. Keep the usage counts through a prompt, and start them from the profile

**Evidence [code].** A prompt's routes are added to the same counts the decode evicts by, and the counts halve
every 32,768 routed experts (`src/core/glm_prefill.cu:3034-3051`). One 12,288-token chunk is 98,304 routes a layer:
three halvings a layer, 225 over the chunk. After a long prompt the counts of all but the last layers are zero, so
the "least used" rule has nothing to go by. At a fresh start the counts are zero too.

**Do.** Halve by tokens, not by routed experts (T5), and add a chunk's routes at a token's scale. Seed the counts
from the expert profile's rank at start, or have the setup switch on the usage file that keeps them between starts
(`MAYA_GLM_USAGE`, off, `docs/SETTINGS.md:100`).

#### D3. Read an expert as one aligned piece

**Evidence [code].** 24 reads and a copy an expert (above). The pack tool can write one contiguous
`[gate|up|down]` blob an expert (`--experts-bin`, `tools/strata/glm_dsa_pack.py:398-408`), but the GLM engine reads
that file's offsets and never uses them (`glm_model.cu:3039-3040`: `off` and `blob` are parsed and dropped).

**Do.** Read the blob - 4 KiB-aligned for this model's expert sizes - straight into the pinned slot with direct IO:
one to eight reads, no bounce copy. It costs the experts' size again on disk. **Try first, no code:** fewer, larger
pieces per request - `bench_http.py --tune '{"read_chunks": 4}'`, then 2 and 1 (the calibration tries only 4 and
2). A DRAM-less NVMe read "1.6 GB/s in the 256 KB pieces ... and 3.0 GB/s in 2 MB ones" (`glm_prefill.cu:2261-2263`)
**[project]**.

#### D4. Use the time a disk read takes

- **The CPU lane waits for the disk.** A route's CPU-lane experts start only after its disk reads return
  (`glm_fast_path.cu:5161-5172`, then `:5214`). Start them first: the route then costs the longer of the two, not
  the sum.
- **Lookahead** (`MAYA_GLM_AHEAD=<n>`, off): the next n layers' routers run on this layer's input and their
  disk-only experts are read ahead. On Flash it was "NOT faster ... only ~40% of disk misses are predictable within
  4 layers" (`bench/results/2026-10-06-glm-spec/README.md:31-33`) **[project]**. GLM-5.3 has an attention between
  every two MoE layers where Flash has one in four, so a read has longer to land: A/B it again, with
  `MAYA_GLM_AHEAD_READ=0` first to see the predictions' coverage without the reads.
- **After a prompt**, read back in the background what the prompt's lending pushed out of VRAM and RAM
  (`glm_prefill.cu:1776-1832`), instead of on the first tokens that route to it.

#### D5. Smaller disk items

- **Load order.** GLM-5.3 warms its tiers on one thread in file order (`MAYA_GLM_WARM_SEQ`, on for this model
  only), a rule made for a hard disk: "the test PC's USB disk gave 39 MB/s, 201 GB in 89 min"
  (`include/strata/core/glm_warm_seq.hpp:7-8`). On an NVMe the parallel order (24 slices on 8 threads) should load
  faster; Flash's Maya-L warmed 134 GB in 44 s that way (`docs/AMD_MAYA.md:288`) **[project]**. The setup already
  tells a spinning disk from an SSD (`maya.py:855`): let it write `MAYA_GLM_WARM_SEQ=0` for an SSD.
- **Say when direct IO was refused.** The fallback to buffered reads is silent (`glm_fast_path.cu:725-733`); with
  it a 239 GB model also fills the page cache.
- **Queue depth.** Decode has at most nine reads in flight a GPU, with no setting. The shards sit in one folder, so
  a second SSD helps only as an OS-level stripe.
- **In a prompt**, a chunk's disk experts are read, copied to the GPU and freed - so every chunk reads its whole
  disk set again (`glm_prefill.cu:3736-3754`). A fifth of the model at 2.9 GB/s is 15 s a chunk **[estimate]**.
  P5 (one expert pass a prompt) matters most here. `MAYA_GLM_DISK_READERS=8` measured +50% and +25% on one SSD and
  -15% on another (`docs/SETTINGS.md:151`): the runbook sweeps it with `MAYA_GLM_DISK_QD`.
- **The lossy way out** is `MAYA_GLM_DISK_SKIP_W` and route affinity: section 5.7.

### 5.7 Turbo: the fastest GLM-5.3 that still answers coherently

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
| `MAYA_GLM_RESIDENT_TOPK=<k>`, `MAYA_GLM_RESIDENT_SKIP_W` | Computes only the k heaviest routed experts, **wherever they sit**; the kept weights are renormalised (`glm_fast.cu:4206-4225`) | Strix Halo, all experts in VRAM: 33.2 -> 38.8 tok/s at k = 4; k = 3 broke the output; `_SKIP_W=0.20` "garbage", `docs/SETTINGS.md:136`. On Flash's 1 - 2 bit quants k = 4 "rambled" and Turbo leaves the cap off (`serve/models.py:796-798`) | **Gorgon Halo**, and everything else |
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
| Gorgon Halo, most experts in VRAM | `MAYA_GLM_RESIDENT_TOPK` 6, then 5, then 4; greedy, lowest thinking level | `MAYA_GLM_ROUTE_AFFINITY` for the experts that do not fit; A2.5 once it exists |

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
  (`glm_prefill.cu:1102-1116`). Not UD-IQ2_M - but a 170 GB quant is exactly where IQ1_M would be chosen
  (section 7.5).

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

## 7. The Gorgon Halo: 160 GB of GPU memory, 32 GB of RAM, a GGUF of about 170 GB

A target of its own (the owner's decision). The project has run GLM engines on exactly this machine - "Ryzen AI Max+
PRO 495 / Radeon 8065S (Gorgon Halo, 192 GB LPDDR5X, 160 GB of it the GPU's), Windows 11"
(`docs/AMD_MAYA.md:284-288`) - but only GLM-5.3's 239 GB quant, which does not fit it, was tried there
(`src/core/glm_mtp.cu:25-28`). A 170 GB GGUF changes the picture: almost all of it fits.

### 7.1 What the engine does on this machine today

**On Windows** the GPU's memory is a fixed carve-out, and the engine sizes it like a discrete card
(`src/core/glm_fast_path.cu:812-822`) **[code]**:

- **The expert pool** is what HIP reports free, less a 3,072 MiB reserve the setup writes (`maya.py:973-979`),
  kept inside the dedicated memory and inside Windows' commit (`glm_fast_path.cu:6930-6988`). HIP reports about
  172 GB for a 160 GB carve-out, and whatever is put past the carve-out pages: "8K-token prompts ran at 62 instead of
  159 tok/s" (`include/strata/platform/dedicated.hpp:3-7`) **[project]**. The page file must be about the GPU's share
  plus 8 GB; the setup asks for 180 GB (`docs/AMD_MAYA.md:268-281`).
- **The RAM tier** is pinned memory, which Windows counts as *shared GPU memory* and caps for all programs together:
  "160 GB carve-out, 15.84 GB shared, 2.56 GB of it held by the desktop: 13.02 GB of pinned buffers ran and 14.02 GB
  failed" (`include/strata/platform/gpu_shared.hpp:10-12`) **[project]**. The engine keeps GLM-5.3's tier under that
  limit (`glm_fast_path.cu:1840-1857`); with the setup's 8 GB headroom it is "a 10.4 GB tier instead of ~3 GB,
  decode +10%" (`maya.py:3526-3532`) **[project]**.
- **Everything else** is read from the NVMe when a token routes to it (section 5.6).

**On Linux** an integrated GPU's memory is the system's (`glm_fast_path.cu:812-822`). The pool is
`min(the experts' bytes, MemAvailable - 16 GiB - 1 GiB)`, measured after the dense weights and the cache are
allocated (`glm_fast_path.cu:1349-1379`, `docs/AMD_MAYA.md:187-198`). When every expert fits, the warm-up fills every
slot, no spare slots are kept, and the RAM tier is only a small staging floor. The project verified this path on a
128 GB Strix Halo, not on the Gorgon Halo.

**On both:** the newer decode attention is on (gfx1151 is the one AMD device that gets it,
`src/kernels/cuda/glmd_dec.cu:265-279`); the all-resident launch skip is on by default but acts only while *every*
expert is in GPU memory (`src/core/glm_fast_state.hpp:588-624`); drafting on one GPU is off for GLM-5.3
(`glm_mtp.cu:85-102`); `--calibrate` refuses a HIP build (`maya.py:3956-3958`), so nothing is tuned per machine.

### 7.2 Does a 170 GB GGUF fit?

**[estimate]** for a file with 19,200 trunk experts of 8.0 MB (153.6 GB), 15.6 GB of dense weights as in UD-IQ2_M,
and a cache of 63,168 bytes a token (the NextN block's share is carved only when the block is loaded,
`src/core/glm_state_layout.cpp:217`):

| | Context | Room for experts | Experts outside GPU memory | Where they are |
| --- | ---: | ---: | ---: | --- |
| Windows, 160 GB carve-out | 32K | ≈ 150 GB | ≈ 650 of 19,200 (430 that do not fit + 225 spare slots kept empty) | all in the RAM tier (it holds ≈ 1,300) |
| Windows, 160 GB carve-out | 128K | ≈ 144 GB | ≈ 1,450 | RAM tier full, ≈ 150 on the NVMe |
| Linux, small carve-out | 32K | ≈ 166 GB | none | - |
| Linux, small carve-out | 128K | ≈ 160 GB | none (with ≈ 4 GB to spare) | - |

So the model runs on this machine either way. On Windows it is *almost* resident, and "almost" has three costs:

1. Every MoE layer of every token still launches the wait, the fetch and the CPU-lane wait, because not *every*
   expert is in GPU memory. On Flash's 42 layers that was "~2.5 ms of a 54 ms ... token on the 8065S"
   (`glm_fast_path.cu:6284-6285`) **[project]**; on 75 layers about 4.5 ms a token **[estimate]**.
2. A long prompt lends about 6 GB of the pool (`MAYA_GLM_PREFILL_MB=6144`), which displaces roughly 800 experts
   into a RAM tier that is already nearly full **[estimate]**; the overflow goes to the NVMe and comes back on the
   first tokens of the answer.
3. Any growth - a longer context, loading the NextN block (about 3 GB) - pushes experts to the NVMe.

### 7.3 What bounds its speed

A token reads about 20 GB (14.9 GB of dense weights, the head, 600 experts of 8 MB). The project measured this
GPU's GEMVs at "220-240 GB/s", bound by the LPDDR5X (`src/kernels/cuda/glmd_mv_ext.cuh:375-376`) **[project]**. So
a token takes 85 - 93 ms - **about 11 tokens/s - with every expert resident and every kernel perfect [estimate]**.
Three quarters of those bytes are dense weights. On this machine the levers are therefore, in order: read the dense
weights once for several tokens (H2), read fewer bytes (H3, section 7.5), and only then anything else.

### 7.4 Recommendations for this machine

#### H1. Choose the operating system and the memory split on purpose

**Evidence.** Section 7.2. On Linux with a small carve-out the whole 170 GB model is resident **[estimate]**: no
RAM tier, no NVMe reads, no empty spare slots, no page-file rule, and the launch skip acts. On Windows it is 3 - 7%
short.

**Do.**
- Measure the model on Linux on this machine (runbook 9.8). This is the largest step that needs no code.
- **A pitfall to avoid:** Linux sizes the pool from the RAM the OS sees. With the BIOS still set to a 160 GB
  carve-out it would see 32 GB and size a pool of about 11 GB **[estimate from `glm_fast_path.cu:1349-1379`]**. On
  Linux the carve-out must be small. Have the setup warn when an integrated GPU's carve-out is larger than the RAM
  left.
- On Windows keep the 180 GB page file, and treat the 3 GiB reserve as measured: lowering it to 2 GiB buys 134
  experts and risks the paging above.

#### H2. Turn on one-GPU drafting for GLM-5.3 here (the verify window)

**What.** `MAYA_GLM_MTP_DECODE=1`: the NextN block drafts up to three tokens and one batched pass checks them. The
rows of that pass share the dense GEMVs, which is the only way past section 7.3's floor without changing the
output.

**Evidence.** It is the default on these APUs for Flash, where it gave "28.4-29.6 tokens/s instead of 19.7" on
this very machine (`CHANGELOG.md:1364-1367`) **[project]**. It is off for GLM-5.3 for a reason that belongs to the
239 GB quant: "a 4-row round read 5x a token's experts from the disk and the decode ran at x0.62"
(`glm_mtp.cu:25-28`) **[project]**. With nearly every expert resident that reason is gone. The round length adapts
by measurement and can fall to zero (`glm_mtp.cu:17-20`), so switching it on cannot make a token much slower.

**Expect [estimate].** Token by token, each token reads 20.2 GB. In a window the rows share 12.6 GB of dense
weights, and each row adds 7.6 GB of its own: its experts (4.8), the head (0.5) and 2.3 GB of `attn_k_b` /
`attn_v_b`, which the window reads once per row (`src/core/glm_dsa_mixer.hpp:421-507`). A draft costs about 0.65 GB.

| Round | Bytes read | Tokens kept at 75% / 85% acceptance | GB a token | Against 20.2 |
| --- | ---: | ---: | ---: | ---: |
| 1 draft, 2 rows | 28.5 GB | 1.75 / 1.85 | 16.3 / 15.4 | 1.24x / 1.31x |
| 3 drafts, 4 rows | 45.0 GB | 2.73 / 3.19 | 16.5 / 14.1 | 1.23x / 1.43x |
| 3 drafts, with `attn_k_b` / `attn_v_b` read once a window (S5) and in Q8_0 (M3) | 37.1 GB | 2.73 / 3.19 | 13.6 / 11.6 | 1.40x / 1.64x (against 19.1) |

75% is the acceptance the code assumes before it has measured (`glm_mtp.cu:535`); nobody has measured GLM-5.3's.
The NextN block costs about 3 GB of GPU memory (≈ 400 experts), which on Windows comes out of the pool.

**Do.**
1. A/B the switch as it is (runbook 9.8).
2. S5's two code items: the headwise GEMVs once for all rows of a window; the tiled multi-column GEMV for Q5_K (G4).
   On AMD the window already runs its rows a layer at a time (`src/core/glm_residual.cu:263-277`).
3. Then make it GLM-5.3's default wherever nearly all routed experts are resident - a rule on the measured hit
   rate, not on the device name - as it is Flash's default on these APUs today.

#### H3. On Windows, shrink what is outside GPU memory

Each expert is 8 MB. **[estimate]**

| Change | Experts gained | See |
| --- | ---: | --- |
| `attn_k_b` / `attn_v_b` in native Q8_0 | +134 (and 1.1 GB fewer bytes every token) | M3 |
| Do not keep empty spare slots where no route uses them | +150 - 225 | T2 |
| FP16 index keys | +22 at 32K, +88 at 128K | M2 |
| A context no larger than needed | +126 for every 16K less | - |
| The NextN block only if H2 pays | +400 when it is not loaded | H2 |

Cache streaming (M1) is *not* on this list: the streamed rows would live in pinned RAM, which is the scarcer of the
two pools here. `--glm-ram-resident` is a useful check, though: it refuses to start unless RAM holds every expert
that GPU memory does not, so a start with it proves that nothing is left on the NVMe.

#### H4. Skip the empty launches layer by layer

**Evidence [code].** The launch skip is all or nothing (`glm_fast_state.hpp:600-624`). A per-layer version exists
(`layer_res`, "every expert of layer il in VRAM"), but only batched rows use it (`glm_fast_state.hpp:595-599`,
`glm_fast_path.cu:6289-6292`), together with a fused wait + fetch launch for the other layers.

**Do.** On the token path use the fused launch (T4, step 3) and fold the CPU-lane wait into it: one launch a layer
instead of three, about 3 of the 4.5 ms a token **[estimate]**. The per-layer skip itself helps only on layers that
are complete. With the shortfall spread evenly over the layers none is - and spreading it is right, because it
leaves out each layer's coldest experts.

#### H5. Prompts on this machine

- **Check the pack's expert types first** (section 7.5). A pack with IQ1_M experts has no prompt path at all.
- **hipBLASLt table.** The projections take hipBLASLt only when a tuning table matches the installed version
  *exactly* (`src/prefill/hipblaslt_tuning.hpp:55-58`). GLM-5.3 has one table, for one version
  (`tools/hip/gfx1151-glm53-hipblaslt-100202.txt`); Flash has tables for 100202, 100401 and 100500. On any other
  version GLM-5.3's projections fall back to plain hipBLAS. On Flash, four tuned rows "took a prompt from 128 to
  168 tok/s" (`docs/AMD_MAYA.md:312`) **[project]**. Generate the table for the installed version
  (`tools/hip/tune_hipblaslt.cpp`, `--shape T,N,K`; the engine's log names each shape that misses).
- **The sub-batch.** The setup writes `MAYA_GLM_PREFILL_SUB=1024` for every AMD GPU - a value chosen for Flash,
  whose own default was 256. It also overrides GLM-5.3's attention sub-batch, which is 2,048 by itself
  (`src/core/glm_prefill.cu:362-364`). A/B 2048.
- **The fused expert kernel.** On Flash it gave, with the attention change, "271 and 278 tokens/s instead of 233
  and 253" on this GPU (`CHANGELOG.md:1376-1379`) **[project]**. It is instantiated for 4096 x 2048 and for
  DeepSeek's 5120 x 2304 only and aborts on anything else (`src/prefill/moe_fused_rdna4.cu:761-776`); a 6144 x 2048
  instance is a template argument away. It takes IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, but not
  IQ1_S or Q2_K (`:731-735`).
- **The prompt budget.** A/B a smaller `MAYA_GLM_PREFILL_MB` on Windows: fewer experts displaced per prompt
  (section 7.2, cost 2) against smaller chunks.

#### H6. Smaller items for this machine

- **The CPU lane.** Its 16 threads spin beside an integrated GPU that shares their memory bandwidth and power
  budget, and no calibration runs on HIP. A/B `MAYA_GLM_CPU_LANE=0` and half the threads by hand (runbook 9.8). On
  Linux, with every expert resident, the lane has nothing to do anyway.
- **Kernels.** Not where this machine's time is: its GEMVs already run at the memory's speed. For the record
  **[code]**: the tuned AMD GEMV takes Q6_K and BF16 only (`src/kernels/cuda/glm_fast.cu:5558-5575`), so UD-IQ2_M's
  Q5_K and Q8_0 weights go through the generic kernel; the indexer's scorer is the F32 one at every context (G3);
  the tuned expert kernels need 4096-wide experts and gave 1.04 - 1.13x where they apply
  (`docs/AMD_EXPERT_KERNELS.md:121-128`).
- **Loading.** `MAYA_GLM_WARM_SEQ=0` on the NVMe (D5).
- **Turbo here** starts with fewer experts a token (`MAYA_GLM_RESIDENT_TOPK`): at 4 of 8 it is 2.4 GB fewer bytes a
  token, about 12% - if answers stay coherent that low on a quant this small (section 5.7).

### 7.5 For the GGUF's author: tensor types and what the engine does with them

The 170 GB target leaves about 8 MB an expert - 1.7 bits a weight - so the choice is between IQ1_S (7.4 MB), IQ1_M
(8.3 MB) and IQ2_XXS (9.7 MB). The engine does not treat them alike. **[code]**

| Expert type | Decode on the GPU | Decode on the CPU lane (one token) | Prompt path |
| --- | --- | --- | --- |
| IQ1_S | yes | ggml's dot; the 4-row kernel if `MAYA_GLM_LANE_ROW4=1` | batched (MMQ), and the light kernels for experts with few rows |
| **IQ1_M** | yes | the engine's own AVX2 rows | **none: the whole prompt is read token by token** (`src/prefill/moe_mmq.cu:97-108`, `src/core/glm_prefill.cu:1102-1116`) |
| IQ2_XXS | yes | the engine's own AVX-512 rows (this CPU has them) | batched (MMQ) and the light kernels |
| IQ2_XS, IQ2_S, IQ3_XXS | yes | own AVX2 rows (IQ2_S: ggml's dot) | batched (MMQ); light kernels for IQ3_XXS only |
| Q2_K | yes | ggml's dot | batched (MMQ), no light kernel |

Sources: `src/kernels/cuda/glm_fast.cu:6941-6942`; `src/kernels/cpu/iq_avx2.cpp:802-821`,
`iq_avx512_one.cpp:152-155`, `glm_lane_row4_avx2.cpp:590-604`; `glm_prefill.cu:2964`.

**The one hard rule: no IQ1_M experts** - unless the fused AMD kernel is first opened to this shape and the prompt
path's type check accepts it (H5). A mix of IQ1_S and IQ2_XXS reaches the same size: 57 layers of IQ1_S and 18 of
IQ2_XXS average 7.9 MB an expert.

**Dense weights.** They are three quarters of a token's bytes on this machine: **each GB of dense weights costs
about 4.3 ms a token [estimate]**.

| Tensor | Guidance |
| --- | --- |
| `attn_k_b`, `attn_v_b` | Q8_0, served natively once M3 is on. Any type the kernels cannot read is widened to BF16 (`tools/strata/glm_dsa_pack.py:128-161`), which doubles it |
| `attn_output`, `attn_q_a`, shared expert, dense FFN | Q4_K or Q6_K rather than Q5_K where the size allows: both have the tiled multi-column GEMV the verify window uses, Q5_K does not (G4); Q6_K also has the tuned AMD GEMV |
| `attn_q_b` | Q8_0 today (2.8 GB a token). Q6_K is 2.1 GB, Q5_K 1.8 GB |
| indexer tensors, router | float (BF16 / F32) only: the prompt path refuses anything else (`src/core/glm_prefill_dsa.cu:94-108`) |
| the head, token embedding | any K-quant; the embedding stays on the host |

## 8. Suggested order of work

The order follows the owner's decisions: the targets are PCs that run the model from VRAM + RAM (+ NVMe) and the
Gorgon Halo; the exact mode first, then Turbo; rigs that hold the model in VRAM last.

### 8.1 Step 1: measure and flip (no new code)

Everything here is a setting or a data file. Section 9 gives each row its procedure and its pass rule, and the model
that carries this report out decides from the result.

| # | Do | For | Runbook | See |
| ---: | --- | --- | --- | --- |
| 1 | Take the baseline: speeds, the engine's own lines, a decode profile, the quality references | every PC | 9.2 | section 3 |
| 2 | Cache streaming: validate it, then set the product flag | RAM- and disk-bound PCs from 64K contexts; long prompts on one card. Not the Halo | 9.3 | M1 |
| 3 | `attn_k_b` / `attn_v_b` in native Q8_0: parity run, pack rebuild | every PC, the Halo most | 9.3 | M3 |
| 4 | FP16 index keys | every PC | 9.3 | M2 |
| 5 | Decode graphs | every PC | 9.3 | G1 |
| 6 | Rebuild the expert profile from decode counts | every PC | 9.4 | T1 |
| 7 | Tier settings: RAM-resident with `PROMOTE_MIN`; on a disk-bound PC `VRAM_EVICT=least`, `RAM_FREE`, the usage file | RAM- and disk-bound | 9.4 | T3, D1, D2 |
| 8 | CPU lane: time each expert type; the 4-row kernel; one shared pool on two GPUs | RAM-bound | 9.4 | C1, C4 |
| 9 | Drafting on a split: acceptance, drafting off, speculative sampling | splits | 9.4 | S2, S3 |
| 10 | Disk: read size, lookahead, the parallel load order, the prompt's readers | disk-bound | 9.4, 9.5 | D3 - D5 |
| 11 | Prompts: pipelined short reads on a split, the warm prompt, the CPU gate, the short-turn threshold | every PC | 9.5 | P1, P6 |
| 12 | Turbo: sweep the existing switches against the coherence gate | every PC | 9.6 | A1, A3 |
| 13 | The Gorgon Halo: Linux against Windows, one-GPU drafting, the hipBLASLt table, the sub-batch, the CPU lane | the Halo | 9.8 | H1 - H6 |

### 8.2 Step 2: small, exact code changes

Days each. None changes the model's output. In the order that serves the targets:

1. **The tiers: T2, D1, T5 with D2.** Free the unused spare slots; let an evicted VRAM expert take the RAM tier's
   coldest slot instead of going to disk; count usage in tokens so a prompt does not wipe it.
2. **The CPU lane: C1, C2, C3.** The kernel per expert type by timing; huge pages on one-socket PCs; calibration
   per expert class.
3. **Prompt memory: P2.** The residual and hand-over buffers at GLM-5.3's real size - a quarter of today's.
4. **Starting CPU work earlier: P3, P4, D4.** The prompt's CPU share from the plan on; the routes before the
   shared expert; the CPU lane beside a route's disk reads.
5. **Drafting: S1.** The six unwired pieces, the statistics first (every later decision needs the acceptance).
6. **Launches: T4 steps 2 and 3, H4.** The missing graph-key field; the fused wait + fetch and the per-layer skip
   on the token path.
7. **The verify window: G4, S5.** The tiled multi-column GEMV for Q5_K; the headwise GEMVs once a window. This is
   the Gorgon Halo's main decode item.
8. **Prompt attention glue: K2, K3, K6.**
9. **Reuse: R1, R2.**
10. **Disk: D3, D5.** An expert read in one piece; the load order by disk type; a message when direct IO is
    refused.
11. **The Gorgon Halo: H1, H5.** A setup warning for a large carve-out on Linux; the fused AMD expert kernel at
    6144 x 2048 (the largest of these, a few days to a week).

### 8.3 Step 3: Turbo's new code

Speed first; coherence is the only gate (section 5.7).

1. **A2, item 1:** the route bias and the miss skip in the prompt path - the largest prompt lever on a RAM-bound
   PC.
2. **A2, item 2:** a per-layer map for `RESIDENT_TOPK`.
3. **A2, item 3:** lenient draft acceptance on a split.
4. **G1, G2:** in Turbo a RAM-bound PC no longer waits for experts, so the GPU path sets its speed - graphs and
   the attention kernels move up.
5. **A2, items 4 - 6:** the deferred CPU add, the smaller dense quant at load, fewer attended cells.
6. **The product side:** offer Turbo for GLM-5.3 with the values the sweep chose (section 9.6).

### 8.4 Step 4: the larger pieces

| Piece | Why it is worth the effort | See |
| --- | --- | --- |
| One `mma.sync` attention kernel at GLM-5.3's row width for decode and prompt (Ampere and newer), and its Turing counterpart | about 30% of a long prompt read; the attention of every Turbo token | G2, K1 |
| One expert pass a prompt instead of one a chunk | 10 s a chunk on one V100; on a disk-bound PC also one disk read of each expert a prompt | P5 |
| The split planned for the speculative decode | up to about 18% on three equal cards | S4 |
| Expert kernels measured and shaped at 6144 | a third of a token's bytes | T7, K5 |
| The indexer's scorer at mid and long context | grows with the context; all of AMD | G3, K4 |
| A smaller dense quant at load, as a normal-mode option | 1 - 2 GB a token | M4 |

### 8.5 Deferred: rigs that hold the model in VRAM

Not a target at this stage: the pipelined window on three or more GPUs (S6; its safety fix is not deferred),
batched `--parallel` rows (S7), and the all-resident launch skip on NVIDIA (T4, step 1).

### 8.6 By kind of PC

| PC | A token mostly waits on | Start with |
| --- | --- | --- |
| **RAM-bound** | the CPU lane and PCIe | M1 - M3 and T1 - T3 (more experts in VRAM), C1 - C3 (a faster lane), S3 (check that drafting pays). Prompts: P1 on a split, P2 on one GPU, P3. Then Turbo (9.6) |
| **Disk-bound** | the NVMe | The same memory items first - every GB freed is a GB not read from disk - then D1 - D4. Turbo's route bias removes the disk from a token altogether |
| **Gorgon Halo** | the memory's bandwidth | H1 (Linux, or a tighter fit on Windows), H2 (one-GPU drafting), M3 and the GGUF's dense types (7.5), H5 for prompts |
| **Holds the model in VRAM** | *deferred* | G1, G2, T4 |

## 9. Validation runbook (for the model that carries this report out)

**Who this is for.** You are the model that runs this on a real PC with the GPUs and the model present. Nothing in
this report was measured. For each item: run the steps, compare with the pass rule, apply the decision that follows,
write the result down. The owner's instruction is that you decide from the results; you do not need to ask before
switching on something that passed its rule.

Every command below was read from the repository's own tools and sources, but **none was run for this report**. If
one fails as written, fix the command, not the rule.

### 9.1 Rules, and the traps in the repository's own tools

**Rules.**

1. **One change at a time**, against a baseline taken the same day on the same PC, pack, context and thinking
   setting.
2. **Speed** is the median of three runs after one warm-up. A change is "faster" when its median is at least 3%
   better and the two ranges do not overlap. Anything less is "no change": leave the default alone.
3. **Every item has a class**, and the class decides its quality check:
   - **E (exact)**: the same greedy tokens as the baseline (9.2, run 3).
   - **R (rounding)**: tokens may differ; the KL against the baseline's own dump stays under the item's bound
     (9.2, run 4), and the loop test finds no looping answer (9.2, run 5).
   - **L (lossy)**: Turbo only; the coherence gate of 9.6.
4. **A start loads the model** (minutes). Prefer per-request knobs (`--tune`) where a setting has one.
5. **Write everything down** (9.9), including failures and commands that did not work.

**Traps.** All **[code]**.

| Trap | What to do |
| --- | --- |
| The setup builds no tests (`-DSTRATA_BUILD_TESTS=OFF`, `maya.py:2056-2072`) | Configure them on once (9.2) |
| Test binaries and `glm_cpu_bench` read only `STRATA_*` names. The engine (`maya`) and the server also take `MAYA_*` (`include/strata/platform/env_alias.hpp:19-40`) | Use the `STRATA_` spelling with test binaries |
| `data/glm-dsa/parity/` ships no token files, so `glm53_gpu_checks.sh kld` and `flash-ab` do nothing unless `--ab-tokens FILE` is given | Make token files (9.2) |
| `glm53_gpu_checks.sh kld` and `loop` print PASS whenever the engine exits 0; `loop_test.py` always exits 0 | Read the numbers yourself: the KL, and "loop test: N looping answer(s)" |
| `glm53_gpu_checks.sh` refuses to run while port 8080 serves; its `long` check uses GPUs 0,1,2 | Stop the server first; call `glm53_long_gate.py` with `--gpus` on other PCs |
| A `--setup` run rebuilds the config's `env` (hand-added entries are lost) and writes `--kv-streaming off` again for GLM-5.3 (`maya.py:3495-3497`, `:3513-3559`) | Test with a copy of the config, or pass settings to the setup as `--env MAYA_X=v` |
| `STRATA_GLM_PROF` keeps the decode graphs off | Never profile and time graphs in one run |
| Request knobs: `"graphs"` can only switch graphs *off*; `"spec": 1` does not reach a split's decode; `"vram_evict"` does nothing | Use the environment and a restart for these |
| With the CPU lane on, two runs of one build can give different tokens (its plan follows timings taken at load, `tools/strata/glm53_gpu_checks.sh:20-23`) | Exactness runs switch the lane off (9.2, run 3) |

### 9.2 Preparation and baseline

**Build, and the tests.**

```sh
./maya.sh --setup --gguf-dir <folder with every GGUF shard>   # builds build/maya and the pack. Windows: START-MAYA.bat. AMD: add --backend hip --gpu 0
.venv/bin/cmake -S . -B build -DSTRATA_BUILD_TESTS=ON          # AMD: -B build-hip
.venv/bin/cmake --build build -j 4
STRATA_PYTHON=$PWD/.venv/bin/python ctest --test-dir build -L gpu -j 1 --output-on-failure --timeout 3600
```

Every test must pass or be skipped. The fixture tests build a synthetic model themselves and need numpy and
gguf-py in that Python.

**Names used below.**

```sh
EXE=build/maya                            # AMD: build-hip/maya
CFG=maya-glm-5.3-<quant>.json             # AMD: maya-glm-5.3-<quant>-hip.json. "args": the engine's arguments; "env": its settings; "log": its stderr
PACK=<the --glm-pack path in $CFG>
ARGS="--glm-pack $PACK --max-context 32768 --expert-cache auto --prefill auto --kv-streaming off --expert-profile <as in $CFG>"
GPUS="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<the config's gpu list>"   # AMD: HIP_VISIBLE_DEVICES
X="STRATA_GLM_CPU_LANE=0 STRATA_GLM_PREFILL_CPU=0 STRATA_GLM_MTP_ADAPT=0 STRATA_GLM_USAGE=0"   # for exactness runs only
```

**Token files.** The engine binary has no tokenizer. A chat prompt with thinking off is
`[gMASK]<sop><|user|>\n<the question><|assistant|>\n</think>` (`maya.py:4841`). Make three files: about 1K, 8K and
30K tokens (a question about a cut of `data/glm-dsa/bench-corpus/prompt_32k.txt` does).

```sh
.venv/bin/python -c "import sys; sys.path.insert(0,'tools'); import strata_tokenizer as ST; \
print(','.join(map(str, ST.Tokenizer.from_dir('$PACK/tokenizer').encode(open('prompt.txt').read(), parse_special=True))))" > ids.txt
```

**The six kinds of run.**

1. **Speed through the server** (what a user sees). `bench_http.py` needs `--api openai`; it sends one user message,
   temperature 0, thinking off. Put `MAYA_GLM_NO_REUSE=1` in the config's `env` for cold prompt numbers.

   ```sh
   ./maya.sh --config $CFG --plain &         # ready when: curl -s http://127.0.0.1:8080/health
   .venv/bin/python tools/strata/bench_http.py --api openai --port 8080 --corpus data/glm-dsa/bench-corpus \
       --tokenizer $PACK/tokenizer --ctx 4096,32768 --prompt-lens 2048,8192,32768 --gen 256 --runs 3 --json base.json
   # a per-request setting, no restart:  --tune '{"lane_row4": true}'
   ```

   The corpus reaches about 52K tokens; a longer context needs a longer one (`tools/strata/make_bench_corpus.py`).

2. **One engine run without the server** (for settings that need a start).

   ```sh
   env $GPUS STRATA_GLM_USAGE=0 <settings> $EXE $ARGS --tokens-file ids.txt --greedy --max-new 256 > run.out 2> run.log
   # sampling as the model's default:  --seed 1 --temperature 1.0 --top-p 0.95   instead of --greedy
   ```

   Read the last line of `run.out`:
   `DONE <produced> <prompt tokens> <prompt ms> <decode ms> <finish> <drafts accepted> <offered> <reused> <vram hits> <lookups>`
   (`src/program/generate.cpp:2476`), and in `run.log` the line
   `glm stat: decode .. ms/tok | vram hit ..% | ram fetches ../tok | disk .. reads/tok .. | promotions ../tok | cpu lane ../tok .. | prompt .. ms/tok`
   (`:2487`). One GPU only: add `STRATA_GLM_SPLIT=0`.

3. **Token identity** (class E). The baseline twice, then the change:

   ```sh
   env $GPUS $X           $EXE $ARGS --tokens-file ids.txt --greedy --max-new 256 > a1.out
   env $GPUS $X           $EXE $ARGS --tokens-file ids.txt --greedy --max-new 256 > a2.out
   env $GPUS $X <setting> $EXE $ARGS --tokens-file ids.txt --greedy --max-new 256 > b.out
   diff <(grep '^T ' a1.out) <(grep '^T ' a2.out) && diff <(grep '^T ' a1.out) <(grep '^T ' b.out) && echo IDENTICAL
   ```

   If the baseline differs from itself, say so in the results and use run 4 instead.

4. **KL against the baseline** (class R). `STRATA_GLM_SCORE=<c>` scores the tokens after the first c
   (`generate.cpp:4060-4067`); use a file of c + 256 tokens (a dump is 620 KB a scored token).

   ```sh
   env $GPUS $X STRATA_GLM_SCORE=<c> STRATA_GLM_SCORE_PREFILL=1 STRATA_GLM_SCORE_DUMP=base.f32 $EXE $ARGS --tokens-file ids.txt
   env $GPUS $X STRATA_GLM_SCORE=<c> STRATA_GLM_SCORE_PREFILL=1 STRATA_GLM_SCORE_REF=base.f32 <setting> $EXE $ARGS --tokens-file ids.txt
   ```

   Read `SCORE n tokens: nll .. ppl .. top1 ..% | vs ref: KL x (max y), same top token z%`. Run the baseline against
   its own dump once: that KL is the noise floor (0 with the lane off; the lane alone moved Flash by about 0.06,
   `docs/SETTINGS.md:134`).

5. **The loop test** (coherence; `tools/maya_quant/loop_test.py`). Five prompts that need long, structured answers
   (a three.js scene, a canvas animation, an explanation, a plan in Portuguese, a proof), each once sampled and once
   greedy, checked for: stopped by itself, closed its reasoning, no 32-token window repeated three times, under 15%
   of its tokens inside repeated 64-grams.

   ```sh
   .venv/bin/pip install tokenizers
   env $GPUS <settings> .venv/bin/python tools/maya_quant/loop_test.py --exe $EXE --pack $PACK --tok $PACK/tokenizer \
       --max-new 3000 --runs 1 --save out/loop-<name>
   ```

   Pass: the last line is `loop test: 0 looping answer(s)`. A `CAP` row (still writing at the limit, no repeat) is
   acceptable; a `BAD` row is not. `--save` keeps every answer as text.

6. **The long-context gate.** No token file ships: make one from a chat prompt over a 16K - 32K-token document.

   ```sh
   .venv/bin/python tools/strata/glm53_long_gate.py --exe $EXE --pack $PACK --tokens long.ids --head 20 --tol 0.25 \
       --gpus <list> [--env STRATA_X=v] --out out/long
   ```

   It prints `PASS|FAIL long gate: long NLL .. | short NLL .. | long - short ..`. **It fails at 32K today**
   (section 3.4), so compare a change with the baseline's own "long - short", not with PASS.

**The baseline to record, once per PC.**

- The bench table of run 1: decode at 4K and 32K, prompt at 2K, 8K and 32K, cold.
- From the engine's log at load: `glm fast: CUDA.. expert pool .. GB, a-b slots/layer`;
  `glm prefill: .. chunks of T tokens`; `glm split auto: ..`;
  `CPU lane: N threads, an expert X ms on the CPU vs Y ms over PCIe`; `glm-dsa mixer: .. decode policy: ..`.
- Per request: the `glm stat:` line and `glm spec: a of b drafts accepted`.
- A decode profile: run 2 with `STRATA_GLM_PROF=8` for 64 tokens prints, at exit, each kernel's ms a token.
  `STRATA_GLM_TIMING=1` adds the slots per layer and the tier counters; `STRATA_GLM_LANE_PROF=1` the lane's phases.
- A prompt profile: the stage line of each chunk (on by default) and `STRATA_GLM_PREFILL_PROF=1`.
- Quality: the baseline's KL against itself, its loop test, its long-gate number.

Then **name the PC's kind**, because it decides what to run: disk reads a token above zero after 200 tokens -
disk-bound; else a VRAM hit rate under about 95% - RAM-bound; else GPU-bound (the Gorgon Halo, or any PC in Turbo).

These numbers are also the ones section 3 could only estimate: the per-kernel decode profile, the draft acceptance,
the lane's speed per expert type, the prompt's split inside the attention.

### 9.3 Features that are built but switched off

#### M1. Cache streaming - class E for the decode, R for a long prompt

1. **There is no test of a streamed GLM-5.3 window.** The nearest is `glm_dsa_prefill_decode_test` with
   `STRATA_GLM_KV_RESIDENT=2096 STRATA_GLM_KV_RESIDENT_TEST_MIN=2096` (context 4,096; a window between 2,048 and
   2,135 makes it evict). Its logits and draft checks apply as they are; its cache-row checks read the arena in
   position order and must first learn the residency map. Write that variant and add it to
   `cmake/glm_dsa.d/55-glm-dsa-wire.cmake`. It must end `glm_dsa_prefill_decode_test: ok`.
2. **Engine, same residents in both arms.** Cap the pool so that streaming does not change which experts fit:
   `STRATA_GLM_POOL_GB=<a little under the resident arm's pool>` in both arms. With a 40K-token prompt and
   `--max-context 65536`: arm A `--kv-streaming off`; arm B `--kv-streaming on --kv-resident 32768`. Run 3: the
   tokens must be identical. The load prints
   `glm pack: CUDA0 KV streaming: N of CTX positions of each of L DSA layers in VRAM (X GB), ..`, and each request
   `glm kv: KV streaming: P% of N block reads hit VRAM ..` - the word `OVERFLOW` there is a failure.
3. **Speed, without the cap.** Decode and prompt at a 100K context, streamed against resident: read tokens/s, the
   slots per layer and the prompt chunk size. At 32K nothing may change (`--kv-streaming on` alone streams only when
   the whole cache does not fit, `src/program/generate.cpp:4909-4939`).

**Pass:** steps 1 and 2 hold, and at the long context the streamed arm is not slower in decode and has a larger
prompt chunk. **Then:** `serve/glm_dsa_product.py:29` `KV_STREAMING = True`. Its consumers follow by themselves
(`maya.py:1912`, `:3497`, `:3584`; `serve/models.py:263`, `:311-312`, `:556-563`; `serve/web/models.js:381`). Update
`serve/test_glm_dsa_models.py:117-190` and `docs/GLM53.md:45`. Not on the Gorgon Halo (H3).

#### M3. Native Q8_0 `attn_k_b` / `attn_v_b` - class R

1. `PYTHONPATH=third_party/llama.cpp/gguf-py python3 -m unittest discover -s tools/strata -p 'test_glm_dsa_pack_native32.py' -v`,
   and the ctest `glm_dsa_prefill_decode_test_native32`.
2. Build a second pack beside the first:
   `STRATA_PACK_NATIVE32=1 STRATA_GGUF_PY=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py --gguf <first shard> --out <gguf folder>/pack-n32 --compat-bf16`.
   Its `compat-bf16.json` must no longer list the two tensors, and the load must print
   `W_kb/W_vb native Q8_0 on 79 layers`.
3. Run 3 with `--max-new 64` and run 4, the new pack against the old.
4. Run 2 on both packs for speed; `build/glmd_dec_bench 40` times the Q8_0 headwise GEMV on its own.

**Pass** - the project's own bar (`tools/strata/glm_dsa_pack.py:115-125`): 64 greedy tokens identical, or KL under
1e-4; and decode not slower. **Then:** make it the default at `glm_dsa_pack.py:124-125`, change the default the
unit test asserts (`test_glm_dsa_pack_native32.py:25-32`), and rebuild installed packs with
`MAYA_PACK_NATIVE32=1 ./maya.sh --setup --repack --gguf-dir <folder>`.

#### M2. FP16 index keys - class R

1. The ctest `glm_dsa_prefill_decode_test_fp16`.
2. Run 4 with `STRATA_GLM_IDX_FP16=1` against the default at contexts of 4K, 16K and 30K (the selection only
   matters above 2,048 keys). Run 6 both ways (`--env STRATA_GLM_IDX_FP16=1`). Run 5 once.

**Pass:** KL at most 0.005 above the noise floor at every context; the long gate's "long - short" no more than 0.05
nats worse; no looping answer. **Then:** the default at `src/core/glm_state_layout.cpp:239-241`. The product side
assumes F32 keys and must follow: `serve/glm_dsa_product.py:43` (`KV_BYTES` 64352 -> 58720), `serve/models.py:326`,
`maya.py:1951`; the test `tests/glm_dsa/glm_idx_fp16_test.cpp:72-74`; `docs/SETTINGS.md:2197`. Conversations saved
under the old layout are refused and read again - expected.

#### G1. Decode graphs - class E

1. The ctests `glmd_graph_test`, `glmd_graph_gen_gpu_test`, `glm_dsa_prefill_decode_test_volta`.
2. Run 3 with `STRATA_GLM_DSA_GRAPHS=1` at 1K, 8K and 30K tokens of prompt, on one GPU (`STRATA_GLM_SPLIT=0`) and
   on the split. Then once more with the CPU lane on, as run 4.
3. Speed: start the server with `MAYA_GLM_DSA_GRAPHS=1`; run the bench twice, plain and with
   `--tune '{"graphs": false}'`.
4. At unload the log prints `glm-dsa decode graphs: CUDA0 a captures, b replays, c invalidations, ..`: a handful of
   captures, replays close to the token count.
5. Before it becomes the default, close T4's step 2, or show it does not bite: after a graph is captured, send a
   request with `"miss_skip_w": 0.2` and check that experts really are skipped.

**Pass:** identical tokens everywhere, and decode at least 3% faster on some target PC with none slower.
**Then:** the default at `src/kernels/cuda/glmd_graph.cu:27-32`.

#### G2. The decode attention kernel - class R

1. `build/glmd_dec_bench 40` on each GPU architecture prints the old and new attention times and
   `mla_dec on this device: yes|no`. On a Radeon other than gfx1151: `STRATA_GLMD_MLA_DEC=1 build-hip/glmd_dec_bench`.
2. NVIDIA: run 4 and run 2 with `STRATA_GLMD_FUSED_MLA=1`; the ctest `glmd_volta_parity`.

**Pass:** KL at most 0.005 above the noise floor and the kernel faster in step 1 and in run 2. **Then:** the
per-device rule in `src/kernels/cuda/glmd_dec.cu:270-279`, or the default at
`include/strata/kernels/glmd_decode_policy.hpp:87`.

### 9.4 Decode settings to sweep

On NVIDIA run `./maya.sh --calibrate` first: it sets the CPU lane's share, its threads and the disk read size for
this PC (it refuses a HIP build). Then, one row at a time:

| Item | How | Read | Keep it when |
| --- | --- | --- | --- |
| **T1** the expert profile | Decode at least 5,000 varied tokens on the split, NextN loaded, with `MAYA_GLM_USAGE=<file>`. Then `.venv/bin/python tools/strata/glm_dsa_profile.py $PACK <file> --base data/expert-profile-glm53.bin --out $PACK/expert-profile.bin`, and set `--expert-profile` in the config by hand | Parse the new file as T1's table does: layers 77 and 78 must hold a normal share of the first 4,800 pairs. After a fresh start: VRAM hit %, decode | hit rate up, decode not slower |
| **T3** RAM-resident tier | `--glm-ram-resident` in `args` (the start fails if RAM cannot hold every off-card expert - that is the test); then `--tune '{"promote_min": N}'` for N = 0, 2, 4, 6, 8 | decode, promotions a token | faster |
| **D1, D2** disk-bound tiers | `MAYA_GLM_VRAM_EVICT=least`; `MAYA_GLM_RAM_FREE=32`; `MAYA_GLM_USAGE=<file>` across a restart | disk reads a token in `glm stat`; `STRATA_GLM_TIER_DIAG=1` | fewer reads and faster |
| **C1** lane kernels | `build/glm_cpu_bench $PACK <layer> <threads> --row4 both`, one layer of each expert type (types: columns 2 and 3 of `$PACK/native_experts.txt`); then `--tune '{"lane_row4": true}'` | ms an expert, and "output bit-identical"; decode | faster for that type |
| **C4** lane pool | two GPUs: `MAYA_GLM_CPU_SHARED=1`; `--tune '{"cpu_split": N}'` | decode, `STRATA_GLM_LANE_PROF=1` | faster |
| **S2, S3** drafting on a split | three starts: default; `MAYA_GLM_NO_SPEC=1`; `MAYA_GLM_SPEC_SAMPLE=1`. Run 2 with the default sampling flags | `glm spec: a of b drafts accepted`, decode; `STRATA_GLM_SPEC_LOG=64` | the fastest of the three becomes this PC's setting |
| **S1** non-Latin answers on 3+ GPUs | a Chinese prompt, default against `MAYA_GLM_MTP_DRAFT_VOCAB=0` | acceptance, decode | (confirms the gap; the fix is code) |
| **D3** read size | `--tune '{"read_chunks": N}'` for N = 8, 4, 2, 1 | disk ms a token | faster |
| **D4** lookahead | `MAYA_GLM_AHEAD=2` with `MAYA_GLM_AHEAD_READ=0`, then with reads | coverage and waste under `STRATA_GLM_TIMING=1`; decode | faster |
| **D5** load order | `MAYA_GLM_WARM_SEQ=0` | `glm fast: .. tiers warm: N experts (X GB) in T s` | faster on this disk |

To check a profile (T1): how many of each layer's experts are among the first 4,800 ranked pairs.

```sh
python3 - "$PACK/expert-profile.bin" <<'EOF'
import struct, sys, collections
b = open(sys.argv[1], 'rb').read()                      # "STRP", then version, layers, experts, slots, ranked pairs
ver, nl, ne, want, nr = struct.unpack('<5I', b[4:24])
raw = struct.unpack('<%dH' % (2 * nr), b[24:24 + 4 * nr])   # (layer, expert) pairs, best first
top = collections.Counter(raw[2 * i] for i in range(min(nr, 4800)))
for layer in range(nl): print(layer, top[layer])       # layers 0-2 have no experts; 3..78 should be alike
EOF
```

### 9.5 Prompt settings to sweep

All measured with run 1's prompt rows, cold. A change in chunking is class R: check run 4 with a long context, and
accept a KL up to 0.02 (re-blocking alone measured 0.013 on Flash, `docs/SETTINGS.md:85`).

| Item | How | Keep it when |
| --- | --- | --- |
| **P1** pipelined short reads (splits) | `MAYA_GLM_PREFILL_MIN_CHUNKS=3,2048`; also `2,2048` and `4,2048`; prompts of 2K, 4K, 8K, 16K | faster at 4K - 16K and not slower at 2K |
| **P6** first prompt | `MAYA_GLM_WARM_PROMPT=64`; time the first request after a start | the first prompt is as fast as the second |
| **P6** CPU gate | `MAYA_GLM_PREFILL_CPU_GATE=0`, then `1` | faster |
| **P6** short turns | `MAYA_GLM_PREFILL_MIN` = 8, 16, 32, 64; time follow-up turns of 5, 20 and 50 new tokens on a restored conversation | the fastest becomes the default for this model |
| **D5** disk readers | `MAYA_GLM_DISK_READERS=8`; `MAYA_GLM_DISK_QD` = 4, 8, 16 | faster on this disk |
| **K5** Volta only | `MAYA_GLM_PREFILL_BLAS` = 0, 128, 512, each with `MAYA_PF_WMMA` = 0, 1 | the fastest pair |
| **K1** Turing only | `STRATA_GLMD_GATHER_RATIO=1` | faster |

For where the time goes: the stage line of each chunk, `STRATA_GLM_PREFILL_PROF=1`, `STRATA_GLM_PREFILL_TRACE=1`.

### 9.6 Turbo: the sweep and the coherence gate

**The rule (the owner's).** Turbo takes the fastest settings whose answers are still coherent. Less detailed or less
accurate is acceptable; incoherent is not.

**Every Turbo arm** runs greedy with thinking off or at its lowest level - the request half the page already
applies (`serve/web/turbo-core.js:6-18`).

**The sweep.** Take the exact mode's speeds and saved loop-test answers first; they are the reference.

1. **Route affinity** (RAM- and disk-bound PCs; a restart for each value): `MAYA_GLM_ROUTE_AFFINITY` = 2, then 1,
   0.5, 0.3, 0.2, 0.1, 0.05. Read decode tokens/s and the VRAM hit rate in `glm stat` (100% means no expert is
   fetched any more). Keep the largest value that passes the gate.
2. **Fewer experts a token**, on top of step 1: `MAYA_GLM_RESIDENT_TOPK` = 6, then 5, then 4. Stop at the first
   that fails. Be warned: on Flash's 1 - 2 bit quants a cap of 4 "invented three.js classes and rambled"
   (`serve/models.py:796-798`), and GLM-5.3's quants are in that range.
3. **If the affinity had to stay low:** `--tune '{"miss_skip_w": w, "miss_skip_from": l}'` for w = 0.08 .. 0.20 and
   l = 0, 23, 40 (no restart), and on a disk-bound PC `MAYA_GLM_DISK_SKIP_W` = 0.12, 0.20.
4. **On a split:** `--tune '{"mtp_keep": 0}'`.

On the Gorgon Halo start at step 2, then step 1 for the experts outside GPU memory.

**The coherence gate.** A candidate passes only if all of these hold:

1. Run 5 with the candidate's settings ends `loop test: 0 looping answer(s)`, with no `BAD` row.
2. No request ends in `ERR`.
3. The saved three.js and canvas answers contain a script that parses (`node --check` on the extracted script -
   the check the project used for Flash's Turbo, `serve/models.py:784-787`), and no answer has broken characters.
4. **You read every saved answer**, beside the exact mode's answer to the same prompt, and judge each: on the
   topic, grammatical, consistent with itself, and finished. Fail the candidate if any answer drifts off the
   topic, contradicts itself, turns into word salad, or stops making sense partway. Do **not** fail it for being
   shorter, plainer or less accurate.
5. One question about a 16K - 30K-token document whose answer is in the text: the reply must be about that
   document.

Record run 4's KL for each candidate, for information only.

**Choose** the fastest candidate that passes; between two within 3%, the one with the lower KL.

**Then wire it** (`serve/models.py`): GLM-5.3's config is in the `glm` family, so removing the gate alone would
apply Flash's values. Add GLM-5.3's own entry and pick it in `turbo_patch` (`:825`) with
`G53.config_is_glm_dsa(cfg)`; make `turbo_hidden` (`:837-840`) return false; decide whether Turbo's 32,768 context
cap (`:770-771`) should apply (it frees VRAM for experts, which suits the route bias). Update `serve/test_turbo.py`
and `serve/test_glm_dsa_models.py:201-221`, `docs/GLM53.md:60`, and document `MAYA_GLM_ROUTE_AFFINITY` in
`docs/SETTINGS.md`. The request half needs nothing.

**Prompts in Turbo** are not affected by any of these switches until A2's first item is written; measure them again
after it.

### 9.7 After a code change

| Area | Run | Must hold |
| --- | --- | --- |
| Any engine change | the ctests (`-L gpu`); run 3 on a short and an 8K prompt, one GPU and the split; run 5 | tests pass; tokens identical (class E changes); no looping answer |
| Tiers (T2, D1, T5, D2) | `glm_tier_plan_test`, `glm_heat_evict_test`; run 3; `glm stat` before and after; a route log (`STRATA_GLM_ROUTE_LOG=<prefix>`) through `tools/glm_tier_replay.py` | identical tokens; more hits or fewer disk reads |
| CPU lane (C1 - C3) | `glm_cpu_bench .. --row4 both`; run 4 with the lane on | "output bit-identical"; KL at the noise floor |
| Drafting (S1) | `glm_ngram_draft_test`, `glm_dsa_spec_test`, `glm_dsa_spec_rollback_test`, `glm_dsa_spec_gpu_test`; run 3 against `MAYA_GLM_NO_SPEC=1` | drafts never change a greedy token |
| Verify window (G4, S5) | `glmd_mv_parity`; run 3 with `MAYA_GLM_MTP_DECODE=1` against the token path | identical tokens |
| Prompt path (P2 - P4, K2, K3) | `glmd_batch_parity`, the `glm_dsa_prefill_decode_test` family; run 4 with a long context | KL at most 0.02 against before (0 for P2, which is exact) |
| Reuse (R1, R2) | `STRATA_TEST_ENGINE=build/maya STRATA_TEST_PACK=$PACK .venv/bin/python -m unittest serve.test_engine_contract -v`; a three-turn conversation with and without `MAYA_GLM_NO_REUSE=1` | the same greedy answers |

### 9.8 The Gorgon Halo

1. **The pack's expert types.** `$PACK/native_experts.txt` (columns 2 and 3) must not hold type 29 (IQ1_M). The
   log must not say `glm prefill: .. has no prompt path (not covered: ..) - token by token`.
2. **The fit.** From the start log: `VRAM before the expert pool`, `expert pool .. GB, a-b slots/layer`; on Windows
   `Windows gives this APU's GPU ..`, `the RAM tier .. GB instead of ..` and any `WARNING: .. GB of experts stay off
   the GPU`; on Linux `unified memory: MemAvailable .., experts need .., pool budget ..`. From `glm stat`: the VRAM
   hit rate, RAM fetches and disk reads a token. Compare with section 7.2.
3. **H1, Linux against Windows.** The same pack and context on both: run 1. On Linux set the BIOS carve-out small
   first. Record whether every expert is resident (hit rate 100%, no RAM fetches).
4. **H2, one-GPU drafting.** `MAYA_GLM_MTP_DECODE=1` (a start). Exactness: run 3 against the token path
   (`STRATA_GLM_MTP_ADAPT=0` is already in `$X`). Speed: run 1, and in the same run `--tune '{"spec": 1}'` for the
   arm without drafting. Read `glm spec: a of b drafts accepted`; `STRATA_GLM_WIN_PROF=1` shows where a round's
   time goes. **Keep it when** decode is at least 3% faster with identical tokens.
5. **H3, a tighter fit on Windows.** The native-Q8_0 pack (9.3); `MAYA_GLM_IDX_FP16=1` (9.3); a start with
   `--glm-ram-resident` (it fails if anything would be left on the NVMe); `MAYA_GLM_RESERVE_MB=2048` only while
   watching prompt speed for the paging symptom of section 7.1 (a drop of more than half).
6. **H5, prompts.** The log names every projection shape that misses the hipBLASLt table: build
   `tools/hip/tune_hipblaslt.cpp`, tune those shapes for the installed version, add the table, measure with
   `tools/hip/bench_prefill.py` and run 1. Then `MAYA_GLM_PREFILL_SUB=2048`, and `MAYA_GLM_PREFILL_MB` = 4096,
   6144, 8192.
7. **H6.** `MAYA_GLM_CPU_LANE=0` and `MAYA_GLM_CPU_LANE=8` against the default; `MAYA_GLM_WARM_SEQ=0`.
8. **Turbo:** 9.6.

### 9.9 What to write down

One entry a result, in `docs/GLM53-SPEED-RESULTS.md` (new):

```
## <item id> <short name> - <date>
PC:        <GPUs, VRAM each, RAM, NVMe, OS, driver>      kind: RAM-bound | disk-bound | Gorgon Halo
Model:     <quant, pack path, native32 yes/no>           context: <n>   thinking: off | <level>
Class:     E | R | L
Commands:  <exactly as run, with the environment>
Baseline:  <three numbers> median <m>
Change:    <three numbers> median <m>   (<+x%>)
Quality:   <identical tokens | KL x against noise y | loop test N | long gate d>
Verdict:   ON (default changed at <file:line>, commit <hash>) | OFF (why) | NEEDS CODE (what)
```

Commit each default that changes on its own, with its numbers in the message.

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
