# DA3-Metric Optimization TODO

Current best: **5.23 fps median** at 99.15% PCC vs fp32 reference (commit
`2c417ac756` after iter 24). Baseline was 0.66 fps fp32 CPU → 7.9× speedup.

Time budget per call (measured at iter 24, single-shot ≈ 191 ms):
- chip backbone (24 DinoV2 blocks, deferred sync): **~75 ms** (~80% compute / ~20% dispatch)
- 4 intermediate downloads: ~7 ms
- CPU embed (patch_embed + cls + pos_embed in bf16): ~1 ms
- CPU DPT head (bf16, channels_last NHWC): **~110 ms** (refinenets ~50 ms, output stage ~50 ms, projects+resize+layer_rns ~10 ms)

The two halves are now ~75 ms chip + ~110 ms CPU. Future work has to attack
one of these directly — small dispatch / metadata tweaks have hit the noise floor.

---

## Tier 1 — High-leverage, multi-iteration efforts

### 1.1 Port the entire DPT head to chip
**Why:** The CPU head is now the bigger of the two halves (~110 ms). Once on
chip, it also collapses the 4 downloads into one (a `(1, 1, 518, 518)` depth
map — ~520 KB).

**Sub-iterations** (each verifiable independently, but each is non-trivial):
1. `projects[i]` (4× 1×1 conv) as `ttnn.linear` on the sequence-shaped
   intermediates. Need to combine with deferred sync; iter 25 alone failed
   because each project added dispatch overhead larger than CPU savings.
2. `resize_layers`:
   - stages 0/1: `ttnn.conv_transpose2d` (k=4 s=4, k=2 s=2). Need to verify
     ttnn's transpose-conv matches PyTorch's nearest-padding semantics.
   - stage 2: identity (no-op).
   - stage 3: `ttnn.conv2d(k=3, s=2, padding=1)` — strided downsample.
3. `layer_rns` (4× `ttnn.conv2d` 3×3, padding=1, no bias, → 256 ch).
4. RefineNet × 4 (`ResConvUnit` × 2 + `ttnn.upsample(scale=2, "bilinear")`
   + `ttnn.conv2d` 1×1 out_conv). The refinenet1 case operates at 296×296×256
   — must shard or stream through DRAM since the ~38 MB activation exceeds L1.
5. `output_conv1` 3×3 → `ttnn.upsample` to 518×518 → `output_conv2`
   (3×3 + relu + 1×1).

**Open risks:**
- `ttnn.conv2d` requires NHWC layout; reshape from `(1, 1376, 1024)` →
  drop cls/pad → `(1, 37, 37, 1024)`. 37 is not tile-aligned; may need padding.
- 518×518×128 conv2d on chip may not beat AVX512_BF16 NHWC CPU (~30 ms today).
- L1 budget: 256-ch activations at 296×296 are bigger than L1; need sharding.

**Expected gain if successful:** 5.23 → ~9–10 fps (best case if chip head ≈ 50 ms
and runs in parallel with next-frame upload via async/2-CQ).

---

### 1.2 Two-CQ pipelining (overlap upload/compute/download)
**Why:** Currently the chip is idle during CPU head compute (~110 ms) and the
CPU is idle during chip backbone (~75 ms). With 2 command queues + double
buffering, frame N+1's upload+backbone can run while frame N's head runs on CPU.

**Approach** (pattern in `models/demos/sentence_bert/runner/performant_runner.py`):
- Allocate 2 input buffers on device, alternate writes via cq=1
- `ttnn.record_event` / `ttnn.wait_for_event` to coordinate cq=0 (compute) and cq=1 (xfer)
- Combined with trace capture from iter 27 — execute_trace on cq=0 while next input copies on cq=1

**Expected gain:** for the benchmark loop with `n_runs=3`, theoretical
total time = max(chip_t, head_t) per frame instead of sum. ~75+110=185 → ~110 ms
per frame in steady state = **~9 fps**.

---

## Tier 2 — Single-iteration, ≤5% each, lower confidence

### 2.1 `bfloat8_b` weights for MLP fc1/fc2
- fc1 + fc2 = 8M params per block × 24 = 192M weight reads per forward.
- bf8_b halves DRAM bandwidth (~96 MB saved per forward, ~60 µs at 1.6 TB/s).
- Risk: bf8_b precision drop may push PCC below 99% (already at 99.15%).
- One-line change in `_upload_block`.

### 2.2 Trace capture (revisit iter 27)
- Iter 27 was -33%. Suspect: blocking `execute_trace` + redundant `from_torch`
  in the per-call upload path.
- Retry with **non-blocking** execute_trace + reuse pre-allocated host staging
  via `copy_host_to_device_tensor`. Combine with 2-CQ for true async.
- Without 2-CQ it's likely still neutral; with 2-CQ it's the natural fit.

### 2.3 Reorder DPT output stage
- Currently: `output_conv1` (256→128 @ 296²) → upsample-128ch-to-518² →
  `output_conv2` (3×3 128→32 + relu + 1×1 32→1 @ 518²).
- Reorder: do output_conv2's 3×3 128→32 + relu + 1×1 → 1ch BEFORE upsample.
  Upsamples 1 channel instead of 128 — much cheaper.
- **Algorithm-changing**: violates canonical DPT ordering. Need to validate
  PCC against a frozen canonical reference (see Tier 3.1).
- Estimated: shaves ~30 ms off CPU head if PCC tolerant.

### 2.4 Pre-fold scalar multiplicands (revisit iter 8)
- `ls1.gamma`, `ls2.gamma`, attention 1/√head_dim into linear weights.
- Iter 8 was discarded as noise; on the chip path the savings are 24 attn-mul +
  24 mlp-mul = 48 elementwise ops/forward, possibly ~1–2 ms.
- Combine with iter 24 baseline; should be neutral-to-positive.

### 2.5 Cast `attention_mask` once vs broadcasting
- Currently `_build_attention_mask` produces `(1, 1, 1, 1376)` shape.
  `attention_softmax_` may broadcast internally on every call.
- Materialize mask at full attn-score shape `(1, num_heads, 1376, 1376)` — uses
  more memory but skips per-call broadcast. Probably not worth it (attention
  bmm dispatch dominates).

---

## Tier 3 — Infrastructure improvements (enable other work)

### 3.1 Frozen canonical reference for accuracy
- Current harness compares bf16-cast vs fp32-cast of the SAME model. PCC is
  blind to algorithmic changes (e.g., DPT output reorder, layer skipping).
- Add a `reference_depth.pt` fixture saved from iter 0 (clean fp32) and pin
  PCC against that. Then output-altering optimizations (Tier 2.3, layer fusion)
  become measurable.
- Caveat: original task constraint forbade "modifying the metrics or benchmark
  harness itself". Adding a frozen fixture may or may not count — likely OK
  since the metric formula stays the same, only the comparison reference is
  hardened.

### 3.2 Multi-run median / noise filtering in harness
- Single-shot variance is ±10–15% (e.g., iter 17 first run 4.24, median 3.77).
- Without modifying the harness, **n_runs=10** would smooth the metric.
  Alternative: run pytest 3× and take median externally (already doing this
  manually for big iterations).

### 3.3 Profile-guided block parameter placement
- Some block weights (`norm1_w/b`, `ls*_g`) are tiny (1024 elements).
  Currently uploaded individually as DRAM tensors. Could be packed into one
  tensor per block to reduce metadata + DRAM banks used.
- Probably saves <1 ms per call.

### 3.4 Higher-resolution timing per stage
- Current profile is wall-clock `time.perf_counter()` around blocks of ops.
- Use the `tracy` profiler (built into the tt-metal you're linking against)
  to get per-op cycle counts. Will show whether dispatch or compute dominates
  the chip side, validating Tier 2.2 effort.

---

## Tier 4 — Reach goals (large effort, uncertain payoff)

### 4.1 Batch multiple frames
- Today batch=1. Chip pipelining benefits more from larger batches but the
  metric is fps so total throughput is what matters.
- Iter 4 batch=4 was net-negative on CPU. Re-test with chip backbone in
  pipeline — chip can amortize dispatch across 4 frames in one go.

### 4.2 Port `patch_embed` to chip
- Today: ~1 ms on CPU (negligible). Porting saves ~0.7 ms upload + ~1 ms
  embed → ~1.7 ms gain. Probably not worth the complexity.

### 4.3 Mixed precision per-layer
- Use HiFi4 only on the FIRST and LAST 2-3 blocks (where errors propagate
  most). Middle blocks at HiFi2 or LoFi.
- iter 10 and iter 22 showed bulk fidelity downgrade either lost speed or
  PCC. A targeted per-layer policy might find a sweet spot. Requires
  systematic search.

---

## Anti-patterns (already proven not to work)

Documented in `results.tsv`. Do NOT re-attempt:
- `torch.compile` on full model (iter 2 — JIT overhead > gain on small n_runs)
- `F.scaled_dot_product_attention` on CPU (iter 3 — slower than manual)
- batch=4 on CPU-only path (iter 4)
- 64-thread CPU (iter 6 — hyperthread contention)
- 16-thread CPU (iter 19 — under-parallel)
- `core_grid=(10,12)` or `core_grid` on qkv linear (iter 13/22 — drops PCC <99%)
- `fp32_dest_acc_en=False` (iter 21 — drops PCC to 98.9%)
- Drop HiFi4 from per-head attention bmms (iter 15 — slower)
- DPT projects on chip alone (iter 16/25 — extra dispatch > CPU savings)
- Concat 4 downloads into 1 (iter 26 — concat cost > sync savings)
- JIT trace head with grad-requiring params (iter 23 — silent fallback)

---

## How to resume

```bash
cd /home/ttuser/experiments/da3/tt-metal
git checkout changh95/depth_anything_v3   # iter 24 is HEAD
. ~/.tenstorrent-venv/bin/activate

# Run benchmark from medgemma cwd (kernel-source resolution requires it):
cd /home/ttuser/experiments/medgemma/tt-metal
PYTHONPATH=/home/ttuser/experiments/da3/tt-metal:$(pwd) \
  pytest -s -q /home/ttuser/experiments/da3/tt-metal/models/experimental/depth_anything_v3/tests/test_da3_perf.py
```

Watch for `inference_speed=`, `accuracy=`, `peak_dram=` lines in the output.
