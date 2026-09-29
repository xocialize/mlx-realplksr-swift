# mlx-realplksr-swift

**RealPLKSR 4× super-resolution for MLXEngine — the web-photo tier.** Weights trained by xocialize on the
provenance-clean **Commons-SR** corpus (`training-resources/WEBPHOTO-SR-PLAN.md`, P3 finished 2026-09-22),
so the whole `imageUpscale` path is clean end-to-end under the conservative reading: corpus, trainer,
architecture, port and weights.

| | |
|---|---|
| capability | `imageUpscale`, native 4× (sub-native `scale` honored by post-downsampling) |
| architecture | RealPLKSR — dim 64 · 28 blocks · 17×17 partial large kernel · split 0.25 · EA · GroupNorm(4) · PixelShuffle(4); 7,389,680 params |
| weights | `4x_webphoto_realplksr.safetensors` (29.6 MB fp32) — the stage-3 EMA of the Commons-SR run, **vendored** (no download) |
| licences | weights **Apache-2.0** (ours) · port code **Apache-2.0** (derived from neosr's `realplksr_arch.py`) — C7/C8 both permissive |
| engine posture | bundled weights (MAT gate), fp32 `QuantConfigured`, split footprint, CAN gate (entry + per-tile checkpoints), C14 inference mode |
| sibling | `mlx-realesrgan-swift` — same capability, ~6× cheaper, for clean/native sources; this package reuses its tile driver |

## Products

- **`RealPLKSRMLX`** — engine-agnostic core: `RealPLKSR` (the network, NHWC, isomorphic to upstream) and
  `RealPLKSR_Playback` (a `PlaybackTier` over the shared `MLXTileProcessor`: 256² tiles, 32 px feathered
  overlap, whole-frame path ≤ `wholeFrameMaxPixels`).
- **`MLXRealPLKSR`** — the MLXEngine `ModelPackage`: `RealPLKSRUpscalePackage` + `RealPLKSRConfiguration`.
- **`realplksr-smoke`** — drives the package through the real `MLXServeEngine` and reports the split footprint.

```swift
let engine = MLXServeEngine()
let id = try await engine.register(RealPLKSRUpscalePackage.registration, configuration: RealPLKSRConfiguration())
let out = try await engine.run(ImageUpscaleRequest(image: image), package: id) as! ImageUpscaleResponse
```

## What the model is

Phips' *4xNomosWebPhoto_RealPLKSR* recipe, reproduced on a corpus we can defend: 5,666 Wikimedia Commons
images (CC0 / public domain / CC-BY, no SA, no NC) → 64,909 lossless 512² tiles (BHI-filtered) × 12
degradation variants (resampling, lens blur, sensor noise, JPEG/WebP, chroma subsampling; 775,308 pairs).
Trained with neosr (commit `31c7022620c682cf0961c8634d60787179145c5b`) on HF Jobs A100s in four stages —
an MS-SSIM pretrain, then GAN + perceptual at GT 128, + LDL/DISTS/FF at GT 256, and a whole-tile polish at
512 — 500k iterations, $147.71. Full receipts: `mlxengine-image/corpus/commons-sr/train/RUNBOOK.md`,
provenance for the card: `…/train/receipts/PROVENANCE.md`.

Hold-out (300 `val_hard` tiles, the fully degraded variant, 128² → 512²): PSNR 29.5 dB vs bicubic 28.95,
DISTS (proper, offline) see the model card; the model beats bicubic on the fidelity metric while adding the
fine texture the perceptual losses target — the recipe's trade, by design.

## Gates

- **Layer parity** vs the PyTorch oracle (upstream's arch file executed on this checkpoint):
  `Tests/RealPLKSRMLXTests` runs the whole forward against the golden on the CPU stream (max|Δ| < 1e-2,
  cos > 0.9999); the per-sub-op harness lives in `mlxengine-image/WIP/realplksr-parity` (AB-R-0178 method).
- **C0–C14** offline in `Tests/MLXRealPLKSRTests` (manifest, licences, requirements, descriptor, codec
  seams), **MAT-1..5** (`MaterializationConformance`), **CAN-1..3** (`CancellationConformance`).
- **Live**: `swift run -c release realplksr-smoke <image.png> <out.png> [webphoto] [wholeFrameMaxPixels]`
  — real engine path, timing, luminance sanity, and an MLX-peak memory report. The manifest's footprint is
  declared on the in-app `phys_footprint` basis instead (0.1 GB + 8.6 GB since v0.1.1, measured in the
  ForgeOptimizer app — see the manifest comment); the smoke's MLX peak under-reads that basis.

```bash
swift build -c release
swift test                         # offline suites — no Metal evaluation
swift run -c release realplksr-smoke in.png out.png webphoto 0   # 0 forces the tiled path
```

## Notes

- Tile overlap buys seam *agreement* here, not receptive-field coverage (the RF is effectively global).
  Throughput is flat at ~121–125 ms per Mpx of output on M5 Max (AB-R-0177): 1024² ≈ 130 ms, 4K ≈ 1 s.
- `DySample` is deliberately not ported (its `groups=` conv has no MLX kernel); this checkpoint does not use it.
- The port's two deviations from upstream (training-branch `PLKConv2d`, hand-rolled NHWC pixel shuffle) are
  documented at the site in `RealPLKSR.swift`.
