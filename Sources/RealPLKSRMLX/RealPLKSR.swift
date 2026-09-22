// RealPLKSR — MLX-Swift port of neosr's `realplksr_arch.py` (Apache-2.0).
//
// Isomorphic to upstream: same class names, same decomposition, same forward order; only
// PyTorch→MLX op substitutions. Layer parity vs the PyTorch oracle: worst max|Δ| 6.4e-05 over 10
// taps on the public fixture (AB-R-0178), and the same harness re-run on THIS package's checkpoint
// (mlxengine-image/WIP/realplksr-parity, `PLK_CKPT=4x_webphoto_realplksr_s3_100k.safetensors`).
//
// Two deliberate deviations from upstream, both documented at the site:
//   1. `PLKConv2d` ports the TRAINING branch (split → conv → concat), not the eval branch's
//      in-place slice assignment. Mathematically identical, functional-style.
//   2. `PixelShuffle` is hand-rolled for NHWC (MLXNN has no channel-last pixel shuffle).
//
// Config of the shipped checkpoint: dim 64, 28 blocks, kernel 17, split 0.25, EA on, 4 norm groups,
// no DySample (its `groups=` conv has no MLX kernel — see mlx-no-grouped-conv3d). 7,389,680 params.

import Foundation
import MLX
import MLXNN

/// Errors from the RealPLKSR core.
public enum RealPLKSRError: Error, Sendable, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)
    /// The checkpoint's parameter set does not cover the module tree exactly — a silently partial
    /// load is the failure mode that produces plausible-but-wrong output, so it is loud.
    case parameterMismatch(missing: [String], extra: [String])

    public var description: String {
        switch self {
        case .weightsNotFound(let p): return "RealPLKSR weights not found: \(p)"
        case .loadFailed(let d): return "RealPLKSR weight load failed: \(d)"
        case .parameterMismatch(let m, let e): return "RealPLKSR parameter mismatch — missing \(m) extra \(e)"
        }
    }
}

// MARK: - ops

/// Numerically stable softplus: max(x,0) + log1p(exp(-|x|)).
@inline(__always) func softplusStable(_ x: MLXArray) -> MLXArray {
    MLX.maximum(x, 0) + MLX.log1p(MLX.exp(-MLX.abs(x)))
}

/// Mish = x * tanh(softplus(x)).
@inline(__always) func mish(_ x: MLXArray) -> MLXArray { x * MLX.tanh(softplusStable(x)) }

/// NHWC pixel shuffle matching `torch.nn.PixelShuffle` on NCHW.
///
/// Torch views (B, C·r·r, H, W) as (B, C, r1, r2, H, W) — channel index `c·r·r + i·r + j` maps to
/// (c, i, j) — then permutes to (B, C, H, r1, W, r2). Channel-last equivalent:
/// (B,H,W,C·r·r) → (B,H,W,C,r1,r2) → (B,H,r1,W,r2,C).
func pixelShuffleNHWC(_ x: MLXArray, _ r: Int) -> MLXArray {
    let (b, h, w, crr) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = crr / (r * r)
    return x.reshaped([b, h, w, c, r, r])
            .transposed(0, 1, 4, 2, 5, 3)      // B, H, r1, W, r2, C
            .reshaped([b, h * r, w * r, c])
}

// MARK: - modules

/// DCCM: Conv3x3(dim→2·dim) → Mish → Conv3x3(2·dim→dim). Upstream names the convs 0 and 2 (1 is Mish).
final class DCCM: Module, UnaryLayer {
    let layers: [Conv2d]
    init(_ dim: Int) {
        layers = [Conv2d(inputChannels: dim, outputChannels: dim * 2, kernelSize: 3, padding: 1),
                  Conv2d(inputChannels: dim * 2, outputChannels: dim, kernelSize: 3, padding: 1)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers[1](mish(layers[0](x))) }
}

/// Partial large kernel: a DENSE k×k conv on the first `pdim` channels, the rest pass through.
/// (`realplksr_arch.py:31` has no `groups=` — the partiality is in the slice, not the grouping.)
final class PLKConv2d: Module {
    let conv: Conv2d
    let idx: Int
    init(_ pdim: Int, _ kernelSize: Int) {
        conv = Conv2d(inputChannels: pdim, outputChannels: pdim,
                      kernelSize: IntOrPair(kernelSize), padding: IntOrPair(kernelSize / 2))
        idx = pdim
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Upstream's eval branch mutates in place; the training branch is ported instead.
        let x1 = x[0..., 0..., 0..., 0 ..< idx]
        let x2 = x[0..., 0..., 0..., idx...]
        return MLX.concatenated([conv(x1), x2], axis: -1)
    }
}

/// Element-wise attention: x · σ(Conv3x3(x)).
final class EA: Module, UnaryLayer {
    let f: Conv2d
    init(_ dim: Int) { f = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(f(x)) }
}

final class PLKBlock: Module, UnaryLayer {
    let channelMixer: DCCM
    let lk: PLKConv2d
    let attn: EA
    let refine: Conv2d
    let norm: GroupNorm

    init(dim: Int, kernelSize: Int, splitRatio: Float, normGroups: Int) {
        channelMixer = DCCM(dim)
        lk = PLKConv2d(Int(Float(dim) * splitRatio), kernelSize)
        attn = EA(dim)
        refine = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        norm = GroupNorm(groupCount: normGroups, dimensions: dim, pytorchCompatible: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let skip = x
        var h = channelMixer(x)
        h = lk(h)
        h = attn(h)
        h = refine(h)
        h = norm(h)
        return h + skip
    }
}

/// The network. Input/output are NHWC RGB float32 in [0, 1]; output is `upscale`× the input.
public final class RealPLKSR: Module {
    let stem: Conv2d
    let blocks: [PLKBlock]
    let tail: Conv2d
    public let upscale: Int

    /// Exact parameter count of the shipped `webphoto` checkpoint (dim 64 / 28 blocks / k17 / split 0.25).
    public static let webphotoParameterCount = 7_389_680

    public init(inCh: Int = 3, outCh: Int = 3, dim: Int = 64, nBlocks: Int = 28,
                upscalingFactor: Int = 4, kernelSize: Int = 17, splitRatio: Float = 0.25,
                normGroups: Int = 4) {
        stem = Conv2d(inputChannels: inCh, outputChannels: dim, kernelSize: 3, padding: 1)
        blocks = (0 ..< nBlocks).map { _ in
            PLKBlock(dim: dim, kernelSize: kernelSize, splitRatio: splitRatio, normGroups: normGroups)
        }
        tail = Conv2d(inputChannels: dim, outputChannels: outCh * upscalingFactor * upscalingFactor,
                      kernelSize: 3, padding: 1)
        upscale = upscalingFactor
        super.init()
        // C14: born in inference mode at the construction choke point. Nothing here is
        // train/eval-dependent (GroupNorm has no batch statistics; upstream's Dropout2d(0) is not
        // ported), but the loaded graph must REPORT `training == false`.
        train(false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = stem(x)
        for b in blocks { h = b(h) }
        h = tail(h)
        // upstream: feats(x) + repeat_interleave(x, upscale², dim=1)
        // repeat_interleave repeats each channel CONSECUTIVELY → MLX `repeated`, not `tiled`.
        let residual = MLX.repeated(x, count: upscale * upscale, axis: -1)
        return pixelShuffleNHWC(h + residual, upscale)
    }
}

// MARK: - weight loading

/// PyTorch Conv2d weight is (O, I, kH, kW); MLX Conv2d wants (O, kH, kW, I).
@inline(__always) func convWeightToMLX(_ w: MLXArray) -> MLXArray { w.transposed(0, 2, 3, 1) }

extension RealPLKSR {
    /// Load a neosr-style safetensors checkpoint (`feats.N.*` keys, 340 tensors) from disk.
    public func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RealPLKSRError.weightsNotFound(url.path)
        }
        let arrays: [String: MLXArray]
        do { arrays = try MLX.loadArrays(url: url) } catch { throw RealPLKSRError.loadFailed(String(describing: error)) }
        try loadWeights(arrays)
    }

    /// Maps the upstream `feats.N.*` checkpoint keys onto this module tree.
    ///
    /// Checkpoint layout (340 tensors): feats.0 = stem, feats.1…28 = PLKBlocks, feats.29 = Dropout2d
    /// (no params), feats.30 = tail.
    public func loadWeights(_ sd: [String: MLXArray]) throws {
        var flat: [(String, MLXArray)] = []
        func conv(_ swiftPath: String, _ ckpt: String) throws {
            guard let w = sd["\(ckpt).weight"], let b = sd["\(ckpt).bias"] else {
                throw RealPLKSRError.loadFailed("checkpoint missing \(ckpt).{weight,bias}")
            }
            flat.append(("\(swiftPath).weight", convWeightToMLX(w)))
            flat.append(("\(swiftPath).bias", b))
        }

        try conv("stem", "feats.0")
        for i in 0 ..< blocks.count {
            let c = "feats.\(i + 1)", s = "blocks.\(i)"
            try conv("\(s).channelMixer.layers.0", "\(c).channel_mixer.0")
            try conv("\(s).channelMixer.layers.1", "\(c).channel_mixer.2")
            try conv("\(s).lk.conv",               "\(c).lk.conv")
            try conv("\(s).attn.f",                "\(c).attn.f.0")
            try conv("\(s).refine",                "\(c).refine")
            guard let nw = sd["\(c).norm.weight"], let nb = sd["\(c).norm.bias"] else {
                throw RealPLKSRError.loadFailed("checkpoint missing \(c).norm.{weight,bias}")
            }
            // GroupNorm affine params are per-channel vectors — no layout change.
            flat.append(("\(s).norm.weight", nw))
            flat.append(("\(s).norm.bias", nb))
        }
        try conv("tail", "feats.\(blocks.count + 2)")

        // Coverage check BEFORE update: a silently partial load is the failure mode that produces
        // plausible-but-wrong output, so make it loud.
        let expected = Set(parameters().flattened().map { $0.0 })
        let provided = Set(flat.map { $0.0 })
        if expected != provided {
            throw RealPLKSRError.parameterMismatch(
                missing: Array(expected.subtracting(provided).sorted().prefix(5)),
                extra: Array(provided.subtracting(expected).sorted().prefix(5)))
        }
        do {
            try update(parameters: ModuleParameters.unflattened(flat), verify: .all)
        } catch {
            throw RealPLKSRError.loadFailed(String(describing: error))
        }
    }
}
