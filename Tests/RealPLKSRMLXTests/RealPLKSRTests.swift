//
//  RealPLKSRTests.swift — core tests: shapes, the exact parameter count, inference mode (C14), the
//  vendored checkpoint's full-coverage load, and LAYER PARITY of the whole forward against the PyTorch
//  oracle golden produced from THIS checkpoint (mlxengine-image/WIP/realplksr-parity/make_goldens.py,
//  upstream's realplksr_arch.py executed on 4x_webphoto_realplksr, input = the harness's 64² probe).
//
//  Everything runs on the CPU stream: from `swift test` the Metal bundle is not always staged into
//  the .xctest bundle and the first GPU op crashes (same wrapper as the Real-ESRGAN sibling). fp32 on
//  CPU is also the numerically strict setting for parity.
//

import Foundation
import Testing
import MLX
import MLXNN
@testable import RealPLKSRMLX

/// Run a closure with the MLX default device pinned to CPU.
private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

private func totalParameterCount(_ module: Module) -> Int {
    var total = 0
    for (_, value) in module.parameters().flattened() { total += value.size }
    return total
}

@Suite("RealPLKSR core")
struct RealPLKSRTests {

    @Test("Forward pass 64×64 → 256×256 NHWC at native 4×")
    func forwardShape() {
        withCPU {
            let model = RealPLKSR()
            let y = model(MLXArray.zeros([1, 64, 64, 3]))
            MLX.eval(y)
            #expect(y.shape == [1, 256, 256, 3])
        }
    }

    @Test("Exactly 7,389,680 parameters — the fixture config, pinned (any drift is a port bug)")
    func parameterCount() {
        withCPU {
            #expect(totalParameterCount(RealPLKSR()) == RealPLKSR.webphotoParameterCount)
        }
    }

    @Test("Born in inference mode (C14): the module graph reports training == false")
    func inferenceMode() {
        withCPU {
            let model = RealPLKSR()
            #expect(model.training == false)
            #expect(model.blocks.allSatisfy { $0.training == false })
        }
    }

    @Test("The vendored webphoto checkpoint is in the bundle and loads with full parameter coverage")
    func bundledWeightsLoad() throws {
        let url = try #require(RealPLKSR_Playback.Variant.webphoto.bundledWeightsURL)
        #expect(FileManager.default.fileExists(atPath: url.path))
        try withCPU {
            let model = RealPLKSR()
            try model.loadWeights(from: url)
            // Something non-trivial actually landed: the stem bias is no longer the zero init.
            let stemBias = model.stem.bias!
            MLX.eval(stemBias)
            #expect(MLX.abs(stemBias).sum().item(Float.self) > 0)
        }
    }

    @Test("A checkpoint that does not cover the tree is refused loudly (no silently partial load)")
    func partialCheckpointIsRefused() throws {
        let url = try #require(RealPLKSR_Playback.Variant.webphoto.bundledWeightsURL)
        try withCPU {
            var sd = try MLX.loadArrays(url: url)
            sd.removeValue(forKey: "feats.30.bias")
            let model = RealPLKSR()
            #expect(throws: RealPLKSRError.self) { try model.loadWeights(sd) }
        }
    }

    @Test("Layer parity vs the PyTorch oracle golden of this checkpoint (fp32, CPU): max|Δ| < 1e-2, cos > 0.9999")
    func goldenParity() throws {
        let goldURL = try #require(Bundle.module.url(forResource: "goldens_webphoto_s3_100k", withExtension: "safetensors"))
        let ckptURL = try #require(RealPLKSR_Playback.Variant.webphoto.bundledWeightsURL)
        try withCPU {
            let gold = try MLX.loadArrays(url: goldURL)
            let model = RealPLKSR()
            try model.loadWeights(from: ckptURL)
            // Goldens are NCHW (as torch produced them); the port is NHWC.
            let x = try #require(gold["input_nchw"]).transposed(0, 2, 3, 1)
            let g = try #require(gold["output"]).transposed(0, 2, 3, 1)
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == g.shape)
            let maxAbs = MLX.abs(g - y).max().item(Float.self)
            let dot = (g * y).sum().item(Float.self)
            let nrm = sqrt(g.square().sum().item(Float.self)) * sqrt(y.square().sum().item(Float.self))
            let cos = nrm > 0 ? Double(dot) / Double(nrm) : 0
            #expect(maxAbs < 1e-2, "max|Δ| \(maxAbs)")
            #expect(cos > 0.9999, "cos \(cos)")
        }
    }

    @Test("Playback tier constructs against the bundle with the documented defaults (256 / 32 / 4×)")
    func playbackTierDefaults() throws {
        let tier = try RealPLKSR_Playback()
        #expect(tier.scaleFactor == 4)
        #expect(tier.inputTileSize == 256 && tier.tileOverlap == 32)
        #expect(tier.outputResolution.width == 1024)
        #expect(tier.name == "realplksr-webphoto-x4")
        #expect(tier.wholeFrameMaxPixels == 1920 * 1080)
    }
}
