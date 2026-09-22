// RealPLKSR_Playback.swift
//
// A `PlaybackTier` over RealPLKSR: model + the SHARED tile driver (RealESRGANMLX.MLXTileProcessor),
// the same pairing the shipped Real-ESRGAN tier uses — so the two backends are drop-in siblings
// behind one call and share the tile compositor's fidelity tests.
//
// Weights: Resources/4x_webphoto_realplksr.safetensors — the stage-3 EMA of the Commons-SR P3 run
// (xocialize, 2026-09-22; sha256 of the .pth 5d84789e…b52). Vendored, loaded lazily on first upscale.
//
// Tile geometry: 256² input tiles with a 32 px feathered overlap (the plan's 1:8 ratio). The
// network's receptive field is effectively global (28 blocks × a 17×17 partial kernel), so the
// overlap buys seam AGREEMENT rather than covering a receptive field; measured throughput is flat
// at ~121–125 ms per Mpx of output across 128²/256² tiles on M5 Max (AB-R-0177), so the larger tile
// costs nothing and halves the seam count. Whole-frame path ≤ `wholeFrameMaxPixels` (default
// 1920×1080, gated by the HOST per machine — same policy as the sibling).

import CoreVideo
import Foundation
import MLX
import MLXNN
import RealESRGANMLX

public final class RealPLKSR_Playback: PlaybackTier, @unchecked Sendable {

    /// Which vendored checkpoint to load. One today; the enum keeps siblings honest if a second
    /// RealPLKSR checkpoint ships (own resource, own name, own tier id).
    public enum Variant: String, Sendable, CaseIterable {
        /// 4x_webphoto_realplksr — web-photo degradation prior (blur · noise · JPEG/WebP · chroma
        /// subsampling), Commons-SR trained. Default.
        case webphoto

        /// Resource stem (no extension) of the vendored safetensors file.
        public var safetensorsName: String {
            switch self {
            case .webphoto: return "4x_webphoto_realplksr"
            }
        }

        /// The vendored checkpoint's URL inside the core's resource bundle (nil = stripped bundle).
        public var bundledWeightsURL: URL? {
            Bundle.module.url(forResource: safetensorsName, withExtension: "safetensors")
        }

        var tierName: String {
            switch self {
            case .webphoto: return "realplksr-webphoto-x4"
            }
        }
    }

    // MARK: - PlaybackTier surface

    public let name: String
    public let scaleFactor: Int
    public let inputTileSize: Int
    public let tileOverlap: Int
    public var inputResolution: (width: Int, height: Int) { (inputTileSize, inputTileSize) }
    public var outputResolution: (width: Int, height: Int) {
        (inputTileSize * scaleFactor, inputTileSize * scaleFactor)
    }
    public let variant: Variant

    /// Input-pixel ceiling for the single-pass (no tiles, no seams) path. See the file header.
    public let wholeFrameMaxPixels: Int

    // MARK: - Internals

    private let model: RealPLKSR
    private let tileProcessor: MLXTileProcessor
    private let weightsURL: URL
    private let loadLock = NSLock()
    private var weightsLoaded = false
    private var compiledForward: (@Sendable (MLXArray) -> MLXArray)?

    // MARK: - Init

    /// Construction validates the vendored checkpoint is PRESENT (a stripped bundle fails here, not
    /// at the first upscale); the tensors load lazily on the first `upscale`.
    public init(variant: Variant = .webphoto, wholeFrameMaxPixels: Int = 1920 * 1080,
                inputTileSize: Int = 256, tileOverlap: Int = 32) throws {
        self.variant = variant
        self.name = variant.tierName
        self.scaleFactor = 4
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap

        guard let url = variant.bundledWeightsURL else {
            throw PlaybackTierError.weightsNotFound(variant.safetensorsName)
        }
        self.weightsURL = url
        self.model = RealPLKSR()
        self.tileProcessor = MLXTileProcessor(tileSize: inputTileSize, overlap: tileOverlap, scale: scaleFactor)
    }

    // MARK: - PlaybackTier impl

    public func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer {
        let run = try ensureReady()
        do {
            return try tileProcessor.processAdaptive(buffer, wholeFrameMaxPixels: wholeFrameMaxPixels) { tile in
                let y = run(tile)
                MLX.eval(y)
                return y
            }
        } catch let err as PlaybackTierError {
            throw err
        } catch is CancellationError {
            // CAN-2: never launder a cancellation — the per-tile checkpoint's CancellationError must
            // reach the engine unchanged.
            throw CancellationError()
        } catch {
            throw PlaybackTierError.inferenceError(String(describing: error))
        }
    }

    // MARK: - Weights + compile

    /// Load weights (once) and build the compiled forward (once) — the same strategy as the
    /// Real-ESRGAN tier so an A/B between the two runs through identical machinery.
    private func ensureReady() throws -> @Sendable (MLXArray) -> MLXArray {
        loadLock.lock()
        defer { loadLock.unlock() }
        if let f = compiledForward { return f }
        if !weightsLoaded {
            do {
                try model.loadWeights(from: weightsURL)
                weightsLoaded = true
            } catch {
                throw PlaybackTierError.modelLoadFailed(String(describing: error))
            }
        }
        let m = model
        let f = compile { x in m(x) }
        compiledForward = f
        return f
    }
}
