import Foundation
import MLXToolKit
import RealPLKSRMLX

/// Which vendored RealPLKSR checkpoint to load (bundled in the core — no download).
public enum RealPLKSRVariant: String, Codable, Sendable, CaseIterable {
    /// 4x_webphoto_realplksr — the Commons-SR web-photo prior. Default (and only, today).
    case webphoto

    var coreVariant: RealPLKSR_Playback.Variant {
        switch self {
        case .webphoto: return .webphoto
        }
    }
}

/// Init-time configuration for `RealPLKSRUpscalePackage` (C9). The checkpoint is **vendored in the
/// core package bundle** — 29.6 MB ships with the code, so a fresh machine never downloads anything
/// and `load()` never touches the network.
///
/// Materialization posture (engine ≥ 0.24.0, contract 1.17): `ModelStorable` so the engine can stamp
/// its store root (MAT-1), and `BundledWeightSourcing` declaring the selected variant's checkpoint as
/// a bundled source — the MAT gate verifies it PRESENT on a fresh machine and `needsDownload` reads
/// `false` on a fresh install. `WeightSourcing` stays undeclared by design (no fresh-machine network
/// source; declaring one would either lie about the missing set or force a download of weights already
/// in the binary). `WeightPrewarming` pages the checkpoint in before `load()`.
public struct RealPLKSRConfiguration: PackageConfiguration, ModelStorable {
    public var variant: RealPLKSRVariant
    /// Engine-chosen models root. Stamped by the engine from its `ModelStore`; unused today (weights
    /// are bundled) but keeps the config store-addressable (MAT-1). Environment-specific → excluded
    /// from `Codable`.
    public var modelsRootDirectory: URL?

    /// Whole-frame fast-path ceiling in INPUT pixels, forwarded to the core.
    ///
    /// `nil` keeps the core's default (`1920*1080`), which prefers the WHOLE-FRAME quality path — no
    /// tiles, no seams. The package deliberately does not cap this: the host gates it per machine.
    /// ⚠️ This network is ~6× the arithmetic of Real-ESRGAN's and builds the same 4× intermediate, so
    /// budget the whole-frame path from the measured footprint in the manifest, not from the sibling's.
    public var wholeFrameMaxPixels: Int?

    /// Tile geometry for the tiled path; `nil` keeps the core defaults (256 / 32).
    public var inputTileSize: Int?
    public var tileOverlap: Int?

    public init(variant: RealPLKSRVariant = .webphoto, modelsRootDirectory: URL? = nil,
                wholeFrameMaxPixels: Int? = nil,
                inputTileSize: Int? = nil, tileOverlap: Int? = nil) {
        self.variant = variant
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant
    }
}

/// The bundled-weights declaration (contract 1.17): the selected variant's vendored checkpoint,
/// verified PRESENT by the MAT gate and read by the engine's `needsDownload` as "nothing to fetch".
extension RealPLKSRConfiguration: BundledWeightSourcing {
    public var bundledWeightSources: [BundledWeightSource] {
        [BundledWeightSource(role: "checkpoint", url: variant.coreVariant.bundledWeightsURL)]
    }
}

/// Cold-start page-in: the prewarmer pages the 29.6 MB checkpoint before first load.
extension RealPLKSRConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        [variant.coreVariant.bundledWeightsURL].compactMap { $0 }
    }
}

/// `QuantConfigured` (engine 1.14): the vendored checkpoint runs at fp32 (the single declared
/// footprint quant), so the governor charges the matching `QuantFootprint(.fp32, …)`.
extension RealPLKSRConfiguration: QuantConfigured {
    public var quant: Quant { .fp32 }
}
