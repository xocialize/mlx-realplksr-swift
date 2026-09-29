import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import RealPLKSRMLX

/// Errors at the RealPLKSR package boundary.
public enum RealPLKSRPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
}

/// An MLXEngine `imageUpscale` package over **RealPLKSR 4×** trained on the provenance-clean
/// Commons-SR corpus (`4x_webphoto_realplksr`): the web-photo tier — a degradation prior that matches
/// what consumers actually feed an upscaler (photos that came off a website, a CDN or a messaging app:
/// resampling, lens blur, sensor noise, JPEG/WebP, chroma subsampling). Sibling of the Real-ESRGAN
/// package (clean/native sources, ~6× cheaper) behind the same capability.
///
/// A thin conformance wrapper over the standalone `RealPLKSRMLX` core; all model logic (the network,
/// 256² tiling with feathered seams via the shared tile driver, NHWC) lives there. The checkpoint is
/// vendored in the core bundle — `load()` involves **no download**.
///
/// Native scale is **4×**. A request's `scale` of `nil` or `≥ 4` runs at native 4×; a sub-native
/// `scale` (e.g. `2`) is honored by post-downsampling the native-4× result to `inputDim * scale`.
/// The response's `appliedScale` always reports what actually ran.
@InferenceActor
public final class RealPLKSRUpscalePackage: ModelPackage {
    public typealias Configuration = RealPLKSRConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weights are OURS — trained on Commons-SR (CC0 / PD / CC-BY, no SA, no NC, credits
            // in the model card's ATTRIBUTION.csv) with neosr (Apache-2.0); published Apache-2.0.
            // C8: the port is derived from neosr's realplksr_arch.py (Apache-2.0) → Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(sourceRepo: "xocialize/4x_webphoto_realplksr",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14). The 29.6 MB of weights are a rounding error; the working
                // set is the activations (64/128-channel feature maps through 28 blocks at the INPUT
                // resolution) plus the 4× output buffer. Measured with `realplksr-smoke`, release, M5 Max,
                // one process per size (MLX peak / floor, 2026-09-22):
                //
                //   input       output       mode          run      MLX peak   floor   process RSS
                //   128²        512²         whole-frame   15.3 s*    312 MB   29 MB   —
                //   512²        2048²        whole-frame    5.5 s    1185 MB   29 MB   —
                //   1024²       4096²        whole-frame   20.3 s    2474 MB   29 MB   647 MB
                //   1024²       4096²        tiled 256/32  26.9 s    2085 MB   29 MB   485 MB
                //   1920×1080   7680×4320    whole-frame   36.9 s    4863 MB   29 MB  1208 MB
                //   (* first run includes ~14 s of MLX compile; the tiled path trades 33% wall-clock for 16% peak)
                //
                // DECLARED on the in-app `phys_footprint` basis the governor compares against — MEASURED
                // 2026-09-29 (AB-T-0019) in the ForgeOptimizer app (Release; ForgeCore 0.27.0; engine 0.63.0 with
                // its 2 GiB MLX pool cap applied; M5 Max 128 GB, macOS 27.2), one fresh process per run through the
                // app's `UpscaleMemoryBench`: prepare → trim the MLX pool → floor → run under a 10 ms phys sampler.
                //
                //   input       output       mode          floor     activation (run peak − floor)    lifetime peak
                //   1080×1920   4320×7680    whole-frame   0.05 GB   6.47 · 6.47 · 6.63 · 6.92 GB     6.52–6.99 GB
                //   1080×1920   2160×3840    whole-frame   0.05 GB   6.47 GB (×2 = native ×4, downsampled)
                //   2160×3840   8640×15360   tiled 256/32  0.05 GB   3.60 GB                          3.69 GB
                //
                // The 1080p-input whole frame is the envelope's worst case: the default `wholeFrameMaxPixels` admits
                // exactly it, and every larger input tiles. Declared with the fleet's harness headroom (×1.2 + 256 MB,
                // as mlx-nerve-swift N6 declares):
                //   residentBytes        0.1 GB   (measured floor 0.05 GB — the app's baseline + 29.6 MB of weights)
                //   peakActivationBytes  6.92 GB × 1.2 + 0.256 GB ≈ 8.6 GB
                // This replaces the PROVISIONAL 1.05 + 11.5 GB of v0.1.0, derived from the MLX peak through
                // Real-ESRGAN's in-app ratios. That figure over-declared by ~1.8×: against a 16 GB Mac's budget
                // (≈ 12.7 GB — the engine's ~74% note) its 12.55 GB left ~0.15 GB of headroom, so the governor could
                // admit this package only after evicting everything else and refused it whenever anything else held
                // memory. 8.7 GB leaves ~4 GB. (v0.1.1's comment said "refused outright … budget ≈ 11.8 GB" — wrong.)
                // ⚠️ The basis assumes a MANAGED MLX pool. A host whose engine runs `.unmanaged` — or a `swift test`
                // process, where the engine's init-time cap write fails on the first MLX touch — keeps a growing pool
                // and reads far higher phys (24.4 GB after a HEART run, measured). That is the pool, not this package.
                // Hosts on small-memory machines can still lower `wholeFrameMaxPixels` (tiled measured ~half the peak).
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 100_000_000, peakActivationBytes: 8_600_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "realplksr-webphoto-upscale",
                    summary: "RealPLKSR 4x super-resolution trained on Commons-SR for web-photo degradations (blur, noise, JPEG/WebP, chroma subsampling); tile-based."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var upscaler: RealPLKSR_Playback?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard upscaler == nil else { return }
        // Weights are vendored in the core bundle; construction validates their presence and the
        // core lazy-loads tensors on the first upscale. `nil` keeps each core default.
        upscaler = try RealPLKSR_Playback(
            variant: configuration.variant.coreVariant,
            wholeFrameMaxPixels: configuration.wholeFrameMaxPixels ?? 1920 * 1080,
            inputTileSize: configuration.inputTileSize ?? 256,
            tileOverlap: configuration.tileOverlap ?? 32)
    }

    public func unload() async {
        upscaler = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        // Mid-run cadence: the shared tile driver checkpoints once per tile, rethrown unchanged.
        try Task.checkCancellation()
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let inPB = try Self.decodeToPixelBuffer(req.image)
        let inW = CVPixelBufferGetWidth(inPB), inH = CVPixelBufferGetHeight(inPB)
        let native = upscaler.scaleFactor

        let nativePB = try await upscaler.upscale(inPB)

        // Honor a requested `scale` below the native factor by post-downsampling the native-4× result
        // (BRIDGE-029). `nil`, the native factor, or any request ≥ native pass through at native scale.
        let outPB: CVPixelBuffer
        let appliedScale: Int
        if let s = req.scale, s > 0, s < native {
            outPB = try Self.resizePixelBuffer(nativePB, toWidth: inW * s, height: inH * s)
            appliedScale = s
        } else {
            outPB = nativePB
            appliedScale = native
        }

        let w = CVPixelBufferGetWidth(outPB), h = CVPixelBufferGetHeight(outPB)
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw RealPLKSRPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw RealPLKSRPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageUpscaleResponse(image: outImage, appliedScale: appliedScale)
    }

    // MARK: - Image codec (identical to the Real-ESRGAN sibling's — the canonical Image ↔ BGRA seam)

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RealPLKSRPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealPLKSRPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw RealPLKSRPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw RealPLKSRPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw RealPLKSRPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealPLKSRPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RealPLKSRPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// High-quality downsample of a 32BGRA `CVPixelBuffer` to `w`×`h` (a new 32BGRA buffer).
    nonisolated static func resizePixelBuffer(_ src: CVPixelBuffer, toWidth w: Int, height h: Int) throws -> CVPixelBuffer {
        guard w > 0, h > 0 else { throw RealPLKSRPackageError.imageEncodeFailed }
        let ci = CIImage(cvPixelBuffer: src)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { throw RealPLKSRPackageError.imageEncodeFailed }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealPLKSRPackageError.imageEncodeFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let outCtx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw RealPLKSRPackageError.imageEncodeFailed
        }
        outCtx.interpolationQuality = .high
        outCtx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension RealPLKSRUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RealPLKSRUpscalePackage.self)
    }
}
