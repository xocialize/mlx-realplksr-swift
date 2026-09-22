import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
@testable import MLXRealPLKSR

/// Offline conformance checks — no Metal evaluation. Live upscaling is proven by `realplksr-smoke`
/// (the real engine path) and in the `MLXEngine Testing` app.
struct RealPLKSRUpscaleTests {

    @Test func manifestIsImageUpscaleAndPermissiveOnBothLayers() {
        let m = RealPLKSRUpscalePackage.manifest
        #expect(m.capabilities == [.imageUpscale])
        #expect(m.license.weightLicense == .apache2)      // C7 — our Commons-SR-trained weights
        #expect(m.license.portCodeLicense == .apache2)    // C8 — derived from neosr's arch (Apache-2.0)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func provenancePointsAtOurNamespace() {
        // Fleet policy (2026-08-03): every weight source names a namespace WE control.
        #expect(RealPLKSRUpscalePackage.manifest.provenance.sourceRepo.hasPrefix("xocialize/"))
    }

    @Test func manifestRequirements() {
        let r = RealPLKSRUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
    }

    /// Efficiency adoption (engine 1.14): the fp32 footprint declares the split (weights floor + the
    /// tile-bounded activation peak), so the engine reserves one shared transient.
    @Test func splitFootprintDeclared() {
        let fp = RealPLKSRUpscalePackage.manifest.requirements.footprints.first { $0.quant == .fp32 }
        #expect(fp?.peakActivationBytes ?? 0 > 0)
        #expect((fp?.peakActivationBytes ?? 0) > (fp?.residentBytes ?? .max))
    }

    /// `QuantConfigured` so the governor charges the declared fp32 footprint, not largest-that-fits.
    @Test func quantConfigured() {
        let cfg: any PackageConfiguration = RealPLKSRConfiguration(variant: .webphoto)
        #expect((cfg as? QuantConfigured)?.quant == .fp32)
    }

    @Test func surfaceIsTheCanonicalUpscaleDescriptor() {
        let s = RealPLKSRUpscalePackage.manifest.surfaces.first
        #expect(s?.capability == .imageUpscale)
        #expect(s?.parameters.first?.kind == .image)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = RealPLKSRUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.imageUpscale])
        let pkg = try reg.makePackage(RealPLKSRConfiguration())
        #expect(pkg is RealPLKSRUpscalePackage)
    }

    @Test func variantMapsToTheCoreCheckpoint() {
        #expect(RealPLKSRConfiguration().variant == .webphoto)
        #expect(RealPLKSRVariant.webphoto.coreVariant.rawValue == "webphoto")
        #expect(RealPLKSRVariant.webphoto.coreVariant.safetensorsName == "4x_webphoto_realplksr")
    }

    @Test func configurationCodableRoundTrips() throws {
        let c = RealPLKSRConfiguration(variant: .webphoto, wholeFrameMaxPixels: 123)
        let back = try JSONDecoder().decode(RealPLKSRConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.variant == .webphoto)
        #expect(back.wholeFrameMaxPixels == nil)   // host-specific knobs are never persisted
    }

    @Test func pngRoundTripsThroughPixelBuffer() throws {
        let png = try #require(Self.makePNG(width: 32, height: 32))
        let image = Image(format: .png, data: png, width: 32, height: 32)
        let pb = try RealPLKSRUpscalePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == 32)
        let back = try #require(RealPLKSRUpscalePackage.encodePNG(pb))
        #expect(back.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    @Test func rawBGRA8RoundTripsBitIdentical() throws {
        let w = 8, h = 4
        let bytes = Data((0..<(w * h * 4)).map { UInt8($0 % 256) })
        let image = Image.rawBGRA8(data: bytes, width: w, height: h)
        let pb = try RealPLKSRUpscalePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
        let back = try #require(RealPLKSRUpscalePackage.encodeRawBGRA8(pb))
        #expect(back.format == .rawBGRA8)
        #expect(back.width == w && back.height == h && back.bytesPerRow == nil)
        #expect(back.data == bytes)
    }

    @Test func rawBGRA8MissingDimensionsThrows() {
        let image = Image(format: .rawBGRA8, data: Data(count: 16))
        #expect(throws: RealPLKSRPackageError.self) {
            _ = try RealPLKSRUpscalePackage.decodeToPixelBuffer(image)
        }
    }

    /// The sub-native `scale` path yields the requested dimensions as a valid 32BGRA buffer.
    @Test func resizePixelBufferProducesRequestedDimensions() throws {
        let png = try #require(Self.makePNG(width: 64, height: 64))
        let nativePB = try RealPLKSRUpscalePackage.decodeToPixelBuffer(
            Image(format: .png, data: png, width: 64, height: 64))
        let scaled = try RealPLKSRUpscalePackage.resizePixelBuffer(nativePB, toWidth: 32, height: 32)
        #expect(CVPixelBufferGetWidth(scaled) == 32 && CVPixelBufferGetHeight(scaled) == 32)
        #expect(CVPixelBufferGetPixelFormatType(scaled) == kCVPixelFormatType_32BGRA)
    }

    static func makePNG(width: Int, height: Int) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0.6, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}
