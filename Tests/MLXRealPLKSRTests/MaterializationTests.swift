//
//  MaterializationTests.swift — the offline MAT gate (engine ≥ 0.24.0, contract 1.17 bundled
//  vocabulary) for a BUNDLED-WEIGHTS package: MAT-1 store-stampable, MAT-2 bundled source declared,
//  MAT-3 hygiene, MAT-4 the vendored checkpoint verified PRESENT on a fresh configuration, and the
//  end-to-end symptom through the REAL engine: a fresh registration must not read as needing a download.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeCore
import MLXServeConformance
@testable import MLXRealPLKSR

final class MaterializationTests: XCTestCase {

    func testFullMATGatePassesPerVariant() {
        for variant in RealPLKSRVariant.allCases {
            let report = MaterializationConformance.check(
                freshConfiguration: RealPLKSRConfiguration(variant: variant))
            XCTAssertTrue(report.passed, "\(variant):\n\(report.summary)")
        }
    }

    /// The checkpoint is actually IN the built bundle — a stripped resource must fail here, not at
    /// the first upscale.
    func testEveryVariantBundlesItsWeights() {
        for variant in RealPLKSRVariant.allCases {
            let url = variant.coreVariant.bundledWeightsURL
            XCTAssertNotNil(url, "\(variant) missing from bundle")
            if let url {
                XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), url.path)
                let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
                XCTAssertGreaterThan(size, 29_000_000, "the 29.6 MB checkpoint, not a stub")
            }
        }
    }

    func testFreshConfigurationPrewarmsFromBundle() {
        for variant in RealPLKSRVariant.allCases {
            let paths = RealPLKSRConfiguration(variant: variant).prewarmPaths
            XCTAssertFalse(paths.isEmpty, "\(variant): empty prewarm paths")
            XCTAssertTrue(paths.allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
                          "\(variant): prewarm path missing on a fresh configuration")
        }
    }

    /// Register is offline — no weights load until `prepare`/`run`.
    func testEngineNeedsDownloadIsFalseOnFreshRegistration() async throws {
        let engine = MLXServeEngine()
        _ = try await engine.register(
            RealPLKSRUpscalePackage.registration, configuration: RealPLKSRConfiguration())
        let needs = await engine.needsDownload(.imageUpscale)
        XCTAssertFalse(needs, "bundled weights present, nothing to download")
    }

    func testCodableRoundTrip() throws {
        let cfg = RealPLKSRConfiguration(variant: .webphoto, modelsRootDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(RealPLKSRConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.variant, .webphoto)
        XCTAssertNil(decoded.modelsRootDirectory)   // environment-specific, never encoded
    }
}
