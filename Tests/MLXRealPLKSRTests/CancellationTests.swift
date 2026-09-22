//
//  CancellationTests.swift — the offline CAN gate (engine ≥ 0.27.0): CAN-1/CAN-2 pre-cancelled run()
//  propagation + classification, and CAN-3 the checkpoint-cadence declaration of record.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeConformance
@testable import MLXRealPLKSR

final class CancellationTests: XCTestCase {

    func testCANGatePreCancelledRun() async {
        // Construction is cheap (C13) and the entry checkpoint throws before validation, decode, or
        // the (bundled) weights are touched, so this is offline-safe.
        let package = RealPLKSRUpscalePackage(configuration: RealPLKSRConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        // peakActivationBytes ≥ 2 GB ⇒ long-run implied; the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: RealPLKSRUpscalePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: RealPLKSRUpscalePackage.manifest,
            posture: .cadence([
                // The shared tile driver checks Task.checkCancellation once per 256² tile
                // (MLXTileProcessor.process, top of the tile loop). "chunk" = one tile. The
                // ≤ wholeFrameMaxPixels whole-frame path is a single MLX eval; there the entry
                // checkpoint is the only seam.
                .init(phase: .upsample, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
