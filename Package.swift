// swift-tools-version: 6.2
import PackageDescription

// mlx-realplksr-swift — RealPLKSR 4× super-resolution for MLXEngine, trained by xocialize on the
// provenance-clean Commons-SR corpus (WEBPHOTO-SR-PLAN P3, finished 2026-09-22). ONE repo, TWO products,
// the mlx-realesrgan-swift shape:
//   • RealPLKSRMLX — engine-agnostic Swift/MLX core: the network (isomorphic to neosr's
//     realplksr_arch.py, Apache-2.0) + a `PlaybackTier` over the SHARED tile driver from RealESRGANMLX
//   • MLXRealPLKSR — the MLXEngine `imageUpscale` ModelPackage over that core
// Vendored weights (29.6 MB fp32 safetensors, the stage-3 EMA `net_g_100000.pth`) — no download.
// Materialization posture: BUNDLED-WEIGHTS (ModelStorable + BundledWeightSourcing; the MAT gate verifies
// the checkpoint PRESENT on a fresh machine; needsDownload reads false). WeightSourcing deliberately NOT
// declared. Cancellation posture (CAN gate): entry checkpoint first act of run(); the shared tile driver
// checkpoints once per tile. Licences: weights Apache-2.0 (ours; corpus credits in the model card),
// port code Apache-2.0 (derived from neosr's arch file). See README.md.
let package = Package(
    name: "mlx-realplksr-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "RealPLKSRMLX", targets: ["RealPLKSRMLX"]),
        .library(name: "MLXRealPLKSR", targets: ["MLXRealPLKSR"]),
        .executable(name: "realplksr-smoke", targets: ["RealPLKSRSmoke"]),  // drive the package + measure the split footprint
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.56.0"),
        .package(url: "https://github.com/xocialize/mlx-realesrgan-swift", from: "0.7.0"),  // MLXTileProcessor + PlaybackTier
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        // Engine-agnostic core — NO MLXToolKit dep. Reuses the shipped tile driver rather than forking it.
        .target(
            name: "RealPLKSRMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "RealESRGANMLX", package: "mlx-realesrgan-swift"),
            ],
            // Per-file .copy so the bundle layout is flat (forge ADR-0011).
            resources: [
                .copy("Resources/4x_webphoto_realplksr.safetensors"),
            ]
        ),
        // MLXEngine `imageUpscale` wrapper over the local core.
        .target(
            name: "MLXRealPLKSR",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "RealPLKSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            // The core's playback tier isn't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning (same posture as the sibling).
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "RealPLKSRMLXTests",
            dependencies: [
                "RealPLKSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources/goldens_webphoto_s3_100k.safetensors"),
            ]
        ),
        .testTarget(
            name: "MLXRealPLKSRTests",
            dependencies: [
                "MLXRealPLKSR",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
        // Drives RealPLKSRUpscalePackage through the REAL MLXServeEngine (register → run) and reports
        // the split footprint (resident floor / tiled activation peak) per the memory harness.
        .executableTarget(
            name: "RealPLKSRSmoke",
            dependencies: [
                "MLXRealPLKSR",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
