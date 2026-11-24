//! Link settings for the Windows executable.

fn main() {
    // Threads created without a stack size take the executable's default.
    // Host graphics drivers compiling the title's large shaders on such
    // threads (vkd3d-proton's and the drivers' own, under Wine too) overflow
    // the linker's default of 1 MiB: reserve 16 MiB instead.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:16777216");
        // Agility SDK redist: d3d12.dll reads these exports and loads
        // D3D12Core.dll / d3d12SDKLayers.dll from the exe's d3d12sdk folder,
        // so the debug layer works without the Graphics Tools feature.
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath");
    }
}
