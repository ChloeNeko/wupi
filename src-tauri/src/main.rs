// In release builds, run on the Windows GUI subsystem so no console window is
// allocated. In debug builds we keep the console (the default) so log output is
// still visible during development.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Windows DLL search directory hook. MUST run before any code that touches
    // a CUDA / VC++ DLL — i.e. before `wupi_lib::run()` pulls in any module.
    //
    // Why: as of v0.3.7 the portable layout moved ALL shipped runtime DLLs
    // (CUDA cublas/cudart/etc. + VC++ msvcp140/vcomp140) out of the install
    // root into a sibling `bin/` subdirectory (AGENTS.md §8B). The 4 PE
    // static-import DLLs (cublas64_13, cudart64_13, msvcp140, vcomp140) are
    // compiled with /DELAYLOAD so they're resolved on FIRST CALL rather than
    // at process start; the other ~6 runtime-loaded CUDA DLLs are loaded by
    // ggml-cuda via LoadLibrary later. Both paths need the loader to search
    // `bin/`. `SetDllDirectoryW` adds one directory to the search path used
    // by LoadLibrary and the delay-load helper — that's the hook.
    //
    // Defensive: if `<exe_dir>\bin` doesn't exist (dev builds run from
    // src-tauri/target/debug), skip the call entirely. The default Windows
    // search path (exe dir + System32 + PATH) still applies, and on the dev
    // box the CUDA Toolkit's bin\x64 is on PATH so the imports resolve there.
    #[cfg(windows)]
    {
        if let Some(bin_path) = exe_bin_dir() {
            // SAFETY: SetDllDirectoryW is a kernel32 function exported since
            // XP. Its only side effect is mutating the per-process DLL
            // search path; passing a valid NUL-terminated wide string is the
            // documented contract. Failure is non-fatal: we ignore the
            // return value (the dev box's PATH already has CUDA, and on a
            // portable install the dir always exists).
            use std::os::windows::ffi::OsStrExt;
            unsafe {
                #[link(name = "kernel32")]
                extern "system" {
                    fn SetDllDirectoryW(lppathname: *const u16) -> i32;
                }
                let wide: Vec<u16> = bin_path
                    .as_os_str()
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect();
                let _ = SetDllDirectoryW(wide.as_ptr());
            }
        }
    }

    wupi_lib::run()
}

/// Resolve `<exe_dir>\bin` if (a) `current_exe()` succeeds and (b) that
/// `bin/` subdir exists on disk. Returns `None` for dev builds run from
/// `target/debug/` (no `bin/` there — the dev box's PATH has CUDA).
#[cfg(windows)]
fn exe_bin_dir() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let bin = dir.join("bin");
    if bin.is_dir() {
        Some(bin)
    } else {
        None
    }
}
