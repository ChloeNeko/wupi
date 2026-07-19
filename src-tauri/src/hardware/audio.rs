//! Core Audio integration (Win32 WASAPI via the `windows` crate).
//!
//! Exposes:
//! - `audio_get_state`: current master volume (0-100), mute, default output.
//! - `audio_set_volume`: set master volume scalar on the default render endpoint.
//! - `audio_list_outputs`: active render endpoints with friendly names + default flag.
//! - `audio_set_default_output`: switch the default render endpoint via
//!   `IPolicyConfig::SetDefaultEndpoint`. That interface isn't in the official
//!   SDK headers (it's the de-facto approach used by EarTrumpet / SoundSwitch);
//!   stable in practice across Win10/11 but not API-stability-guaranteed by
//!   Microsoft. Declared here via the `#[interface]` macro with the known GUID.
//!
//! All COM work runs on a dedicated blocking thread with COINIT_MULTITHREADED
//! so it never tangles with Tauri's async apartment.

use serde::{Deserialize, Serialize};
use tauri::State;
use windows::core::{GUID, PCWSTR, PROPVARIANT};
use windows::Win32::Foundation::BOOL;
use windows::Win32::Media::Audio::{
    eCommunications, eConsole, eMultimedia, eRender, ERole, DEVICE_STATE_ACTIVE,
    IMMDevice, IMMDeviceCollection, IMMDeviceEnumerator, MMDeviceEnumerator,
};
use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED, STGM_READ,
};
use windows::Win32::UI::Shell::PropertiesSystem::{IPropertyStore, PROPERTYKEY};

// PKEY_Device_FriendlyName: the canonical friendly-name property key.
// {a45c254e-df1c-4efd-8020-67d146a850e0}, 14
const PKEY_DEVICE_FRIENDLYNAME: PROPERTYKEY = PROPERTYKEY {
    fmtid: GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0),
    pid: 14,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct AudioOutput {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Serialize, Deserialize)]
pub struct AudioState {
    pub volume: u8,            // 0-100
    pub muted: bool,
    pub default_output: Option<AudioOutput>,
}

// GUIDs for switching the default audio endpoint:
//   CLSID_CPolicyConfigClient  {870AF99C-171D-4F9E-AF0D-E63D-FAF4C664}
//   IID_IPolicyConfig          {F8679F50-850A-41CF-9C72-430F290290C8}
// `#[interface]` generates the vtable + wrapper; only SetDefaultEndpoint is
// declared because it's the sole method we call.
#[windows::core::interface("F8679F50-850A-41CF-9C72-430F290290C8")]
unsafe trait IPolicyConfig: windows::core::IUnknown {
    fn set_default_endpoint(&self, device_id: PCWSTR, role: ERole) -> windows::core::HRESULT;
}
const CLSID_CPOLICY_CONFIG_CLIENT: GUID =
    GUID::from_u128(0x870af99c_171d_4f9e_af0d_e63dfaf4c664);


/// Run a COM-using closure on a fresh thread with its own MTA init. COM
/// apartments don't mix with async runtimes, so each call gets a clean
/// thread: CoInitializeEx on enter, CoUninitialize on exit.
fn with_com<T, F>(f: F) -> windows::core::Result<T>
where
    F: FnOnce() -> windows::core::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let init = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        let init_ok = init.is_ok();
        // RPC_E_CHANGED_MODE: this thread was already inited in a different
        // apartment: rare for a fresh thread, but tolerate by still running.
        let result = if init_ok || init == windows::Win32::Foundation::RPC_E_CHANGED_MODE {
            f()
        } else {
            Err(windows::core::Error::from(init))
        };
        if init_ok {
            unsafe { CoUninitialize() };
        }
        let _ = tx.send(result);
    });
    rx.recv()
        .map_err(|e| windows::core::Error::new(windows::core::HRESULT(-1), e.to_string()))?
}

fn enumerator() -> windows::core::Result<IMMDeviceEnumerator> {
    // CoCreateInstance has signature <P0, T>(rclsid, punkouter: P0, ctx) in 0.58.
    unsafe { CoCreateInstance::<_, IMMDeviceEnumerator>(&MMDeviceEnumerator, None, CLSCTX_ALL) }
}

/// Read the friendly name from a device's property store. The value is a
/// VT_LPWSTR (wide string) inside a PROPVARIANT.
fn device_name(dev: &IMMDevice) -> windows::core::Result<String> {
    unsafe {
        let store: IPropertyStore = dev.OpenPropertyStore(STGM_READ)?;
        let prop: PROPVARIANT = store.GetValue(&PKEY_DEVICE_FRIENDLYNAME as *const PROPERTYKEY)?;
        // PROPVARIANT layout in windows 0.58: vt is the tag; the union holds
        // the pointer for VT_LPWSTR. Read it via the documented accessor path.
        // We use the fact that PROPVARIANT is a transparent-ish wrapper; the
        // safe way is `from_abi`-style, but GetValue already returned it by
        // value. Fall back to the raw field names this crate exposes.
        //
        // The crate exposes `.vt()`-style accessors only on newer versions;
        // for 0.58 we read the wide string pointer out of the union directly.
        let raw: &PROPVARIANT = &prop;
        // SAFETY: we only read after checking vt == VT_LPWSTR (31). The union
        // field holding a wide-string pointer is at a fixed offset; we access
        // it through the crate's Anonymous layout. If the layout name isn't
        // stable, we instead use the well-known fact that a PROPVARIANT for a
        // string is `{ vt: 31, _reserved: ..., pwsz_val: *const u16 }`.
        let vt = std::ptr::addr_of!(*raw).cast::<u16>();
        if std::ptr::read_unaligned(vt) != 31 {
            return Ok(String::from("(unnamed)"));
        }
        // The string pointer is at offset 8 (after vt:u16 + reserved:u16 +
        // padding) on 64-bit: PROPVARIANT is 16 bytes for the scalar forms,
        // but VT_LPWSTR places the pointer in the union. Access via the
        // crate's field if present; otherwise compute it.
        let ptr_field: *const *const u16 = (std::ptr::addr_of!(*raw) as *const u8).add(8).cast();
        let ptr = std::ptr::read_unaligned(ptr_field);
        if ptr.is_null() {
            return Ok(String::from("(unnamed)"));
        }
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        Ok(String::from_utf16_lossy(slice))
    }
}

/// Device id as a String. IMMDevice::GetId returns a PWSTR we must copy out
/// (and it's CoTaskMemAlloc'd: the crate frees it on drop of the PWSTR).
fn device_id(dev: &IMMDevice) -> windows::core::Result<String> {
    unsafe {
        let pwstr = dev.GetId()?;
        // PWSTR → PCWSTR → to_string. The PWSTR owns the allocation in this crate.
        let s = PCWSTR(pwstr.0).to_string()?;
        Ok(s)
    }
}


pub struct AudioRegistry;
pub fn new_registry() -> AudioRegistry {
    AudioRegistry
}

#[tauri::command]
pub fn audio_get_state(_reg: State<'_, AudioRegistry>) -> Result<AudioState, String> {
    with_com(|| {
        let en = enumerator()?;
        let dev = unsafe { en.GetDefaultAudioEndpoint(eRender, eConsole)? };
        let vol_iface: IAudioEndpointVolume = unsafe { dev.Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None)? };
        let scalar = unsafe { vol_iface.GetMasterVolumeLevelScalar()? };
        // GetMute resolves to the BOOL-returning overload; .as_bool() handles either.
        let muted_val = unsafe { vol_iface.GetMute()? };
        let default_output = Some(AudioOutput {
            id: device_id(&dev).unwrap_or_default(),
            name: device_name(&dev).unwrap_or_else(|_| "(audio device)".into()),
            is_default: true,
        });
        let muted: bool = muted_val.as_bool();
        Ok(AudioState {
            volume: (scalar.clamp(0.0, 1.0) * 100.0).round() as u8,
            muted,
            default_output,
        })
    })
    .map_err(|e| format!("audio_get_state: {e}"))
}

#[tauri::command]
pub fn audio_set_volume(volume: u8, _reg: State<'_, AudioRegistry>) -> Result<(), String> {
    let scalar = (volume as f32).clamp(0.0, 100.0) / 100.0;
    // `move` so the closure owns scalar + volume (both Copy, but the closure
    // still needs 'static bounds for the worker thread).
    with_com(move || {
        let en = enumerator()?;
        let dev = unsafe { en.GetDefaultAudioEndpoint(eRender, eConsole)? };
        let vol_iface: IAudioEndpointVolume =
            unsafe { dev.Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None)? };
        unsafe { vol_iface.SetMasterVolumeLevelScalar(scalar, std::ptr::null())? };
        // If the user nudged volume up from 0, also unmute (matches Windows behavior).
        if volume > 0 {
            // SetMute takes Param<BOOL>; pass FALSE to unmute.
            unsafe { vol_iface.SetMute(BOOL(0), std::ptr::null())? };
        }
        Ok(())
    })
    .map_err(|e| format!("audio_set_volume: {e}"))
}

#[tauri::command]
pub fn audio_list_outputs(_reg: State<'_, AudioRegistry>) -> Result<Vec<AudioOutput>, String> {
    with_com(|| {
        let en = enumerator()?;
        let default_id = unsafe {
            match en.GetDefaultAudioEndpoint(eRender, eConsole) {
                Ok(d) => device_id(&d).unwrap_or_default(),
                Err(_) => String::new(),
            }
        };
        let coll: IMMDeviceCollection =
            unsafe { en.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)? };
        let count = unsafe { coll.GetCount()? };
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let dev = unsafe { coll.Item(i)? };
            let id = device_id(&dev).unwrap_or_default();
            let name = device_name(&dev).unwrap_or_else(|_| "(audio device)".into());
            out.push(AudioOutput {
                is_default: id == default_id,
                id,
                name,
            });
        }
        Ok(out)
    })
    .map_err(|e| format!("audio_list_outputs: {e}"))
}

#[tauri::command]
pub fn audio_set_default_output(id: String, _reg: State<'_, AudioRegistry>) -> Result<(), String> {
    // The HSTRING must be created inside the worker thread (it isn't Send);
    // `move` so the closure takes ownership of `id`.
    with_com(move || {
        let hstr = windows::core::HSTRING::from(&id);
        let pcw = PCWSTR(hstr.as_ptr());
        let policy: IPolicyConfig =
            unsafe { CoCreateInstance::<_, IPolicyConfig>(&CLSID_CPOLICY_CONFIG_CLIENT, None, CLSCTX_ALL)? };
        // Set for console + multimedia + communications roles (covers all apps).
        for role in [eConsole, eMultimedia, eCommunications] {
            let hr = unsafe { policy.set_default_endpoint(pcw, role) };
            hr.ok()?;
        }
        Ok(())
    })
    .map_err(|e| format!("audio_set_default_output: {e}"))
}
