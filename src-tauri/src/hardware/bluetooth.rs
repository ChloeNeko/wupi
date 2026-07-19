//! Bluetooth integration (WinRT via the `windows` crate).
//!
//! Exposes:
//! - `bluetooth_get_state`: whether the Bluetooth radio is on.
//! - `bluetooth_toggle_radio`: turn the radio on/off (WinRT Radio.SetStateAsync).
//! - `bluetooth_list_devices`: paired/known devices via DeviceInformation.
//! - `bluetooth_pair`: pair a device; Windows owns the PIN/confirmation UI.
//!
//! WinRT is apartment-aware, so each command runs on a worker thread with an
//! MTA init (CoInitializeEx), and blocks on the IAsyncOperation via `.get()`
//! (the crate's built-in blocking wait: simpler than wiring the Future impl
//! through tokio). Tauri's own runtime is left untouched.

use serde::{Deserialize, Serialize};
use windows::core::HSTRING;
use windows::Devices::Enumeration::{
    DeviceInformation, DevicePairingResultStatus,
};
use windows::Devices::Radios::{Radio, RadioKind, RadioState};

#[derive(Serialize, Deserialize)]
pub struct BluetoothState {
    pub radio_on: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BluetoothDevice {
    pub id: String,
    pub name: String,
    pub connected: bool,
    pub paired: bool,
}

/// Run a WinRT closure on a dedicated worker thread with MTA init. WinRT
/// requires an apartment; we isolate it from Tauri's runtime. Operations block
/// via the crate's `IAsyncOperation::get()` inside the closure.
fn with_winrt<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let init = unsafe {
            windows::Win32::System::Com::CoInitializeEx(
                None,
                windows::Win32::System::Com::COINIT_MULTITHREADED,
            )
        };
        let init_ok = init.is_ok();
        let result = f();
        if init_ok {
            unsafe { windows::Win32::System::Com::CoUninitialize() };
        }
        let _ = tx.send(result);
    });
    rx.recv().map_err(|e| format!("winrt worker disconnected: {e}"))?
}


#[tauri::command]
pub fn bluetooth_get_state() -> Result<BluetoothState, String> {
    with_winrt(|| {
        let op = Radio::GetRadiosAsync().map_err(|e| format!("GetRadiosAsync: {e}"))?;
        let radios = op.get().map_err(|e| format!("GetRadiosAsync.get: {e}"))?;
        for radio in radios {
            let kind = radio.Kind().map_err(|e| format!("Radio.Kind: {e}"))?;
            if kind == RadioKind::Bluetooth {
                let state = radio.State().map_err(|e| format!("Radio.State: {e}"))?;
                return Ok(BluetoothState {
                    radio_on: state == RadioState::On,
                });
            }
        }
        Ok(BluetoothState { radio_on: false })
    })
}

#[tauri::command]
pub fn bluetooth_toggle_radio(on: bool) -> Result<(), String> {
    with_winrt(move || {
        let op = Radio::GetRadiosAsync().map_err(|e| format!("GetRadiosAsync: {e}"))?;
        let radios = op.get().map_err(|e| format!("GetRadiosAsync.get: {e}"))?;
        // RadioState::On == 1, Off == 0. The enum has `from(u32)`.
        let target = if on { RadioState::On } else { RadioState::Off };
        let mut found = false;
        for radio in radios {
            let kind = radio.Kind().map_err(|e| format!("Radio.Kind: {e}"))?;
            if kind == RadioKind::Bluetooth {
                found = true;
                let sop = radio
                    .SetStateAsync(target)
                    .map_err(|e| format!("SetStateAsync: {e}"))?;
                sop.get().map_err(|e| format!("SetStateAsync.get: {e}"))?;
            }
        }
        if !found {
            return Err("no Bluetooth radio found".into());
        }
        Ok(())
    })
}

#[tauri::command]
pub fn bluetooth_list_devices() -> Result<Vec<BluetoothDevice>, String> {
    with_winrt(|| {
        // DeviceClass has no Bluetooth variant, so use the canonical Bluetooth
        // AQS selector via FindAllAsyncAqsFilter. The GUID
        // {e0cbf06c-cd8b-4647-bb8a-263b43f0f974} is the Bluetooth device
        // interface class: this returns ONLY Bluetooth devices, not the whole
        // PnP tree (which is why the panel previously showed "a million
        // devices": FindAllAsync returns every HID/USB/audio/etc. device).
        let aqs = HSTRING::from(
            "System.Devices.Aep.ProtocolId:=\"{e0cbf06c-cd8b-4647-bb8a-263b43f0f974}\"",
        );
        let op = DeviceInformation::FindAllAsyncAqsFilter(&aqs)
            .map_err(|e| format!("FindAllAsyncAqsFilter: {e}"))?;
        let coll = op.get().map_err(|e| format!("FindAllAsync.get: {e}"))?;
        let mut out = Vec::new();
        let size = coll.Size().map_err(|e| format!("Size: {e}"))?;
        for i in 0..size {
            let dev = coll.GetAt(i).map_err(|e| format!("GetAt: {e}"))?;
            let name = dev.Name().map_err(|e| format!("Name: {e}"))?.to_string();
            let id = dev.Id().map_err(|e| format!("Id: {e}"))?.to_string();
            let pairing = dev.Pairing().map_err(|e| format!("Pairing: {e}"))?;
            let paired = pairing.IsPaired().map_err(|e| format!("IsPaired: {e}"))?;
            // "My Devices" = paired only. Unpaired entries are noise here;
            // pairing new devices is a separate flow (bluetooth_pair).
            if !paired {
                continue;
            }
            // Skip empty-named placeholder entries.
            if name.trim().is_empty() {
                continue;
            }
            out.push(BluetoothDevice {
                id,
                name,
                connected: false, // needs extra property queries; deferred
                paired,
            });
        }
        Ok(out)
    })
}

/// Discover in-range, unpaired Bluetooth devices (for the "Add Device" flow).
/// Same AQS filter as bluetooth_list_devices but returns UNPAIRED entries -
/// these are devices actively advertising and available to pair. Windows
/// handles the actual PIN/confirmation handshake when bluetooth_pair is called.
#[tauri::command]
pub fn bluetooth_discover() -> Result<Vec<BluetoothDevice>, String> {
    with_winrt(|| {
        let aqs = HSTRING::from(
            "System.Devices.Aep.ProtocolId:=\"{e0cbf06c-cd8b-4647-bb8a-263b43f0f974}\"",
        );
        let op = DeviceInformation::FindAllAsyncAqsFilter(&aqs)
            .map_err(|e| format!("FindAllAsyncAqsFilter: {e}"))?;
        let coll = op.get().map_err(|e| format!("FindAllAsync.get: {e}"))?;
        let mut out = Vec::new();
        let size = coll.Size().map_err(|e| format!("Size: {e}"))?;
        for i in 0..size {
            let dev = coll.GetAt(i).map_err(|e| format!("GetAt: {e}"))?;
            let name = dev.Name().map_err(|e| format!("Name: {e}"))?.to_string();
            let id = dev.Id().map_err(|e| format!("Id: {e}"))?.to_string();
            let pairing = dev.Pairing().map_err(|e| format!("Pairing: {e}"))?;
            let paired = pairing.IsPaired().map_err(|e| format!("IsPaired: {e}"))?;
            // Discover = unpaired + named (filter empty placeholder names).
            if paired || name.trim().is_empty() {
                continue;
            }
            out.push(BluetoothDevice {
                id,
                name,
                connected: false,
                paired,
            });
        }
        Ok(out)
    })
}

#[tauri::command]
pub fn bluetooth_pair(device_id: String) -> Result<bool, String> {
    with_winrt(move || {
        let id_h = HSTRING::from(&device_id);
        // CreateFromIdAsync returns an IAsyncOperation<DeviceInformation>.
        let op = DeviceInformation::CreateFromIdAsync(&id_h)
            .map_err(|e| format!("CreateFromIdAsync: {e}"))?;
        let dev = op.get().map_err(|e| format!("CreateFromIdAsync.get: {e}"))?;
        let pairing = dev.Pairing().map_err(|e| format!("Pairing: {e}"))?;
        // PairAsync surfaces the native Windows PIN/confirmation dialog when
        // the device requires it: we don't reimplement the handshake.
        let pop = pairing
            .PairAsync()
            .map_err(|e| format!("PairAsync: {e}"))?;
        let result = pop.get().map_err(|e| format!("PairAsync.get: {e}"))?;
        let status = result.Status().map_err(|e| format!("Pair Status: {e}"))?;
        Ok(status == DevicePairingResultStatus::Paired)
    })
}
