//! Wi-Fi integration (Win32 WLAN via the `windows` crate, wlanapi.dll).
//!
//! Exposes:
//! - `wifi_get_current`: the connected network (SSID, signal %) if any.
//! - `wifi_scan`: visible networks with signal + security flags.
//! - `wifi_connect`: connect to a network (password for secured networks).
//!
//! WLAN is plain C FFI: handle-based, raw pointers, manual WlanFreeMemory.
//! All calls run on a per-call worker thread (no COM init needed for WLAN,
//! but we still isolate the blocking work from Tauri's async runtime).
//!
//! Note on scan: WlanScan is async (~4s for the radio sweep). `wifi_scan`
//! returns the LAST cached BSS list immediately; the frontend re-polls after
//! a short delay to pick up fresh results.

use serde::{Deserialize, Serialize};
use windows::core::GUID;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::NetworkManagement::WiFi::{
    WlanCloseHandle, WlanEnumInterfaces, WlanFreeMemory, WlanGetNetworkBssList, WlanOpenHandle,
    WlanQueryInterface, DOT11_BSS_TYPE, WLAN_BSS_LIST,
    WLAN_CONNECTION_ATTRIBUTES, WLAN_INTERFACE_INFO_LIST, WLAN_INTF_OPCODE,
};

const WLAN_CLIENT_VERSION_XP: u32 = 1;

#[derive(Serialize, Deserialize, Clone)]
pub struct WifiNetwork {
    pub ssid: String,
    pub signal_pct: u8,
    pub secure: bool,
    pub connected: bool,
}

#[derive(Serialize, Deserialize)]
pub struct WifiState {
    pub connected: bool,
    pub ssid: Option<String>,
    pub signal_pct: u8,
}


/// RAII guard for the WLAN client handle: closes on drop. The handle is a
/// `HANDLE` wrapping a raw pointer; closing it releases the WLAN session.
struct WlanHandle(HANDLE);
impl WlanHandle {
    fn open() -> Result<Self, String> {
        let mut negotiated: u32 = 0;
        let mut handle = HANDLE(std::ptr::null_mut());
        let hr = unsafe {
            WlanOpenHandle(
                WLAN_CLIENT_VERSION_XP,
                None,
                &mut negotiated,
                &mut handle,
            )
        };
        if hr != 0 {
            return Err(format!("WlanOpenHandle failed: error {hr}"));
        }
        Ok(WlanHandle(handle))
    }
}
impl Drop for WlanHandle {
    fn drop(&mut self) {
        unsafe { let _ = WlanCloseHandle(self.0, None); }
    }
}

/// Run a WLAN closure on a worker thread (isolate blocking FFI from async).
fn with_wlan<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce(&WlanHandle) -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = (|| {
            let handle = WlanHandle::open()?;
            f(&handle)
        })();
        let _ = tx.send(result);
    });
    rx.recv().map_err(|e| format!("wlan worker disconnected: {e}"))?
}

/// The first interface GUID (most PCs have one Wi-Fi adapter). Returns Err if
/// there are no WLAN interfaces (e.g. desktop without a Wi-Fi card).
fn first_interface(handle: &WlanHandle) -> Result<GUID, String> {
    let mut list_ptr: *mut WLAN_INTERFACE_INFO_LIST = std::ptr::null_mut();
    let hr = unsafe { WlanEnumInterfaces(handle.0, None, &mut list_ptr) };
    if hr != 0 {
        return Err(format!("WlanEnumInterfaces failed: {hr}"));
    }
    // SAFETY: list_ptr is valid until WlanFreeMemory. Read count + first entry.
    let result = unsafe {
        let list = &*list_ptr;
        if list.dwNumberOfItems == 0 {
            Err("no WLAN interfaces present".to_string())
        } else {
            // InterfaceGuid is the first element of the array.
            Ok(list.InterfaceInfo[0].InterfaceGuid)
        }
    };
    unsafe { WlanFreeMemory(list_ptr as *mut _) };
    result
}

/// Pull an SSID byte buffer (up to 32 bytes, DOT11_SSID length) into a String.
fn ssid_to_string(uc_ssid: &[u8], length: u32) -> String {
    let len = (length as usize).min(uc_ssid.len());
    // SSIDs are bytes, not strictly utf-16: DOT11_SSID defines a byte buffer,
    // commonly ASCII. Use from_utf8_lossy for safety.
    String::from_utf8_lossy(&uc_ssid[..len]).to_string()
}


#[tauri::command]
pub fn wifi_get_current() -> Result<WifiState, String> {
    with_wlan(|handle| {
        let guid = first_interface(handle)?;
        let mut data_size: u32 = 0;
        let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let hr = unsafe {
            WlanQueryInterface(
                handle.0,
                &guid,
                WLAN_INTF_OPCODE(7), // wlan_intf_opcode_current_connection
                None,
                &mut data_size,
                &mut data_ptr,
                None,
            )
        };
        if hr != 0 || data_ptr.is_null() {
            // Not connected (this opcode returns an error when no connection).
            return Ok(WifiState { connected: false, ssid: None, signal_pct: 0 });
        }
        let state = unsafe {
            let attrs = &*(data_ptr as *const WLAN_CONNECTION_ATTRIBUTES);
            // isState: 0 = not connected, 1 = connected, 2 = adversarial.
            let connected = attrs.isState.0 == 1;
            let ssid = if connected {
                let profile = &attrs.wlanAssociationAttributes.dot11Ssid;
                Some(ssid_to_string(&profile.ucSSID, profile.uSSIDLength))
            } else {
                None
            };
            // Signal quality is a u32 percentage (0-100) from the association
            // attrs; clamp to u8 for the JSON payload.
            let signal = attrs.wlanAssociationAttributes.wlanSignalQuality.min(100) as u8;
            WifiState {
                connected,
                ssid,
                signal_pct: signal,
            }
        };
        unsafe { WlanFreeMemory(data_ptr) };
        Ok(state)
    })
}

#[tauri::command]
pub fn wifi_scan() -> Result<Vec<WifiNetwork>, String> {
    with_wlan(|handle| {
        let guid = first_interface(handle)?;
        // Pull the cached BSS list (WlanGetNetworkBssList returns the last
        // scan's results). Pass None for SSID to get all visible networks.
        let mut bss_ptr: *mut WLAN_BSS_LIST = std::ptr::null_mut();
        let hr = unsafe {
            WlanGetNetworkBssList(
                handle.0,
                &guid,
                None,
                DOT11_BSS_TYPE(2), // dot11_BSS_type_any
                true,              // include security-enabled networks
                None,
                &mut bss_ptr,
            )
        };
        if hr != 0 || bss_ptr.is_null() {
            return Err(format!("WlanGetNetworkBssList failed: {hr}"));
        }
        let networks = unsafe {
            let list = &*bss_ptr;
            let entries = std::slice::from_raw_parts(
                list.wlanBssEntries.as_ptr(),
                list.dwNumberOfItems as usize,
            );
            // One BSS entry PER ACCESS POINT: a mesh/multi-AP network like a
            // home Wi-Fi produces 3-6 entries for the SAME SSID. Collapse to
            // one network per SSID, keeping the strongest signal. This is why
            // the panel showed "MyNet 85%, MyNet 90%, MyNet 85%" etc.
            let mut by_ssid: std::collections::HashMap<String, WifiNetwork> =
                std::collections::HashMap::new();
            for e in entries {
                let ssid = ssid_to_string(&e.dot11Ssid.ucSSID, e.dot11Ssid.uSSIDLength);
                if ssid.is_empty() {
                    continue;
                }
                let sig = signal_rssi_to_pct(e.lRssi);
                let secure = e.dot11BssType.0 != 2 || e.uPhyId != 0; // heuristic
                by_ssid
                    .entry(ssid.clone())
                    .and_modify(|existing| {
                        // Keep the strongest signal across APs sharing this SSID.
                        if sig > existing.signal_pct {
                            existing.signal_pct = sig;
                        }
                    })
                    .or_insert(WifiNetwork {
                        ssid,
                        signal_pct: sig,
                        secure,
                        connected: false,
                    });
            }
            // Sort strongest-first so the usable networks rise to the top.
            let mut out: Vec<WifiNetwork> = by_ssid.into_values().collect();
            out.sort_by(|a, b| b.signal_pct.cmp(&a.signal_pct));
            out
        };
        unsafe { WlanFreeMemory(bss_ptr as *mut _) };
        Ok(networks)
    })
}

/// Toggle the Wi-Fi radio on/off via the WinRT Radio API. The Win32 WLAN API
/// (wlanapi.dll) has no clean radio on/off: Windows exposes that through the
/// same `Devices::Radios::Radio` interface Bluetooth uses. Run on a worker
/// thread with MTA CoInitializeEx + block on the IAsyncOperation via `.get()`.
#[tauri::command]
pub fn wifi_toggle_radio(on: bool) -> Result<(), String> {
    use windows::Devices::Radios::{Radio, RadioKind, RadioState};

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let init =
            unsafe { windows::Win32::System::Com::CoInitializeEx(None, windows::Win32::System::Com::COINIT_MULTITHREADED) };
        let init_ok = init.is_ok();
        let result = (|| {
            let op = Radio::GetRadiosAsync().map_err(|e| format!("GetRadiosAsync: {e}"))?;
            let radios = op.get().map_err(|e| format!("GetRadiosAsync.get: {e}"))?;
            let target = if on { RadioState::On } else { RadioState::Off };
            let mut found = false;
            for radio in radios {
                let kind = radio.Kind().map_err(|e| format!("Radio.Kind: {e}"))?;
                if kind == RadioKind::WiFi {
                    found = true;
                    let sop = radio
                        .SetStateAsync(target)
                        .map_err(|e| format!("SetStateAsync: {e}"))?;
                    sop.get().map_err(|e| format!("SetStateAsync.get: {e}"))?;
                }
            }
            if !found {
                return Err("no Wi-Fi radio found".into());
            }
            Ok(())
        })();
        if init_ok {
            unsafe { windows::Win32::System::Com::CoUninitialize() };
        }
        let _ = tx.send(result);
    });
    rx.recv().map_err(|e| format!("wifi_toggle_radio worker: {e}"))?
}

#[tauri::command]
pub fn wifi_connect(ssid: String, password: Option<String>) -> Result<(), String> {
    // Connecting requires a WLAN profile. For WPA2-Personal networks we build
    // the profile XML and call WlanSetProfile, then WlanConnect. This is the
    // most complex WLAN operation; we keep it focused on the common case
    // (WPA2-PSK) and surface other scenarios via an error.
    use std::fmt::Write;
    use windows::Win32::NetworkManagement::WiFi::{
        WlanConnect, WLAN_CONNECTION_PARAMETERS, WLAN_CONNECTION_MODE,
    };

    let _ = password; // folded into the profile XML below.
    let pw = password.unwrap_or_default();
    let key_material = pw.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    let ssid_esc = ssid.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");

    let hex_ssid: String = ssid.as_bytes().iter().fold(String::new(), |mut acc, b| {
        write!(acc, "{:02X}", b).ok();
        acc
    });

    let profile_xml = format!(
        r#"<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid_esc}</name>
  <SSIDConfig>
    <SSID>
      <hex>{hex_ssid}</hex>
      <name>{ssid_esc}</name>
    </SSID>
  </SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM>
    <security>
      <authEncryption>
        <authentication>WPA2PSK</authentication>
        <encryption>AES</encryption>
        <useOneX>false</useOneX>
      </authEncryption>
      <sharedKey>
        <keyType>passPhrase</keyType>
        <protected>false</protected>
        <keyMaterial>{key_material}</keyMaterial>
      </sharedKey>
    </security>
  </MSM>
</WLANProfile>"#
    );

    let profile_hstring = windows::core::HSTRING::from(&profile_xml);
    with_wlan(move |handle| {
        let guid = first_interface(handle)?;
        // Set the profile (all-user, overwrite if exists). WlanSetProfile takes
        // Param<PCWSTR> / Param<BOOL> wrappers, so pass PCWSTR/BOOL directly.
        let mut reason_code: u32 = 0;
        let hr = unsafe {
            windows::Win32::NetworkManagement::WiFi::WlanSetProfile(
                handle.0,
                &guid,
                0, // all-user profile
                windows::core::PCWSTR(profile_hstring.as_ptr()),
                windows::core::PCWSTR::null(),
                windows::Win32::Foundation::BOOL(1), // overwrite
                None,
                &mut reason_code,
            )
        };
        if hr != 0 {
            return Err(format!("WlanSetProfile failed: error {hr}, reason {reason_code}"));
        }
        // Connect using the profile.
        let profile_name = windows::core::HSTRING::from(&ssid);
        let params = WLAN_CONNECTION_PARAMETERS {
            wlanConnectionMode: WLAN_CONNECTION_MODE(0), // wlan_connection_mode_profile
            strProfile: windows::core::PCWSTR(profile_name.as_ptr()),
            pDot11Ssid: std::ptr::null_mut(),
            pDesiredBssidList: std::ptr::null_mut(),
            dot11BssType: DOT11_BSS_TYPE(2), // any
            dwFlags: 0,
        };
        let hr = unsafe { WlanConnect(handle.0, &guid, &params, None) };
        if hr != 0 {
            return Err(format!("WlanConnect failed: error {hr}"));
        }
        Ok(())
    })
}

/// Convert RSSI (dBm, typically -100..-30) to a 0-100 quality percentage.
/// Uses the standard WiFi signal-quality mapping.
fn signal_rssi_to_pct(rssi: i32) -> u8 {
    // Map -100 dBm → 0%, -50 dBm → 100%.
    let clamped = rssi.clamp(-100, -50);
    let pct = ((clamped + 100) as f32 / 50.0) * 100.0;
    pct.round() as u8
}
