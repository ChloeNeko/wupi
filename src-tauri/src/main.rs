// In release builds, run on the Windows GUI subsystem so no console window is
// allocated. In debug builds we keep the console (the default) so log output is
// still visible during development.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    wupi_os_lib::run()
}
