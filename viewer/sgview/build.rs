fn main() {
    // assets/robot/* (URDF model: GLB meshes + robot.json) are embedded into the binary: the page needs no files next to it
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("robot_assets.rs");
    let dir = std::path::Path::new("assets/robot");
    let mut code = String::from("pub const ROBOT_FILES: &[(&str, &[u8])] = &[\n");
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut names: Vec<_> = rd.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
        names.sort();
        for n in names {
            let abs = std::fs::canonicalize(dir.join(&n)).unwrap();
            code.push_str(&format!("    ({:?}, include_bytes!({:?})),\n", n, abs.to_string_lossy()));
        }
    }
    code.push_str("];\n");
    std::fs::write(out, code).unwrap();
    println!("cargo:rerun-if-changed=assets/robot");
}
