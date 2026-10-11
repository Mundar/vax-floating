use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
use version_check;

fn main() {
    let path = Path::new(&env::var("OUT_DIR").unwrap()).join("to_from_bits.rs");
    let mut file = BufWriter::new(File::create(&path).unwrap());

    write!(&mut file, r#"
/// Convert from a floating point type into an integer bits representation.
#[allow(unused_macros)]
macro_rules! fp_to_bits {{
    ($fx: ident, $fu: ident, $fp: ident) => {{"#).unwrap();
    if version_check::is_min_version("1.83.0").unwrap_or(false) {
        write!(&mut file, r#"
        $fp.to_bits()"#).unwrap();
    }
    else {
        write!(&mut file, r#"
        unsafe {{ std::mem::transmute::<$fx, $fu>($fp) }}"#).unwrap();
    }
    write!(&mut file, r#"
    }};
}}

/// Convert from an integer bits representation into a floating point type.
#[allow(unused_macros)]
macro_rules! fp_from_bits {{
    ($fx: ident, $fu: ident, $bits: expr) => {{"#).unwrap();
    if version_check::is_min_version("1.83.0").unwrap_or(false) {
        write!(&mut file, r#"
        $fx::from_bits($bits)"#).unwrap();
    }
    else {
        write!(&mut file, r#"
        unsafe {{ std::mem::transmute::<$fu, $fx>($bits) }}"#).unwrap();
    }
    write!(&mut file, r#"
    }};
}}"#).unwrap();
}
