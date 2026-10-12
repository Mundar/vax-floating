use vax_floating::{FFloating, DFloating, GFloating, HFloating};
macro_rules! display_consts {
    ($ftype:ident) => {
        {
            let ftype = stringify!($ftype);
            println!("{}::RADIX = {}", ftype, $ftype::RADIX);
            println!("{}::MANTISSA_DIGITS = {}", ftype, $ftype::MANTISSA_DIGITS);
            println!("{}::DIGITS = {}", ftype, $ftype::DIGITS);
            println!("{}::EPSILON = {:e}", ftype, $ftype::EPSILON);
            println!("{}::MIN = {:e}", ftype, $ftype::MIN);
            println!("{}::MIN_POSITIVE = {:e}", ftype, $ftype::MIN_POSITIVE);
            println!("{}::MAX = {:e}", ftype, $ftype::MAX);
            println!("{}::MIN_EXP = {}", ftype, $ftype::MIN_EXP);
            println!("{}::MAX_EXP = {}", ftype, $ftype::MAX_EXP);
            println!("{}::MIN_10_EXP = {}", ftype, $ftype::MIN_10_EXP);
            println!("{}::MAX_10_EXP = {}", ftype, $ftype::MAX_10_EXP);
            println!("{}::BITS = {}", ftype, $ftype::BITS);
        }
    };
}

fn main() {
    display_consts!(FFloating);
    println!();
    display_consts!(DFloating);
    println!();
    display_consts!(GFloating);
    println!();
    display_consts!(HFloating);
}
