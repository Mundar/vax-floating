#![doc = include_str!("../README.md")]
//!
#![doc = include_str!("doc/encoded_reserved.doc")]
//!
#![doc = include_str!("doc/vax_ieee_754_diffs.doc")]

#![forbid(future_incompatible)]
#![warn(missing_docs, missing_debug_implementations, bare_trait_objects)]

use forward_ref::{
    forward_ref_binop,
    forward_ref_op_assign,
    forward_ref_unop,
};
use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter, LowerExp, UpperExp},
    hash::{Hash, Hasher},
    ops::{
        Add,
        AddAssign,
        Div,
        DivAssign,
        Mul,
        MulAssign,
        Sub,
        SubAssign,
        Neg,
        Shl,
        ShlAssign,
        Shr,
        ShrAssign,
    },
    str::FromStr,
};

pub mod error;
pub mod arithmetic;
#[cfg(any(test, feature = "proptest"))]
pub mod proptest;

pub use crate::{
    error::{Error, Result},
    arithmetic::{Fault, Sign, VaxFloatingPoint},
};

/// Implement the `swap_words` function for a given unsigned integer type.
macro_rules! swap_words_impl {
    (u32) => {
        /// Reverses the (16-bit) word order of a 32-bit integer (`u32`).
        const fn swap_words(value: u32) -> u32 {
            value.rotate_right(16)
        }
    };
    (u64) => {
        /// Reverses the (16-bit) word order of a 64-bit integer (`u64`).
        const fn swap_words(value: u64) -> u64 {
            let low = value as u32;
            let low = low.rotate_right(16) as u64;
            let high = (value >> 32) as u32;
            let high = high.rotate_right(16) as u64;
            (low << 32) | high
        }
    };
    (u128) => {
        /// Reverses the (16-bit) word order of a 128-bit integer (`u128`).
        const fn swap_words(value: u128) -> u128 {
            let low_low = value as u32;
            let low_low = low_low.rotate_right(16) as u128;
            let low_high = (value >> 32) as u32;
            let low_high = low_high.rotate_right(16) as u128;
            let high_low = (value >> 64) as u32;
            let high_low = high_low.rotate_right(16) as u128;
            let high_high = (value >> 96) as u32;
            let high_high = high_high.rotate_right(16) as u128;
            (low_low << 96) | (low_high << 64) | (high_low << 32) | high_high
        }
    };
}

/// The format specifier for the bits of a specified VAX floating-point type.
macro_rules! zero_ext_hex {
    (FFloating) => {"{:#010X}"};
    (DFloating) => {"{:#018X}"};
    (GFloating) => {"{:#018X}"};
    (HFloating) => {"{:#034X}"};
}

/// Example bits values for a specified VAX floating-point type. Used for examples.
macro_rules! vax_fp_bits {
    (FFloating, 1.5) => {"0x000040C0"};
    (DFloating, 1.5) => {"0x00000000000040C0"};
    (GFloating, 1.5) => {"0x0000000000004018"};
    (HFloating, 1.5) => {"0x0000000000000000000000000080004001"};
    (FFloating, 12.5) => {"0x00004248"};
    (DFloating, 12.5) => {"0x0000000000004248"};
    (GFloating, 12.5) => {"0x0000000000004049"};
    (HFloating, 12.5) => {"0x0000000000000000000000000090004004"};
    (FFloating, 10.0) => {"0x00004220"};
    (DFloating, 10.0) => {"0x0000000000004220"};
    (GFloating, 10.0) => {"0x0000000000004044"};
    (HFloating, 10.0) => {"0x00000000000000000000000040004004"};
    (FFloating, -10.0) => {"0x0000C220"};
    (DFloating, -10.0) => {"0x000000000000C220"};
    (GFloating, -10.0) => {"0x000000000000C044"};
    (HFloating, -10.0) => {"0x0000000000000000000000004000C004"};
}

/// Implement the functions that support converting to and from Rust floating point types (`f32` and
/// `f64`).
///
/// This creates the constant functions 'to_f32()` and `from_f32()` for all VAX floating-point
/// types, and `to_f64()` and 'from_f64()` for VAX floating-point types with a large enough
/// fraction. USAGE: `to_from_rust_fp_impl!(<unsigned type of fraction>, <VAX FP Struct>);`
///
/// This also creates the implementations for `From<f32>` for all VAX floating-point types and
/// `From<f64>` for VAX floating-point types with a large enough fraction. It also creates the
/// inverse `From<(F|D|G|H)Floating>` for `f32` and `f64`. USAGE:
/// `to_from_rust_fp_impl!(From, <unsigned type of fraction>, <VAX FP Type>);`
macro_rules! to_from_rust_fp_impl {
    (u32, $SelfT: ident) => {
        to_from_rust_fp_impl!(u32, $SelfT, f32, to_f32, from_f32);
    };
    ($ux: ident, $SelfT: ident) => {
        to_from_rust_fp_impl!($ux, $SelfT, f32, to_f32, from_f32);
        to_from_rust_fp_impl!($ux, $SelfT, f64, to_f64, from_f64);
    };
    (From, u32, $SelfT: ident) => {
        to_from_rust_fp_impl!(From, u32, $SelfT, f32, to_f32, from_f32);
    };
    (From, $ux: ident, $SelfT: ident) => {
        to_from_rust_fp_impl!(From, $ux, $SelfT, f32, to_f32, from_f32);
        to_from_rust_fp_impl!(From, $ux, $SelfT, f64, to_f64, from_f64);
    };
    ($ux: ident, $SelfT: ident, $fx: ident, $to_func: ident, $from_func: ident) => {
            #[doc = concat!("Convert from [`", stringify!($fx), "`] to a `", stringify!($SelfT), "`.")]
            ///
            /// Can be used to define constants.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT), "::",
                stringify!($from_func), "(0_", stringify!($fx), ");")]
            #[doc = concat!("const THREE_HALVES: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::", stringify!($from_func), "(1.5);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0), ZERO);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(", vax_fp_bits!($SelfT, 1.5), "), THREE_HALVES);")]
            /// ```
            ///
            #[doc = concat!("`From<", stringify!($fx), ">` cannot be used to define constants.")]
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from(0_", stringify!($fx), ");")]
            /// ```
            pub const fn $from_func(src: $fx) -> Self {
                Self::from_fp(VaxFloatingPoint::<$ux>::$from_func(src))
            }

            #[doc = concat!("Convert from a `", stringify!($SelfT), "` to [`", stringify!($fx), "`].")]
            ///
            /// Can be used to define constants.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const ZERO: ", stringify!($fx), " = ", stringify!($SelfT),
                "::from_bits(0).", stringify!($to_func), "();")]
            #[doc = concat!("const THREE_HALVES: ", stringify!($fx), " = ", stringify!($SelfT),
                "::from_bits(", vax_fp_bits!($SelfT, 1.5), ").", stringify!($to_func), "();")]
            #[doc = concat!("assert_eq!(ZERO, 0.0_", stringify!($fx), ");")]
            #[doc = concat!("assert_eq!(THREE_HALVES, 1.5_", stringify!($fx), ");")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(",
                vax_fp_bits!($SelfT, 1.5), ").", stringify!($to_func), "(), 1.5_",
                stringify!($fx), ");")]
            /// ```
            ///
            #[doc = concat!("`From<", stringify!($SelfT), ">` cannot be used to define constants.")]
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const ZERO: ", stringify!($fx), " = ", stringify!($fx),
                "::from(", stringify!($SelfT), "::from_bits(0));")]
            /// ```
            pub const fn $to_func(&self) -> $fx {
                self.to_fp().$to_func()
            }
    };
    (From, $ux: ident, $SelfT: ident, $fx: ident, $to_func: ident, $from_func: ident) => {
        impl From<&$fx> for $SelfT {
            fn from(src: &$fx) -> Self {
                Self::$from_func(*src)
            }
        }

        impl From<$fx> for $SelfT {
            fn from(src: $fx) -> Self {
                Self::$from_func(src)
            }
        }

        impl From<&$SelfT> for $fx {
            fn from(src: &$SelfT) -> Self {
                src.$to_func()
            }
        }

        impl From<$SelfT> for $fx {
            fn from(src: $SelfT) -> Self {
                src.$to_func()
            }
        }
    };
}

macro_rules! vax_float_use_line {
    (FFloating, FFloating) => {
        "# use vax_floating::FFloating;"
    };
    (DFloating, DFloating) => {
        "# use vax_floating::DFloating;"
    };
    (GFloating, GFloating) => {
        "# use vax_floating::GFloating;"
    };
    (HFloating, HFloating) => {
        "# use vax_floating::HFloating;"
    };
    ($SelfT: ident, $ToSelfT: ident) => {
        concat!("# use vax_floating::{", stringify!($SelfT), ", ", stringify!($ToSelfT), "};")
    };
}

/// Implement the functions that support converting to and from Rust floating point types (`f32` and
/// `f64`).
///
/// This creates the constant functions 'to_f32()` and `from_f32()` for all VAX floating-point
/// types, and `to_f64()` and 'from_f64()` for VAX floating-point types with a large enough
/// fraction. USAGE: `to_from_vax_float_impl!(<unsigned type of fraction>, <VAX FP Struct>);`
///
/// This also creates the implementations for `From<f32>` for all VAX floating-point types and
/// `From<f64>` for VAX floating-point types with a large enough fraction. It also creates the
/// inverse `From<(F|D|G|H)Floating>` for `f32` and `f64`. USAGE:
/// `to_from_rust_fp_impl!(From, <unsigned type of fraction>, <VAX FP Type>);`
macro_rules! to_from_vax_float_impl {
    (From, FFloating) => {
        to_from_vax_float_impl!(From, FFloating, DFloating, to_f_floating, to_d_floating);
        to_from_vax_float_impl!(From, FFloating, GFloating, to_f_floating, to_g_floating);
        to_from_vax_float_impl!(From, FFloating, HFloating, to_f_floating, to_h_floating);
    };
    (From, DFloating) => {
        to_from_vax_float_impl!(From, DFloating, GFloating, to_d_floating, to_g_floating);
        to_from_vax_float_impl!(From, DFloating, HFloating, to_d_floating, to_h_floating);
    };
    (From, GFloating) => {
        to_from_vax_float_impl!(From, GFloating, HFloating, to_g_floating, to_h_floating);
    };
    (From, HFloating) => {};
    ($ux: ident, $SelfT: ident) => {
        to_from_vax_float_impl!($ux, $SelfT, FFloating, to_f_floating, to_vfp_32, from_f_floating, from_vfp_32);
        to_from_vax_float_impl!($ux, $SelfT, DFloating, to_d_floating, to_vfp_64, from_d_floating, from_vfp_64);
        to_from_vax_float_impl!($ux, $SelfT, GFloating, to_g_floating, to_vfp_64, from_g_floating, from_vfp_64);
        to_from_vax_float_impl!($ux, $SelfT, HFloating, to_h_floating, to_vfp_128, from_h_floating, from_vfp_128);
    };
    ($ux: ident, $SelfT: ident, $ToSelfT: ident, $to_fp_func: ident, $to_vfp: ident, $from_fp_func: ident, $from_vfp: ident) => {
            #[doc = concat!("Convert from [`", stringify!($ToSelfT), "`] to a `", stringify!($SelfT), "`.")]
            ///
            /// Can be used to define constants.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = vax_float_use_line!($SelfT, $ToSelfT)]
            #[doc = concat!("const FROM_ZERO: ", stringify!($ToSelfT), " = ", stringify!($ToSelfT),
                "::from_bits(0);")]
            #[doc = concat!("const FROM_THREE_HALVES: ", stringify!($ToSelfT), " = ",
                stringify!($ToSelfT), "::from_f32(1.5);")]
            #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT), "::",
                stringify!($from_fp_func), "(FROM_ZERO);")]
            #[doc = concat!("const THREE_HALVES: ", stringify!($SelfT), " = ", stringify!($SelfT), "::",
                stringify!($from_fp_func), "(FROM_THREE_HALVES);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0), ZERO);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(", vax_fp_bits!($SelfT, 1.5), "), THREE_HALVES);")]
            /// ```
            ///
            #[doc = concat!("`From<", stringify!($ToSelfT), ">` cannot be used to define constants.")]
            ///
            /// ```compile_fail
            #[doc = vax_float_use_line!($SelfT, $ToSelfT)]
            #[doc = concat!("const FROM_ZERO: ", stringify!($ToSelfT), " = ", stringify!($ToSelfT),
                "::from_bits(0);")]
            #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from(FROM_ZERO);")]
            /// ```
            pub const fn $from_fp_func(src: $ToSelfT) -> Self {
                Self::from_fp(VaxFloatingPoint::<$ux>::$from_vfp(src.to_fp()))
            }

            #[doc = concat!("Convert from a `", stringify!($SelfT), "` to [`",
                stringify!($ToSelfT), "`].")]
            ///
            /// Can be used to define constants.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = vax_float_use_line!($SelfT, $ToSelfT)]
            #[doc = concat!("const FROM_ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_bits(0);")]
            #[doc = concat!("const FROM_THREE_HALVES: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_bits(", vax_fp_bits!($SelfT, 1.5), ");")]
            #[doc = concat!("const ZERO: ", stringify!($ToSelfT), " = FROM_ZERO.",
                stringify!($to_fp_func), "();")]
            #[doc = concat!("const THREE_HALVES: ", stringify!($ToSelfT), " = FROM_THREE_HALVES.",
                stringify!($to_fp_func), "();")]
            #[doc = concat!("assert_eq!(ZERO, ", stringify!($ToSelfT), "::from_bits(0));")]
            #[doc = concat!("assert_eq!(THREE_HALVES, ", stringify!($ToSelfT), "::from_f32(1.5));")]
            /// ```
            ///
            #[doc = concat!("`From<", stringify!($SelfT), ">` cannot be used to define constants.")]
            ///
            /// ```compile_fail
            #[doc = vax_float_use_line!($SelfT, $ToSelfT)]
            #[doc = concat!("const FROM_ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_bits(0);")]
            #[doc = concat!("const ZERO: ", stringify!($ToSelfT), " = ", stringify!($ToSelfT),
                "::from(FROM_ZERO);")]
            /// ```
            pub const fn $to_fp_func(&self) -> $ToSelfT {
                $ToSelfT::from_fp(self.to_fp().$to_vfp())
            }
    };
    (From, $SelfT: ident, $OtherT: ident, $to_self: ident, $to_other: ident) => {
        impl From<&$OtherT> for $SelfT {
            fn from(src: &$OtherT) -> Self {
                src.$to_self()
            }
        }

        impl From<$OtherT> for $SelfT {
            fn from(src: $OtherT) -> Self {
                src.$to_self()
            }
        }

        impl From<&$SelfT> for $OtherT {
            fn from(src: &$SelfT) -> Self {
                src.$to_other()
            }
        }

        impl From<$SelfT> for $OtherT {
            fn from(src: $SelfT) -> Self {
                src.$to_other()
            }
        }
    };
}

/// The documentation to display for lossy `from_*()` and `From<*>` conversions to
/// VAX floating point types.
macro_rules! from_int_lossy_doc {
    ($SelfT: ident) => {
        concat!("**Note**: Only the most significant set bits that fit into the number of [`",
            stringify!($SelfT), "::MANTISSA_DIGITS`] will be preserved. This will result in a loss
            of precision.")
    };
}

/// Implement the functions that support converting from Rust integer types.
///
/// This creates the constant functions `from_<type>()` for all integer types that are smaller than
/// the fraction size of the VAX floating-point type.
/// USAGE: `from_rust_int_impl!(<unsigned type of fraction>, <VAX FP Struct>);`
///
/// This also creates the implementations for `From` for all VAX floating-point types and
/// `From<f64>` for VAX floating-point types with a large enough fraction. It also creates the
/// inverse `From<(F|D|G|H)Floating>` for `f32` and `f64`. USAGE:
/// `from_rust_int_impl!(From, <unsigned type of fraction>, <VAX FP Type>);`
macro_rules! from_rust_int_impl {
    (u32, $SelfT: ident) => {
        from_rust_int_impl!(to_u32, u32, $SelfT);
        from_rust_int_impl!(lossy_u64, u32, $SelfT);
        from_rust_int_impl!(lossy_u128, u32, $SelfT);
    };
    (u64, $SelfT: ident) => {
        from_rust_int_impl!(to_u32, u64, $SelfT);
        from_rust_int_impl!(to_u64, u64, $SelfT);
        from_rust_int_impl!(lossy_u128, u64, $SelfT);
    };
    (u128, $SelfT: ident) => {
        from_rust_int_impl!(to_u32, u128, $SelfT);
        from_rust_int_impl!(to_u64, u128, $SelfT);
        from_rust_int_impl!(to_u128, u128, $SelfT);
    };
    (From, u32, $SelfT: ident) => {
        from_rust_int_impl!(From, to_u32, u32, $SelfT);
        from_rust_int_impl!(From, lossy_u64, u32, $SelfT);
        from_rust_int_impl!(From, lossy_u128, u32, $SelfT);
    };
    (From, u64, $SelfT: ident) => {
        from_rust_int_impl!(From, to_u32, u64, $SelfT);
        from_rust_int_impl!(From, to_u64, u64, $SelfT);
        from_rust_int_impl!(From, lossy_u128, u64, $SelfT);
    };
    (From, u128, $SelfT: ident) => {
        from_rust_int_impl!(From, to_u32, u128, $SelfT);
        from_rust_int_impl!(From, to_u64, u128, $SelfT);
        from_rust_int_impl!(From, to_u128, u128, $SelfT);
    };
    (to_u32, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!($ux, $SelfT, u8, from_u8, "");
        from_rust_int_impl!($ux, $SelfT, i8, from_i8, "");
        from_rust_int_impl!($ux, $SelfT, u16, from_u16, "");
        from_rust_int_impl!($ux, $SelfT, i16, from_i16, "");
    };
    (to_u64, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!($ux, $SelfT, u32, from_u32, "");
        from_rust_int_impl!($ux, $SelfT, i32, from_i32, "");
    };
    (to_u128, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!($ux, $SelfT, u64, from_u64, "");
        from_rust_int_impl!($ux, $SelfT, i64, from_i64, "");
        from_rust_int_impl!($ux, $SelfT, usize, from_usize, "");
        from_rust_int_impl!($ux, $SelfT, isize, from_isize, "");
        from_rust_int_impl!($ux, $SelfT, u128, from_u128, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, i128, from_i128, from_int_lossy_doc!($SelfT));
    };
    (lossy_u64, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!($ux, $SelfT, u32, from_u32, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, i32, from_i32, from_int_lossy_doc!($SelfT));
    };
    (lossy_u128, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!($ux, $SelfT, u64, from_u64, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, i64, from_i64, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, usize, from_usize, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, isize, from_isize, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, u128, from_u128, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!($ux, $SelfT, i128, from_i128, from_int_lossy_doc!($SelfT));
    };
    (From, to_u32, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!(From, $ux, $SelfT, u8, from_u8, "");
        from_rust_int_impl!(From, $ux, $SelfT, i8, from_i8, "");
        from_rust_int_impl!(From, $ux, $SelfT, u16, from_u16, "");
        from_rust_int_impl!(From, $ux, $SelfT, i16, from_i16, "");
    };
    (From, to_u64, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!(From, $ux, $SelfT, u32, from_u32, "");
        from_rust_int_impl!(From, $ux, $SelfT, i32, from_i32, "");
    };
    (From, to_u128, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!(From, $ux, $SelfT, u64, from_u64, "");
        from_rust_int_impl!(From, $ux, $SelfT, i64, from_i64, "");
        from_rust_int_impl!(From, $ux, $SelfT, usize, from_usize, "");
        from_rust_int_impl!(From, $ux, $SelfT, isize, from_isize, "");
        from_rust_int_impl!(From, $ux, $SelfT, u128, from_u128, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, i128, from_i128, from_int_lossy_doc!($SelfT));
    };
    (From, lossy_u64, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!(From, $ux, $SelfT, u32, from_u32, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, i32, from_i32, from_int_lossy_doc!($SelfT));
    };
    (From, lossy_u128, $ux: ident, $SelfT: ident) => {
        from_rust_int_impl!(From, $ux, $SelfT, u64, from_u64, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, i64, from_i64, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, usize, from_usize, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, isize, from_isize, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, u128, from_u128, from_int_lossy_doc!($SelfT));
        from_rust_int_impl!(From, $ux, $SelfT, i128, from_i128, from_int_lossy_doc!($SelfT));
    };
    ($ux: ident, $SelfT: ident, $uy: ident, $from_func: ident, $lossy_doc: expr) => {
        #[doc = concat!("Convert from [`", stringify!($uy), "`] to a `", stringify!($SelfT), "`.")]
        ///
        /// Can be used to define constants.
        ///
        #[doc = $lossy_doc]
        ///
        /// # Examples
        ///
        /// ```rust
        #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
        #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT), "::",
            stringify!($from_func), "(0_", stringify!($uy), ");")]
        #[doc = concat!("const TEN: ", stringify!($SelfT), " = ", stringify!($SelfT),
            "::", stringify!($from_func), "(10);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0), ZERO);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(", vax_fp_bits!($SelfT, 10.0), "), TEN);")]
        /// ```
        ///
        #[doc = concat!("`From<", stringify!($uy), ">` cannot be used to define constants.")]
        ///
        /// ```compile_fail
        #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
        #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
            "::from(0_", stringify!($uy), ");")]
        /// ```
        pub const fn $from_func(src: $uy) -> Self {
            Self::from_fp(VaxFloatingPoint::<$ux>::$from_func(src))
        }
    };
    (From, $ux: ident, $SelfT: ident, $fx: ident, $from_func: ident, $lossy_doc: expr) => {
        impl From<&$fx> for $SelfT {
            /// Converts to this type from the input type.
            ///
            #[doc = $lossy_doc]
            fn from(src: &$fx) -> Self {
                Self::$from_func(*src)
            }
        }

        impl From<$fx> for $SelfT {
            /// Converts to this type from the input type.
            ///
            #[doc = $lossy_doc]
            fn from(src: $fx) -> Self {
                Self::$from_func(src)
            }
        }
    };
}

/// Implement the Shr, Shl, ShrAssign, and ShlAssign traits.
macro_rules! sh_impl {
    ($t: ident) => {
        sh_impl! { $t, u32, i32 }
    };
    ($t: ident, $uf: ident, $if: ident) => {
        impl Shl<$uf> for $t {
            type Output = $t;

            #[inline]
            fn shl(self, other: $uf) -> $t {
                Self::from_fp(self.to_fp().shift_left_unsigned(other))
            }
        }
        forward_ref_binop! { impl Shl, shl for $t, $uf }

        impl Shr<$uf> for $t {
            type Output = $t;

            #[inline]
            fn shr(self, other: $uf) -> $t {
                Self::from_fp(self.to_fp().shift_right_unsigned(other))
            }
        }
        forward_ref_binop! { impl Shr, shr for $t, $uf }

        impl ShlAssign<$uf> for $t {
            #[inline]
            fn shl_assign(&mut self, other: $uf) {
                *self = *self << other;
            }
        }
        forward_ref_op_assign! { impl ShlAssign, shl_assign for $t, $uf }

        impl ShrAssign<$uf> for $t {
            #[inline]
            fn shr_assign(&mut self, other: $uf) {
                *self = *self >> other;
            }
        }
        forward_ref_op_assign! { impl ShrAssign, shr_assign for $t, $uf }

        impl Shl<$if> for $t {
            type Output = $t;

            #[inline]
            fn shl(self, other: $if) -> $t {
                Self::from_fp(self.to_fp().shift_left(other))
            }
        }
        forward_ref_binop! { impl Shl, shl for $t, $if }

        impl Shr<$if> for $t {
            type Output = $t;

            #[inline]
            fn shr(self, other: $if) -> $t {
                Self::from_fp(self.to_fp().shift_right(other))
            }
        }
        forward_ref_binop! { impl Shr, shr for $t, $if }
    };
}

/// Define and implement a VAX floating-point type given a set of parameters.
///
/// # Examples
///
/// ```text
/// floating_impl!{
///     Self = FFloating,
///     ActualT = u32,
///     ExpBits = 8,
///     VaxName = "F_floating",
///     le_bytes = "[0xC8, 0x40, 0x00, 0x00]",
///     be_bytes = "[0x00, 0x00, 0x40, 0xC8]",
/// }
/// ```
macro_rules! floating_impl {
    (
        Self = $SelfT: ident,
        ActualT = $ux: ident,
        ExpBits = $exp: literal,
        VaxName = $VaxName: literal,
        swapped = $swapped: literal,
        le_bytes = $le_bytes: literal,
        be_bytes = $be_bytes: literal,
    ) => {
        #[doc = concat!("# The VAX ", $VaxName, " type.")]
        ///
        /// ## Reference Documentation
        ///
        /// Here are excerpts from the **VAX Architecture Reference Manual** and the **VAX MACRO
        /// and Instruction Set Reference Manual** for the VAX
        #[doc = concat!($VaxName, " floating-point type.")]
        ///
        #[doc = include_str!(concat!("doc/", stringify!($SelfT), "_vax.doc"))]
        #[derive(Copy, Clone, Default, Eq)]
        pub struct $SelfT($ux);

        impl $SelfT {
            #[doc = concat!("The radix or base of the internal representation of `", stringify!($SelfT), "`.")]
            pub const RADIX: u32 = 2;

            /// Number of significant digits in base 2.
            pub const MANTISSA_DIGITS: u32 = <$ux>::BITS - $exp;

            #[doc = concat!("The mask used by `from_ascii()` to determine when new digits won't ",
                "change the `", stringify!($SelfT), "` fraction value.")]
            const DIV_PRECISION: u32 = Self::MANTISSA_DIGITS + 2;

            /// Approximate number of significant digits in base 10.
            pub const DIGITS: u32 = {
                let value: $ux = 1 << Self::MANTISSA_DIGITS;
                // I chose not to make this change because similar changes would be needed in
                // src/arithmetic.rs as well that happen during runtime. I'm leaving it here in
                // case I change my mind later.
                //
                // This should use the ilog10, but the feature was unstable until Rust version
                // 1.67.1, which prevents it from being used to define a constant. This change
                // enables support for more versions of the rust compiler.
                //
                // The simple replacement is slow, but since it is run only once at compile time,
                // I'm not going to bother finding a more efficient one.
                //
                // error[E0658]: use of unstable library feature 'int_log'
                //     --> src/lib.rs:523:23
                //      |
                //      |                   value.ilog10()
                //                                ^^^^^^
                //let mut slow_ilog = 0_u32;
                //while 10 <= value {
                //    slow_ilog += 1;
                //    value /= 10;
                //}
                //slow_ilog
                value.ilog10()
            };

            #[doc = concat!("[Machine epsilon] value for `", stringify!($SelfT), "`.")]
            ///
            /// This is the difference between `1.0` and the next larger representable number.
            ///
            /// [Machine epsilon]: https://en.wikipedia.org/wiki/Machine_epsilon
            pub const EPSILON: $SelfT = $SelfT::from_parts(Sign::Positive, Self::EXP_BIAS + 1 - (Self::MANTISSA_DIGITS as i32), 0);

            #[doc = concat!("Smallest finite `", stringify!($SelfT), "` value.")]
            pub const MIN: $SelfT =
                $SelfT::from_parts(Sign::Negative, Self::MAX_EXP, (1 << Self::MANTISSA_DIGITS) - 1);
            #[doc = concat!("Smallest positive normal `", stringify!($SelfT), "` value.")]
            pub const MIN_POSITIVE: $SelfT =
                $SelfT::from_parts(Sign::Positive, Self::MIN_EXP, 0);
            #[doc = concat!("Largest finite `", stringify!($SelfT), "` value.")]
            pub const MAX: $SelfT =
                $SelfT::from_parts(Sign::Positive, Self::MAX_EXP, (1 << Self::MANTISSA_DIGITS) - 1);

            /// One greater than the minimum possible normal power of 2 exponent.
            pub const MIN_EXP: i32 = 1 - (Self::EXP_BIAS);
            /// Maximum possible power of 2 exponent.
            pub const MAX_EXP: i32 = (Self::EXP_BIAS) - 1;

            /// Minimum possible normal power of 10 exponent.
            pub const MIN_10_EXP: i32 = {
                const TEN: VaxFloatingPoint::<$ux> = VaxFloatingPoint::<$ux>::from_f32(10.0);
                let mut temp = Self::MIN_POSITIVE.to_fp();
                let mut tens = 1;
                while temp.exponent() <= 0 {
                    temp = temp.multiply_by(TEN);
                    tens -= 1;
                }
                tens
            };
            /// Maximum possible power of 10 exponent.
            pub const MAX_10_EXP: i32 = {
                const TENTH: VaxFloatingPoint::<$ux> = VaxFloatingPoint::<$ux>::from_f32(1.0)
                    .divide_by(VaxFloatingPoint::<$ux>::from_f32(10.0), <$ux>::BITS);
                let mut temp = Self::MAX.to_fp();
                let mut tens = -1;
                while temp.exponent() > 0 {
                    temp = temp.multiply_by(TENTH);
                    tens += 1;
                }
                tens
            };

            #[doc = concat!("The size of the VAX `", $VaxName, "` type in bits.")]
            pub const BITS: u32 = <$ux>::BITS;

            #[doc = concat!("[Exponent bias] of the `", stringify!($SelfT), "` type.")]
            ///
            /// The value subtracted from the exponent to get the actual exponent.
            ///
            /// [Exponent bias]: https://en.wikipedia.org/wiki/Exponent_bias
            const EXP_BIAS: i32 = 1 << ($exp - 1);

            #[doc = concat!("The size of the exponent in the VAX `", $VaxName, "` type in bits.")]
            const EXP_BITS: u32 = $exp;

            #[doc = concat!("The number of bits the exponent of the VAX `", $VaxName,
                "` type is shifted.")]
            ///
            /// Because of the unique ordering of the bytes in VAX floating point types, the
            /// exponent is always in the first (lowest addressed) 16-bits of binary representation.
            const EXP_SHIFT: u32 = 15 - Self::EXP_BITS;

            #[doc = concat!("The mask for the exponent of the VAX `", $VaxName, "` type.")]
            ///
            /// Because of the unique ordering of the bytes in VAX floating point types, the
            /// exponent is always in the first (lowest addressed) 16-bits of binary representation.
            const EXP_MASK: $ux = ((1 << Self::EXP_BITS) - 1) << Self::EXP_SHIFT;

            #[doc = concat!("The sign mask for the VAX `", $VaxName, "` type.")]
            ///
            /// Because of the unique ordering of the bytes in VAX floating point types, the sign
            /// bit is always bit 15.
            const SIGN: $ux = 1 << 15;

            #[doc = concat!("The fraction mask for the VAX `", $VaxName, "` type.")]
            const FRAC_MASK: $ux = !(Self::EXP_MASK | Self::SIGN);

            #[doc = concat!("Number of bits to shift when converting from the word swapped `",
                stringify!($SelfT), "` to `VaxFloatingPoint<", stringify!($ux),
                ">` fraction value.")]
            const FP_FRAC_SHIFT: u32 = $exp;

            #[doc = concat!("The rounding bit mask added to the fraction of `VaxFloatingPoint<",
                stringify!($ux), "> when converting back to `", stringify!($SelfT), "`.")]
            const FP_FRAC_ROUND: $ux = 1 << (Self::FP_FRAC_SHIFT - 1);

            #[doc = concat!(
                "The mask used by `from_ascii()` to determine when new digits won't change the `",
                stringify!($SelfT), "` fraction value.")]
            const ASCII_MASK: $ux = ((1 << (Self::MANTISSA_DIGITS + 1)) - 1) << Self::FP_FRAC_SHIFT;

            /// Reserved shifts to the top two bits of the fraction.
            const RESERVED_SHIFT: u32 = if 14 > Self::EXP_BITS {
                    // This shouldn't need the wrapping_sub because the if statement should block
                    // any Self::EXP_BITS values above 13, however Rust versions after 1.67.1 and
                    // before 1.70.0 seem to evaluate this even though it is unused and trigger an
                    // overflow error. This change enables support for more versions of the rust
                    // compiler.
                    //
                    // 13 - Self::EXP_BITS
                    // ^^^^^^^^^^^^^^^^^^^ attempt to compute `13_u32 - 15_u32`, which would overflow
                    13_u32.wrapping_sub(Self::EXP_BITS)
                }
                else {
                   // If there is no room in the first 16 bits (h_floating), use top of the second 16-bits.
                   30
                };
            /// The overflow or underflow shift moves a 16-bit value into the 16-bit area after the
            /// most significant two bits.
            const RESERVED_OVER_UNDER_SHIFT: u32 = if 14 > Self::EXP_BITS { 16 } else { 32 };

            // Implement the swap_words function for this VAX floating point type.
            swap_words_impl!($ux);

            #[doc = concat!("Raw transmutation from the `", stringify!($SelfT), "` type to `",
                stringify!($ux), "`.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(0_f32).to_bits(), 0_",
                stringify!($ux), ");")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(1.5).to_bits(), ",
                vax_fp_bits!($SelfT, 1.5), "_", stringify!($ux), ");")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5).to_bits(), ",
                vax_fp_bits!($SelfT, 12.5), "_", stringify!($ux), ");")]
            /// ```
            #[inline]
            pub const fn to_bits(self) -> $ux { self.0 }

            #[doc = concat!("Raw transmutation from a `", stringify!($ux), "` the `",
                stringify!($SelfT), "` type.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0), ",
                stringify!($SelfT), "::from_f32(0_f32));")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(",
                vax_fp_bits!($SelfT, 1.5), "), ", stringify!($SelfT), "::from_f32(1.5));")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(",
                vax_fp_bits!($SelfT, 12.5), "), ", stringify!($SelfT), "::from_f32(12.5));")]
            /// ```
            #[inline]
            pub const fn from_bits(bits: $ux) -> Self { Self(bits) }

            #[doc = concat!("Return the memory representation of the `", stringify!($SelfT),
                "` type as a byte array in little-endian byte order.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("let bytes = ", stringify!($SelfT), "::from_f32(12.5).to_le_bytes();")]
            #[doc = concat!("assert_eq!(bytes, ", $le_bytes, ");")]
            /// ```
            #[inline]
            pub const fn to_le_bytes(&self) -> [u8; std::mem::size_of::<$ux>()] { self.0.to_le_bytes() }

            #[doc = concat!("Return the memory representation of the `", stringify!($SelfT),
                "` type as a byte array in big-endian (network) byte order.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("let bytes = ", stringify!($SelfT), "::from_f32(12.5).to_be_bytes();")]
            #[doc = concat!("assert_eq!(bytes, ", $be_bytes, ");")]
            /// ```
            #[inline]
            pub const fn to_be_bytes(&self) -> [u8; std::mem::size_of::<$ux>()] { self.0.to_be_bytes() }

            #[doc = concat!("Return the memory representation of the `", stringify!($SelfT),
                "` type as a byte array in native byte order.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("let bytes = ", stringify!($SelfT), "::from_f32(12.5).to_ne_bytes();")]
            /// assert_eq!(
            ///     bytes,
            ///     if cfg!(target_endian = "big") {
            #[doc = concat!("   ", $be_bytes)]
            ///     } else {
            #[doc = concat!("   ", $le_bytes)]
            ///     }
            /// );
            /// ```
            #[inline]
            pub const fn to_ne_bytes(&self) -> [u8; std::mem::size_of::<$ux>()] { self.0.to_ne_bytes() }

            #[doc = concat!("Create a `", stringify!($SelfT),
                "` type from its representation as a byte array in little endian.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("let float = ", stringify!($SelfT), "::from_le_bytes(", $le_bytes, ");")]
            #[doc = concat!("assert_eq!(float, ", stringify!($SelfT), "::from_f32(12.5));")]
            /// ```
            #[inline]
            pub const fn from_le_bytes(bytes: [u8; std::mem::size_of::<$ux>()]) -> Self {
                Self(<$ux>::from_le_bytes(bytes))
            }

            #[doc = concat!("Create a `", stringify!($SelfT),
                "` type from its representation as a byte array in big endian.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("let float = ", stringify!($SelfT), "::from_be_bytes(", $be_bytes, ");")]
            #[doc = concat!("assert_eq!(float, ", stringify!($SelfT), "::from_f32(12.5));")]
            /// ```
            #[inline]
            pub const fn from_be_bytes(bytes: [u8; std::mem::size_of::<$ux>()]) -> Self {
                Self(<$ux>::from_be_bytes(bytes))
            }

            #[doc = concat!("Create a `", stringify!($SelfT),
                "` type from its representation as a byte array in native endianness.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("let float = ", stringify!($SelfT), "::from_ne_bytes(")]
            ///     if cfg!(target_endian = "big") {
            #[doc = concat!("   ", $be_bytes)]
            ///     } else {
            #[doc = concat!("   ", $le_bytes)]
            ///     }
            /// );
            #[doc = concat!("assert_eq!(float, ", stringify!($SelfT), "::from_f32(12.5));")]
            /// ```
            #[inline]
            pub const fn from_ne_bytes(bytes: [u8; std::mem::size_of::<$ux>()]) -> Self {
                Self(<$ux>::from_ne_bytes(bytes))
            }

            #[doc = concat!("Reverses the (16-bit) word order of the raw transmutation of a `",
                stringify!($ux), "` into the `", stringify!($SelfT), "` type.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(",
                vax_fp_bits!($SelfT, 12.5), ").to_swapped(), ", $swapped, "_", stringify!($ux), ");")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5).to_swapped(), ",
                $swapped, "_", stringify!($ux), ");")]
            /// ```
            #[inline]
            pub const fn to_swapped(self) -> $ux {
                Self::swap_words(self.0)
            }

            #[doc = concat!("Reverses the (16-bit) word order of the raw transmutation of the `",
                stringify!($SelfT), "` type into a `", stringify!($ux), "`.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_swapped(", $swapped, "), ",
                stringify!($SelfT), "::from_f32(12.5));")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_swapped(", $swapped, "), ",
                stringify!($SelfT), "::from_bits(", vax_fp_bits!($SelfT, 12.5), "));")]
            /// ```
            #[inline]
            pub const fn from_swapped(swapped: $ux) -> Self {
                Self(Self::swap_words(swapped))
            }

            #[doc = concat!("Create a `", stringify!($SelfT),
                "` type from the sign, (base 2) exponent, and fraction value.")]
            #[inline]
            const fn from_parts(sign: Sign, exp: i32, frac: $ux) -> Self {
                Self((Self::swap_words(frac) & Self::FRAC_MASK) |
                    ((((exp + Self::EXP_BIAS) as $ux) << Self::EXP_SHIFT) & Self::EXP_MASK) |
                    if sign.is_negative() { Self::SIGN } else { 0 })
            }

            /// Returns a number that represents the sign of `self`.
            ///
            /// - `1.0` if the number is positive, `+0.0`
            /// - `-1.0` if the number is negative
            /// - `Reserved` if the number is `Reserved`
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const ONE: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_i8(1);")]
            #[doc = concat!("const NEG: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_i8(-1);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0).signum(), ONE);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN_POSITIVE.signum(), ONE);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.signum(), ONE);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.signum(), NEG);")]
            #[doc = concat!("assert!(", stringify!($SelfT),
                "::from_bits(0x8000).signum().is_reserved());")]
            /// ```
            #[inline]
            pub const fn signum(self) -> Self {
                if self.is_reserved() { self }
                else {
                    Self((self.0 & Self::SIGN) |
                        (((Self::EXP_BIAS as $ux) + 1) << Self::EXP_SHIFT))
                }
            }

            #[doc = concat!("Return `true` if the `", stringify!($SelfT), "` is zero.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0).is_zero(), true);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN_POSITIVE.is_zero(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.is_zero(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.is_zero(), false);")]
            /// // As long as the sign and exponent is zero, it is considered to be zero.
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0xFFFF0000_",
                stringify!($ux), ").is_zero(), true);")]
            /// ```
            #[inline]
            pub const fn is_zero(&self) -> bool { 0 == (self.0 & (Self::SIGN | Self::EXP_MASK)) }

            #[doc = concat!("Return `true` if the `", stringify!($SelfT), "` is negative.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0).is_negative(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN_POSITIVE.is_negative(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.is_negative(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.is_negative(), true);")]
            /// // All reserved values have the sign bit set, but are not negative.
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0x8000_",
                stringify!($ux), ").is_negative(), false);")]
            /// ```
            #[inline]
            pub const fn is_negative(&self) -> bool {
                0 != ((self.0 & Self::SIGN)) && (0 != (self.0 & Self::EXP_MASK))
            }

            #[doc = concat!("Return `true` if the `", stringify!($SelfT), "` is reserved.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0).is_reserved(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN_POSITIVE.is_reserved(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.is_reserved(), false);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.is_reserved(), false);")]
            /// // As long as the sign is negative and exponent is zero, it is considered reserved.
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0x8000_",
                stringify!($ux), ").is_reserved(), true);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0xFFFF8000_",
                stringify!($ux), ").is_reserved(), true);")]
            /// ```
            #[inline]
            pub const fn is_reserved(&self) -> bool { Self::SIGN == (self.0 & (Self::SIGN | Self::EXP_MASK)) }

            #[doc = concat!("Force the sign of the `", stringify!($SelfT), "` to positive.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            /// for (case, abs) in [
            ///     (0_f32, 0_f32), // Zero doesn't have a sign.
            ///     (1.5, 1.5),    // Positive isn't changed.
            ///     (-3.14, 3.14),  // Negative to positive.
            ///     (-0.04, 0.04),  // Negative to positive.
            /// ].iter() {
            #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_f32(*case).abs(), ", stringify!($SelfT), "::from_f32(*abs));")]
            /// }
            /// ```
            #[inline]
            pub const fn abs(self) -> Self { Self(self.0 & !Self::SIGN) }

            #[doc = concat!("Negate the sign of the `", stringify!($SelfT), "` value.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            /// for (case, neg) in [
            ///     (0_f32, 0_f32), // Zero doesn't have a sign.
            ///     (1.5, -1.5),    // Positive to negative.
            ///     (-3.14, 3.14),  // Negative to positive.
            /// ].iter() {
            #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_f32(*case).negate(), ", stringify!($SelfT), "::from_f32(*neg));")]
            /// }
            /// ```
            pub const fn negate(self) -> Self {
                if 0 != (self.0 & Self::EXP_MASK) { Self(self.0 ^ Self::SIGN) }
                else { self }
            }

            #[doc = concat!("Return the sign of the `", stringify!($SelfT), "` value.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::{", stringify!($SelfT), ", arithmetic::Sign};")]
            /// for (case, sign) in [
            ///     (-0.0_f32, Sign::Positive),
            ///     (1.5, Sign::Positive),
            ///     (-3.14, Sign::Negative),
            /// ].iter() {
            #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_f32(*case).sign(), *sign);")]
            /// }
            /// ```
            pub const fn sign(&self) -> Sign {
                if 0 == (self.0 & Self::SIGN) {
                    Sign::Positive
                }
                else {
                    Sign::Negative
                }
            }

            #[doc = concat!("Add a `", stringify!($SelfT), "` to another `",
                stringify!($SelfT), "`.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const NINETEEN_POINT_FIVE: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(17).add_to(", stringify!($SelfT),
                "::from_f32(2.5));")]
            #[doc = concat!("let seventeen = ", stringify!($SelfT), "::from_u8(17);")]
            #[doc = concat!("let two_point_five = ", stringify!($SelfT), "::from_f32(2.5);")]
            /// assert_eq!(seventeen + two_point_five, NINETEEN_POINT_FIVE);
            /// ```
            ///
            /// This is the same as the addition (`+`) operator, except it can be used to
            /// define constants.
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const NINETEEN_POINT_FIVE: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(17) + ", stringify!($SelfT), "::from_f32(2.5);")]
            /// ```
            pub const fn add_to(self, other: Self) -> Self {
                Self::from_fp(self.to_fp().add_to(other.to_fp(), false))
            }

            #[doc = concat!("Subtract a `", stringify!($SelfT), "` from another `",
                stringify!($SelfT), "`.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const FOURTEEN_POINT_FIVE: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(17).subtract_by(", stringify!($SelfT),
                "::from_f32(2.5));")]
            #[doc = concat!("let seventeen = ", stringify!($SelfT), "::from_u8(17);")]
            #[doc = concat!("let two_point_five = ", stringify!($SelfT), "::from_f32(2.5);")]
            /// assert_eq!(seventeen - two_point_five, FOURTEEN_POINT_FIVE);
            /// ```
            ///
            /// This is the same as the subtraction (`-`) operator, except it can be used to
            /// define constants.
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const FOURTEEN_POINT_FIVE: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(17) - ", stringify!($SelfT), "::from_f32(2.5);")]
            /// ```
            pub const fn subtract_by(self, other: Self) -> Self {
                Self::from_fp(self.to_fp().add_to(other.to_fp(), true))
            }

            #[doc = concat!("Multiply a `", stringify!($SelfT), "` by another `",
                stringify!($SelfT), "`.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const SEVENTEEN_TIMES_TWENTY_THREE: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(17).multiply_by(", stringify!($SelfT),
                "::from_u8(23));")]
            #[doc = concat!("let seventeen = ", stringify!($SelfT), "::from_u8(17);")]
            #[doc = concat!("let twenty_three = ", stringify!($SelfT), "::from_u8(23);")]
            /// assert_eq!(seventeen * twenty_three, SEVENTEEN_TIMES_TWENTY_THREE);
            /// ```
            ///
            /// This is the same as the multiplication (`*`) operator, except it can be used to
            /// define constants.
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const SEVENTEEN_TIMES_TWENTY_THREE: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(17) * ", stringify!($SelfT), "::from_u8(23);")]
            /// ```
            pub const fn multiply_by(self, multiplier: Self) -> Self {
                Self::from_fp(self.to_fp().multiply_by(multiplier.to_fp()))
            }

            #[doc = concat!("Divide a `", stringify!($SelfT), "` by another `",
                stringify!($SelfT), "`.")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWENTY_TWO_SEVENTHS: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::from_u8(22).divide_by(", stringify!($SelfT),
                "::from_u8(7));")]
            #[doc = concat!("let twenty_two = ", stringify!($SelfT), "::from_u8(22);")]
            #[doc = concat!("let seven = ", stringify!($SelfT), "::from_u8(7);")]
            /// assert_eq!(twenty_two / seven, TWENTY_TWO_SEVENTHS);
            /// ```
            ///
            /// This is the same as the division (`/`) operator, except it can be used to define
            /// constants.
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWENTY_TWO_SEVENTHS: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_u8(22) / ", stringify!($SelfT), "::from_u8(7);")]
            /// ```
            pub const fn divide_by(self, divisor: Self) -> Self {
                Self::from_fp(self.to_fp().divide_by(divisor.to_fp(), Self::DIV_PRECISION))
            }

            #[doc = concat!("Convert from a `", stringify!($SelfT), "` to a `VaxFloatingPoint<", stringify!($ux), ">`.")]
            ///
            /// VaxFloatingPoint is used internally for performing mathmatical operations. Since
            #[doc = concat!("the `VaxFloatingPoint<", stringify!($ux),
                ">` type has more precision (it uses the entire", stringify!($ux), ")")]
            /// and supports exponent values outside the range of the
            #[doc = concat!(stringify!($SelfT), "(", $VaxName, ")")]
            /// floating-point type, it may be useful for some calculations.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::{", stringify!($SelfT), ", VaxFloatingPoint};")]
            #[doc = concat!("const TWO: VaxFloatingPoint<", stringify!($ux), "> = ", stringify!($SelfT),
                "::from_ascii(\"2\").to_fp();")]
            #[doc = concat!("const THREE: VaxFloatingPoint<", stringify!($ux),
                "> = VaxFloatingPoint::<", stringify!($ux), ">::from_u8(3);")]
            #[doc = concat!("const TWO_THIRDS_MAX: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::MAX.divide_by(", stringify!($SelfT),
                "::from_u8(3)).multiply_by(", stringify!($SelfT), "::from_u8(2));")]
            #[doc = concat!("let fp = ", stringify!($SelfT), "::MAX.to_fp();")]
            /// let invalid = fp * TWO;
            /// let two_thirds = invalid / THREE;
            #[doc = concat!("assert_eq!(", stringify!($SelfT),
                "::from_fp(two_thirds), TWO_THIRDS_MAX);")]
            #[doc = concat!("assert!(", stringify!($SelfT), "::from_fp(invalid).is_reserved());")]
            /// ```
            pub const fn to_fp(&self) -> VaxFloatingPoint<$ux> {
                match (self.0 & Self::EXP_MASK) >> Self::EXP_SHIFT {
                    0 => match self.to_fault() {
                        None => VaxFloatingPoint::<$ux>::ZERO,
                        Some(fault) => VaxFloatingPoint::<$ux>::from_fault(fault),
                    }
                    exp => unsafe { VaxFloatingPoint::<$ux>::new_unchecked(
                            self.sign(),
                            (exp as i32) - Self::EXP_BIAS,
                            Self::swap_words(self.0 & Self::FRAC_MASK) << Self::FP_FRAC_SHIFT |
                                (1 << (<$ux>::BITS - 1)),
                    ) },
                }
            }

            #[doc = concat!("Convert from a `VaxFloatingPoint<", stringify!($ux), ">` to a `",
                stringify!($SelfT), "`.")]
            ///
            /// VaxFloatingPoint is used internally for performing mathmatical operations. Since
            #[doc = concat!("the `VaxFloatingPoint<", stringify!($ux),
                ">` type has more precision (it uses the entire", stringify!($ux), ")")]
            /// and supports exponent values outside the range of the
            #[doc = concat!(stringify!($SelfT), "(", $VaxName, ")")]
            /// floating-point type, it may be useful for some calculations.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::{", stringify!($SelfT), ", VaxFloatingPoint};")]
            #[doc = concat!("const TWO: VaxFloatingPoint<", stringify!($ux), "> = ", stringify!($SelfT),
                "::from_ascii(\"2\").to_fp();")]
            #[doc = concat!("const THREE: VaxFloatingPoint<", stringify!($ux),
                "> = VaxFloatingPoint::<", stringify!($ux), ">::from_u8(3);")]
            #[doc = concat!("const TWO_THIRDS_MAX: ", stringify!($SelfT), " = ",
                stringify!($SelfT), "::MAX.divide_by(", stringify!($SelfT),
                "::from_u8(3)).multiply_by(", stringify!($SelfT), "::from_u8(2));")]
            #[doc = concat!("let fp = ", stringify!($SelfT), "::MAX.to_fp();")]
            /// let invalid = fp * TWO;
            /// let two_thirds = invalid / THREE;
            #[doc = concat!("assert_eq!(", stringify!($SelfT),
                "::from_fp(two_thirds), TWO_THIRDS_MAX);")]
            #[doc = concat!("assert!(", stringify!($SelfT), "::from_fp(invalid).is_reserved());")]
            /// ```
            pub const fn from_fp(fp: VaxFloatingPoint<$ux>) -> Self {
                if let Some(fault) = fp.fault() { Self::from_fault(fault) }
                else if fp.is_zero() { Self(0) }
                else {
                    let fp = fp.round_fraction(Self::FP_FRAC_ROUND);
                    let exp = ((fp.exponent() + Self::EXP_BIAS) << Self::EXP_SHIFT) as $ux;
                    if 0 == (exp & Self::EXP_MASK) || 0 != (exp & !Self::EXP_MASK) {
                        if 0 >= (fp.exponent() + Self::EXP_BIAS) {
                            Self::from_underflow(Some(fp.exponent()))
                        }
                        else {
                            Self::from_overflow(Some(fp.exponent()))
                        }
                    }
                    else {
                        Self((Self::swap_words(fp.fraction() >> Self::FP_FRAC_SHIFT) & Self::FRAC_MASK) |
                            exp | if fp.sign().is_negative() { Self::SIGN } else { 0 })
                    }
                }
            }

            #[doc = concat!("Parse a string slice into a `", stringify!($SelfT), "`.")]
            ///
            /// # Panics
            ///
            /// This will panic if it fails to parse the string.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWELVE_DOT_FIVE: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_ascii(\"12.5\");")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5), TWELVE_DOT_FIVE);")]
            /// ```
            ///
            /// Invalid input strings will fail to compile.
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWO_DECIMAL_POINTS: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_ascii(\"..\");")]
            /// ```
            ///
            /// Unlike [`FromStr::from_str`], `from_ascii` can be used to define constants.
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            /// # use std::str::FromStr;
            #[doc = concat!("const TWELVE_DOT_FIVE: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_str(\"12.5\").unwrap();")]
            /// ```
            pub const fn from_ascii(text: &str) -> $SelfT {
                match Self::from_ascii_inner(text) {
                    Ok(me) => me,
                    Err(_) => { panic!("Failed to parse input string within from_ascii()"); }
                }
            }

            #[doc = concat!("Internal function that Parses a string slice into a `",
                stringify!($SelfT), "`.")]
            const fn from_ascii_inner(text: &str) -> std::result::Result<$SelfT, &str> {
                match VaxFloatingPoint::<$ux>::from_ascii(text, Self::ASCII_MASK) {
                    Ok(vfp) => Ok(Self::from_fp(vfp)),
                    Err(s) => Err(s),
                }
            }

            #[doc = concat!("Convert an [`Error::Underflow`] to the corresponding reserved value of a `",
                stringify!($SelfT), "`.")]
            const fn from_underflow(exp: Option<i32>) -> $SelfT {
                const UNDERFLOW: $ux = $SelfT::SIGN | (0b01 << $SelfT::RESERVED_SHIFT);
                match exp {
                    None => Self(UNDERFLOW),
                    Some(exp) => Self(UNDERFLOW | if exp < (i16::MIN as i32) {
                        0
                    }
                    else {
                        ((exp as $ux) & 0xFFFF) << Self::RESERVED_OVER_UNDER_SHIFT
                    }),
                }
            }

            #[doc = concat!("Convert an [`Error::Overflow`] to the corresponding reserved value of a `",
                stringify!($SelfT), "`.")]
            const fn from_overflow(exp: Option<i32>) -> $SelfT {
                const OVERFLOW: $ux = $SelfT::SIGN | (0b10 << $SelfT::RESERVED_SHIFT);
                match exp {
                    None => Self(OVERFLOW),
                    Some(exp) => Self(OVERFLOW | if exp > (u16::MAX as i32) {
                        0
                    }
                    else {
                        ((exp as $ux) & 0xFFFF) << Self::RESERVED_OVER_UNDER_SHIFT
                    }),
                }
            }

            #[doc = concat!("Convert an [`Error`] to the corresponding reserved value of a `",
                stringify!($SelfT), "`.")]
            const fn from_error(err: &Error) -> $SelfT {
                const DIV_BY_ZERO: $ux = $SelfT::SIGN;
                const RESERVED: $ux = $SelfT::SIGN | (0b11 << $SelfT::RESERVED_SHIFT);
                match err {
                    Error::DivByZero => $SelfT(DIV_BY_ZERO),
                    Error::Underflow(exp) => $SelfT::from_underflow(*exp),
                    Error::Overflow(exp) => $SelfT::from_overflow(*exp),
                    Error::Reserved | Error::InvalidStr(_) => $SelfT(RESERVED),
                }
            }

            #[doc = concat!("Convert a [`Fault`] to the corresponding reserved value of a `",
                stringify!($SelfT), "`.")]
            const fn from_fault(fault: Fault) -> $SelfT {
                const DIV_BY_ZERO: $ux = $SelfT::SIGN;
                const UNDERFLOW: $ux = $SelfT::SIGN | (0b01 << $SelfT::RESERVED_SHIFT);
                const OVERFLOW: $ux = $SelfT::SIGN | (0b10 << $SelfT::RESERVED_SHIFT);
                const RESERVED: $ux = $SelfT::SIGN | (0b11 << $SelfT::RESERVED_SHIFT);
                match fault {
                    Fault::DivByZero => $SelfT(DIV_BY_ZERO),
                    Fault::Underflow => $SelfT(UNDERFLOW),
                    Fault::Overflow => $SelfT(OVERFLOW),
                    Fault::Reserved => $SelfT(RESERVED),
                }
            }

            #[doc = concat!("Convert a `", stringify!($SelfT), "` to a [`Result`].")]
            ///
            /// All valid floating point values will be `Ok`, and encoded reserved values will
            /// return the corresponding `Err([Error])`.
            const fn to_result(self) -> Result<$SelfT> {
                if self.is_reserved() {
                    match (self.0 >> $SelfT::RESERVED_SHIFT) & 3 {
                        0b00 => Err(Error::DivByZero),
                        0b01 => Err(Error::Underflow(
                            match (self.0 >> $SelfT::RESERVED_OVER_UNDER_SHIFT) & 0xFFFF {
                                0 => None,
                                value => Some((value | 0xFFFF0000) as i32),
                            })),
                        0b10 => Err(Error::Overflow(
                            match (self.0 >> $SelfT::RESERVED_OVER_UNDER_SHIFT) & 0xFFFF {
                                0 => None,
                                value => Some(value as i32),
                            })),
                        0b11 => Err(Error::Reserved),
                        _ => unreachable!(),
                    }
                }
                else {
                    Ok(self)
                }
            }

            #[doc = concat!("Convert a [`Fault`] to the corresponding reserved value of a `",
                stringify!($SelfT), "`.")]
            const fn to_fault(self) -> Option<Fault> {
                if self.is_reserved() {
                    match (self.0 >> $SelfT::RESERVED_SHIFT) & 3 {
                        0b00 => Some(Fault::DivByZero),
                        0b01 => Some(Fault::Underflow),
                        0b10 => Some(Fault::Overflow),
                        0b11 => Some(Fault::Reserved),
                        _ => unreachable!(),
                    }
                }
                else {
                    None
                }
            }

            #[doc = concat!("Panic if the `",
                stringify!($SelfT), "` is not a valid value (i.e. reserved).")]
            ///
            /// This should be used when defining constants to check for errors.
            ///
            /// # Panics
            ///
            /// Panics if the value is reserved (i.e. sign bit set with exponent value of zero).
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWELVE_DOT_FIVE: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_f32(12.5).unwrap();")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5), TWELVE_DOT_FIVE);")]
            /// ```
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const OVERFLOW: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::MAX.add_to(", stringify!($SelfT), "::MAX).unwrap();")]
            /// ```
            ///
            /// ```
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const DIV_BY_ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::MAX.divide_by(", stringify!($SelfT), "::from_bits(0));")]
            /// // Without unwrap, sets constant to divide-by-zero encoded reserved value.
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_bits(0x8000), DIV_BY_ZERO);")]
            /// ```
            ///
            /// ```compile_fail
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const DIV_BY_ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::MAX.divide_by(", stringify!($SelfT), "::from_bits(0)).unwrap();")]
            /// ```
            pub const fn unwrap(self) -> Self {
                if self.is_reserved() {
                    match (self.0 >> $SelfT::RESERVED_SHIFT) & 3 {
                        0b00 => { panic!("Divide by zero error"); }
                        0b01 => { panic!("Underflow error"); }
                        0b10 => { panic!("Overflow error"); }
                        0b11 => { panic!("Reserved operand fault"); }
                        _ => unreachable!(),
                    }
                }
                else {
                    self
                }
            }

            #[doc = concat!("Return the defualt value if the `",
                stringify!($SelfT), "` is not valid (i.e. reserved).")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWELVE_DOT_FIVE: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_f32(12.5).unwrap_or_default();")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5), TWELVE_DOT_FIVE);")]
            ///
            #[doc = concat!("const OVERFLOW: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::MAX.add_to(", stringify!($SelfT), "::MAX).unwrap_or_default();")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::default(), OVERFLOW);")]
            /// ```
            pub const fn unwrap_or_default(self) -> Self {
                if self.is_reserved() {
                    // default() is not const, but the following is. The `verify_float_defaults`
                    // test verifies that this is correct.
                    Self::from_bits(0)
                }
                else {
                    self
                }
            }

            #[doc = concat!("Return an alternate value if the `",
                stringify!($SelfT), "` is not valid (i.e. reserved).")]
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            #[doc = concat!("const TWELVE_DOT_FIVE: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_f32(12.5).unwrap_or(", stringify!($SelfT), "::MAX);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5), TWELVE_DOT_FIVE);")]
            ///
            #[doc = concat!("const OVERFLOW: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::MAX.add_to(", stringify!($SelfT), "::MAX).unwrap_or(", stringify!($SelfT), "::MAX);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX, OVERFLOW);")]
            /// ```
            pub const fn unwrap_or(self, default: Self) -> Self {
                if self.is_reserved() {
                    default
                }
                else {
                    self
                }
            }

            #[doc = concat!("Returns the result from a closure if the `",
                stringify!($SelfT), "` is not valid (i.e. reserved).")]
            ///
            /// This is included for completeness, but it isn't const like the other `unwrap`
            /// functions.
            ///
            /// # Examples
            ///
            /// ```rust
            #[doc = concat!("# use vax_floating::", stringify!($SelfT), ";")]
            /// use vax_floating::Result;
            #[doc = concat!("fn saturate_float(float: ", stringify!($SelfT), ") -> ",
                stringify!($SelfT), " {")]
            ///     use vax_floating::Error::*;
            #[doc = concat!("    match <Result<", stringify!($SelfT), ">>::from(float) {")]
            ///         Ok(float) => float,
            #[doc = concat!("        Err(Overflow(_)) | Err(DivByZero) => ", stringify!($SelfT), "::MAX,")]
            #[doc = concat!("        Err(Underflow(_)) => ", stringify!($SelfT), "::MIN_POSITIVE,")]
            #[doc = concat!("        Err(_) => ", stringify!($SelfT), "::MIN,")]
            ///     }
            /// }
            ///
            #[doc = concat!("let twelve_dot_five: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::from_f32(12.5).unwrap_or_else(saturate_float);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::from_f32(12.5), twelve_dot_five);")]
            ///
            #[doc = concat!("let overflow: ", stringify!($SelfT), " = ", stringify!($SelfT),
                "::MAX.add_to(", stringify!($SelfT), "::MAX).unwrap_or_else(saturate_float);")]
            #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX, overflow);")]
            /// ```
            pub fn unwrap_or_else<F: FnOnce(Self) -> Self>(self, op: F) -> Self {
                if self.is_reserved() {
                    op(self)
                }
                else {
                    self
                }
            }

            from_rust_int_impl!($ux, $SelfT);

            to_from_rust_fp_impl!($ux, $SelfT);

            to_from_vax_float_impl!($ux, $SelfT);
        }

        from_rust_int_impl!(From, $ux, $SelfT);

        to_from_rust_fp_impl!(From, $ux, $SelfT);

        to_from_vax_float_impl!(From, $SelfT);

        impl From<$SelfT> for Result<$SelfT> {
            fn from(float: $SelfT) -> Result<$SelfT> {
                float.to_result()
            }
        }

        impl From<Result<$SelfT>> for $SelfT {
            fn from(result: Result<$SelfT>) -> $SelfT {
                match result {
                    Ok(float) => float,
                    Err(err) => $SelfT::from_error(&err),
                }
            }
        }

        impl Hash for $SelfT {
            fn hash<H: Hasher>(&self, state: &mut H) {
                const ZERO: $ux = 0;
                if self.is_zero() { ZERO.hash(state); }
                else { self.0.hash(state) }
            }
        }

        impl PartialEq for $SelfT {
            fn eq(&self, other: &Self) -> bool {
                // All zeroes are equally zero.
                if self.is_zero() && other.is_zero() { true }
                else { self.0 == other.0 }
            }
        }

        impl PartialOrd for $SelfT {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                use Ordering::*;
                if self.is_reserved() || other.is_reserved() {
                    if self.0 == other.0 { return Some(Equal); }
                    return None;
                }
                match (self.is_zero(), other.is_zero()) {
                    (true, true) => Some(Equal),
                    (false, true) => match 0 == (self.0 & Self::SIGN) {
                        true => Some(Greater),
                        false => Some(Less),
                    }
                    (true, false) => match 0 == (other.0 & Self::SIGN) {
                        true => Some(Less),
                        false => Some(Greater),
                    }
                    (false, false) => {
                        if (0 != (self.0 & Self::SIGN)) ^ (0 != (other.0 & Self::SIGN)) {
                            match (0 != (self.0 & Self::SIGN)) {
                                false => Some(Greater),
                                true => Some(Less),
                            }
                        }
                        else {
                            match 0 != (self.0 & Self::SIGN) {
                                false => {
                                    Self::swap_words(self.0 & !Self::SIGN)
                                        .partial_cmp(&Self::swap_words(other.0 & !Self::SIGN))
                                }
                                true => {
                                    Self::swap_words(other.0 & !Self::SIGN)
                                        .partial_cmp(&Self::swap_words(self.0 & !Self::SIGN))
                                }
                            }
                        }
                    }
                }
            }
        }

        impl Add for $SelfT {
            type Output = $SelfT;

            fn add(self, rhs: Self) -> Self::Output {
                Self::from_fp(self.to_fp().add_to(rhs.to_fp(), false))
            }
        }
        forward_ref_binop!(impl Add, add for $SelfT, $SelfT);

        impl AddAssign for $SelfT {
            #[inline]
            fn add_assign(&mut self, other: $SelfT) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl AddAssign, add_assign for $SelfT, $SelfT }

        impl Sub for $SelfT {
            type Output = $SelfT;

            fn sub(self, rhs: Self) -> Self::Output {
                Self::from_fp(self.to_fp().add_to(rhs.to_fp(), true))
            }
        }
        forward_ref_binop!(impl Sub, sub for $SelfT, $SelfT);

        impl SubAssign for $SelfT {
            #[inline]
            fn sub_assign(&mut self, other: $SelfT) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl SubAssign, sub_assign for $SelfT, $SelfT }

        impl Div for $SelfT {
            type Output = $SelfT;

            fn div(self, rhs: Self) -> Self::Output {
                Self::from_fp(self.to_fp().divide_by(rhs.to_fp(), Self::DIV_PRECISION))
            }
        }
        forward_ref_binop!(impl Div, div for $SelfT, $SelfT);

        impl DivAssign for $SelfT {
            #[inline]
            fn div_assign(&mut self, other: $SelfT) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl DivAssign, div_assign for $SelfT, $SelfT }

        impl Mul for $SelfT {
            type Output = $SelfT;

            fn mul(self, rhs: Self) -> Self::Output {
                Self::from_fp(self.to_fp().multiply_by(rhs.to_fp()))
            }
        }
        forward_ref_binop!(impl Mul, mul for $SelfT, $SelfT);

        impl MulAssign for $SelfT {
            #[inline]
            fn mul_assign(&mut self, other: $SelfT) {
                *self = *self * other;
            }
        }
        forward_ref_op_assign! { impl MulAssign, mul_assign for $SelfT, $SelfT }

        sh_impl!($SelfT);

        impl Neg for $SelfT {
            type Output = $SelfT;

            #[inline]
            fn neg(self) -> Self::Output
            {
                self.negate()
            }
        }
        forward_ref_unop! { impl Neg, neg for $SelfT }

        impl FromStr for $SelfT {
            type Err = Error;

            fn from_str(s: &str) -> Result<$SelfT> {
                Ok($SelfT::from_ascii_inner(s)?)
            }
        }

        impl Debug for $SelfT {
            fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
                let fp = self.to_fp();
                fmt.debug_struct(stringify!($SelfT))
                    .field("bits", &format_args!(zero_ext_hex!($SelfT), self.clone().to_bits()))
                    .field("sign", &format_args!("{:?}", fp.sign()))
                    .field("exponent", &format_args!("{0}", fp.exponent()))
                    .field("frac", &format_args!("{:#X}", fp.fraction()))
                    .finish()
            }
        }

        impl Display for $SelfT {
            fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
                self.to_fp().float_to_decimal_display(fmt, Self::MANTISSA_DIGITS)
            }
        }

        impl LowerExp for $SelfT {
            fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
                self.to_fp().float_to_exponential_common(fmt, Self::MANTISSA_DIGITS, false)
            }
        }

        impl UpperExp for $SelfT {
            fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
                self.to_fp().float_to_exponential_common(fmt, Self::MANTISSA_DIGITS, true)
            }
        }
    };
}

floating_impl!{
    Self = FFloating,
    ActualT = u32,
    ExpBits = 8,
    VaxName = "F_floating",
    swapped = "0x42480000",
    le_bytes = "[0x48, 0x42, 0x00, 0x00]",
    be_bytes = "[0x00, 0x00, 0x42, 0x48]",
}

floating_impl!{
    Self = DFloating,
    ActualT = u64,
    ExpBits = 8,
    VaxName = "D_floating",
    swapped = "0x4248000000000000",
    le_bytes = "[0x48, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]",
    be_bytes = "[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42, 0x48]",
}

floating_impl!{
    Self = GFloating,
    ActualT = u64,
    ExpBits = 11,
    VaxName = "G_floating",
    swapped = "0x4049000000000000",
    le_bytes = "[0x49, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]",
    be_bytes = "[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x49]",
}

floating_impl!{
    Self = HFloating,
    ActualT = u128,
    ExpBits = 15,
    VaxName = "H_floating",
    swapped = "0x40049000000000000000000000000000",
    le_bytes = "[0x04, 0x40, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]",
    be_bytes = "[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x90, 0x00, 0x40, 0x04]",
}

#[cfg(test)]
mod tests {
    use super::{
        FFloating,
        DFloating,
        GFloating,
        HFloating,
        Sign,
        Error,
    };
    use proptest::prelude::*;
    use std::{
        cmp::{min, max},
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
        str::FromStr,
    };

    fn pi_str(size: usize) -> &'static str {
        static PI_STR: &'static str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989";
        if 0 == size { &PI_STR[..=0] }
        else if size < (PI_STR.len()-2) { &PI_STR[..(size+2)] }
        else { PI_STR }
    }

    macro_rules! create_pi_test {
        ($name: ident, $floating: ident, $start: expr) => {
            #[test]
            fn $name() {
                let expected_pi = $floating::from_str(pi_str($start as usize)).unwrap();
                for i in (($start+1) as usize)..1000 {
                    let pi = $floating::from_str(pi_str(i)).unwrap();
                    assert_eq!(pi, expected_pi);
                }
            }
        };
    }

    create_pi_test!(f_floating_pi_test, FFloating, FFloating::DIGITS+1);
    create_pi_test!(d_floating_pi_test, DFloating, DFloating::DIGITS+1);
    create_pi_test!(g_floating_pi_test, GFloating, GFloating::DIGITS);
    create_pi_test!(h_floating_pi_test, HFloating, HFloating::DIGITS);

    const MAX_FFLOATING: f32 = unsafe { std::mem::transmute::<u32, f32>(0x7EFFFFFF) };
    const MIN_FFLOATING: f32 = unsafe { std::mem::transmute::<u32, f32>(0xFEFFFFFF) };

    prop_compose! {
        fn ffloating_f32_range(min: f32, max: f32)(float in min..=max) -> f32 {
            float
        }
    }

    prop_compose! {
        fn ffloating_range(min: f32, max: f32)(float in ffloating_f32_range(min, max)) -> FFloating {
            FFloating::from_f32(float)
        }
    }

    prop_compose! {
        fn ffloating_f32()(float in ffloating_f32_range(MIN_FFLOATING, MAX_FFLOATING)) -> f32 {
            float
        }
    }

    prop_compose! {
        fn ffloating()(float in ffloating_f32_range(MAX_FFLOATING, MIN_FFLOATING)) -> FFloating {
            FFloating::from_f32(float)
        }
    }

    macro_rules! to_from_test {
        ($float: ident, $floating: ident) => {
            let floating = $floating::from_f32($float);
            let to_f32 = floating.to_f32();
            assert_eq!($float, to_f32, "{:?}, float bits = {:#X}, to_from bits = {:#X}", floating,
                $float.to_bits(), to_f32.to_bits());
        };
    }

    proptest! {
        #[test]
        fn to_from_f32(float in ffloating_f32()) {
            to_from_test!(float, FFloating);
            to_from_test!(float, DFloating);
            to_from_test!(float, GFloating);
            to_from_test!(float, HFloating);
        }
    }

    #[test]
    fn verify_float_defaults() {
        assert_eq!(FFloating::default(), FFloating::from_bits(0));
        assert_eq!(DFloating::default(), DFloating::from_bits(0));
        assert_eq!(GFloating::default(), GFloating::from_bits(0));
        assert_eq!(HFloating::default(), HFloating::from_bits(0));
    }

    macro_rules! swap_words_test {
        ($ux: ident, $bytes: literal, $vax_type: ident, $start: ident, $swapped: ident) => {
            let mut start = [0_u8; $bytes];
            start.copy_from_slice(&$start[0..$bytes]);
            let start = $ux::from_ne_bytes(start);
            let mut swapped = [0_u8; $bytes];
            swapped.copy_from_slice(&$swapped[(16-$bytes)..16]);
            let swapped = $ux::from_ne_bytes(swapped);
            assert_eq!($vax_type::swap_words(start), swapped)//,
                //"start = {:X}; swapped = {:X}", start, swapped);
        };
    }

    proptest! {
        #[test]
        fn swap_words(words in proptest::collection::vec(u16::MIN..=u16::MAX, 8..=8)) {
            let start_bytes: Vec<u8> = words.iter().map(|w| w.to_ne_bytes()).flatten().collect();
            let swapped_bytes: Vec<u8> = words.iter().rev().map(|w| w.to_ne_bytes()).flatten().collect();
            swap_words_test!(u128, 16, HFloating, start_bytes, swapped_bytes);
            swap_words_test!(u64, 8, GFloating, start_bytes, swapped_bytes);
            swap_words_test!(u64, 8, DFloating, start_bytes, swapped_bytes);
            swap_words_test!(u32, 4, FFloating, start_bytes, swapped_bytes);
        }
    }

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    macro_rules! create_hash_eq_test {
        ($name: ident, $floating: ident, $ux: ident) => {
            proptest! {
                #[test]
                fn $name(bits1 in <$ux>::MIN..=<$ux>::MAX, bits2 in <$ux>::MIN..=<$ux>::MAX) {
                    let zero1 = bits1 & $floating::FRAC_MASK;
                    let zero2 = bits2 & $floating::FRAC_MASK;
                    let float1 = $floating::from_bits(bits1);
                    let float2 = $floating::from_bits(bits2);
                    let float_zero1 = $floating::from_bits(zero1);
                    let float_zero2 = $floating::from_bits(zero2);
                    if zero1 != bits1 {
                        assert_ne!(float1, float_zero1, "float1 should not be equal to float_zero1");
                    }
                    if zero2 != bits2 {
                        assert_ne!(float2, float_zero2, "float2 should not be equal to float_zero2");
                    }
                    if bits1 != bits2 {
                        assert_ne!(float1, float2, "float1 should not be equal to float2");
                    }
                    else {
                        assert_eq!(float1, float2, "float1 should be equal to float2");
                        assert_eq!(calculate_hash(&float1), calculate_hash(&float2),
                            "float1 hash should be equal to float2 hash");
                    }
                    // All zeroes are equal to each other.
                    assert_eq!(float_zero1, float_zero2, "float_zero1 should be equal to float_zero2");
                    assert_eq!(calculate_hash(&float_zero1), calculate_hash(&float_zero2),
                        "float_zero1 hash should be equal to float_zero2 hash");
                }
            }
        };
    }

    create_hash_eq_test!(test_f_floating_hash_eq, FFloating, u32);
    create_hash_eq_test!(test_d_floating_hash_eq, DFloating, u64);
    create_hash_eq_test!(test_g_floating_hash_eq, GFloating, u64);
    create_hash_eq_test!(test_h_floating_hash_eq, HFloating, u128);

    macro_rules! create_ordering_test {
        ($name: ident, $floating: ident, $ux: ident) => {
            #[test]
            fn $name() {
                use Sign::*;
                const MAX_FRAC: $ux = (1 << $floating::MANTISSA_DIGITS) - 1;
                static ORDERED_LIST: &'static [$floating] = &[
                    $floating::from_parts(Negative, $floating::MAX_EXP, MAX_FRAC),  // Minimum
                    $floating::from_parts(Negative, $floating::MAX_EXP, 0),
                    $floating::from_parts(Negative, 0, MAX_FRAC),  // -0.99999999
                    $floating::from_parts(Negative, 0, 0),  // -0.5
                    $floating::from_parts(Negative, $floating::MIN_EXP, MAX_FRAC),
                    $floating::from_parts(Negative, $floating::MIN_EXP, 0), // Max negative
                    $floating(0),   // 0
                    $floating::from_parts(Positive, $floating::MIN_EXP, 0), // Max positive
                    $floating::from_parts(Positive, $floating::MIN_EXP, MAX_FRAC),
                    $floating::from_parts(Positive, 0, 0),  // 0.5
                    $floating::from_parts(Positive, 0, MAX_FRAC),  // 0.99999999
                    $floating::from_parts(Positive, $floating::MAX_EXP, 0),
                    $floating::from_parts(Positive, $floating::MAX_EXP, MAX_FRAC),  // Maximum
                ];
                for i in 0..ORDERED_LIST.len() {
                    let leq = ORDERED_LIST[i];
                    for j in i..ORDERED_LIST.len() {
                        let geq = ORDERED_LIST[j];
                        assert!(leq <= geq,
                            "Comparison failed: {:X?} should be less than {:X?}, but it wasn't",
                            leq, geq);
                        assert!(geq >= leq,
                            "Comparison failed: {:X?} should be less than {:X?}, but it wasn't",
                            leq, geq);
                    }
                }
            }
        };
    }

    create_ordering_test!(test_f_floating_ordering, FFloating, u32);
    create_ordering_test!(test_d_floating_ordering, DFloating, u64);
    create_ordering_test!(test_g_floating_ordering, GFloating, u64);
    create_ordering_test!(test_h_floating_ordering, HFloating, u128);

    macro_rules! create_convert_test {
        ($name: ident, $floating: ident, $ux: ident) => {
            proptest! {
                #[test]
                fn $name(
                    frac in 0..((1 as $ux) << ($floating::MANTISSA_DIGITS)),
                    sign in 0..=1,
                ) {
                    let sign = if sign == 0 { '+' } else { '-' };
                    let int_text = format!("{}{}", sign, frac);
                    let float_text = format!("{}{}.0", sign, frac);
                    let exp_text = {
                        let (before, after) = int_text.split_at(2);
                        format!("{}.{}e{}", before, after, after.len())
                    };
                    let from_int_text = $floating::from_str(&int_text).unwrap();
                    let from_float_text = $floating::from_str(&float_text).unwrap();
                    let from_exp_text = $floating::from_str(&exp_text).unwrap();
                    let mut from_frac = $floating::from(frac);
                    if '-' == sign { from_frac = -from_frac; }
                    prop_assert_eq!(from_int_text, from_float_text);
                    prop_assert_eq!(from_int_text, from_exp_text);
                    prop_assert_eq!(from_int_text, from_frac);
                }
            }
        };
    }

    create_convert_test!(convert_to_f_floating, FFloating, u32);
    create_convert_test!(convert_to_d_floating, DFloating, u64);
    create_convert_test!(convert_to_g_floating, GFloating, u64);
    create_convert_test!(convert_to_h_floating, HFloating, u128);

    // This compares two display outputs to see if they are close enough to each other. This is to
    // compare the f32 to FFloating and f64 to GFloating. There are some rounding differences
    // between the VAX floating-point Display functions. This function compares the two strings and
    // check to see if they are within 2 of each other at the least significant digit.
    pub fn display_close_enough(disp1: &str, disp2: &str, ok_diff: usize, fail_msg: &str) -> u128 {
        if disp1 != disp2 {
            close_enough_inner(disp1.to_string(), disp2.to_string(), disp1, disp2, ok_diff, fail_msg)
        }
        else { 0 }
    }

    // This compares two display outputs to see if they are close enough to each other. This is to
    // compare the f32 to FFloating and f64 to GFloating. There are some rounding differences
    // between the VAX floating-point Display functions. This function compares the two strings and
    // check to see if they are within 2 of each other at the least significant digit.
    //
    // There is probably a better way to do this, but this works for now.
    pub fn close_enough_inner(
        mut disp1: String,
        mut disp2: String,
        orig1: &str,
        orig2: &str,
        ok_diff: usize,
        fail_msg: &str,
    ) -> u128 {
        // Test and remove the sign.
        match disp1.get(0..=0) {
            Some("+") => if disp1.get(0..=0) == Some("+") {
                disp1.remove(0);
                disp2.remove(0);
            }
            else {
                panic!("Signs don't match don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
            Some("-") => if disp1.get(0..=0) == Some("-") {
                disp1.remove(0);
                disp2.remove(0);
            }
            else {
                panic!("Signs don't match don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
            _ => {}
        }
        // Check and remove exponent.
        if disp1.contains(['e', 'E']) {
            let (rem1, exp1) = disp1.rsplit_once(['e', 'E']).unwrap_or((&disp1, ""));
            let (rem2, exp2) = disp2.rsplit_once(['e', 'E']).unwrap_or((&disp2, ""));
            if exp1 != exp2 {
                panic!("Exponents don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
            disp1.truncate(rem1.len());
            disp2.truncate(rem2.len());
        }
        // Remove decimal places.
        match(disp1.find('.'), disp2.find('.')) {
            (Some(p1), Some(p2)) => {
                if p1 != p2 {
                    panic!("Decimal points don't line up: {}\n left: {:?}\nright: {:?}", fail_msg,
                        orig1, orig2);
                }
                disp1.remove(p1);
                disp2.remove(p2);
            }
            (None, Some(p2)) => {
                if disp1.len() != p2 {
                    panic!("Decimal points don't line up: {}\n left: {:?}\nright: {:?}", fail_msg,
                        orig1, orig2);
                }
                disp2.remove(p2);
            }
            (Some(p1), None) => {
                if p1 != disp2.len() {
                    panic!("Decimal points don't line up: {}\n left: {:?}\nright: {:?}", fail_msg,
                        orig1, orig2);
                }
                disp1.remove(p1);
            }
            (None, None) => {}
        }
        // Remove leading zeroes and verify that they are the same size.
        let no_lead1 = disp1.trim_start_matches('0');
        let no_lead2 = disp2.trim_start_matches('0');
        if 1 < (disp1.len() - no_lead1.len()).abs_diff(disp2.len() - no_lead2.len()) {
            panic!("Leading zeroes don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
        }
        // Trim trailing zeroes and replace any size difference with zeroes.
        let no_tail1 = disp1.trim_end_matches('0');
        let no_tail2 = disp2.trim_end_matches('0');
        let remove_tail = min(disp1.len() - no_tail1.len(), disp2.len() - no_tail2.len());
        disp1.truncate(disp1.len() - remove_tail);
        disp2.truncate(disp2.len() - remove_tail);
        if disp1.len() > disp2.len() {
            for _ in 0..(disp1.len()-disp2.len()) {
                disp2.push('0');
            }
        }
        else if disp1.len() < disp2.len() {
            for _ in 0..(disp2.len()-disp1.len()) {
                disp1.push('0');
            }
        }
        let val1 = i128::from_str_radix(&disp1, 10)
            .expect(&format!("Failed to convert string to number ({:?}): {}\n left: {:?}\nright: {:?}",
                &disp1, fail_msg, orig1, orig2));
        let val2 = i128::from_str_radix(&disp2, 10)
            .expect(&format!("Failed to convert string to number ({:?}): {}\n left: {:?}\nright: {:?}",
                &disp2, fail_msg, orig1, orig2));
        let diff = val1.abs_diff(val2);
        if diff > (ok_diff as u128) {
            panic!("Values are too far apart ({}): {}\n left: {:?}\nright: {:?}", diff, fail_msg,
                orig1, orig2);
        }
        diff
    }

    // This compares two display outputs to see if they are close enough to each other. This is to
    // compare the f32 to FFloating and f64 to GFloating. There are some rounding differences
    // between the VAX floating-point Display functions. This function compares the two strings and
    // and only compare the `digits` nost-significant digits.
    pub fn display_close_enough_2(disp1: &str, disp2: &str, digits: usize, precision: bool, fail_msg: &str) -> u128 {
        if disp1 != disp2 {
            close_enough_inner_2(disp1.to_string(), disp2.to_string(), disp1, disp2, digits, precision, fail_msg)
        }
        else { 0 }
    }

    // This compares two display outputs to see if they are close enough to each other. This is to
    // compare the f32 to FFloating and f64 to GFloating. There are some rounding differences
    // between the VAX floating-point Display functions. This function compares the two strings and
    // and only compare the `digits` nost-significant digits.
    //
    // This is attempt number two to write this function. The original was having problems with
    // precision, because there seem to be issues with the Rust implementations with precision:
    //
    // Input: "+1.0e23" Format: "{:.1}"
    // g_floating: "100000000000000000000000.0"
    // f64:         "99999999999999991611392.0"
    // 15 digits:    ^^^^^^^^^^^^^^^
    //
    // It looks like it is not restricting the display digits to the number of valid digits (15)
    // and rounding appropriately. What we want to do extract the portion of the string that we
    // care about and just compare those.
    pub fn close_enough_inner_2(
        mut disp1: String,
        mut disp2: String,
        orig1: &str,
        orig2: &str,
        digits: usize,
        precision: bool,
        fail_msg: &str,
    ) -> u128 {
        // Test and remove the sign.
        match disp1.get(0..=0) {
            Some("+") => if disp1.get(0..=0) == Some("+") {
                disp1.remove(0);
                disp2.remove(0);
            }
            else {
                panic!("Signs don't match don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
            Some("-") => if disp1.get(0..=0) == Some("-") {
                disp1.remove(0);
                disp2.remove(0);
            }
            else {
                panic!("Signs don't match don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
            _ => {}
        }
        // Check and remove exponent.
        if disp1.contains(['e', 'E']) {
            let (rem1, exp1) = disp1.rsplit_once(['e', 'E']).unwrap_or((&disp1, ""));
            let (rem2, exp2) = disp2.rsplit_once(['e', 'E']).unwrap_or((&disp2, ""));
            if exp1 != exp2 {
                panic!("Exponents don't match: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
            disp1.truncate(rem1.len());
            disp2.truncate(rem2.len());
        }
        // Find location of decimal place and first non-zero digit.
        let fp1 = match disp1.find('.') {
            None => disp1.len(),
            Some(fp1) => {
                disp1.remove(fp1);
                fp1
            }
        };
        let fp2 = match disp2.find('.') {
            None => disp2.len(),
            Some(fp2) => {
                disp2.remove(fp2);
                fp2
            }
        };
        if precision {
            if (disp1.len() - fp1) != (disp2.len() - fp2) {
                panic!("Precision mismatch: {}\n left: {:?}\nright: {:?}", fail_msg, orig1, orig2);
            }
        }
        let (val1, val2) = match (disp1.find(|c: char| ('1' <= c) && ('9' >= c)),
            disp2.find(|c: char| ('1' <= c) && ('9' >= c)))
        {
            (None, None) => (0, 0),
            (Some(digit1), None) => (
                u128::from_str_radix(&disp1[digit1..min(digit1+digits, disp1.len())], 10)
                    .expect(&format!("Failed to convert string to number ({:?}): {}\n left: {:?}\nright: {:?}",
                    &disp1, fail_msg, orig1, orig2)),
                0),
            (None, Some(digit2)) => (0,
                u128::from_str_radix(&disp2[digit2..min(digit2+digits, disp2.len())], 10)
                    .expect(&format!("Failed to convert string to number ({:?}): {}\n left: {:?}\nright: {:?}",
                    &disp2, fail_msg, orig1, orig2))),
            (Some(digit1), Some(digit2)) => {
                let offset = min((digit1 as isize)-(fp1 as isize), (digit2 as isize)-(fp2 as isize));
                let start1 = if 0 > ((fp1 as isize)+offset) {
                    for _ in 0..((fp1 as isize)+offset).unsigned_abs() {
                        disp1.insert(0, '0');
                    }
                    0
                }
                else {
                    ((fp1 as isize)+offset) as usize
                };
                let start2 = if 0 > ((fp2 as isize)+offset) {
                    for _ in 0..((fp2 as isize)+offset).unsigned_abs() {
                        disp2.insert(0, '0');
                    }
                    0
                }
                else {
                    ((fp2 as isize)+offset) as usize
                };
                let mut target_len = max(disp1.len()-start1, disp2.len()-start2);
                if target_len > digits { target_len = digits; }
                if disp1.len()-start1 > target_len {
                    disp1.truncate(start1 + target_len);
                }
                else if disp1.len()-start1 < target_len {
                    for _ in 0..(target_len+start1-disp1.len()) {
                        disp1.push('0');
                    }
                }
                if disp2.len()-start2 > target_len {
                    disp2.truncate(start2 + target_len);
                }
                else if disp2.len()-start2 < target_len {
                    for _ in 0..(target_len+start2-disp2.len()) {
                        disp2.push('0');
                    }
                }
                (
                    u128::from_str_radix(&disp1[start1..], 10)
                        .expect(&format!("Failed to convert string to number ({:?}): {}\n left: {:?}\nright: {:?}",
                        &disp1, fail_msg, orig1, orig2)),
                    u128::from_str_radix(&disp2[start2..], 10)
                        .expect(&format!("Failed to convert string to number ({:?}): {}\n left: {:?}\nright: {:?}",
                        &disp1, fail_msg, orig1, orig2))
                )
            }
        };
        let diff = val1.abs_diff(val2);
        if diff > (1 as u128) {
            panic!("Values are too far apart ({}): {}\n left: {:?}\nright: {:?}", diff, fail_msg,
                orig1, orig2);
        }
        diff
    }

    macro_rules! display_case {
        ($fmt: literal, $floating: ident, $fx: ident, $text: ident) => {
            let vax_float = format!($fmt, $floating);
            let rust_float = format!($fmt, $fx);
            dbg!(&$text);
            display_close_enough(&vax_float, &rust_float, 2, &format!("input string = {:?}", $text));
        };
    }

    macro_rules! create_display_test {
        ($name: ident, $floating: ident, $ux: ident, $fx: ident) => {
            create_display_test!($name, $floating, $ux, $fx, {
                "{}",
                "{:e}",
                "{:E}",
                "{:.0e}",
                "{:.0E}",
                "{:.1e}",
                "{:.1E}",
                "{:.2e}",
                "{:.2E}",
                "{:.3e}",
                "{:.3E}",
                "{:.4e}",
                "{:.4E}",
                "{:.5e}",
                "{:.5E}",
                "{:.6e}",
                "{:.6E}"
            });
        };
        ($name: ident, $floating: ident, $ux: ident, $fx: ident, $fmts: tt) => {
            create_display_test!($name, $floating, $ux, $fx, $fmts, $floating::MIN_10_EXP+1, $floating::MAX_10_EXP);
        };
        (
            $name: ident,
            $floating: ident,
            $ux: ident,
            $fx: ident,
            {$($fmt: literal),+},
            $min_exp: expr,
            $max_exp: expr
        ) => {
            proptest! {
                #[test]
                fn $name(
                    frac in 9..((1 as $ux) << ($floating::MANTISSA_DIGITS)),
                    exp in $min_exp..$max_exp,
                    sign in 0..=1,
                ) {
                    let sign = if sign == 0 { '+' } else { '-' };
                    let int_text = if 9 == frac { "0".to_string() } else { format!("{}", frac) };
                    let exp_text = {
                        let (before, mut after) = int_text.split_at(1);
                        if "" == after { after = "0"; }
                        format!("{}{}.{}e{}", sign, before, after, exp)
                    };
                    let mut rust_float = $fx::from_str(&exp_text).unwrap();
                    if rust_float == -0.0 { rust_float = 0.0; }
                    let vax_float = $floating::from_str(&exp_text).unwrap();
                    $(
                        display_case!($fmt, vax_float, rust_float, exp_text);
                    )+
                }
            }
        };
        ($name: ident, $floating: ident, $ux: ident) => {
            create_display_test!($name, $floating, $ux, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17});
        };
        ($name: ident, $floating: ident, $ux: ident, $fmts: tt) => {
            create_display_test!($name, $floating, $ux, $fmts, $floating::MIN_10_EXP+1, $floating::MAX_10_EXP);
        };
        (
            $name: ident,
            $floating: ident,
            $ux: ident,
            {$($prec: literal),+},
            $min_exp: expr,
            $max_exp: expr
        ) => {
            proptest! {
                #[test]
                fn $name(
                    frac in 9..((1 as $ux) << ($floating::MANTISSA_DIGITS)),
                    exp in $min_exp..$max_exp,
                    sign in 0..=1,
                ) {
                    let mut sign = if sign == 0 { "" } else { "-" };
                    let int_text = if 9 == frac {
                        sign = "";
                        "0".to_string()
                    } else { format!("{}", frac) };
                    let exp_text = {
                        let (before, mut after) = int_text.split_at(1);
                        if "" == after { after = "0"; }
                        format!("{}{}.{}e{}", sign, before, after, exp)
                    };
                    let vax_float = $floating::from_str(&exp_text).unwrap();
                    $(
                    {
                        let mut expected = if 0 > exp {
                            let leading_zeroes = (exp.unsigned_abs() - 1) as usize;
                            format!("{0}0.{3:0>1$}{2}", sign, leading_zeroes, int_text, "")
                        }
                        else {
                            let insert = (exp as usize) + 1;
                            if insert > int_text.len() {
                                let trailing_zeroes = if "0" == int_text { 0 } else {
                                    insert - int_text.len()
                                };
                                format!("{0}{1}{3:0>2$}.0", sign, int_text, trailing_zeroes, "")
                            }
                            else {
                                let (before, after) = int_text.split_at(insert);
                                format!("{0}{1}.{2}", sign, before, after)
                            }
                        };
                        let end = expected.find('.').unwrap_or(expected.len()) + $prec + 1;
                        if 0 == $prec { expected.truncate(end - 1); }
                        else if end < expected.len() { expected.truncate(end); }
                        else if end > expected.len() {
                            for _ in expected.len()..end {
                                expected.push('0');
                            }
                        }
                        let vax_float = format!("{0:.1$}", vax_float, $prec);
                        display_close_enough_2(&vax_float, &expected, $floating::DIGITS as usize, true, &format!("input string = {:?}", &exp_text));
                    }
                    )+
                }
            }
        };
    }

    create_display_test!(display_f_floating, FFloating, u32, f32);
    create_display_test!(display_g_floating, GFloating, u64, f64);
    create_display_test!(display_g_floating_extra, GFloating, u64, f64, {
        "{:.7e}",
        "{:.7E}",
        "{:.8e}",
        "{:.8E}",
        "{:.9e}",
        "{:.9E}",
        "{:.10e}",
        "{:.10E}",
        "{:.11e}",
        "{:.11E}",
        "{:.12e}",
        "{:.12E}",
        "{:.13e}",
        "{:.13E}",
        "{:.14e}",
        "{:.14E}",
        "{:.15e}",
        "{:.15E}"
    });
    create_display_test!(display_d_floating, DFloating, u64, f64, {
        "{:.0e}",
        "{:.0E}",
        "{:.1e}",
        "{:.1E}",
        "{:.2e}",
        "{:.2E}",
        "{:.3e}",
        "{:.3E}",
        "{:.4e}",
        "{:.4E}",
        "{:.5e}",
        "{:.5E}",
        "{:.6e}",
        "{:.6E}",
        "{:.7e}",
        "{:.7E}",
        "{:.8e}",
        "{:.8E}",
        "{:.9e}",
        "{:.9E}",
        "{:.10e}",
        "{:.10E}",
        "{:.11e}",
        "{:.11E}",
        "{:.12e}",
        "{:.12E}",
        "{:.13e}",
        "{:.13E}",
        "{:.14e}",
        "{:.14E}",
        "{:.15e}",
        "{:.15E}"
    });
    create_display_test!(display_h_floating, HFloating, u128, f64, {
        "{:.0e}",
        "{:.0E}",
        "{:.1e}",
        "{:.1E}",
        "{:.2e}",
        "{:.2E}",
        "{:.3e}",
        "{:.3E}",
        "{:.4e}",
        "{:.4E}",
        "{:.5e}",
        "{:.5E}",
        "{:.6e}",
        "{:.6E}",
        "{:.7e}",
        "{:.7E}",
        "{:.8e}",
        "{:.8E}",
        "{:.9e}",
        "{:.9E}",
        "{:.10e}",
        "{:.10E}",
        "{:.11e}",
        "{:.11E}",
        "{:.12e}",
        "{:.12E}",
        "{:.13e}",
        "{:.13E}",
        "{:.14e}",
        "{:.14E}",
        "{:.15e}",
        "{:.15E}"
    }, f64::MIN_10_EXP+1, f64::MAX_10_EXP);
    //create_display_test!(display_f_floating_0, FFloating, u32, f32, {"{:.0}"}, FFloating::MIN_10_EXP+1, 7);
    //create_display_test!(display_f_floating_1, FFloating, u32, f32, {"{:.1}"}, FFloating::MIN_10_EXP+1, 7);
    //create_display_test!(display_f_floating_2, FFloating, u32, f32, {"{:.2}"}, FFloating::MIN_10_EXP+1, 7);
    create_display_test!(display_f_floating_precision, FFloating, u32);
    create_display_test!(display_d_floating_precision, DFloating, u64);
    create_display_test!(display_g_floating_precision, GFloating, u64);
    create_display_test!(display_h_floating_precision, HFloating, u128);

    #[test]
    fn vax_reserved_and_results() {
        assert_eq!(FFloating::from_bits(0x8000).to_result(), Err(Error::DivByZero));
        assert_eq!(DFloating::from_bits(0x8000).to_result(), Err(Error::DivByZero));
        assert_eq!(GFloating::from_bits(0x8000).to_result(), Err(Error::DivByZero));
        assert_eq!(HFloating::from_bits(0x8000).to_result(), Err(Error::DivByZero));
        assert_eq!(FFloating::from_bits(0x8060).to_result(), Err(Error::Reserved));
        assert_eq!(DFloating::from_bits(0x8060).to_result(), Err(Error::Reserved));
        assert_eq!(GFloating::from_bits(0x800C).to_result(), Err(Error::Reserved));
        assert_eq!(HFloating::from_bits(0xC0008000).to_result(), Err(Error::Reserved));
        assert_eq!(FFloating::from_bits(0x8040).to_result(), Err(Error::Overflow(None)));
        assert_eq!(DFloating::from_bits(0x8040).to_result(), Err(Error::Overflow(None)));
        assert_eq!(GFloating::from_bits(0x8008).to_result(), Err(Error::Overflow(None)));
        assert_eq!(HFloating::from_bits(0x80008000).to_result(), Err(Error::Overflow(None)));
        assert_eq!(FFloating::from_bits(0x8020).to_result(), Err(Error::Underflow(None)));
        assert_eq!(DFloating::from_bits(0x8020).to_result(), Err(Error::Underflow(None)));
        assert_eq!(GFloating::from_bits(0x8004).to_result(), Err(Error::Underflow(None)));
        assert_eq!(HFloating::from_bits(0x40008000).to_result(), Err(Error::Underflow(None)));
        assert_eq!(FFloating::from_bits(0xFFFF8040).to_result(), Err(Error::Overflow(Some(65535))));
        assert_eq!(DFloating::from_bits(0xFFFF8040).to_result(), Err(Error::Overflow(Some(65535))));
        assert_eq!(GFloating::from_bits(0xFFFF8008).to_result(), Err(Error::Overflow(Some(65535))));
        assert_eq!(HFloating::from_bits(0xFFFF80008000).to_result(), Err(Error::Overflow(Some(65535))));
        assert_eq!(FFloating::from_bits(0x80008020).to_result(), Err(Error::Underflow(Some(-32768))));
        assert_eq!(DFloating::from_bits(0x80008020).to_result(), Err(Error::Underflow(Some(-32768))));
        assert_eq!(GFloating::from_bits(0x80008004).to_result(), Err(Error::Underflow(Some(-32768))));
        assert_eq!(HFloating::from_bits(0x800040008000).to_result(), Err(Error::Underflow(Some(-32768))));
    }

    macro_rules! ilog10_test {
        ($ux: ty) => {
            let mut value: $ux = <$ux>::MAX;
            let ilog = value.ilog10();
            let display = format!("{}", value).len() - 1;
            let mut slow = 0_u32;
            while 10 <= value {
                slow += 1;
                value /= 10;
            }
            assert_eq!(ilog, slow);
            assert_eq!(ilog, display as u32);
        };
        ($ux: ty, $floating: ident) => {
            let mut value: $ux = 1 << $floating::MANTISSA_DIGITS;
            let ilog = value.ilog10();
            let display = format!("{}", value).len() - 1;
            let mut slow = 0_u32;
            while 10 <= value {
                slow += 1;
                value /= 10;
            }
            assert_eq!(ilog, slow);
            assert_eq!(ilog, display as u32);
        };
    }

    #[test]
    fn alternate_ilog10() {
        ilog10_test!(u32, FFloating);
        ilog10_test!(u64, DFloating);
        ilog10_test!(u64, GFloating);
        ilog10_test!(u128, HFloating);
        ilog10_test!(u32);
        ilog10_test!(u64);
        ilog10_test!(u128);
    }

    #[test]
    #[ignore]
    fn minor_display_bug_1() {
        const TENTH: DFloating = DFloating::from_ascii("0.1");
        const ONE_HUNDRED: FFloating = FFloating::from_u8(100);
        const MANY_ZEROES: HFloating = HFloating::from_u128(
            100_000_000_000_000_000_000_000_000_000_000u128);
        assert_eq!(&format!("{:e}", TENTH), "1e-1");
        assert_eq!(&format!("{:e}", ONE_HUNDRED), "1e2");   // Fails with 1.00e2
        assert_eq!(&format!("{:E}", MANY_ZEROES), "1E32");  // Fails with 1.00000000000000000000000000000000E32
    }

    proptest! {
        fn from_h_floating_tests(
            h_floating in any::<HFloating>(),
        ) {
            let float_as_text = h_floating.to_string();
            let f_float = h_floating.to_f_floating();
            let d_float = h_floating.to_d_floating();
            let g_float = h_floating.to_g_floating();
            assert_eq!(f_float, FFloating::from_str(&float_as_text).unwrap());
            assert_eq!(d_float, DFloating::from_str(&float_as_text).unwrap());
            assert_eq!(g_float, GFloating::from_str(&float_as_text).unwrap());
        }
    }

    proptest! {
        fn from_g_floating_tests(
            g_floating in any::<GFloating>(),
        ) {
            let float_as_text = g_floating.to_string();
            let f_float = g_floating.to_f_floating();
            let d_float = g_floating.to_d_floating();
            let h_float = g_floating.to_h_floating();
            assert_eq!(f_float, FFloating::from_str(&float_as_text).unwrap());
            assert_eq!(d_float, DFloating::from_str(&float_as_text).unwrap());
            assert_eq!(h_float, HFloating::from_str(&float_as_text).unwrap());
        }
    }

    proptest! {
        fn from_d_floating_tests(
            d_floating in any::<DFloating>(),
        ) {
            let float_as_text = d_floating.to_string();
            let f_float = d_floating.to_f_floating();
            let g_float = d_floating.to_g_floating();
            let h_float = d_floating.to_h_floating();
            assert_eq!(f_float, FFloating::from_str(&float_as_text).unwrap());
            assert_eq!(g_float, GFloating::from_str(&float_as_text).unwrap());
            assert_eq!(h_float, HFloating::from_str(&float_as_text).unwrap());
        }
    }

    proptest! {
        fn from_f_floating_tests(
            f_floating in any::<FFloating>(),
        ) {
            let float_as_text = f_floating.to_string();
            let d_float = f_floating.to_d_floating();
            let g_float = f_floating.to_g_floating();
            let h_float = f_floating.to_h_floating();
            assert_eq!(d_float, DFloating::from_str(&float_as_text).unwrap());
            assert_eq!(g_float, GFloating::from_str(&float_as_text).unwrap());
            assert_eq!(h_float, HFloating::from_str(&float_as_text).unwrap());
        }
    }
}
