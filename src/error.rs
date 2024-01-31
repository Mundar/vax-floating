//! # VAX Floating-point Errors
//!
#![doc = include_str!("doc/encoded_reserved.doc")]
//!
//! # Examples
//!
//! ```rust
//! use vax_floating::{FFloating, Error, Result};
//! use std::str::FromStr;
//!
//! const ZERO: FFloating = FFloating::from_u8(0);
//! const TWO: FFloating = FFloating::from_u8(2);
//!
//! // Failures return a reserved type with the error encoded.
//! let div_by_zero = TWO / ZERO;
//! let overflow = FFloating::MAX * TWO;
//! let underflow = FFloating::MIN_POSITIVE / TWO;
//!
//! // You can convert a reserved value to the matching error.
//! assert_eq!(<Result<FFloating>>::from(div_by_zero), Err(Error::DivByZero));
//! assert_eq!(<Result<FFloating>>::from(overflow),
//!     Err(Error::Overflow(Some(FFloating::MAX_EXP+1))));
//! assert_eq!(<Result<FFloating>>::from(underflow),
//!     Err(Error::Underflow(Some(FFloating::MIN_EXP-1))));
//!
//! assert_eq!(FFloating::from_str("0.0.0"), Err(Error::InvalidStr("0.0.0".to_string())));
//! ```

use std::fmt::{self, Debug, Display, Formatter};

/// # VAX floating-point errors
///
/// # Examples
///
/// ```rust
/// use vax_floating::{FFloating, Error, Result};
/// use std::str::FromStr;
///
/// const ZERO: FFloating = FFloating::from_u8(0);
/// const TWO: FFloating = FFloating::from_u8(2);
///
/// // Failures return a reserved type with the error encoded.
/// let div_by_zero = TWO / ZERO;
/// let overflow = FFloating::MAX * TWO;
/// let underflow = FFloating::MIN_POSITIVE / TWO;
///
/// // You can convert a reserved value to the matching error.
/// assert_eq!(<Result<FFloating>>::from(div_by_zero), Err(Error::DivByZero));
/// assert_eq!(<Result<FFloating>>::from(overflow),
///     Err(Error::Overflow(Some(FFloating::MAX_EXP+1))));
/// assert_eq!(<Result<FFloating>>::from(underflow),
///     Err(Error::Underflow(Some(FFloating::MIN_EXP-1))));
///
/// assert_eq!(FFloating::from_str("0.0.0"), Err(Error::InvalidStr("0.0.0".to_string())));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Exponent too large for floating-point type.
    Overflow(Option<i32>),

    /// Exponent too small for floating-point type.
    Underflow(Option<i32>),


    /// Set as the result of a division operation where the denominator is zero.
    ///
    /// If converted into a Rust floating-point type, this will be converted into an `Infinity` value. For
    /// VAX floating-point types, it will be converted to the Reserved value (negative zero).
    DivByZero,

    /// VAX floating point numbers encoded with a negative zero trigger a reserved operand fault.
    ///
    /// If converted into a Rust floating-point type, this will be converted into a `NaN` value. For
    /// VAX floating-point types, it will be converted to the Reserved value (negative zero).
    Reserved,

    /// Invalid string passed to `from_str` or `from_ascii`.
    InvalidStr(String),
}

impl From<&str> for Error {
    fn from(text: &str) -> Self {
        Error::InvalidStr(text.to_string())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Error::Overflow(exp) =>
                write!(f, "Floating-point overflow error (exponent = {:?})", exp),
            Error::Underflow(exp) =>
                write!(f, "Floating point underflow error (exponent = {:?})", exp),
            Error::DivByZero => f.write_str("Divide by zero error"),
            Error::Reserved => f.write_str("Reserved floating point value"),
            Error::InvalidStr(string) =>
                write!(f, "Failed conversion from {:?} to floating-point type", string),
        }
    }
}

impl std::error::Error for Error {}

/// Error handling with the `Result` type for the [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;
