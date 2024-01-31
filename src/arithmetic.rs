//! # Vax floating-point arithmetic
//!
//! In order to perform most of the arithmetic on VAX floating-point types, they need to be
//! converted into an intermediate form. The intermediate form is the [VaxFloatingPoint] type.
//!
//! The [VaxFloatingPoint] structure separates the sign, exponent, and fractional protions of the
//! underlying floating-point type. The floating-point arithmetic is based off the implementation
//! in SimH VAX. This project assumes that the SimH VAX floating-point implementation would match
//! the VAX hardware.
//!
//! [VaxFloatingPoint] supports adding, subtracting, multiplication, and division. It also
//! supports shift right and left operations which just modify the exponent value. A shift left
//! multiplies the value of the float by 2 and a shift right divides it by 2. It also suports
//! reading from a string and displaying to a string.
//!
//! [VaxFloatingPoint] provides higher precision then the native floating point types, because the
//! fractional value uses all of the bits of the underlying floating point type. For example, the
//! F_floating type uses 23 bits for the fractional value, but when converted into a
//! [VaxFloatingPoint], the fractional value uses 32 bits. If the higher precision is needed, then
//! [VaxFloatingPoint] values can be used directly.
//!
//! Constant versions of all mathmatical operations and conversions from the Rust data types
//! (including from an ASCII string slice) are available. Since the arithmetic operations are not
//! particularly fast, doing them at compile time as constant operations is preferable.
//!
//! The Display (and LowerExp and UpperExp) implementations fall into the slow but accurate
//! category, since they use the VAX mathmatical operations to do the conversions.
//!
#![doc = include_str!("doc/vax_ieee_754_diffs.doc")]

use forward_ref::{
    forward_ref_binop,
    forward_ref_op_assign,
};
use std::{
    cmp::{Ordering, min},
    fmt::{self, Debug, Display, Formatter, LowerExp, UpperHex, UpperExp},
    ops::{
        Add,
        AddAssign,
        BitXor,
        BitXorAssign,
        Div,
        DivAssign,
        Mul,
        MulAssign,
        Sub,
        SubAssign,
    },
    str::FromStr,
};
use crate::{
    error::{Error, Result},
};

/// VAX floating-point number fault
///
/// Internally, none of the arithmetic functions will panic or return an error, but the error state
/// is recored as the `Fault`. Depending on how the [`VaxFloatingPoint`] type is converted, it
/// will either return an [`Error`] of the appropriate type, or be converted into an appropriate
/// type.
///
/// Once a fault has been set, it prevents further arithmetic operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Set as the result of a division operation where the denominator is zero.
    ///
    /// If converted into a Rust floating-point type, this will be converted into an `Infinity` value. For
    /// VAX floating-point types, it will be converted to the Reserved value (negative zero).
    ///
    /// It will return an [`Error::DivByZero`] error for conversions that return a [`Result`].
    DivByZero,
    /// VAX floating point numbers encoded with a negative zero trigger a reserved operand fault.
    ///
    /// If converted into a Rust floating-point type, this will be converted into a `NaN` value. For
    /// VAX floating-point types, it will be converted to the Reserved value (negative zero).
    ///
    /// It will return an [`Error::Reserved`] error for conversions that return a [`Result`].
    Reserved,
    /// Set as a result of overflowing the VaxFloatingPoint exponent.
    ///
    /// If converted into a Rust floating-point type, this will be converted into a `NaN` value. For
    /// VAX floating-point types, it will be converted to the Reserved value (negative zero).
    ///
    /// It will return an [`Error::Overflow(i32::MAX)`][Error::Overflow] error for conversions that
    /// return a [`Result`].
    Overflow,
    /// Set as a result of overflowing the VaxFloatingPoint exponent.
    ///
    /// If converted into a Rust floating-point type, this will be converted into a `NaN` value. For
    /// VAX floating-point types, it will be converted to the Reserved value (negative zero).
    ///
    /// It will return an [`Error::Underflow(i32::MIN)`][Error::Underflow] error for conversions
    /// that return a [`Result`].
    Underflow,
}

impl From<Fault> for Error {
    fn from(fault: Fault) -> Self {
        match fault {
            Fault::DivByZero => Error::DivByZero,
            Fault::Reserved => Error::Reserved,
            Fault::Overflow => Error::Overflow(None),
            Fault::Underflow => Error::Underflow(None),
        }
    }
}

/// Sign of the floating-point number.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Sign {
    #[default]
    /// Indicate positive floating-point numbers.
    Positive,
    /// Indicate negative floating-point numbers.
    Negative,
}

impl Sign {
    /// Returns the display of this character as a &'static str.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::Sign;
    /// assert_eq!(Sign::Positive.as_str(false), "");
    /// assert_eq!(Sign::Positive.as_str(true), "+");
    /// assert_eq!(Sign::Negative.as_str(false), "-");
    /// assert_eq!(Sign::Negative.as_str(true), "-");
    /// ```
    pub const fn as_str(&self, force_sign: bool) -> &'static str {
        match self {
            Sign::Positive => match force_sign {
                false => "",
                true => "+",
            }
            Sign::Negative => "-",
        }
    }

    /// Performs a NOT operation on the `Sign` type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::Sign;
    /// assert_eq!(Sign::Positive.negate(), Sign::Negative);
    /// assert_eq!(Sign::Negative.negate(), Sign::Positive);
    /// ```
    ///
    /// `Sign::negate` is used in `const` functions.
    ///
    /// ```compile_fail
    /// # use vax_floating::arithmetic::Sign;
    /// const fn negate(sign: Sign) -> Sign {
    ///     !sign
    /// }
    /// ```
    #[inline]
    pub const fn negate(self) -> Self {
        use Sign::*;
        match self {
            Positive => Negative,
            Negative => Positive,
        }
    }

    /// Performs an XOR operation on two `Sign` types.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::Sign;
    /// assert_eq!(Sign::Positive.xor(Sign::Positive), Sign::Positive);
    /// assert_eq!(Sign::Positive.xor(Sign::Negative), Sign::Negative);
    /// assert_eq!(Sign::Negative.xor(Sign::Positive), Sign::Negative);
    /// assert_eq!(Sign::Negative.xor(Sign::Negative), Sign::Positive);
    /// ```
    ///
    /// `Sign::xor` is used in `const` functions.
    ///
    /// ```compile_fail
    /// # use vax_floating::arithmetic::Sign;
    /// const fn xor(sign1: Sign, sign2: Sign) -> Sign {
    ///     sign1 ^ sign2
    /// }
    /// ```
    #[inline]
    pub const fn xor(self, other: Self) -> Self {
        use Sign::*;
        match self {
            Positive => other,
            Negative => match other {
                Positive => Negative,
                Negative => Positive,
            },
        }
    }

    /// Returns true of the `Sign` is `Positive`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::Sign;
    /// let positive = Sign::Positive;
    /// let negative = Sign::Negative;
    ///
    /// assert_eq!(Sign::Positive.is_positive(), true);
    /// assert_eq!(negative.is_positive(), false);
    /// ```
    ///
    /// `Sign::is_positive` is used in `const` functions.
    ///
    /// ```compile_fail
    /// # use vax_floating::arithmetic::Sign;
    /// const fn is_positive(sign: Sign) -> bool {
    ///     Sign::Positive == sign
    /// }
    /// ```
    #[inline]
    pub const fn is_positive(&self) -> bool {
        match self {
            Sign::Positive => true,
            Sign::Negative => false,
        }
    }

    /// Returns true of the `Sign` is `Negative`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::Sign;
    /// let positive = Sign::Positive;
    /// let negative = Sign::Negative;
    ///
    /// assert_eq!(Sign::Positive.is_negative(), false);
    /// assert_eq!(negative.is_negative(), true);
    /// ```
    ///
    /// `Sign::is_negative` is used in `const` functions.
    ///
    /// ```compile_fail
    /// # use vax_floating::arithmetic::Sign;
    /// const fn is_negative(sign: Sign) -> bool {
    ///     Sign::Negative == sign
    /// }
    /// ```
    #[inline]
    pub const fn is_negative(&self) -> bool {
        match self {
            Sign::Positive => false,
            Sign::Negative => true,
        }
    }

    /// Returns the `Sign` based on an ASCII byte.
    ///
    /// Expects '+' or '-'. Invalid values return `Sign::Positive`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::Sign;
    /// assert_eq!(Sign::from_byte(b'+'), Sign::Positive);
    /// assert_eq!(Sign::from_byte(b'-'), Sign::Negative);
    /// ```
    #[inline]
    pub const fn from_byte(ch: u8) -> Self {
        match ch {
            b'-' => Sign::Negative,
            _ => Sign::Positive,
        }
    }
}

impl BitXor for Sign {
    type Output = Self;

    #[inline]
    fn bitxor(self, other: Self) -> Self {
        if self == other { Self::Positive }
        else { Self::Negative }
    }
}

impl BitXorAssign for Sign {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        if *self == other { *self = Self::Positive; }
        else { *self = Self::Negative; }
    }
}

/// Intermediate floating-point type used to perform arithmetic operations for the VAX
/// floating-point types.
///
#[doc = include_str!("doc/vax_ieee_754_diffs.doc")]
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct VaxFloatingPoint<T> {
    sign: Sign,
    exp: i32,
    frac: T,
    fault: Option<Fault>,
}

impl<T: Copy + PartialOrd> VaxFloatingPoint<T> {
    /// Create a new VaxFloatingPoint value from the component sign, exponent, and fraction.
    ///
    /// # Safety
    ///
    /// The fraction value `frac` must be normalized, meaning that the highest bit must be set. If
    /// the fraction is not normalized in, the the behavior of other functions is unpredictable.
    /// The only time the highest bit of the fraction will not be set is when the fraction is zero.
    ///
    /// You should use the `VaxFloatingPoint::new()` implementation first which panics if given an
    /// invalid fraction, and then switch to this once tested.
    pub(crate) const unsafe fn new_unchecked(sign: Sign, exp: i32, frac: T) -> Self {
        Self {
            sign,
            exp,
            frac,
            fault: None,
        }
    }

    /// Return the [`Sign`] of the `VaxFloatingPoint`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::{Sign, VaxFloatingPoint};
    /// assert_eq!(VaxFloatingPoint::<u32>::from_f32(-1.0).sign(), Sign::Negative);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(0.0).sign(), Sign::Positive);
    /// assert_eq!(VaxFloatingPoint::<u128>::from_f32(1.0).sign(), Sign::Positive);
    /// // Zero is always positive, because VaxFloatingPoint doesn't support negative zero.
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f64(-0.0).sign(), Sign::Positive);
    /// ```
    #[inline]
    pub const fn sign(&self) -> Sign { self.sign }

    /// Return the exponent of the `VaxFloatingPoint`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// assert_eq!(VaxFloatingPoint::<u32>::from_f32(-1.0).exponent(), 1);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(0.5).exponent(), 0);
    /// assert_eq!(VaxFloatingPoint::<u128>::from_f32(0.25).exponent(), -1);
    /// ```
    #[inline]
    pub const fn exponent(&self) -> i32 { self.exp }

    /// Return the fraction of the `VaxFloatingPoint`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// assert_eq!(VaxFloatingPoint::<u32>::from_f32(-1.0).fraction(), 1 << 31);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(0.5).fraction(), 1 << 63);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(0.0).fraction(), 0);
    /// assert_eq!(VaxFloatingPoint::<u128>::from_f32(0.25).fraction(), 1 << 127);
    /// ```
    #[inline]
    pub const fn fraction(&self) -> T { self.frac }

    /// Return the fault of the `VaxFloatingPoint`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::{Fault, VaxFloatingPoint};
    /// assert_eq!(VaxFloatingPoint::<u32>::from_f32(-1.0).fault(), None);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(f32::INFINITY).fault(),
    ///     Some(Fault::DivByZero));
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(f32::NAN).fault(),
    ///     Some(Fault::Reserved));
    /// assert_eq!(VaxFloatingPoint::<u128>::from_f32(0.25).fault(), None);
    /// ```
    #[inline]
    pub const fn fault(&self) -> Option<Fault> { self.fault }

    /// Return true if the fault is not None of the `VaxFloatingPoint`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// assert_eq!(VaxFloatingPoint::<u32>::from_f32(-1.0).is_error(), false);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(f32::INFINITY).is_error(), true);
    /// assert_eq!(VaxFloatingPoint::<u64>::from_f32(f32::NAN).is_error(), true);
    /// assert_eq!(VaxFloatingPoint::<u128>::from_f32(0.25).is_error(), false);
    /// ```
    #[inline]
    pub const fn is_error(&self) -> bool { self.fault.is_some() }

    /// Multiply the `VaxFloatingPoint` by 2<sup>shift</sup>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// for (start, shift, result) in [
    ///     (-1.0, -1, -0.5),
    ///     (0.5, 1, 1.0),
    ///     (2048.0, -11, 1.0),
    ///     (2.0, 10, 2048.0),
    /// ].iter() {
    ///     assert_eq!(VaxFloatingPoint::<u32>::from_f32(*start).shift_left(*shift),
    ///         VaxFloatingPoint::<u32>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u64>::from_f32(*start).shift_left(*shift),
    ///         VaxFloatingPoint::<u64>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u128>::from_f32(*start).shift_left(*shift),
    ///         VaxFloatingPoint::<u128>::from_f32(*result));
    /// }
    /// ```
    #[inline]
    pub const fn shift_left(mut self, shift: i32) -> Self {
        match self.exp.checked_add(shift) {
            Some(exp) => { self.exp = exp; }
            None => {
                (self.exp, self.fault) = Self::over_under(shift, false);
            }
        }
        self
    }

    /// Divide the `VaxFloatingPoint` by 2<sup>shift</sup>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// for (start, shift, result) in [
    ///     (-1.0, 1, -0.5),
    ///     (0.5, 1, 0.25),
    ///     (1024.0, 10, 1.0),
    /// ].iter() {
    ///     assert_eq!(VaxFloatingPoint::<u32>::from_f32(*start).shift_right(*shift),
    ///         VaxFloatingPoint::<u32>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u64>::from_f32(*start).shift_right(*shift),
    ///         VaxFloatingPoint::<u64>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u128>::from_f32(*start).shift_right(*shift),
    ///         VaxFloatingPoint::<u128>::from_f32(*result));
    /// }
    /// ```
    #[inline]
    pub const fn shift_right(mut self, shift: i32) -> Self {
        match self.exp.checked_sub(shift) {
            Some(exp) => { self.exp = exp; }
            None => {
                (self.exp, self.fault) = Self::over_under(shift, true);
            }
        }
        self
    }

    /// Multiply the `VaxFloatingPoint` by 2<sup>shift</sup>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// for (start, shift, result) in [
    ///     (-1.0, 1, -2.0),
    ///     (0.5, 1, 1.0),
    ///     (2.0, 10, 2048.0),
    /// ].iter() {
    ///     assert_eq!(VaxFloatingPoint::<u32>::from_f32(*start).shift_left_unsigned(*shift),
    ///         VaxFloatingPoint::<u32>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u64>::from_f32(*start).shift_left_unsigned(*shift),
    ///         VaxFloatingPoint::<u64>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u128>::from_f32(*start).shift_left_unsigned(*shift),
    ///         VaxFloatingPoint::<u128>::from_f32(*result));
    /// }
    /// ```
    #[inline]
    pub const fn shift_left_unsigned(mut self, shift: u32) -> Self {
        match self.exp.checked_add_unsigned(shift) {
            Some(exp) => { self.exp = exp; }
            None => {
                self.exp = i32::MAX;
                self.fault = Some(Fault::Overflow)
            }
        }
        self
    }

    /// Divide the `VaxFloatingPoint` by 2<sup>shift</sup>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use vax_floating::arithmetic::VaxFloatingPoint;
    /// for (start, shift, result) in [
    ///     (-1.0, 1, -0.5),
    ///     (0.5, 1, 0.25),
    ///     (1024.0, 10, 1.0),
    /// ].iter() {
    ///     assert_eq!(VaxFloatingPoint::<u32>::from_f32(*start).shift_right_unsigned(*shift),
    ///         VaxFloatingPoint::<u32>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u64>::from_f32(*start).shift_right_unsigned(*shift),
    ///         VaxFloatingPoint::<u64>::from_f32(*result));
    ///     assert_eq!(VaxFloatingPoint::<u128>::from_f32(*start).shift_right_unsigned(*shift),
    ///         VaxFloatingPoint::<u128>::from_f32(*result));
    /// }
    /// ```
    #[inline]
    pub const fn shift_right_unsigned(mut self, shift: u32) -> Self {
        match self.exp.checked_sub_unsigned(shift) {
            Some(exp) => { self.exp = exp; }
            None => {
                self.exp = i32::MIN;
                self.fault = Some(Fault::Underflow)
            }
        }
        self
    }

    /// Used internally to negate a floating point value. Must not be used on zero values.
    #[inline]
    const fn negate_inner(mut self) -> Self {
        self.sign = self.sign.negate();
        self
    }

    /// Used internally when a `checked_add` or `checked_sub` fails to determine an underflow
    /// or overflow.
    ///
    /// The `exp` parameter is the value added or subtracted that caused the underflow or overflow,
    /// and the `sub` parameter is true if the `checked_sub` was used instead of `checked_add`.
    ///
    /// It returns the value to set the `VaxFloatingPoint` `exp` and `fault` values.
    const fn over_under(exp: i32, sub: bool) -> (i32, Option<Fault>) {
        match sub {
            true => if exp > 0 {
                (i32::MIN, Some(Fault::Underflow))
            }
            else {
                (i32::MAX, Some(Fault::Overflow))
            }
            false => if exp > 0 {
                (i32::MAX, Some(Fault::Overflow))
            }
            else {
                (i32::MIN, Some(Fault::Underflow))
            }
        }
    }
}

impl<T: UpperHex> Debug for VaxFloatingPoint<T> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("VaxFloatingPoint")
            .field("sign", &self.sign)
            .field("exponent", &format_args!("{0}", self.exp))
            .field("frac", &format_args!("{:#X}", self.frac))
            .field("fault", &self.fault)
            .finish()
    }
}

/// Implement the `VaxFloatingPoint` type for a specific fraction size.
macro_rules! vfp_impl {
    (
        FracT = $ux: ident,
        SignedT = $ix: ident,
    ) => {
        impl VaxFloatingPoint<$ux> {
            #[doc = concat!("Mask with the most significant bit of the `VaxFloatingPoint<", stringify!($ux),
                ">` fraction value set.")]
            pub const FRAC_HIGH_BIT: $ux = 1 << (<$ux>::BITS - 1);
            #[doc = concat!("The precision value to use for raw `VaxFloatingPoint<",
                stringify!($ux), ">` values.")]
            const DIV_PRECISION: u32 = <$ux>::BITS;
            #[doc = concat!("The mask used by `from_ascii()` to determine when new digits won't ",
                "change the `VaxFloatingPoint<", stringify!($ux), ">` fraction value.")]
            const ASCII_MASK: $ux = <$ux>::MAX;
            /// Approximate number of significant digits in base 10.
            pub const DIGITS: u32 = {
                let value = <$ux>::MAX;
                value.ilog10()
            };
            #[doc = concat!("A `VaxFloatingPoint<", stringify!($ux), ">` equal to zero.")]
            pub const ZERO: Self = Self { sign: Sign::Positive, exp: 0, frac: 0, fault: None };

            #[doc = concat!("Create a new `VaxFloatingPoint<", stringify!($ux),
                ">` from the sign, exponent, and fraction.")]
            ///
            /// It expects a normalized the fraction value.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::{Sign, VaxFloatingPoint};
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            /// assert_eq!(VFP::new(Sign::Positive, 0, 0), VFP::ZERO);
            /// assert_eq!(VFP::new(Sign::Positive, 0, VFP::FRAC_HIGH_BIT), VFP::from_f32(0.5));
            /// assert_eq!(VFP::new(Sign::Positive, 1, VFP::FRAC_HIGH_BIT), VFP::from_f32(1.0));
            /// assert_eq!(VFP::new(Sign::Positive, -2, VFP::FRAC_HIGH_BIT), VFP::from_f32(0.125));
            /// ```
            pub const fn new(sign: Sign, exp: i32, frac: $ux) -> Self {
                debug_assert!((0 != (Self::FRAC_HIGH_BIT & frac)) || (0 == frac),
                    concat!("VaxFloatingPoint<", stringify!($ux),
                        ">::new called with a non-normalized fraction"));
                Self {
                    sign,
                    exp,
                    frac,
                    fault: None,
                }
            }

            #[doc = concat!("Create a new `VaxFloatingPoint<", stringify!($ux),
                ">` in a fault state.")]
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::{Fault, VaxFloatingPoint};
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            /// assert_eq!(VFP::from_fault(Fault::DivByZero).fault(), Some(Fault::DivByZero));
            /// assert_eq!(VFP::from_fault(Fault::Reserved).fault(), Some(Fault::Reserved));
            /// assert_eq!(VFP::from_fault(Fault::Overflow).fault(), Some(Fault::Overflow));
            /// assert_eq!(VFP::from_fault(Fault::Underflow).fault(), Some(Fault::Underflow));
            /// ```
            pub const fn from_fault(fault: Fault) -> Self {
                Self {
                    sign: Sign::Positive,
                    exp: 0,
                    frac: 0,
                    fault: Some(fault),
                }
            }

            /// Set the value of the `VaxFloatingPoint` to zero.
            #[inline]
            const fn to_zero(self) -> Self {
                Self {
                    sign: Sign::Positive,
                    exp: 0,
                    frac: 0,
                    fault: self.fault,
                }
            }

            /// Check to see if the value is zero.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            ///
            /// for case in [
            ///     1.0_f32,
            ///     123.456,
            ///     -300.0
            /// ].iter() {
            ///     assert_eq!(VFP::from_f32(*case).is_zero(), false);
            /// }
            /// assert_eq!(VFP::ZERO.is_zero(), true);
            /// ```
            #[inline]
            pub const fn is_zero(&self) -> bool {
                0 == self.frac
            }

            /// Normalize the fration portion of the `VaxFloatingPoint`.
            ///
            /// Floating point values are always normalized, meaning that the highest set bit is
            /// shifted to the most significant bit in the fraction, and the exponent is adjusted
            /// accordingly.
            const fn normalize(self) -> Self {
                if 0 == self.frac {
                    self.to_zero()
                }
                else {
                    let adjust = self.frac.leading_zeros();
                    let (exp, fault) = match self.exp.checked_sub(adjust as i32) {
                        Some(exp) => (exp, self.fault),
                        None => (i32::MIN, Some(Fault::Underflow)),
                    };
                    Self {
                        sign: self.sign,
                        exp,
                        frac: self.frac << adjust,
                        fault,
                    }
                }
            }

            /// Negate the `VaxFloatingPoint` value.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            ///
            /// for (case, neg) in [
            ///     (1.0, -1.0),
            ///     (-1.0, 1.0),
            ///     (123.456, -123.456),
            ///     (-300.0, 300.0),
            /// ].iter() {
            ///     assert_eq!(VFP::from_f32(*case).negate(), VFP::from_f32(*neg));
            /// }
            /// assert_eq!(VFP::ZERO.negate(), VFP::ZERO);
            /// ```
            #[inline]
            pub const fn negate(self) -> Self {
                if !self.is_zero() { self.negate_inner() } else {self}
            }

            /// Round the fractional portion of the `VaxFloatingPoint`.
            ///
            /// This is used internally to round up the fractional value. It is given the rounding
            /// value, which is a mask with the bit right below the mantissa digits for the
            /// floating point type we are rounding for. Done immediately before converting back
            /// to a floating point type with a smaller fractional value.
            pub(crate) const fn round_fraction(mut self, round: $ux) -> Self {
                self.frac = self.frac.wrapping_add(round);
                if 0 == Self::FRAC_HIGH_BIT & self.frac {
                    // If wrapping caused carry
                    self.frac = (self.frac >> 1) | Self::FRAC_HIGH_BIT;
                    match self.exp.checked_add(1) {
                        Some(exp) => { self.exp = exp; }
                        None => { self.fault = Some(Fault::Overflow); }
                    }
                }
                self
            }

            /// Add (or subtract) two `VaxFloatingPoint` numbers.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            ///
            /// for (a, b, add, sub) in [
            ///     (1.0, 0.0, 1.0, 1.0),
            ///     (0.0, 1.0, 1.0, -1.0),
            ///     (1.0, 1.0, 2.0, 0.0),
            ///     (2.5, 1.0, 3.5, 1.5),
            ///     (2.5, 3.0, 5.5, -0.5),
            /// ].iter() {
            ///     let fp1 = VFP::from_f32(*a);
            ///     let fp2 = VFP::from_f32(*b);
            ///     assert_eq!(fp1.add_to(fp2, false).to_f32(), *add);
            ///     assert_eq!(fp1.add_to(fp2, true).to_f32(), *sub);
            /// }
            /// ```
            pub const fn add_to(mut self, mut other: Self, sub: bool) -> Self {
                if self.fault.is_some() { return self; }
                if other.fault.is_some() { return other; }
                if other.is_zero() { return self; }
                if sub { other = other.negate() }
                if self.is_zero() { return other; }

                if self.exp < other.exp {
                    let temp = self;
                    self = other;
                    other = temp;
                }

                debug_assert!(other.exp <= self.exp);   // Pretty sure that this is impossible.
                let exp_diff = (self.exp - other.exp) as u32;

                if self.sign.xor(other.sign).is_negative() {
                    if 0 < exp_diff {
                        let frac = if <$ux>::BITS <= exp_diff {
                            <$ux>::MAX
                        }
                        else {
                            ((other.frac as $ix).wrapping_neg() >> exp_diff) as $ux | (<$ux>::MAX << (<$ux>::BITS - exp_diff))
                        };
                        self.frac = self.frac.wrapping_add(frac);
                    }
                    else {
                        if self.frac < other.frac {
                            self.frac = other.frac - self.frac;
                            self.sign = other.sign;
                        }
                        else {
                            self.frac -= other.frac;
                        }
                    }
                    self.normalize()
                }
                else {
                    let frac = if <$ux>::BITS <= exp_diff {
                        0
                    }
                    else {
                        other.frac >> exp_diff
                    };
                    self.frac = self.frac.wrapping_add(frac);
                    if self.frac < frac {
                        self.frac = (1 << (<$ux>::BITS - 1)) | (self.frac >> 1);
                        match self.exp.checked_add(1) {
                            Some(exp) => { self.exp = exp; }
                            None => { self.fault = Some(Fault::Overflow); }
                        }
                    }
                    self
                }
            }

            /// Multiply two `VaxFloatingPoint` numbers.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            ///
            /// for (a, b, quo) in [
            ///     (1.0, 0.0, 0.0),
            ///     (0.0, 1.0, 0.0),
            ///     (1.0, 1.0, 1.0),
            ///     (2.5, -2.0, -5.0),
            ///     (2.5, 3.0, 7.5),
            ///     (10.0, 0.1, 1.0),
            /// ].iter() {
            ///     let fp1 = VFP::from_f32(*a);
            ///     let fp2 = VFP::from_f32(*b);
            ///     assert_eq!(fp1.multiply_by(fp2).to_f32(), *quo);
            /// }
            /// ```
            pub const fn multiply_by(
                mut self,
                multiplier: Self,
            ) -> Self {
                if self.fault.is_some() { return self; }
                if multiplier.fault.is_some() { return multiplier; }
                if self.is_zero() || multiplier.is_zero() { return self.to_zero(); }

                self.sign = self.sign.xor(multiplier.sign);
                self.exp = match self.exp.checked_add(multiplier.exp) {
                    Some(exp) => exp,
                    None => {
                        let (exp, fault) = Self::over_under(multiplier.exp, false);
                        self.fault = fault;
                        exp
                    }
                };
                multiply_impl!(self, multiplier, $ux);
                self.normalize()
            }

            /// Divide `VaxFloatingPoint` number by another.
            ///
            /// The `precision` parameter is the number of bits to perform the division on. Under
            /// normal circumstances, it should be 2 more than the number of mantissa digits for
            /// rounding when converted back into the original floating-point type.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            ///
            /// for (a, b, div) in [
            ///     (0.0, 1.0, 0.0),
            ///     (1.0, 10.0, 0.1),
            ///     (2.5, -2.0, -1.25),
            ///     (-2.5, 0.5, -5.0),
            ///     (-10.0, -5.0, 2.0),
            /// ].iter() {
            ///     let fp1 = VFP::from_f32(*a);
            ///     let fp2 = VFP::from_f32(*b);
            #[doc = concat!("    assert_eq!(fp1.divide_by(fp2, ", stringify!($ux),
                "::BITS).to_f32(), *div);")]
            /// }
            /// ```
            pub const fn divide_by(
                mut self,
                denominator: Self,
                precision: u32,
            ) -> Self {
                if self.is_error() { return self; }
                if denominator.is_error() { return denominator; }
                if denominator.is_zero() { self.fault = Some(Fault::DivByZero); return self; }

                self.sign = self.sign.xor(denominator.sign);
                self.exp = match self.exp.checked_sub(denominator.exp) {
                    Some(exp_minus_one) => {
                        match exp_minus_one.checked_add(1) {
                            Some(exp) => exp,
                            None => {
                                self.fault = Some(Fault::Overflow);
                                i32::MAX
                            }
                        }
                    }
                    None => {
                        let (exp, fault) = Self::over_under(denominator.exp, true);
                        self.fault = fault;
                        exp
                    }
                };
                let denom = denominator.frac >> 1;
                let mut numer = self.frac >> 1;

                let mut quotient = 0;
                let mut i = 0;
                while i < precision {
                    quotient <<= 1;
                    if numer >= denom {
                        numer -= denom;
                        quotient += 1;
                    }
                    numer <<= 1;
                    i += 1;
                }

                self.frac = quotient << (<$ux>::BITS - i);
                self.normalize()
            }

            /// Returns the whole number portion of the floating-point number and removes it.
            fn pop_whole(&mut self) -> $ux {
                if self.exp <= 0 { return 0; }
                debug_assert!(self.exp > 0);
                let adjust = self.exp as u32;
                let whole = self.frac >> (<$ux>::BITS - adjust);
                self.frac <<= adjust;
                self.exp -= adjust as i32;
                *self = self.normalize();
                whole
            }

            /// Convert the floating point data into the printable string.
            ///
            /// It returns an integer value with the displayable digits and the tens exponent.
            fn float_to_decimal_shared(&self, mantissa: u32) -> ($ux, i32) {
                const TEN: VaxFloatingPoint<$ux> = VaxFloatingPoint::<$ux>::from_u8(10);
                const TENTH: VaxFloatingPoint<$ux> = VaxFloatingPoint::<$ux>::from_u8(1)
                    .divide_by(TEN, <$ux>::BITS);
                let limit = (((1 as $ux) << mantissa)/10)-10;

                if self.is_zero() { (0, 0) }
                else {
                    let mut tens: i32;
                    let mut temp = *self;
                    let mut number = 0;
                    let mut last_digit = 0;
                    if (0 < self.exp) && ((mantissa as i32) >= self.exp) {
                        number = temp.pop_whole();
                        // Check for no fraction.
                        if temp.is_zero() { return (number, 0); }
                        temp *= TEN;
                        tens = 0;
                        while (number < limit) && !temp.is_zero() {
                            last_digit = temp.pop_whole();
                            number = (number * 10) + last_digit;
                            temp *= TEN;
                            tens -= 1;
                        }
                    }
                    else {
                        tens = 1;
                        // Find the most significant decimal digit.
                        while temp.exp > 0 {    // If the exponent is positive, this will find the
                            temp *= TENTH;
                            tens += 1;
                        }
                        while temp.exp <= 0 {
                            temp *= TEN;
                            tens -= 1;
                        }
                        // Now we have the first digit in front of the decimal place.
                        while (number < limit) && !temp.is_zero() {
                            last_digit = temp.pop_whole();
                            number = (number * 10) + last_digit;
                            temp *= TEN;
                            tens -= 1;
                        }
                    }
                    // Round the number based on the last digit and the digit after the last one.
                    let next_digit = temp.pop_whole();
                    if ((9 == last_digit) && (6 <= next_digit)) || (5 <= next_digit) {
                        number += 1;
                    }
                    while 0 == (number % 10) {
                        number /= 10;
                        tens += 1;
                    }
                    (number, tens)
                }
            }

            #[doc = concat!("The implementation of [`Display`] for the `VaxFloatingPoint<",
                stringify!($ux), ">` type.")]
            pub(crate) fn float_to_decimal_display(&self, fmt: &mut Formatter<'_>, mantissa: u32) -> fmt::Result
            {
                if self.is_error() {
                    match self.fault.unwrap() {
                        Fault::DivByZero => fmt.write_str("Infinity"),
                        Fault::Reserved => fmt.write_str("Reserved"),
                        Fault::Overflow => fmt.write_str("Overflowed"),
                        Fault::Underflow => fmt.write_str("Underflowed"),
                    }
                }
                else {
                    let sign = self.sign.as_str(fmt.sign_plus());
                    let (mut number, mut tens) = self.float_to_decimal_shared(mantissa);
                    let mut width = if 0 != number { (number.ilog10() as i32) + 1 } else { 1 };
                    let mut zeroes = 0;
                    if let Some(precision) = fmt.precision() {
                        let change = tens.saturating_add_unsigned(precision as u32);
                        if 0 > change {
                            if change.abs() > width {
                                width = 1;
                                number = 0;
                                tens = 0;
                                zeroes = precision;
                            }
                            else if change.abs() == width {
                                let divisor: $ux = 5 * (10 as $ux).pow((width as u32) - 1);
                                number = ((number / divisor) + 1) >> 1;
                                tens -= change;
                                width = 1;
                                if 0 == number {
                                    tens = 0;
                                    zeroes = precision;
                                }
                            }
                            else {
                                let divisor: $ux = 5 * (10 as $ux).pow(change.unsigned_abs() - 1);
                                number = ((number / divisor) + 1) >> 1;
                                tens -= change;
                                width += change;
                                // If rounded number increases width.
                                let new_width = if 0 != number { (number.ilog10() as i32) + 1 } else { 1 };
                                if width != new_width {
                                    number /= 10;
                                    tens += 1;
                                    if tens > -(precision as i32) {
                                        zeroes = min(precision.saturating_add_signed(tens as isize), precision);
                                    }
                                }
                            }
                        }
                        else if 0 < change {
                            zeroes = min(change as usize, precision);
                        }
                    }
                    let tens_start = tens + width;
                    let old_number = format!("{}", number);
                    let tens_start_old = (old_number.len() as i32) + tens;
                    assert_eq!(tens_start_old, tens_start, "tens_start doesn't match!: tens = {}; width = {}, old_number = {:?}, tens_start_old = {}", tens, width, old_number, tens_start_old);
                    if 0 >= tens_start {    // Decimal point before number (0.<0>number)
                        if (0 == number) && (0 == zeroes) {
                            write!(fmt, "{0}0", sign)
                        }
                        else {
                            write!(fmt, "{0}0.{1:0>2$}{4:0>3$}", sign, number, tens.abs() as usize, zeroes, "")
                        }
                    }
                    else if 0 > tens {     // Decimal point in number
                        let num_str = format!("{}", number);
                        let (before, after) = num_str.split_at(tens_start as usize);
                        write!(fmt, "{0}{1}.{2}{4:0>3$}", sign, before, after, zeroes, "")
                    }
                    else if 0 == tens {     // Just number
                        if 0 == zeroes {
                            write!(fmt, "{}{}", sign, number)
                        }
                        else {
                             write!(fmt, "{0}{1}.{3:0<2$}", sign, number, zeroes, "")
                        }
                    }
                    else {                  // Number with trailing zeroes
                        if 0 == zeroes {
                            write!(fmt, "{0}{1}{3:0<2$}", sign, number, tens as usize, "")
                        }
                        else {
                            write!(fmt, "{0}{1}{4:0<2$}.{4:0<3$}", sign, number, tens as usize,
                                zeroes, "")
                        }
                    }
                }
            }

            // Common code of floating point LowerExp and UpperExp.
            #[doc = concat!("Common code for the [`LowerExp`] and [`UpperExp`] traits for the `VaxFloatingPoint<",
                stringify!($ux), ">` type.")]
            pub(crate) fn float_to_exponential_common(
                &self,
                fmt: &mut Formatter<'_>,
                mantissa: u32,
                upper: bool,
            ) -> fmt::Result {
                if self.is_error() {
                    match self.fault.unwrap() {
                        Fault::DivByZero => fmt.write_str("Infinity"),
                        Fault::Reserved => fmt.write_str("Reserved"),
                        Fault::Overflow => fmt.write_str("Overflowed"),
                        Fault::Underflow => fmt.write_str("Underflowed"),
                    }
                }
                else {
                    let sign = self.sign.as_str(fmt.sign_plus());
                    let (mut number, mut tens) = self.float_to_decimal_shared(mantissa);
                    let mut width = if 0 != number { (number.ilog10() as i32) } else { 0 };
                    let mut zeroes = 0;
                    if let Some(precision) = fmt.precision() {
                        let change = (precision as i32) - width;
                        if 0 > change {
                            let divisor: $ux = 5 * (10 as $ux).pow(change.unsigned_abs() - 1);
                            number = ((number / divisor) + 1) >> 1;
                            width += change;
                            tens -= change;
                            // If rounded number increases width.
                            let new_width = if 0 != number { (number.ilog10() as i32) } else { 0 };
                            if width != new_width {
                                number /= 10;
                                tens += 1;
                            }
                        }
                        else if 0 < change {
                            zeroes = change as usize;
                        }
                    }
                    let exp = tens + width;
                    let number = format!("{}", number);
                    if (0 == width) && (0 == zeroes) {
                        write!(fmt, "{}{}{}{}", sign, number, if upper {"E"} else {"e"}, exp)
                    }
                    else {
                        let (before, after) = number.split_at(1);
                        write!(fmt, "{0}{1}.{2}{6:0>5$}{3}{4}", sign, before, after,
                            if upper {"E"} else {"e"}, exp, zeroes, "")
                    }
                }
            }

            #[doc = concat!("Convert from a string slice to a `VaxFloatingPoint<", stringify!($ux),
                ">`. A constant version of [`FromStr::from_str`].")]
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            #[doc = concat!("type VFP = VaxFloatingPoint<", stringify!($ux), ">;")]
            ///
            #[doc = concat!("let F32_MASK: ", stringify!($ux), " = 0x1FFFFFF << (<",
                stringify!($ux), ">::BITS - f32::MANTISSA_DIGITS - 1);")]
            /// let TEN: VFP = VFP::from_ascii("10", F32_MASK).unwrap();
            /// assert_eq!(TEN.to_f32(), 10.0_f32);
            ///
            /// for (text, value) in [
            ///     ("0", 0.0),
            ///     ("-0", 0.0),
            ///     ("+0.0e0", 0.0),
            ///     ("2.5", 2.5),
            ///     ("1e1", 10.0),
            ///     ("-10E-1", -1.0),
            ///     ("1,234,567", 1234567.0),
            ///     ("1.75", 1.75),
            ///     ("+175e-2", 1.75),
            ///     ("123_456_789_123_456_789_123_456_789", 123456789123456789123456789.0),
            ///     ("0.123456789123456789123456789", 0.123456789123456789123456789),
            ///     ("1.23e37", 1.23e37),
            ///     ("1.23e-36", 1.23e-36),
            /// ].iter() {
            ///     assert_eq!(VFP::from_ascii(text, F32_MASK).unwrap().to_f32(), *value);
            /// }
            ///
            /// for err_text in [
            ///     "e",
            ///     "e10",
            ///     "1-0",
            ///     "1.2.3",
            ///     "1e2e3",
            ///     "102e3-",
            /// ].iter() {
            ///     assert!(VFP::from_ascii(err_text, F32_MASK).is_err());
            /// }
            /// ```
            ///
            #[doc = concat!("[`FromStr`] cannot be used to define constants.")]
            ///
            /// ```compile_fail
            /// # use vax_floating::arithmetic::VaxFloatingPoint;
            /// # use std::str::FromStr;
            #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux),
                "> = VaxFloatingPoint::<", stringify!($ux), ">::from_str(\"0\").unwrap();")]
            /// ```
            pub const fn from_ascii(text: &str, mask: $ux) -> std::result::Result<Self, &str> {
                enum State {
                    Start,
                    BeforePoint,
                    AfterPoint,
                    Insignificant,
                    InsignificantAP, // Insignificant After Point
                    AfterExp,
                    ExpDigits,
                }
                use State::*;
                const ONE: VaxFloatingPoint<$ux> = VaxFloatingPoint::<$ux>::from_u8(1);
                const TEN: VaxFloatingPoint<$ux> = VaxFloatingPoint::<$ux>::from_u8(10);
                const TENTH: VaxFloatingPoint<$ux> =
                    ONE.divide_by(TEN, VaxFloatingPoint::<$ux>::DIV_PRECISION);
                let bytes = text.as_bytes();
                let mut result = Self::ZERO;
                let mut state = Start;
                let mut sign = Sign::Positive;
                let mut multiplier = ONE;
                let mut tens_sign = Sign::Positive;
                let mut tens = 0;
                let mut i = 0;
                while i < bytes.len() {
                    match bytes[i] {
                        b'-' | b'+' => match state {
                            Start => {
                                sign = Sign::from_byte(bytes[i]);
                                state = BeforePoint;
                            }
                            AfterExp => {
                                tens_sign = Sign::from_byte(bytes[i]);
                                state = ExpDigits;
                            }
                            _ => { return Err(text); }
                        }
                        b'.' => match state {
                            Start | BeforePoint => {
                                state = AfterPoint;
                            }
                            Insignificant => {
                                state = InsignificantAP;
                            }
                            _ => { return Err(text); }
                        }
                        b'e' | b'E' => match state {
                            BeforePoint | AfterPoint | Insignificant | InsignificantAP => {
                                state = AfterExp;
                            }
                            _ => { return Err(text); }
                        }
                        b'0'..=b'9' => {
                            match state {
                                Start => {
                                    result = result.add_to(VaxFloatingPoint::<$ux>::from_u8(bytes[i] - b'0'), false);
                                    state = BeforePoint;
                                }
                                BeforePoint => {
                                    result = result.multiply_by(TEN);
                                    let last_frac = result.frac & mask;
                                    result = result.add_to(Self::from_u8(bytes[i] - b'0'), false);
                                    if (b'0' != bytes[i]) && (last_frac == (result.frac & mask)) {
                                        state = Insignificant;
                                    }
                                }
                                AfterPoint => {
                                    let last_frac = result.frac & mask;
                                    multiplier = multiplier.multiply_by(TENTH);
                                    result = result
                                        .add_to(VaxFloatingPoint::<$ux>::from_u8(bytes[i] - b'0')
                                        .multiply_by(multiplier), false);
                                    if (b'0' != bytes[i]) && (last_frac == (result.frac & mask)) {
                                        state = InsignificantAP;
                                    }
                                }
                                Insignificant => {
                                    result = result.multiply_by(TEN);
                                }
                                InsignificantAP => {}
                                AfterExp => {
                                    tens = (bytes[i] - b'0') as usize;
                                    state = ExpDigits;
                                }
                                ExpDigits => {
                                    tens = (tens * 10) + ((bytes[i] - b'0') as usize);
                                }
                            }
                        }
                        b',' | b'_' => {}
                        _ => { return Err(text); }
                    }
                    i += 1;
                }
                if sign.is_negative() { result = result.negate(); }
                if tens != 0 && !result.is_zero() {
                    if tens_sign.is_negative() {
                        while 0 < tens {
                            result = result.multiply_by(TENTH);
                            tens -= 1;
                        }
                    }
                    else {
                        while 0 < tens {
                            result = result.multiply_by(TEN);
                            tens -= 1;
                        }
                    }
                }
                Ok(result)
            }

            from_rust_int_impl!($ux);

            to_from_rust_fp_impl!($ux);
        }

        from_rust_int_impl!(From, $ux);

        to_from_rust_fp_impl!(From, $ux);

        impl PartialOrd for VaxFloatingPoint<$ux> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                use Ordering::*;
                if self.fault.is_some() || other.fault.is_some() { return None; }
                match (self.is_zero(), other.is_zero()) {
                    (true, true) => Some(Equal),
                    (false, true) => match self.sign {
                        Sign::Positive => Some(Greater),
                        Sign::Negative => Some(Less),
                    }
                    (true, false) => match other.sign {
                        Sign::Positive => Some(Less),
                        Sign::Negative => Some(Greater),
                    }
                    (false, false) => {
                        if self.sign ^ other.sign == Sign::Negative {
                            match self.sign {
                                Sign::Positive => Some(Greater),
                                Sign::Negative => Some(Less),
                            }
                        }
                        else {
                            match self.sign {
                                Sign::Positive => {
                                    match self.exp.partial_cmp(&other.exp) {
                                        Some(Equal) => self.frac.partial_cmp(&other.frac),
                                        otherwise => otherwise,
                                    }
                                }
                                Sign::Negative => {
                                    match other.exp.partial_cmp(&self.exp) {
                                        Some(Equal) => other.frac.partial_cmp(&self.frac),
                                        otherwise => otherwise,
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        impl Add for VaxFloatingPoint<$ux> {
            type Output = VaxFloatingPoint<$ux>;

            fn add(self, rhs: Self) -> Self::Output {
                self.add_to(rhs, false)
            }
        }
        forward_ref_binop!(impl Add, add for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux>);

        impl AddAssign for VaxFloatingPoint<$ux> {
            #[inline]
            fn add_assign(&mut self, other: VaxFloatingPoint<$ux>) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl AddAssign, add_assign for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux> }

        impl Sub for VaxFloatingPoint<$ux> {
            type Output = VaxFloatingPoint<$ux>;

            fn sub(self, rhs: Self) -> Self::Output {
                self.add_to(rhs, true)
            }
        }
        forward_ref_binop!(impl Sub, sub for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux>);

        impl SubAssign for VaxFloatingPoint<$ux> {
            #[inline]
            fn sub_assign(&mut self, other: VaxFloatingPoint<$ux>) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl SubAssign, sub_assign for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux> }

        impl Div for VaxFloatingPoint<$ux> {
            type Output = VaxFloatingPoint<$ux>;

            fn div(self, rhs: Self) -> Self::Output {
                self.divide_by(rhs, Self::DIV_PRECISION)
            }
        }
        forward_ref_binop!(impl Div, div for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux>);

        impl DivAssign for VaxFloatingPoint<$ux> {
            #[inline]
            fn div_assign(&mut self, other: VaxFloatingPoint<$ux>) {
                *self = *self / other;
            }
        }
        forward_ref_op_assign! { impl DivAssign, div_assign for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux> }

        impl Mul for VaxFloatingPoint<$ux> {
            type Output = VaxFloatingPoint<$ux>;

            fn mul(self, rhs: Self) -> Self::Output {
                self.multiply_by(rhs)
            }
        }
        forward_ref_binop!(impl Mul, mul for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux>);

        impl MulAssign for VaxFloatingPoint<$ux> {
            #[inline]
            fn mul_assign(&mut self, other: VaxFloatingPoint<$ux>) {
                *self = *self * other;
            }
        }
        forward_ref_op_assign! { impl MulAssign, mul_assign for VaxFloatingPoint<$ux>, VaxFloatingPoint<$ux> }

        impl FromStr for VaxFloatingPoint<$ux> {
            type Err = Error;

            fn from_str(s: &str) -> Result<Self> {
                Ok(Self::from_ascii(s, Self::ASCII_MASK)?)
            }
        }

        impl Display for VaxFloatingPoint<$ux> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                self.float_to_decimal_display(f, <$ux>::BITS)
            }
        }

        impl LowerExp for VaxFloatingPoint<$ux> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                self.float_to_exponential_common(f, <$ux>::BITS, false)
            }
        }

        impl UpperExp for VaxFloatingPoint<$ux> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                self.float_to_exponential_common(f, <$ux>::BITS, true)
            }
        }
    };
}

/// Implement multiply_by for a specific size of fraction.
///
/// It performs a multiply with overflow and returns the overflow. For integer types that have an
/// equivalent with double the bits, it will multiply the two numbers converted into the larger
/// integer, and then return the high half of the larger integer. For example, multiplying a u32
/// would convert the two u32 values into u64 values, multiply them, and then take the high 32-bits
/// of the u64 and convert that back into a u32 value.
///
/// For integer types that don't have a bigger type, we perform a more complicated multiplication
/// using half of the integer type and performing multiple multiplications and additions.
macro_rules! multiply_impl {
    ($a: ident, $b: ident, u32) => {multiply_impl!($a, $b, u32, u64)};
    ($a: ident, $b: ident, u64) => {multiply_impl!($a, $b, u64, u128)};
    ($a: ident, $b: ident, $ux: ident) => {
        const HALF_BITS: u32 = $ux::BITS / 2;
        const HALF_MASK: $ux = (1 << HALF_BITS) - 1;
        let ah = ($a.frac >> HALF_BITS) & HALF_MASK;
        let bh = ($b.frac >> HALF_BITS) & HALF_MASK;
        let al = $a.frac & HALF_MASK;
        let bl = $b.frac & HALF_MASK;
        $a.frac = ah * bh;
        let mut mid1 = ah * bl;
        let mut mid2 = al * bh;
        let rlo = al * bl;
        $a.frac += ((mid1 >> HALF_BITS) & HALF_MASK) + ((mid2 >> HALF_BITS) & HALF_MASK);
        mid1 = rlo.wrapping_add(mid1 << HALF_BITS);
        if mid1 < rlo { $a.frac += 1; }
        mid2 = mid1.wrapping_add(mid2 << HALF_BITS);
        if mid2 < mid1 { $a.frac += 1; }
    };
    ($a: ident, $b: ident, $ux: ty, $uy: ty) => {
        let ah = $a.frac as $uy;
        let bh = $b.frac as $uy;
        $a.frac = ((ah * bh) >> <$ux>::BITS) as $ux;
    };
}

/// Implement the functions that support converting to and from Rust floating point types (`f32` and
/// `f64`).
///
/// This creates the constant functions 'to_f32()` and `from_f32()` for all `VaxFloatingPoint`
/// types, and `to_f64()` and 'from_f64()` for `VaxFloatingPoint` types with a large enough
/// fraction. USAGE: `to_from_rust_fp_impl!(<unsigned type of fraction>);`
///
/// This also creates the implementations for `From<f32>` for all `VaxFloatingPoint` types and
/// `From<f64>` for `VaxFloatingPoint` types with a large enough fraction. It also creates the
/// inverse `From<VaxFloatingPoint>` for `f32` and `f64`. USAGE:
/// `to_from_rust_fp_impl!(From, <unsigned type of fraction>);`
macro_rules! to_from_rust_fp_impl {
    (u32) => {
        to_from_rust_fp_impl!(f32, u32, u32, 8, from_f32, to_f32);
    };
    ($ux: ident) => {
        to_from_rust_fp_impl!(f32, $ux, u32, 8, from_f32, to_f32);
        to_from_rust_fp_impl!(f64, $ux, u64, 11, from_f64, to_f64);
    };
    (From, u32) => {
        to_from_rust_fp_impl!(From, f32, u32, u32, 8, from_f32, to_f32);
    };
    (From, $ux: ident) => {
        to_from_rust_fp_impl!(From, f32, $ux, u32, 8, from_f32, to_f32);
        to_from_rust_fp_impl!(From, f64, $ux, u64, 11, from_f64, to_f64);
    };
    ($fx: ident, $ux: ident, $fu: ident, $exp: literal, $from_fx: ident, $to_fx: ident) => {
        #[doc = concat!("Convert from [`", stringify!($fx), "`] to a `VaxFloatingPoint<",
            stringify!($ux), ">`.")]
        ///
        /// Can be used to define constants.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::",
            stringify!($from_fx), "(0_", stringify!($fx), ");")]
        #[doc = concat!("const TEN: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::",
            stringify!($from_fx), "(10.0);")]
        #[doc = concat!("assert_eq!(VaxFloatingPoint::<", stringify!($ux), ">::from_u8(0), ZERO);")]
        #[doc = concat!("assert_eq!(VaxFloatingPoint::<", stringify!($ux), ">::from_u8(10), TEN);")]
        /// ```
        ///
        #[doc = concat!("`From<", stringify!($fx), ">` cannot be used to define constants.")]
        ///
        /// ```compile_fail
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::from(0_", stringify!($fx), ");")]
        /// ```
        pub const fn $from_fx(fp: $fx) -> Self {
            // Constants used to convert from f(32/64).
            const SIGN: $fu = 1 << (<$fu>::BITS - 1);
            const EXP_SHIFT: u32 = <$fu>::BITS - $exp - 1;
            const EXP_MASK: $fu = (1 << $exp) - 1;
            const EXP_BIAS: i32 = (1 << ($exp - 1)) - 2;
            const FRAC_MASK: $fu = (1 << EXP_SHIFT) - 1;

            // SAFETY: According to the documentation for f(32/64)::to_bits, it is identical to
            // `transmute::<f(32/64), u(32/64)>(self)` on all platforms. f(32/64)::to_bits is not
            // const.
            let bits = unsafe { std::mem::transmute::<$fx, $fu>(fp) };
            if 0 == (bits & !SIGN) { return Self::ZERO; }
            let sign = if 0 == (bits & SIGN) { Sign::Positive } else { Sign::Negative };
            match (bits >> EXP_SHIFT) & EXP_MASK {
                // Infinity or NaN
                EXP_MASK => {
                    let mut frac =
                        ((bits & FRAC_MASK) as $ux) << (<$ux>::BITS - $fx::MANTISSA_DIGITS);
                    let fault = if 0 == frac {
                        Some(Fault::DivByZero)
                    }
                    else {
                        frac |= Self::FRAC_HIGH_BIT;
                        Some(Fault::Reserved)
                    };
                    Self {
                        sign,
                        exp: 0,
                        frac,
                        fault,
                    }
                },
                // Subnormal numbers
                0 => Self {
                    sign,
                    exp: $fx::MIN_EXP as i32,
                    frac: ((bits & FRAC_MASK) as $ux) << (<$ux>::BITS - $fx::MANTISSA_DIGITS),
                    fault: None,
                }.normalize(),
                exp => Self {
                    sign,
                    exp: (exp as i32) - EXP_BIAS,
                    frac: Self::FRAC_HIGH_BIT |
                        (((bits & FRAC_MASK) as $ux) << (<$ux>::BITS - $fx::MANTISSA_DIGITS)),
                    fault: None,
                },
            }
        }

        #[doc = concat!("Convert from a `VaxFloatingPoint<", stringify!($ux), ">` to a [`",
            stringify!($fx), "`].")]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<",
            stringify!($ux), ">::from_u8(0);")]
        #[doc = concat!("const TEN: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<",
            stringify!($ux), ">::from_u8(10);")]
        #[doc = concat!("assert_eq!(ZERO.", stringify!($to_fx), "(), 0_", stringify!($fx), ");")]
        #[doc = concat!("assert_eq!(TEN.", stringify!($to_fx), "(), 10_", stringify!($fx), ");")]
        /// ```
        pub const fn $to_fx(mut self) -> $fx {
            // Constants used to convert to f(32/64).
            const SIGN: $fu = 1 << (<$fu>::BITS - 1);
            const EXP_SHIFT: u32 = <$fu>::BITS - $exp - 1;
            const EXP_MASK: $fu = (1 << $exp) - 1;
            const EXP_BIAS: i32 = (1 << ($exp - 1)) - 2;
            const FRAC_MASK: $fu = (1 << EXP_SHIFT) - 1;
            const MAX_NEGATIVE: $fx = unsafe { std::mem::transmute::<$fu, $fx>(SIGN+1) };
            const FRAC_SHIFT: u32 = <$ux>::BITS - $fx::MANTISSA_DIGITS;
            const ROUND: $ux = 1 << (FRAC_SHIFT - 1);
            const HIGH_BIT: $ux = 1 << (<$ux>::BITS - 1);

            match self.fault {
                Some(Fault::DivByZero) => match self.sign {
                    Sign::Positive => $fx::INFINITY,
                    Sign::Negative => $fx::NEG_INFINITY,
                }
                Some(Fault::Reserved) | Some(Fault::Overflow) | Some(Fault::Underflow) => $fx::NAN,
                None => {
                    if self.is_zero() { 0.0 }
                    else if $fx::MAX_EXP < (self.exp as i32) {
                        // If the exponent is too large, set to maximum.
                        match self.sign {
                            Sign::Positive => $fx::MAX,
                            Sign::Negative => $fx::MIN,
                        }
                    }
                    else {
                        if $fx::MIN_EXP > (self.exp as i32) { // Could be submormal
                            // Un-normalize the fraction for subnormal numbers.
                            let adjust = ($fx::MIN_EXP as i32) - self.exp;
                            if adjust >= ($fx::MANTISSA_DIGITS as i32) {
                                match self.sign {
                                    Sign::Positive => { return $fx::MIN_POSITIVE; }
                                    Sign::Negative => { return MAX_NEGATIVE; }
                                }
                            }
                            else {
                                debug_assert!(adjust > 0);
                                self.exp += adjust;
                                self.frac >>= adjust;
                                self.frac += ROUND;
                            }
                        }
                        else {
                            // Round the fraction
                            self.frac = self.frac.wrapping_add(ROUND);
                            if 0 == HIGH_BIT & self.frac {
                                // If wrapping caused carry
                                self.frac >>= 1;
                                self.exp += 1;
                                // We aren't worried about the high bit, because it gets striped
                                // out anyway.
                            }

                        }
                        let bits = (if self.sign.is_negative() { SIGN } else { 0 }) |
                            ((((self.exp + EXP_BIAS) as $fu) & EXP_MASK) << EXP_SHIFT) |
                            (((self.frac >> FRAC_SHIFT) as $fu) & FRAC_MASK);
                        // SAFETY: According to the documentation for f(32/64)::from_bits, it is
                        // identical to `transmute::<u(32/64), f(32/64)>(self)` on all platforms.
                        // f(32/64)::from_bits is not const.
                        unsafe { std::mem::transmute::<$fu, $fx>(bits) }
                    }
                }
            }
        }
    };
    (From, $fx: ident, $ux: ident, $fu: ident, $exp: literal, $from_fx: ident, $to_fx: ident) => {
        impl From<VaxFloatingPoint<$ux>> for $fx {
            fn from(vfp: VaxFloatingPoint<$ux>) -> Self {
                vfp.$to_fx()
            }
        }

        impl From<&VaxFloatingPoint<$ux>> for $fx {
            fn from(vfp: &VaxFloatingPoint<$ux>) -> Self {
                vfp.$to_fx()
            }
        }

        impl From<$fx> for VaxFloatingPoint<$ux> {
            fn from(fp: $fx) -> Self {
                VaxFloatingPoint::<$ux>::$from_fx(fp)
            }
        }

        impl From<&$fx> for VaxFloatingPoint<$ux> {
            fn from(fp: &$fx) -> Self {
                VaxFloatingPoint::<$ux>::$from_fx(*fp)
            }
        }
    };
}

/// The documentation to display for lossy `from_*()` and `From<*>` conversions to
/// `VaxFloatingPoint`.
macro_rules! from_int_lossy_doc {
    ($ux: ident) => {
        concat!("**Note**: Only the most significant set bits that fit into the number of [`",
            stringify!($ux), "::BITS`] will be preserved. This will result in a loss
            of precision.")
    };
}

/// Implement the functions that support converting from Rust integer types.
///
/// This creates the constant functions `from_<type>()` for all integer types that are smaller than
/// the fraction size of the `VaxFloatingPoint` type.
/// USAGE: `from_rust_int_impl!(<unsigned type of fraction>, <VAX FP Struct>);`
///
/// This also creates the implementations for `From` for all `VaxFloatingPoint` types and
/// `From<f64>` for `VaxFloatingPoint` types with a large enough fraction.
/// USAGE: `from_rust_int_impl!(From, <unsigned type of fraction>, <VAX FP Type>);`
macro_rules! from_rust_int_impl {
    (u32) => {
        from_rust_int_impl!(to_u32, u32);
        from_rust_int_impl!(lossy_u64, u32);
        from_rust_int_impl!(lossy_u128, u32);
    };
    (u64) => {
        from_rust_int_impl!(to_u32, u64);
        from_rust_int_impl!(to_u64, u64);
        from_rust_int_impl!(lossy_u128, u64);
    };
    (u128) => {
        from_rust_int_impl!(to_u32, u128);
        from_rust_int_impl!(to_u64, u128);
        from_rust_int_impl!(to_u128, u128);
    };
    (From, u32) => {
        from_rust_int_impl!(From, to_u32, u32);
        from_rust_int_impl!(From, lossy_u64, u32);
        from_rust_int_impl!(From, lossy_u128, u32);
    };
    (From, u64) => {
        from_rust_int_impl!(From, to_u32, u64);
        from_rust_int_impl!(From, to_u64, u64);
        from_rust_int_impl!(From, lossy_u128, u64);
    };
    (From, u128) => {
        from_rust_int_impl!(From, to_u32, u128);
        from_rust_int_impl!(From, to_u64, u128);
        from_rust_int_impl!(From, to_u128, u128);
    };
    (to_u32, $ux: ident) => {
        from_rust_int_impl!($ux, u8, i8, from_u8, from_i8, "");
        from_rust_int_impl!($ux, u16, i16, from_u16, from_i16, "");
        from_rust_int_impl!($ux, u32, i32, from_u32, from_i32, "");
    };
    (to_u64, $ux: ident) => {
        from_rust_int_impl!($ux, u64, i64, from_u64, from_i64, "");
        from_rust_int_impl!($ux, usize, isize, from_usize, from_isize, "");
    };
    (to_u128, $ux: ident) => {
        from_rust_int_impl!($ux, u128, i128, from_u128, from_i128, "");
    };
    (lossy_u64, $ux: ident) => {
        from_rust_int_impl!($ux, u64, i64, from_u64, from_i64, from_int_lossy_doc!($ux));
        from_rust_int_impl!($ux, usize, isize, from_usize, from_isize, from_int_lossy_doc!($ux));
    };
    (lossy_u128, $ux: ident) => {
        from_rust_int_impl!($ux, u128, i128, from_u128, from_i128, from_int_lossy_doc!($ux));
    };
    (From, to_u32, $ux: ident) => {
        from_rust_int_impl!(From, $ux, u8, i8, from_u8, from_i8, "");
        from_rust_int_impl!(From, $ux, u16, i16, from_u16, from_i16, "");
        from_rust_int_impl!(From, $ux, u32, i32, from_u32, from_i32, "");
    };
    (From, to_u64, $ux: ident) => {
        from_rust_int_impl!(From, $ux, u64, i64, from_u64, from_i64, "");
        from_rust_int_impl!(From, $ux, usize, isize, from_usize, from_isize, "");
    };
    (From, to_u128, $ux: ident) => {
        from_rust_int_impl!(From, $ux, u128, i128, from_u128, from_i128, "");
    };
    (From, lossy_u64, $ux: ident) => {
        from_rust_int_impl!(From, $ux, u64, i64, from_u64, from_i64, from_int_lossy_doc!($ux));
        from_rust_int_impl!(From, $ux, usize, isize, from_usize, from_isize,
            from_int_lossy_doc!($ux));
    };
    (From, lossy_u128, $ux: ident) => {
        from_rust_int_impl!(From, $ux, u128, i128, from_u128, from_i128, from_int_lossy_doc!($ux));
    };
    ($ux: ident, $uy: ident, $iy: ident, $from_func: ident, $from_func_i: ident, $lossy_doc: expr) => {
        #[doc = concat!("Convert from [`", stringify!($uy), "`] to a `VaxFloatingPoint<",
            stringify!($ux), ">`.")]
        ///
        /// Can be used to define constants.
        ///
        #[doc = $lossy_doc]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::",
            stringify!($from_func), "(0_", stringify!($uy), ");")]
        #[doc = concat!("const TEN: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::",
            stringify!($from_func), "(10);")]
        #[doc = concat!("assert_eq!(VaxFloatingPoint::<", stringify!($ux), ">::from_f32(0.0), ZERO);")]
        #[doc = concat!("assert_eq!(VaxFloatingPoint::<", stringify!($ux), ">::from_f32(10.0), TEN);")]
        /// ```
        ///
        #[doc = concat!("`From<", stringify!($uy), ">` cannot be used to define constants.")]
        ///
        /// ```compile_fail
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<",
            stringify!($ux), ">::from(0_", stringify!($uy), ");")]
        /// ```
        pub const fn $from_func(src: $uy) -> Self {
            Self {
                sign: Sign::Positive,
                exp: <$ux>::BITS as i32,
                frac: src as $ux,
                fault: None,
            }.normalize()
        }

        #[doc = concat!("Convert from [`", stringify!($iy), "`] to a `VaxFloatingPoint<",
            stringify!($ux), ">`.")]
        ///
        /// Can be used to define constants.
        ///
        #[doc = $lossy_doc]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::",
            stringify!($from_func_i), "(0_", stringify!($iy), ");")]
        #[doc = concat!("const MINUS_TEN: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<", stringify!($ux), ">::",
            stringify!($from_func_i), "(-10);")]
        #[doc = concat!("assert_eq!(VaxFloatingPoint::<", stringify!($ux), ">::from_f32(0.0), ZERO);")]
        #[doc = concat!("assert_eq!(VaxFloatingPoint::<", stringify!($ux), ">::from_f32(-10.0), MINUS_TEN);")]
        /// ```
        ///
        #[doc = concat!("`From<", stringify!($iy), ">` cannot be used to define constants.")]
        ///
        /// ```compile_fail
        /// # use vax_floating::arithmetic::VaxFloatingPoint;
        #[doc = concat!("const ZERO: VaxFloatingPoint<", stringify!($ux), "> = VaxFloatingPoint::<",
            stringify!($ux), ">::from(0_", stringify!($iy), ");")]
        /// ```
        pub const fn $from_func_i(src: $iy) -> Self {
            Self {
                sign: if src < 0 { Sign::Negative } else { Sign::Positive },
                exp: <$ux>::BITS as i32,
                frac: src.wrapping_abs() as $ux,
                fault: None,
            }.normalize()
        }
    };
    (From, $ux: ident, $uy: ident, $iy: ident, $from_func: ident, $from_func_i: ident, $lossy_doc: expr) => {
        impl From<&$uy> for VaxFloatingPoint<$ux> {
            /// Converts to this type from the input type.
            ///
            #[doc = $lossy_doc]
            fn from(src: &$uy) -> Self {
                Self::$from_func(*src)
            }
        }

        impl From<$uy> for VaxFloatingPoint<$ux> {
            /// Converts to this type from the input type.
            ///
            #[doc = $lossy_doc]
            fn from(src: $uy) -> Self {
                Self::$from_func(src)
            }
        }

        impl From<&$iy> for VaxFloatingPoint<$ux> {
            /// Converts to this type from the input type.
            ///
            #[doc = $lossy_doc]
            fn from(src: &$iy) -> Self {
                Self::$from_func_i(*src)
            }
        }

        impl From<$iy> for VaxFloatingPoint<$ux> {
            /// Converts to this type from the input type.
            ///
            #[doc = $lossy_doc]
            fn from(src: $iy) -> Self {
                Self::$from_func_i(src)
            }
        }
    };
}

vfp_impl!{
    FracT = u32,
    SignedT = i32,
}

vfp_impl!{
    FracT = u64,
    SignedT = i64,
}

vfp_impl!{
    FracT = u128,
    SignedT = i128,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use crate::{
        FFloating,
        DFloating,
        GFloating,
        HFloating,
    };

    mod f32_consts {
        // Constants used to convert from f32.
        pub const SIGN: u32 = 0x80000000;
        pub const FRAC: u32 = 0x007FFFFF;
        pub const EXP_SHIFT: u32 = 23;
        pub const EXP_MASK: u32 = 0xFF;
        pub const EXP_BIAS: i32 = 0x7E;
        pub const MAX_NEGATIVE: f32 = unsafe { std::mem::transmute::<u32, f32>(0x80000001) };
    }

    mod f64_consts {
        // Constants used to convert from f64.
        pub const SIGN: u64 = 0x8000000000000000;
        pub const FRAC: u64 = 0x000FFFFFFFFFFFFF;
        pub const EXP_SHIFT: u32 = 52;
        pub const EXP_MASK: u64 = 0x7FF;
        pub const EXP_BIAS: i32 = 0x3FE;
        pub const MAX_NEGATIVE: f64 = unsafe { std::mem::transmute::<u64, f64>(0x8000000000000001) };
    }

    macro_rules! verify_calc_consts {
        ($fx: ident, $fu: ident, $exp: literal, $mod: ident) => {
            {
                const SIGN: $fu = 1 << (<$fu>::BITS - 1);
                const EXP_SHIFT: u32 = <$fu>::BITS - $exp - 1;
                const EXP_MASK: $fu = (1 << $exp) - 1;
                const EXP_BIAS: i32 = (1 << ($exp - 1)) - 2;
                const FRAC_MASK: $fu = (1 << EXP_SHIFT) - 1;
                const MAX_NEGATIVE: $fx = unsafe { std::mem::transmute::<$fu, $fx>(SIGN+1) };

                assert_eq!(SIGN, $mod::SIGN);
                assert_eq!(EXP_SHIFT, $mod::EXP_SHIFT);
                assert_eq!(EXP_MASK, $mod::EXP_MASK);
                assert_eq!(EXP_BIAS, $mod::EXP_BIAS);
                assert_eq!(FRAC_MASK, $mod::FRAC);
                assert_eq!(MAX_NEGATIVE, $mod::MAX_NEGATIVE);
            }
        };
    }

    #[test]
    fn verify_calculated_consts() {
        // This is just a sanity check to make sure my auto-generated constants match the manual
        // constants.
        verify_calc_consts!(f32, u32, 8, f32_consts);
        verify_calc_consts!(f64, u64, 11, f64_consts);
    }

    macro_rules! create_vfp_ordering_test {
        ($name: ident, $ux: ident, $exp: literal) => {
            #[test]
            fn $name() {
                type VFP = VaxFloatingPoint<$ux>;
                use Sign::*;
                static ORDERED_LIST: &'static [VFP] = &[
                    VFP { sign: Negative, exp: $exp, frac: <$ux>::MAX, fault: None, }, // Minimum
                    VFP { sign: Negative, exp: $exp, frac: 1 << (<$ux>::BITS - 1), fault: None, },
                    VFP { sign: Negative, exp: 0, frac: <$ux>::MAX, fault: None, },   // -0.99999999
                    VFP { sign: Negative, exp: 0, frac: 1 << (<$ux>::BITS - 1), fault: None, },   // -0.5
                    VFP { sign: Negative, exp: -$exp, frac: <$ux>::MAX, fault: None, },
                    VFP { sign: Negative, exp: -$exp, frac: 1 << (<$ux>::BITS - 1), fault: None, }, // Max negative
                    VFP { sign: Positive, exp: 0, frac: 0, fault: None, },   // 0
                    VFP { sign: Positive, exp: -$exp, frac: 1 << (<$ux>::BITS - 1), fault: None, }, // Min posative
                    VFP { sign: Positive, exp: -$exp, frac: <$ux>::MAX, fault: None, },
                    VFP { sign: Positive, exp: 0, frac: 1 << (<$ux>::BITS - 1), fault: None, },   // 0.5
                    VFP { sign: Positive, exp: 0, frac: <$ux>::MAX, fault: None, },   // 0.99999999
                    VFP { sign: Positive, exp: $exp, frac: 1 << (<$ux>::BITS - 1), fault: None, },
                    VFP { sign: Positive, exp: $exp, frac: <$ux>::MAX, fault: None, }, // Maximum
                ];
                for i in 0..ORDERED_LIST.len() {
                    let leq = ORDERED_LIST[i];
                    for j in i..ORDERED_LIST.len() {
                        let geq = ORDERED_LIST[j];
                        assert!(leq <= geq,
                            "Comparison failed: {:X?} should be less than {:X?}, but it wasn't",
                            leq, geq);
                        assert!(geq >= leq,
                            "Comparison failed: {:X?} should be greater than {:X?}, but it wasn't",
                            geq, leq);
                    }
                }
            }
        };
    }

    create_vfp_ordering_test!(test_ordering_u32, u32, 127);
    create_vfp_ordering_test!(test_ordering_u64, u64, 1023);
    create_vfp_ordering_test!(test_ordering_u128, u128, 16383);

    #[test]
    fn sign_tests() {
        assert_eq!(Sign::Positive ^ Sign::Positive, Sign::Positive);
        assert_eq!(Sign::Negative ^ Sign::Negative, Sign::Positive);
        assert_eq!(Sign::Positive ^ Sign::Negative, Sign::Negative);
        assert_eq!(Sign::Negative ^ Sign::Positive, Sign::Negative);
    }

    macro_rules! from_f_test {
        (f32, $ux: ident) => {
            assert_eq!(VaxFloatingPoint::<$ux>::from_f32(0.0_f32), VaxFloatingPoint::<$ux>::ZERO);
            assert_eq!(VaxFloatingPoint::<$ux>::from_f32(-0.0_f32), VaxFloatingPoint::<$ux>::ZERO);
            assert_eq!(VaxFloatingPoint::<$ux>::from_f32(1.0_f32),
                VaxFloatingPoint::<$ux>{exp: 1, frac: 1 << (<$ux>::BITS - 1), ..Default::default()});
        };
    }

    #[test]
    fn to_from_rust_fp_tests() {
        from_f_test!(f32, u32);
        from_f_test!(f32, u64);
        from_f_test!(f32, u128);
    }

    macro_rules! to_from_f32_test {
        ($ux: ident, $float: ident) => {
            let intermediate = VaxFloatingPoint::<$ux>::from_f32($float);
            assert_eq!($float, intermediate.to_f32(), concat!("intermediate value (",
                stringify!($ux), ") = {:?}"), intermediate);
        };
    }

    proptest! {
        #[test]
        fn to_from_f32_tests(float in f32::MIN..=f32::MAX) {
            to_from_f32_test!(u32, float);
            to_from_f32_test!(u64, float);
            to_from_f32_test!(u128, float);
        }
    }

    trait NextUp {
        fn next_fp(self) -> Self;
    }

    impl NextUp for f32 {
        // Shamelessly stolen from core::num::f32::next_up.
        fn next_fp(self) -> f32 {
            // We must use strictly integer arithmetic to prevent denormals from
            // flushing to zero after an arithmetic operation on some platforms.
            const TINY_BITS: u32 = 0x1; // Smallest positive f32.
            const CLEAR_SIGN_MASK: u32 = 0x7fff_ffff;

            let bits = self.to_bits();
            if self.is_nan() || bits == f32::INFINITY.to_bits() {
                return self;
            }

            let abs = bits & CLEAR_SIGN_MASK;
            let next_bits = if abs == 0 {
                TINY_BITS
            } else if bits == abs {
                bits + 1
            } else {
                bits - 1
            };
            f32::from_bits(next_bits)
        }
    }

    impl NextUp for f64 {
        // Shamelessly stolen from core::num::f64::next_up.
        fn next_fp(self) -> f64 {
            // We must use strictly integer arithmetic to prevent denormals from
            // flushing to zero after an arithmetic operation on some platforms.
            const TINY_BITS: u64 = 0x1; // Smallest positive f64.
            const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

            let bits = self.to_bits();
            if self.is_nan() || bits == f64::INFINITY.to_bits() {
                return self;
            }

            let abs = bits & CLEAR_SIGN_MASK;
            let next_bits = if abs == 0 {
                TINY_BITS
            } else if bits == abs {
                bits + 1
            } else {
                bits - 1
            };
            f64::from_bits(next_bits)
        }
    }

    /// This is the close enough (ce) floating-point assertion macro.
    ///
    /// There are differences in the way that rust floating-point types and the VAX floating-point
    /// types handle the rounding of the results. This will trigger an assertion error if the
    /// floating point values are within range of a rounding difference.
    macro_rules! assert_ce {
        ($v1: expr, $v2: expr) => {
            if $v1 > $v2 {
                assert_eq!($v1, $v2::next_fp(), "value not close enough: {:?} ~!= {:?}", $v1, $v2);
            }
            else if $v1 < $v2 {
                assert_eq!($v1.next_fp($v1), $v2, "value not close enough: {:?} ~!= {:?}", $v1, $v2);
            }
        };
        ($fx: ident, $v1: expr, $v2: expr, $fmt: expr, $($param: expr),*) => {
            if $v1 > $v2 {
                assert_eq!($v1, $v2.next_fp(), $fmt, $($param),*);
            }
            else if $v1 < $v2 {
                assert_eq!($v1.next_fp(), $v2, $fmt, $($param),*);
            }
        };
    }

    macro_rules! to_from_f64_test {
        ($ux: ident, $float: ident) => {
            let intermediate = VaxFloatingPoint::<$ux>::from_f64($float);
            assert_eq!($float, intermediate.to_f64(), concat!("intermediate value (",
                stringify!($ux), ") = {:?}"), intermediate);
        };
    }

    proptest! {
        #[test]
        fn to_from_f64_tests(float in f64::MIN..=f64::MAX) {
            to_from_f64_test!(u64, float);
            to_from_f64_test!(u128, float);
        }
    }

    macro_rules! create_convert_test {
        ($name: ident, $ux: ident, $floating: ident) => {
            proptest! {
                #[test]
                fn $name(frac in (0 as $ux)..(1 << ($floating::MANTISSA_DIGITS))) {
                    let int_text = format!("{:02}", frac);
                    let float_text = format!("{}.0", frac);
                    let exp_text = {
                        let (before, after) = int_text.split_at(1);
                        format!("{}.{}e{}", before, after, after.len())
                    };
                    let from_float = $floating::from_ascii(&float_text).unwrap();
                    let from_exp = $floating::from_ascii(&exp_text).unwrap();
                    assert_eq!(from_float, from_exp);
                    let vfp = from_float.to_fp();
                    let from_to = $floating::from_fp(vfp);
                    assert_eq!(from_float, from_to);
                    assert_eq!(from_exp, from_to);
                }
            }
        };
    }

    create_convert_test!(convert_f_floating, u32, FFloating);
    create_convert_test!(convert_d_floating, u64, DFloating);
    create_convert_test!(convert_g_floating, u64, GFloating);
    create_convert_test!(convert_h_floating, u128, HFloating);

    macro_rules! add_to_test {
        ($fx: ident, $ux: ident, $float1: ident, $float2: ident) => {
            let v1 = VaxFloatingPoint::<$ux>::from($float1);
            let v2 = VaxFloatingPoint::<$ux>::from($float2);
            let add = v1.add_to(v2, false);
            let sub = v1.add_to(v2, true);
            assert_ce!($fx, $fx::from(add), $float1 + $float2,
                "add failed: v1 = {:?}; v2 = {:?}, add = {:?}, float bits = {:#X}, add_bits ={:#X}", v1, v2, add,
                ($float1 + $float2).to_bits(), <$fx>::from(add).to_bits());
            assert_ce!($fx, $fx::from(sub), $float1 - $float2,
                "sub failed: v1 = {:?}; v2 = {:?}, sub = {:?}, float bits = {:#X}, sub_bits ={:#X}", v1, v2, sub,
                ($float1 - $float2).to_bits(), <$fx>::from(sub).to_bits());
        };
    }

    prop_compose! {
        fn f32_float(min: f32, max: f32)(float in min..=max) -> f32 {
            float
        }
    }

    prop_compose! {
        fn f64_float(min: f64, max: f64)(float in min..=max) -> f64 {
            float
        }
    }

    prop_compose! {
        fn add_f32_float()(float in f32_float(f32::MIN/2.0, f32::MAX/2.0)) -> f32 {
            float
        }
    }

    proptest! {
        #[test]
        fn add_sub_f32_tests(float1 in add_f32_float(), float2 in add_f32_float()) {
            add_to_test!(f32, u32, float1, float2);
            add_to_test!(f32, u64, float1, float2);
            add_to_test!(f32, u128, float1, float2);
        }
    }

    prop_compose! {
        fn add_f64_float()(float in f64_float(f64::MIN/2.0, f64::MAX/2.0)) -> f64 {
            float
        }
    }

    proptest! {
        #[test]
        fn add_sub_f64_tests(float1 in add_f64_float(), float2 in add_f64_float()) {
            add_to_test!(f64, u64, float1, float2);
            add_to_test!(f64, u128, float1, float2);
        }
    }

    const F32_DIV_PRECISION: u32 = f32::MANTISSA_DIGITS + 2;

    macro_rules! multiply_by_test {
        ($fx: ident, $ux: ident, $float1: ident, $float2: ident) => {
            let v1 = VaxFloatingPoint::<$ux>::from_f32($float1);
            let v2 = VaxFloatingPoint::<$ux>::from_f32($float2);
            let mul = v1.multiply_by(v2);
            let div = v1.divide_by(v2,  F32_DIV_PRECISION);
            assert_ce!($fx, mul.to_f32(), $float1 * $float2,
                "mul failed: v1 = {:?}; v2 = {:?}, mul = {:?}, float bits = {:#X}", v1, v2, mul,
                ($float1 + $float2).to_bits());
            if $float2 != 0_f32 {
                assert_ce!($fx, div.to_f32(), $float1 / $float2,
                    "div failed: v1 = {:?}; v2 = {:?}, div = {:?}, float bits = {:#X}", v1, v2,
                    div, ($float1 / $float2).to_bits());
            }
        };
    }

    prop_compose! {
        fn mul_f32_float()(float in f32_float(-f32::MAX.sqrt(), f32::MAX.sqrt())) -> f32 {
            float
        }
    }

    proptest! {
        #[test]
        fn mul_div_f32_tests(float1 in mul_f32_float(), float2 in mul_f32_float()) {
            multiply_by_test!(f32, u32, float1, float2);
            multiply_by_test!(f32, u64, float1, float2);
            multiply_by_test!(f32, u128, float1, float2);
        }
    }
}
