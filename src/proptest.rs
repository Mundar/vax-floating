//! Implement [proptest] Arbitrary, Strategy, and ValueTree
//!
// This code is largely based on macros defined in `proptest::num` and `proptest::arbitrary`, but
// modified for our use.

use bitflags::bitflags;
use crate::{
    FFloating,
    DFloating,
    GFloating,
    HFloating,
};

/// This is originally from `proptest::num` and used unmodified.
///
/// # Original Version Copyright
///
/// Copyright 2017, 2018 Jason Lingle
///
/// Licensed under the [Apache License, Version 2.0] or the [MIT license], at your option. This
/// file may not be copied, modified, or distributed except according to those terms.
macro_rules! arbitrary {
    ([$($bounds : tt)*] $typ: ty, $strat: ty, $params: ty;
        $args: ident => $logic: expr) => {
        impl<$($bounds)*> proptest::arbitrary::Arbitrary for $typ {
            type Parameters = $params;
            type Strategy = $strat;
            fn arbitrary_with($args: Self::Parameters) -> Self::Strategy {
                $logic
            }
        }
    };
    ([$($bounds : tt)*] $typ: ty, $strat: ty; $logic: expr) => {
        arbitrary!([$($bounds)*] $typ, $strat, (); _args => $logic);
    };
    ([$($bounds : tt)*] $typ: ty; $logic: expr) => {
        arbitrary!([$($bounds)*] $typ,
            proptest::strategy::Just<Self>, ();
            _args => proptest::strategy::Just($logic)
        );
    };
    ($typ: ty, $strat: ty, $params: ty; $args: ident => $logic: expr) => {
        arbitrary!([] $typ, $strat, $params; $args => $logic);
    };
    ($typ: ty, $strat: ty; $logic: expr) => {
        arbitrary!([] $typ, $strat; $logic);
    };
    ($strat: ty; $logic: expr) => {
        arbitrary!([] $strat; $logic);
    };
    ($($typ: ident),*) => {
        $(arbitrary!($typ, $typ::Any; $typ::ANY);)*
    };
}

/// Generate the proptest Strategy for a VAX floating-point type.
///
/// This is originally from `proptest::num` and modified for the `vax_floating` types.
///
/// # Original Version Copyright
///
/// Copyright 2017, 2018 Jason Lingle
///
/// Licensed under the [Apache License, Version 2.0] or the [MIT license], at your option. This
/// file may not be copied, modified, or distributed except according to those terms.
///
/// # Modified Version Copyright
///
/// Copyright 2024 Thomas Mundar
///
/// Licensed under the [MIT license]. This file may not be copied, modified, or distributed except
/// according to those terms.
///
/// [Apache License, Version 2.0]: http://www.apache.org/licenses/LICENSE-2.0
/// [MIT license]: http://opensource.org/licenses/MIT
macro_rules! numeric_api {
    ($typ:ident, $mod: ident, $range: ident) => {
        #[doc = concat!("[`Strategy`] for the [`", stringify!($typ), "`] type.")]
        ///
        /// # Examples
        ///
        /// ```rust
        /// use proptest::prelude::*;
        #[doc = concat!("use vax_floating::proptest::", stringify!($mod), "::", stringify!($range), ";")]
        #[doc = concat!("use vax_floating::", stringify!($typ), ";")]
        ///
        #[doc = concat!("const ZERO: ", stringify!($typ), " = ", stringify!($typ), "::from_bits(0);")]
        #[doc = concat!("const NEG_ONE: ", stringify!($typ), " = ", stringify!($typ), "::from_i8(-1);")]
        ///
        /// proptest! {
        #[doc = concat!("    fn ascii_test(float in ", stringify!($range), "::from(\"-1.0\"..\"0\")) {")]
        ///         println!("ascii_test: float = {}", float);
        ///         prop_assert!(float >= NEG_ONE);
        ///         prop_assert!(float < ZERO);
        ///    }
        /// }
        ///
        /// proptest! {
        #[doc = concat!("    fn float_test(float in ", stringify!($range), "::from(NEG_ONE..ZERO)) {")]
        ///         println!("float_test: float = {}", float);
        ///         prop_assert!(float >= NEG_ONE);
        ///         prop_assert!(float < ZERO);
        ///    }
        /// }
        ///
        /// // Run the tests manually for doc tests
        /// ascii_test();
        /// float_test();
        /// ```
        #[derive(Copy, Clone, Debug)]
        pub struct $range {
            start: $typ,
            end: $typ,
            include_end: bool,
        }

        impl<S: AsRef<str>> From<::core::ops::Range<S>> for $range {
            fn from(range: ::core::ops::Range<S>) -> Self {
                Self {
                    start: $typ::from_ascii(range.start.as_ref()).unwrap(),
                    end: $typ::from_ascii(range.end.as_ref()).unwrap(),
                    include_end: false,
                }
            }
        }

        impl From<::core::ops::Range<$typ>> for $range {
            fn from(range: ::core::ops::Range<$typ>) -> Self {
                Self {
                    start: range.start,
                    end: range.end,
                    include_end: false,
                }
            }
        }

        impl<S: AsRef<str>> From<::core::ops::RangeInclusive<S>> for $range {
            fn from(range: ::core::ops::RangeInclusive<S>) -> Self {
                Self {
                    start: $typ::from_ascii(range.start().as_ref()).unwrap(),
                    end: $typ::from_ascii(range.end().as_ref()).unwrap(),
                    include_end: true,
                }
            }
        }

        impl From<::core::ops::RangeInclusive<$typ>> for $range {
            fn from(range: ::core::ops::RangeInclusive<$typ>) -> Self {
                Self {
                    start: *range.start(),
                    end: *range.end(),
                    include_end: true,
                }
            }
        }

        impl<S: AsRef<str>> From<::core::ops::RangeFrom<S>> for $range {
            fn from(range: ::core::ops::RangeFrom<S>) -> Self {
                Self {
                    start: $typ::from_ascii(range.start.as_ref()).unwrap(),
                    end: $typ::MAX,
                    include_end: true,
                }
            }
        }

        impl From<::core::ops::RangeFrom<$typ>> for $range {
            fn from(range: ::core::ops::RangeFrom<$typ>) -> Self {
                Self {
                    start: range.start,
                    end: $typ::MAX,
                    include_end: true,
                }
            }
        }

        impl<S: AsRef<str>> From<::core::ops::RangeTo<S>> for $range {
            fn from(range: ::core::ops::RangeTo<S>) -> Self {
                Self {
                    start: $typ::MIN,
                    end: $typ::from_ascii(range.end.as_ref()).unwrap(),
                    include_end: false,
                }
            }
        }

        impl From<::core::ops::RangeTo<$typ>> for $range {
            fn from(range: ::core::ops::RangeTo<$typ>) -> Self {
                Self {
                    start: $typ::MIN,
                    end: range.end,
                    include_end: false,
                }
            }
        }

        impl<S: AsRef<str>> From<::core::ops::RangeToInclusive<S>> for $range {
            fn from(range: ::core::ops::RangeToInclusive<S>) -> Self {
                Self {
                    start: $typ::MIN,
                    end: $typ::from_ascii(range.end.as_ref()).unwrap(),
                    include_end: true,
                }
            }
        }

        impl From<::core::ops::RangeToInclusive<$typ>> for $range {
            fn from(range: ::core::ops::RangeToInclusive<$typ>) -> Self {
                Self {
                    start: $typ::MIN,
                    end: range.end,
                    include_end: true,
                }
            }
        }

        impl Strategy for $range {
            type Tree = BinarySearch;
            type Value = $typ;

            fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
                Ok(BinarySearch::from_range(
                    runner,
                    self.start,
                    self.end,
                    self.include_end,
                ))
            }
        }
    };
}

bitflags! {
    /// Bit flags used to categorize different types of VAX floating-point values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct VaxFloatTypes: u32 {
        const POSITIVE          = 0b0000_0001;
        const NEGATIVE          = 0b0000_0010;
        const NORMAL            = 0b0000_0100;
        const ZERO              = 0b0000_1000;
        const RESERVED          = 0b0001_0000;
        const ANY =
            Self::POSITIVE.bits() |
            Self::NEGATIVE.bits() |
            Self::NORMAL.bits() |
            Self::ZERO.bits() |
            Self::RESERVED.bits();
    }
}

impl VaxFloatTypes {
    fn normalise(mut self) -> Self {
        if !self.intersects(VaxFloatTypes::POSITIVE | VaxFloatTypes::NEGATIVE) {
            self |= VaxFloatTypes::POSITIVE;
        }

        if !self.intersects(
            VaxFloatTypes::NORMAL
                | VaxFloatTypes::ZERO
                | VaxFloatTypes::RESERVED
        ) {
            self |= VaxFloatTypes::NORMAL;
        }
        self
    }
}

/// Constants used to parse VAX floating-point types.
trait VaxFloatLayout
{
    type Bits: Copy;

    const SIGN_MASK: Self::Bits;
    const EXP_MASK: Self::Bits;
    const EXP_ZERO: Self::Bits;
    const MANTISSA_MASK: Self::Bits;
}

/// Define the Any type for a VAX floating-point type.
macro_rules! float_any {
    ($typ:ident) => {
        /// Strategies which produce floating-point values from particular
        /// classes. See the various `Any`-typed constants in this module.
        ///
        /// Note that this usage is fairly advanced and primarily useful to
        /// implementors of algorithms that need to handle wild values in a
        /// particular way. For testing things like graphics processing or game
        /// physics, simply using ranges (e.g., `-1.0..2.0`) will often be more
        /// practical.
        ///
        /// `Any` can be OR'ed to combine multiple classes. For example,
        /// `POSITIVE | RESERVED` will generate arbitrary positive, valid
        /// floats, and reserved.
        /// course).
        ///
        /// If neither `POSITIVE` nor `NEGATIVE` has been OR'ed into an `Any`
        /// but a type to be generated requires a sign, `POSITIVE` is assumed.
        /// If no classes are OR'ed into an `Any` (i.e., only `POSITIVE` and/or
        /// `NEGATIVE` are given), `NORMAL` is assumed.
        ///
        /// The various float classes are assigned fixed weights for generation
        /// which are believed to be reasonable for most applications. Roughly:
        ///
        /// - If `POSITIVE | NEGATIVE`, the sign is evenly distributed between
        ///   both options.
        ///
        /// - Classes are weighted as follows, in descending order:
        ///   `NORMAL` > `ZERO` > `RESERVED`
        #[derive(Clone, Copy, Debug)]
        #[must_use = "strategies do nothing unless used"]
        pub struct Any(VaxFloatTypes);

        /*
        #[cfg(test)]
        impl Any {
            pub(crate) fn from_bits(bits: u32) -> Self {
                Any(VaxFloatTypes::from_bits_truncate(bits))
            }

            pub(crate) fn normal_bits(&self) -> VaxFloatTypes {
                self.0.normalise()
            }
        }
        */

        impl ops::BitOr for Any {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self {
                Any(self.0 | rhs.0)
            }
        }

        impl ops::BitOrAssign for Any {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0
            }
        }

        /// Generates positive floats
        ///
        /// By itself, implies the `NORMAL` class, unless another class is
        /// OR'ed in. That is, using `POSITIVE` as a strategy by itself will
        /// generate arbitrary values between the type's `MIN_POSITIVE` and
        /// `MAX`.
        pub const POSITIVE: Any = Any(VaxFloatTypes::POSITIVE);
        /// Generates negative floats.
        ///
        /// By itself, implies the `NORMAL` class, unless another class is
        /// OR'ed in. That is, using `POSITIVE` as a strategy by itself will
        /// generate arbitrary values between the type's `MIN` and
        /// `-MIN_POSITIVE`.
        pub const NEGATIVE: Any = Any(VaxFloatTypes::NEGATIVE);
        /// Generates "normal" floats.
        ///
        /// These are finite values where the first bit of the mantissa is an
        /// implied `1`. When positive, this represents the range
        /// `MIN_POSITIVE` through `MAX`, both inclusive.
        ///
        /// Generated values are uniform over the discrete floating-point
        /// space, which means the numeric distribution is an inverse
        /// exponential step function. For example, values between 1.0 and 2.0
        /// are generated with the same frequency as values between 2.0 and
        /// 4.0, even though the latter covers twice the numeric range.
        ///
        /// If neither `POSITIVE` nor `NEGATIVE` is OR'ed with this constant,
        /// `POSITIVE` is implied.
        pub const NORMAL: Any = Any(VaxFloatTypes::NORMAL);
        /// Generates zero-valued floats.
        ///
        /// There are no `NEGATIVE` zero floats, so if the sign is ignored for `ZERO`.
        pub const ZERO: Any = Any(VaxFloatTypes::ZERO);
        /// Generates reserved floats.
        pub const RESERVED: Any = Any(VaxFloatTypes::RESERVED);

        /// Generates literally arbitrary VAX floating-point values, including
        /// Reserved values.
        ///
        /// Equivalent to `POSITIVE | NEGATIVE | NORMAL | ZERO | RESERVED
        pub const ANY: Any = Any(VaxFloatTypes::ANY);

        impl Strategy for Any {
            type Tree = BinarySearch;
            type Value = $typ;

            fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
                let flags = self.0.normalise();
                let sign_mask = if flags.contains(VaxFloatTypes::NEGATIVE) {
                    $typ::SIGN_MASK
                } else {
                    0
                };
                let sign_or = if flags.contains(VaxFloatTypes::POSITIVE) {
                    0
                } else {
                    $typ::SIGN_MASK
                };

                macro_rules! weight {
                    ($case:ident, $weight:expr) => {
                        if flags.contains(VaxFloatTypes::$case) {
                            $weight
                        } else {
                            0
                        }
                    }
                }

                let (class_mask, class_or, allow_edge_exp, allow_sign) =
                    prop_oneof![
                        weight!(NORMAL, 20) => Just(
                            (<$typ as VaxFloatLayout>::EXP_MASK | $typ::MANTISSA_MASK, 0,
                             false, true)),
                        weight!(ZERO, 4) => Just(
                            (0, 0, true, false)),
                        weight!(RESERVED, 2) => Just(
                            (!<$typ as VaxFloatLayout>::EXP_MASK, $typ::SIGN_MASK, true, true)),
                    ].new_tree(runner)?.current();

                let mut generated_value: <$typ as VaxFloatLayout>::Bits =
                    runner.rng().gen();
                generated_value &= sign_mask | class_mask;
                generated_value |= sign_or | class_or;
                let exp = generated_value & <$typ as VaxFloatLayout>::EXP_MASK;
                if !allow_edge_exp && (0 == exp) {
                    generated_value &= !<$typ as VaxFloatLayout>::EXP_MASK;
                    generated_value |= $typ::EXP_ZERO;
                }
                if !allow_sign {
                    generated_value &= !$typ::SIGN_MASK;
                }

                Ok(BinarySearch::new_with_types(
                    $typ::from_swapped(generated_value), flags))
            }
        }
    }
}

/// Implement the proptest Arbitrary, Strategy, and ValueTree for a VAX floating-point type.
macro_rules! proptest_impl {
    (
        Self = $SelfT: ident,
        Strategy = $StrategyT: ident,
        ActualT = $ux: ident,
        SignedT = $ix: ident,
        ExpBits = $exp: literal,
        Mod = $mod: ident,
    ) => {
        impl VaxFloatLayout for $SelfT {
            type Bits = $ux;

            const SIGN_MASK: $ux = 1 << ($ux::BITS - 1);
            const EXP_MASK: $ux = ((1 << $exp) - 1) << ($ux::BITS - 1 - $exp);
            const EXP_ZERO: $ux = (($SelfT::EXP_BIAS as $ux) + 1) << ($ux::BITS - 1 - $exp);
            const MANTISSA_MASK: $ux = (1 << ($SelfT::MANTISSA_DIGITS - 1)) - 1;
        }

        #[doc = concat!("Module for proptest structures for the ", stringify!($mod), " (",
            stringify!($SelfT), ") VAX floating-point type.")]
        ///
        /// # Examples
        ///
        /// ```rust
        /// use proptest::prelude::*;
        /// use vax_floating::{
        #[doc = concat!("    ", stringify!($SelfT), ",")]
        #[doc = concat!("    proptest::", stringify!($mod), "::", stringify!($SelfT), "Strategy,")]
        /// };
        ///
        #[doc = concat!("// Implements the Arbitrary trait for ", stringify!($SelfT), ".")]
        /// proptest! {
        #[doc = concat!("    fn arbitrary(float in any::<", stringify!($SelfT), ">()) {")]
        ///         // Any should never return a reserved value.
        ///         prop_assert!(!float.is_reserved());
        ///         // It will never return a zero value that is not from_bits(0)
        ///         if float.is_zero() {
        ///             prop_assert_eq!(float.to_bits(), 0);
        ///         }
        ///     }
        /// }
        ///
        #[doc = concat!("const ZERO: ", stringify!($SelfT), " = ", stringify!($SelfT), "::from_bits(0);")]
        #[doc = concat!("const ONE: ", stringify!($SelfT), " = ", stringify!($SelfT), "::from_u8(1);")]
        ///
        #[doc = concat!("// ", stringify!($SelfT), "Strategy can be used to define a range using string slices.")]
        /// proptest! {
        #[doc = concat!("    fn str_range(float in ", stringify!($SelfT), "Strategy::from(\"0\"..\"1.0\")) {")]
        ///         prop_assert!(float >= ZERO);
        ///         prop_assert!(float < ONE);
        ///     }
        /// }
        ///
        #[doc = concat!("// ", stringify!($SelfT), "Strategy can be used to define a range using ", stringify!($SelfT), ".")]
        /// proptest! {
        #[doc = concat!("    fn float_range(float in ", stringify!($SelfT), "Strategy::from(ZERO..ONE)) {")]
        ///         prop_assert!(float >= ZERO);
        ///         prop_assert!(float < ONE);
        ///     }
        /// }
        ///
        /// // Manually run proptest tests within this doc test
        /// arbitrary();
        /// str_range();
        /// float_range();
        /// ```
        pub mod $mod {
            use core::ops;

            use rand::Rng;

            use super::{VaxFloatLayout, VaxFloatTypes};
            use proptest::strategy::*;
            use proptest::test_runner::TestRunner;
            use proptest::prop_oneof;
            use crate::$SelfT;
            use rand::distributions::{
                uniform::Uniform,
                Distribution,
            };

            float_any!($SelfT);

            /// Shrinks a float towards 0, using binary search to find boundary
            /// points.
            ///
            /// Non-finite values immediately shrink to 0.
            // f_floating example
            // positive floats  7FFFFFFF-00800000 7F800000-00000001 +7FFFFF
            // unused zeroes    007FFFFF-00000001
            // zero             00000000
            // negative floats  80800000-FFFFFFFF
            // reserved         807FFFFF-80000000
            #[derive(Clone, Copy, Debug)]
            pub struct BinarySearch {
                lo: $ix,
                curr: $ix,
                hi: $ix,
            }

            impl BinarySearch {
                const POS_ADJ: $ix = (1 << $SelfT::MANTISSA_DIGITS) - 1;
                const MAX_RESERVED: $ix = <$ix>::MIN + Self::POS_ADJ;

                /// Creates a new binary searcher starting at the given value.
                pub fn new(start: $SelfT) -> Self {
                    let start = Self::float_to_int(start);
                    BinarySearch {
                        lo: 0,
                        curr: start,
                        hi: start,
                    }
                }

                fn new_with_types(start: $SelfT, _allowed: VaxFloatTypes) -> Self {
                    let start = Self::float_to_int(start);
                    BinarySearch {
                        lo: 0,
                        curr: start,
                        hi: start,
                    }
                }

                /// Creates a new binary searcher which will not produce values
                /// on the other side of `lo` or `hi` from `start`. `lo` is
                /// inclusive, `hi` is exclusive.
                fn from_range(
                    runner: &mut TestRunner,
                    lo: $SelfT,
                    hi: $SelfT,
                    inclusive: bool,
                ) -> Self {
                    let lo_int = Self::float_to_int(lo);
                    let hi_int = Self::float_to_int(hi);
                    let start_int = if inclusive {
                        Uniform::new_inclusive(lo_int, hi_int).sample(runner.rng())
                    }
                    else {
                        Uniform::new(lo_int, hi_int).sample(runner.rng())
                    };
                    BinarySearch {
                        lo: if start_int < 0 {
                            std::cmp::min(0, hi_int - 1)
                        }
                        else {
                            std::cmp::max(0, lo_int)
                        },
                        hi: start_int,
                        curr: start_int,
                    }
                }

                #[doc = concat!("Convert from the internal representation (", stringify!($ix),
                    ") to the external representation (", stringify!($SelfT), ").")]
                fn int_to_float(int: $ix) -> $SelfT {
                    if 0 < int {
                        $SelfT::from_swapped(int.saturating_add(Self::POS_ADJ) as $ux)
                    }
                    else if 0 == int {
                        $SelfT::from_bits(0)
                    }
                    else if int > Self::MAX_RESERVED {
                        $SelfT::from_swapped((int.abs().saturating_add(Self::POS_ADJ) as $ux)
                            | (1 << (<$ux>::BITS - 1)))
                    }
                    else {
                        $SelfT::from_swapped(int as $ux)
                    }
                }


                #[doc = concat!("Convert from the external representation (", stringify!($SelfT),
                    ") to the internal representation (", stringify!($ix), ").")]
                fn float_to_int(float: $SelfT) -> $ix {
                    if float.is_zero() { 0 }
                    else if float.is_negative() {
                        -(((float.to_swapped() & ((1 << (<$ux>::BITS - 1)) - 1)) as $ix) - Self::POS_ADJ)
                    }
                    else if float.is_reserved() {
                        float.to_swapped() as $ix
                    }
                    else {
                        (float.to_swapped() as $ix) - Self::POS_ADJ
                    }
                }

                #[doc = concat!("Return the current value as a ", stringify!($SelfT), ".")]
                pub fn curr(&self) -> $SelfT {
                    Self::int_to_float(self.curr)
                }

                /*
                fn current_allowed(&self) -> bool {
                    // Don't reposition if the new value is not allowed
                    let class_allowed = if self.curr <= Self::MAX_RESERVED {
                        self.allowed.contains(VaxFloatTypes::RESERVED)
                    } else if 0 == self.curr {
                        self.allowed.contains(VaxFloatTypes::ZERO)
                    } else {
                        self.allowed.contains(VaxFloatTypes::NORMAL)
                    };
                    let sign_allowed = if self.curr <= Self::MAX_RESERVED { true }
                    else if 0 > self.curr {
                        self.allowed.contains(VaxFloatTypes::NEGATIVE)
                    } else {
                        self.allowed.contains(VaxFloatTypes::POSITIVE)
                    };

                    class_allowed && sign_allowed
                }
                */

                fn reposition(&mut self) -> bool {
                    // Won't ever overflow since lo starts at 0 and advances
                    // towards hi.
                    let interval = self.hi - self.lo;
                    let new_mid = self.lo + interval / 2;

                    if new_mid == self.curr {
                        false
                    } else {
                        self.curr = new_mid;
                        true
                    }
                }

                fn magnitude_greater(lhs: $ix, rhs: $ix) -> bool {
                    if 0 == lhs {
                        false
                    } else if lhs < 0 {
                        lhs < rhs
                    } else {
                        lhs > rhs
                    }
                }
            }
            impl ValueTree for BinarySearch {
                type Value = $SelfT;

                fn current(&self) -> $SelfT {
                    self.curr()
                }

                fn simplify(&mut self) -> bool {
                    if !BinarySearch::magnitude_greater(self.hi, self.lo) {
                        return false;
                    }

                    self.hi = self.curr;
                    self.reposition()
                }

                fn complicate(&mut self) -> bool {
                    if !BinarySearch::magnitude_greater(self.hi, self.lo) {
                        return false;
                    }

                    self.lo = self.curr + if self.hi < 0 { -1 } else { 1 };

                    self.reposition()
                }
            }

            numeric_api!($SelfT, $mod, $StrategyT);
        }

        arbitrary!($SelfT, $mod::Any; {
            $mod::POSITIVE | $mod::NEGATIVE | $mod::ZERO | $mod::NORMAL
        });
    };
}

proptest_impl!{
    Self = FFloating,
    Strategy = FFloatingStrategy,
    ActualT = u32,
    SignedT = i32,
    ExpBits = 8,
    Mod = f_floating,
}

proptest_impl!{
    Self = DFloating,
    Strategy = DFloatingStrategy,
    ActualT = u64,
    SignedT = i64,
    ExpBits = 8,
    Mod = d_floating,
}

proptest_impl!{
    Self = GFloating,
    Strategy = GFloatingStrategy,
    ActualT = u64,
    SignedT = i64,
    ExpBits = 11,
    Mod = g_floating,
}

proptest_impl!{
    Self = HFloating,
    Strategy = HFloatingStrategy,
    ActualT = u128,
    SignedT = i128,
    ExpBits = 15,
    Mod = h_floating,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Verify that the calculated constants are as expected.
    #[test]
    fn vax_floating_layout_verify() {
        assert_eq!(<FFloating as VaxFloatLayout>::SIGN_MASK, 0x8000_0000);
        assert_eq!(<FFloating as VaxFloatLayout>::EXP_MASK, 0x7F80_0000);
        assert_eq!(<FFloating as VaxFloatLayout>::EXP_ZERO, 0x4080_0000);
        assert_eq!(<FFloating as VaxFloatLayout>::MANTISSA_MASK, 0x007F_FFFF);
        assert_eq!(<DFloating as VaxFloatLayout>::SIGN_MASK, 0x8000_0000_0000_0000);
        assert_eq!(<DFloating as VaxFloatLayout>::EXP_MASK, 0x7F80_0000_0000_0000);
        assert_eq!(<DFloating as VaxFloatLayout>::EXP_ZERO, 0x4080_0000_0000_0000);
        assert_eq!(<DFloating as VaxFloatLayout>::MANTISSA_MASK, 0x007F_FFFF_FFFF_FFFF);
        assert_eq!(<GFloating as VaxFloatLayout>::SIGN_MASK, 0x8000_0000_0000_0000);
        assert_eq!(<GFloating as VaxFloatLayout>::EXP_MASK, 0x7FF0_0000_0000_0000);
        assert_eq!(<GFloating as VaxFloatLayout>::EXP_ZERO, 0x4010_0000_0000_0000);
        assert_eq!(<GFloating as VaxFloatLayout>::MANTISSA_MASK, 0x000F_FFFF_FFFF_FFFF);
        assert_eq!(<HFloating as VaxFloatLayout>::SIGN_MASK, 0x8000_0000_0000_0000_0000_0000_0000_0000);
        assert_eq!(<HFloating as VaxFloatLayout>::EXP_MASK, 0x7FFF_0000_0000_0000_0000_0000_0000_0000);
        assert_eq!(<HFloating as VaxFloatLayout>::EXP_ZERO, 0x4001_0000_0000_0000_0000_0000_0000_0000);
        assert_eq!(<HFloating as VaxFloatLayout>::MANTISSA_MASK, 0x0000_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF);
    }

    macro_rules! impl_prop_tests {
        ($floating: ident, $mod: ident, $range: ident) => {
            mod $mod {
                use super::*;
                use std::{
                    cell::RefCell,
                    collections::HashMap,
                    sync::Mutex,
                };
                use crate::proptest::$mod::$range;
                use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};

                const NEG_ONE: $floating = $floating::from_i8(-1);
                const ZERO: $floating = $floating::from_bits(0);
                const ONE: $floating = $floating::from_u8(1);

                proptest! {
                    #[test]
                    fn arbitrary(float in any::<$floating>()) {
                        // Any should never return a reserved value.
                        prop_assert!(!float.is_reserved());
                        // It will never return a zero value that is not from_bits(0)
                        if float.is_zero() {
                            prop_assert_eq!(float.to_bits(), 0);
                        }
                    }
                }

                // This tests makes sure that we have a good random distribution of random values.
                #[test]
                fn distribution() {
                    let match_counts: Mutex<RefCell<HashMap<$floating, usize>>> =
                        Mutex::new(RefCell::new(HashMap::new()));
                    let mut runner = TestRunner::new(Config {
                        failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
                        .. Config::default()
                    });
                    runner.run(&(any::<$floating>()), |f| {
                        let lock = match_counts.lock().unwrap();
                        let mut counts = lock.borrow_mut();
                        println!("f = {0} {0:?}", f);
                        match counts.get_mut(&f) {
                            Some(count) => { *count += 1; }
                            None => { counts.insert(f, 1); }
                        }
                        Ok(())
                    }).unwrap();
                    let mut total = 0;
                    let lock = match_counts.lock().unwrap();
                    let counts = lock.borrow();
                    for (float, count) in counts.iter() {
                        total += count;
                        if !float.is_zero() {   // Zero appears more often than random.
                            if 1 < *count {
                                println!("{0}::distribution: Warning: float {1} {1:?} was returned {2} times", stringify!($mod), float, count);
                            }
                            assert!(3 > *count, "float {0} {0:?} was returned {1} times", float, count);
                        }
                    }
                    println!("{}::distribution: Test ran {} times", stringify!($mod), total);
                }

                proptest! {
                    #[test]
                    fn range_positive_ascii(float in $range::from("0".."1.0")) {
                        const ZERO: $floating = $floating::from_bits(0);
                        const ONE: $floating = $floating::from_u8(1);
                        prop_assert!(ZERO <= float);
                        prop_assert!(ONE > float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_positive(float in $range::from(ZERO..ONE)) {
                        const ZERO: $floating = $floating::from_bits(0);
                        const ONE: $floating = $floating::from_u8(1);
                        prop_assert!(ZERO <= float);
                        prop_assert!(ONE > float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_ascii(float in $range::from("-1.0".."0.0")) {
                        const NEG_ONE: $floating = $floating::from_i8(-1);
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(NEG_ONE <= float);
                        prop_assert!(ZERO > float);
                    }
                }

                proptest! {
                    #[test]
                    fn range(float in $range::from(NEG_ONE..ZERO)) {
                        const NEG_ONE: $floating = $floating::from_i8(-1);
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(NEG_ONE <= float);
                        prop_assert!(ZERO > float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_inclusive_ascii(float in $range::from("-1.0"..="0.0")) {
                        const NEG_ONE: $floating = $floating::from_i8(-1);
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(NEG_ONE <= float);
                        prop_assert!(ZERO >= float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_inclusive(float in $range::from(NEG_ONE..=ZERO)) {
                        const NEG_ONE: $floating = $floating::from_i8(-1);
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(NEG_ONE <= float);
                        prop_assert!(ZERO >= float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_from_ascii(float in $range::from("1.0"..)) {
                        const ONE: $floating = $floating::from_u8(1);
                        prop_assert!(ONE <= float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_from(float in $range::from(ONE..)) {
                        const ONE: $floating = $floating::from_u8(1);
                        prop_assert!(ONE <= float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_to_ascii(float in $range::from(.."0.0")) {
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(ZERO > float);
                    }
                }


                proptest! {
                    #[test]
                    fn range_to(float in $range::from(..ZERO)) {
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(ZERO > float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_to_inclusive_ascii(float in $range::from(..="0.0")) {
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(ZERO >= float);
                    }
                }

                proptest! {
                    #[test]
                    fn range_to_inclusive(float in $range::from(..=ZERO)) {
                        const ZERO: $floating = $floating::from_bits(0);
                        prop_assert!(ZERO >= float);
                    }
                }
            }
        }
    }

    impl_prop_tests!(FFloating, f_floating, FFloatingStrategy);
    impl_prop_tests!(DFloating, d_floating, DFloatingStrategy);
    impl_prop_tests!(GFloating, g_floating, GFloatingStrategy);
    impl_prop_tests!(HFloating, h_floating, HFloatingStrategy);
}
