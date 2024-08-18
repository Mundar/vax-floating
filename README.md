# vax-floating - VAX Floating-Point Types

This is a Rust implementation of the VAX floating-point types documented in the *VAX Architecture Reference Manual*.

- Supports conversion from rust data types.
- Supports conversion from rust data types to constants.
- Supports conversion between vax floating types (both constant and runtime).
- Supports standard mathematical operators.
- Supports constant mathematical operators.
- Supports display, and lowercase and uppercase exponential output.

## Features

**proptest** - Include support for the `proptest` testing crate.

## Supported VAX floating-Point Types

VAX Type   | Size     | Exponent size | Exponent range
-----------|----------|---------------|---------------
F_floating | 32-bits  |  8-bits       | 2<sup>127</sup> to 2<sup>-127</sup>
D_floating | 64-bits  |  8-bits       | 2<sup>127</sup> to 2<sup>-127</sup>
G_floating | 64-bits  |  11-bits      | 2<sup>1,023</sup> to 2<sup>-1,023</sup>
H_floating | 128-bits |  15-bits      | 2<sup>16,383</sup> to 2<sup>-16,383</sup>

## Examples

```rust
use vax_floating::{FFloating, DFloating, GFloating, HFloating};
use std::str::FromStr;

// Supports conversion from rust data types.
let ten = FFloating::from(10_u8);
let three_hundred = DFloating::from(300_u16);
let twelve_point_five = GFloating::from(12.5_f32);
let very_small = HFloating::from_str("1e-1000").unwrap();

assert_eq!(ten, FFloating::from(10_u64));
assert_eq!(three_hundred, DFloating::from(300_u32));
assert_eq!(twelve_point_five, GFloating::from_str("12.5").unwrap());
assert_eq!(very_small, HFloating::from_u8(1) / HFloating::from_str("1e1000").unwrap());

// Supports conversion from rust data types to constants.
const TEN: FFloating = FFloating::from_u8(10);
const ONE_FIFTY: DFloating = DFloating::from_u16(150);
const PI: GFloating = GFloating::from_ascii("3.1415926535897932384626433832");
const MANY_ZEROES: HFloating = HFloating::from_u128(
    100_000_000_000_000_000_000_000_000_000_000u128);

assert_eq!(ten, TEN);
assert_eq!(ONE_FIFTY, DFloating::from_i32(150));
assert_eq!(PI, GFloating::from_f64(std::f64::consts::PI));
assert_eq!(MANY_ZEROES, HFloating::from_str("1.0e32").unwrap());

// Supports conversion between VAX floating point types
let ten_h = HFloating::from(ten);
let three_hundred_g = GFloating::from(three_hundred);
let twelve_point_five_f = FFloating::from(twelve_point_five);
let pi_d = DFloating::from(PI);

assert_eq!(ten_h, HFloating::from(10_u64));
assert_eq!(three_hundred_g, GFloating::from(300_u32));
assert_eq!(twelve_point_five_f, FFloating::from_str("12.5").unwrap());
assert_eq!(pi_d, DFloating::from_f64(std::f64::consts::PI));

// Supports conversion between VAX floating point types to constants
const TEN_G: GFloating = TEN.to_g_floating();
const ONE_FIFTY_H: HFloating = ONE_FIFTY.to_h_floating();
const PI_F: FFloating = PI.to_f_floating();
const MANY_ZEROES_D: DFloating = MANY_ZEROES.to_d_floating();

assert_eq!(TEN_G, GFloating::from_u8(10));
assert_eq!(ONE_FIFTY_H, HFloating::from_i32(150));
assert_eq!(PI_F, FFloating::from_f32(std::f32::consts::PI));
assert_eq!(MANY_ZEROES_D, DFloating::from_str("1.0e32").unwrap());

// Supports standard mathematical operators.
let one = TEN / ten;
let four_fifty = three_hundred + ONE_FIFTY;
let two_pi = PI * GFloating::from(2_i8);
let many_zeroes = MANY_ZEROES - very_small;

assert_eq!(one, FFloating::from_i128(1));
assert_eq!(four_fifty, DFloating::from(450_u64));
assert_eq!(two_pi, GFloating::from_f64(std::f64::consts::PI * 2.0));
assert_eq!(many_zeroes, MANY_ZEROES);

// Supports constant mathematical operators.
const TENTH: FFloating = FFloating::from_u8(1).divide_by(FFloating::from_u8(10));
const NEG_ONE_FIFTY: DFloating = DFloating::from_bits(0).subtract_by(ONE_FIFTY);
const TWO_PI: GFloating = PI.multiply_by(GFloating::from_i64(2));
const TWO_HUNDRED_NONILLION: HFloating = MANY_ZEROES.add_to(MANY_ZEROES);

assert_eq!(TENTH, FFloating::from_str("0.1").unwrap());
assert_eq!(NEG_ONE_FIFTY, -ONE_FIFTY);
assert_eq!(TWO_PI, two_pi);
assert_eq!(TWO_HUNDRED_NONILLION, HFloating::from_str("200,000,000,000,000,000,000,000,000,000,000").unwrap());

// Supports display, and lowercase and uppercase exponential output.
assert_eq!(&format!("{:.4}", TENTH), "0.1000");
assert_eq!(&format!("{}", PI), "3.141592653589793");
assert_eq!(&format!("{:e}", very_small), "1e-1000");
assert_eq!(&format!("{:.1E}", MANY_ZEROES), "1.0E32");
assert_eq!(&format!("{:.3e}", four_fifty), "4.500e2");
assert_eq!(&format!("{:.3}", four_fifty), "450.000");
```
