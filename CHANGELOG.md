# Change Log

## [Unreleased]

## [0.2.1] - 2026-04-28

### Added

- Added tests for the reported bugs:
    - AddAssign and SubAssign divide
    - EPSILON is wrong for all four types
    - abs() turns a reserved operand into zero
    - to_f32/to_f64 are wrong in the IEEE subnormal range
- Added vax_float_consts example executable to display vax floating constant
    values (since they don't get displayed in the docs).
- Added exponent display of value to the Debug output.
- Proptest regressions since proptest found an edge case in the "close enough"
    calculation comparing floating point displays.

### Changed

- Changed how floating-point numbers are created from the raw bits in a const
    context. Prevents warnings on newer compilers while maintaining backwards
    compatibility.
- Changed the test output for the 'close_enough_inner' and
    'close_enough_inner_2' functions so they show what the crate produces and
    the 'expected' value from Rust.

### Fixed

- AddAssign and SubAssign fixed to perform the correct operation.
- EPSILON constant corrected.
- abs() no longer converts a divide by zero error into a zero.
- Fixed the conversion of VAX floating types into subnormal range and saturate
    underflows to 0.0.
- Fixed the test for comparing the display output to accept 1.000e-20 and
    9.999e-21 as close enough due to rounding differences between the
    vax_floating crate and Rust's built in floating point types.

## [0.2.0] - 2024-08-18

### Added

- Added conversion functions to switch between `VaxFloatingPoint` sizes.
- Added conversion functions for all of the VAX floating point types to each
    other.
