# Change Log

## [Unreleased]

## [0.3.0] - 2026-04-26

### Changed

- Changed calls from std::mem::transmute into f32 and f64 to_bits and from_bits. These functions were made stable for const in rust version 1.83.0, and newer compilers complain about the use of std::mem::transmute. This bumps up the minimum supported rust version from 1.67 to 1.83.0.

## [0.2.0] - 2024-08-18

### Added

- Added conversion functions to switch between `VaxFloatingPoint` sizes.
- Added conversion functions for all of the VAX floating point types to each
    other.
