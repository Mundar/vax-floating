use criterion::{criterion_group, criterion_main, Criterion, black_box};

fn u32_rotate(value: u32) -> u32 {
    value.rotate_right(16)
}

fn u64_rotate(value: u64) -> u64 {
    let low = value as u32;
    let low = low.rotate_right(16) as u64;
    let high = (value >> 32) as u32;
    let high = high.rotate_right(16) as u64;
    (low << 32) | high
}

fn u64_shift(value: u64) -> u64 {
    ((value & 0xFFFF) << 48) |
        ((value & 0xFFFF0000) << 16) |
        ((value & 0xFFFF00000000) >> 16) |
        ((value >> 48) & 0xFFFF)
}

fn u64_transmute_swap(value: u64) -> u64 {
    let mut words = unsafe { std::mem::transmute::<u64, [u16; 4]>(value) };
    words.swap(0, 3);
    words.swap(1, 2);
    unsafe { std::mem::transmute::<[u16; 4], u64>(words) }
}

fn u64_transmute_rotate(value: u64) -> u64 {
    let value = value.rotate_right(32);
    let mut longs = unsafe { std::mem::transmute::<u64, [u32; 2]>(value) };
    longs[0] = longs[0].rotate_right(16);
    longs[1] = longs[1].rotate_right(16);
    unsafe { std::mem::transmute::<[u32; 2], u64>(longs) }
}

// This is probably the safest because it doesn't rely on the endianess of the target machine.
fn u64_copy_from_slice(value: u64) -> u64 {
    let mut output = [0; 8];
    let value_bytes = value.to_le_bytes();
    output[0..=1].copy_from_slice(&value_bytes[6..=7]);
    output[2..=3].copy_from_slice(&value_bytes[4..=5]);
    output[4..=5].copy_from_slice(&value_bytes[2..=3]);
    output[6..=7].copy_from_slice(&value_bytes[0..=1]);
    u64::from_le_bytes(output)
}

fn u128_rotate(value: u128) -> u128 {
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

fn u128_shift(value: u128) -> u128 {
    ((value & 0xFFFF) << 112) |
        ((value & 0xFFFF0000) << 80) |
        ((value & 0xFFFF00000000) << 48) |
        ((value & 0xFFFF000000000000) << 16) |
        ((value >> 16) & 0xFFFF000000000000) |
        ((value >> 48) & 0xFFFF00000000) |
        ((value >> 80) & 0xFFFF0000) |
        ((value >> 112) & 0xFFFF)
}

fn u128_transmute_swap(value: u128) -> u128 {
    let mut words = unsafe { std::mem::transmute::<u128, [u16; 8]>(value) };
    words.swap(0, 7);
    words.swap(1, 6);
    words.swap(2, 5);
    words.swap(3, 4);
    unsafe { std::mem::transmute::<[u16; 8], u128>(words) }
}

fn u128_transmute_rotate(value: u128) -> u128 {
    let value = value.rotate_right(64);
    let mut quads = unsafe { std::mem::transmute::<u128, [u64; 2]>(value) };
    quads[0] = quads[0].rotate_right(32);
    quads[1] = quads[1].rotate_right(32);
    let mut longs = unsafe { std::mem::transmute::<[u64; 2], [u32; 4]>(quads) };
    longs[0] = longs[0].rotate_right(16);
    longs[1] = longs[1].rotate_right(16);
    longs[2] = longs[2].rotate_right(16);
    longs[3] = longs[3].rotate_right(16);
    unsafe { std::mem::transmute::<[u32; 4], u128>(longs) }
}

// This is probably the safest because it doesn't rely on the endianess of the target machine.
fn u128_copy_from_slice(value: u128) -> u128 {
    let mut output = [0; 16];
    let value_bytes = value.to_le_bytes();
    output[0..=1].copy_from_slice(&value_bytes[14..=15]);
    output[2..=3].copy_from_slice(&value_bytes[12..=13]);
    output[4..=5].copy_from_slice(&value_bytes[10..=11]);
    output[6..=7].copy_from_slice(&value_bytes[8..=9]);
    output[8..=9].copy_from_slice(&value_bytes[6..=7]);
    output[10..=11].copy_from_slice(&value_bytes[4..=5]);
    output[12..=13].copy_from_slice(&value_bytes[2..=3]);
    output[14..=15].copy_from_slice(&value_bytes[0..=1]);
    u128::from_le_bytes(output)
}

macro_rules! test_and_bench {
    ($group: ident, $name: ident, $start: ident, $end: ident) => {
        assert_eq!($name($start), $end, concat!("Verification failed for ", stringify!($name),
            ":\nleft =  {:#X},\nright = {:#X}"), $name($start), $end);
        $group.bench_function(stringify!($name), |b| b.iter(|| $name(black_box($start))));
    };
}

fn bench_flip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Swap Words");
    let start_u32 = 0x11223344_u32;
    let end_u32 = 0x33441122_u32;
    test_and_bench!(group, u32_rotate, start_u32, end_u32);
    let start_u64 = 0x1122334455667788_u64;
    let end_u64 = 0x7788556633441122_u64;
    test_and_bench!(group, u64_rotate, start_u64, end_u64);
    test_and_bench!(group, u64_shift, start_u64, end_u64);
    test_and_bench!(group, u64_transmute_swap, start_u64, end_u64);
    test_and_bench!(group, u64_transmute_rotate, start_u64, end_u64);
    test_and_bench!(group, u64_copy_from_slice, start_u64, end_u64);
    let start_u128 = 0x00112233445566778899AABBCCDDEEFF_u128;
    let end_u128 = 0xEEFFCCDDAABB88996677445522330011_u128;
    test_and_bench!(group, u128_rotate, start_u128, end_u128);
    test_and_bench!(group, u128_shift, start_u128, end_u128);
    test_and_bench!(group, u128_transmute_swap, start_u128, end_u128);
    test_and_bench!(group, u128_transmute_rotate, start_u128, end_u128);
    test_and_bench!(group, u128_copy_from_slice, start_u128, end_u128);
    group.finish();
}

criterion_group!(benches, bench_flip);
criterion_main!(benches);
