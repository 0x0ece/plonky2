// Gates have `new` methods that return `GateRef`s.
#![allow(clippy::new_ret_no_self)]

pub mod arithmetic_base;
pub mod arithmetic_extension;
pub mod base_sum;
pub mod constant;
pub mod coset_interpolation;
pub mod exponentiation;
pub mod gate;
pub mod multiplication_extension;
pub mod noop;
pub mod packed_util;
#[cfg(feature = "poseidon1")]
pub mod poseidon;
#[cfg(not(feature = "poseidon1"))]
pub mod poseidon2;
#[cfg(not(feature = "poseidon1"))]
pub mod poseidon2_mds;
#[cfg(feature = "poseidon1")]
pub mod poseidon_mds;
pub mod public_input;
pub mod random_access;
pub mod reducing;
pub mod reducing_extension;
pub(crate) mod selectors;
pub mod util;

// Can't use #[cfg(test)] here because it needs to be visible to other crates.
// See https://github.com/rust-lang/cargo/issues/8379
#[cfg(any(feature = "gate_testing", test))]
pub mod gate_testing;

#[cfg(not(feature = "poseidon1"))]
pub use poseidon2 as poseidon;
#[cfg(not(feature = "poseidon1"))]
pub use poseidon2_mds as poseidon_mds;
