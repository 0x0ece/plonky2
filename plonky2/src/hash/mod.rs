mod arch;
pub mod hash_types;
pub mod hashing;
pub mod keccak;
pub mod merkle_proofs;
pub mod merkle_tree;
pub mod path_compression;
#[cfg(feature = "poseidon1")]
pub mod poseidon;
#[cfg(not(feature = "poseidon1"))]
pub mod poseidon2;
#[cfg(not(feature = "poseidon1"))]
pub mod poseidon2_goldilocks;
#[cfg(feature = "poseidon1")]
pub mod poseidon_goldilocks;

#[cfg(not(feature = "poseidon1"))]
pub use poseidon2 as poseidon;
#[cfg(not(feature = "poseidon1"))]
pub use poseidon2_goldilocks as poseidon_goldilocks;
