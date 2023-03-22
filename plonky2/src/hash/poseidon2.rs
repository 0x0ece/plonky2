//! Implementation of the Poseidon2 hash function, as described in
//! <https://eprint.iacr.org/2023/323.pdf>

use alloc::vec;
use alloc::vec::Vec;

use unroll::unroll_for_loops;

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::{Field, PrimeField64};
use crate::gates::gate::Gate;
use crate::gates::poseidon::PoseidonGate;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation, SPONGE_WIDTH};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

// NB: Changing any of these values will require regenerating all of
// the precomputed constant arrays in this file.
pub const HALF_N_FULL_ROUNDS: usize = 4;
pub(crate) const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const N_ROUNDS: usize = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;
const MAX_WIDTH: usize = 12;

/// Note that these work for the Goldilocks field, but not necessarily others.
#[rustfmt::skip]
pub const ALL_ROUND_CONSTANTS: [u64; MAX_WIDTH * N_ROUNDS]  = [
    0xe034a8785fd284a7, 0xe2463f1ea42e1b80, 0x048742e681ae290a, 0xe4af50ade990154c, 0x8b13ffaaf4f78f8a, 0xe3fbead7dccd8d63, 0x631a47705eb92bf8, 0x88fbbb8698548659, 0x74cd2003b0f349c9, 0xe16a3df6764a3f5d, 0x57ce63971a71aaa2, 0xdc1f7fd3e7823051,
    0xbb8423be34c18d7a, 0xf8bc5a2a0c1b3d6d, 0xf1a01bbd6f7123e5, 0xed960a080f5e348b, 0x1b9c0c1e87e2390e, 0x18c83caf729a613e, 0x671ab9fe037a72c4, 0x508565f67d4c276a, 0x4d2cd8827a482590, 0xa48e11e84dd3500b, 0x825a8c955fc2442b, 0xf573a6ee07cddc68,
    0x7dd3f19c73a39e0b, 0xcc0f13537a796fa6, 0x1d9006bfaedac57f, 0x4705f69b68b0b7de, 0x5b62bfb718bcc57f, 0x879d821770563827, 0x3da5ccb7f8dff0e3, 0xb49d6a706923fc5b, 0xb6a0babe883a969d, 0x2984f9b055401960, 0xcd3496f05511d79d, 0x4791da5d63854fc5,
    0xdb7344d0580a39d4, 0x5aedc4dad1de120a, 0x5e1bdc1fb8e1abf0, 0x3904c09a0e46747c, 0xb54a0e23ab85ddcd, 0xc0c3cf05bccbdb3a, 0xb362076a73baf7e9, 0x212c953d81a5d5ba, 0x212d4cc965d898bd, 0xdd44ddd0f41509b9, 0x8931329fa67823c0, 0xc65510f4d2a873be,
    0xe3ecbb6ba1e16211, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x70f5b3266792bbb6, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xe7560e690634757e, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xafd0202bc7eaf66e, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x349f4c5871f220fd, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x3697eb3e31529e0d, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x7735d5b0622d9900, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x5f5b58b9cf997668, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x645534b6548af9d9, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x4232d29d91a426a8, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xb987278aed485d35, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x6dabeef669bb406e, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x35ee78288b749d40, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x6dcd560f14af0fc3, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x71ed3dc007ea6383, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x8b6b51caab7f5b6f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xcf2e8cc4181dbfa8, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xa01d3f1c306f825a, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0xccee646a5d8ddb87, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x70df6f277cbaffeb, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x64ec0a6556b8f45c, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x6f68c9664fda6e37, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x387356e4516fab6f, 0x35310dce33903e67, 0x45f3e5251d30f912, 0x7c97f480ca428f45, 0x74d5874c20b50de2, 0xff1d5b7cee3dc67f, 0xa04d5d5ac0ff3de9, 0x1cefb5eb7d24580e, 0xf685e1bfcc0104ad, 0x6204dd95db22ead4, 0x8265c6c57c73c440, 0x4f708ab0b4e1e382,
    0xcfc60c7a52fbffa7, 0x9c0c1951d8910306, 0x4d06df27c89819f2, 0x621bdb0e75eca660, 0x343adffd079cee57, 0xa760f0e5debde398, 0xe3110fefd97b188a, 0x0ed6584e6b150297, 0x2b10e625d0d079c0, 0xefa493442057264f, 0xebcfaa7b3f26a2b6, 0xf36bcda28e343e2a,
    0xa1183cb63b67aa9e, 0x40f3e415d5e5b0ba, 0xc51fc2367eff7b15, 0xe07fe5f3aebc649f, 0xc9cb2be56968e8aa, 0x648600db69078a0e, 0x4e9135ab1256edb9, 0x00382c73435556c2, 0x1d78cafac9150ddf, 0xb8df60ab6215a233, 0xa7a65ba31f8fcd9a, 0x907d436dd964006b,
    0x3bdf7fd528633b97, 0x265adb359c0cc0f8, 0xf16cfc4034b39614, 0x71f0751b08fa0947, 0x3165eda4b5403a37, 0xca30fc5680467e46, 0x4c743354d37777c5, 0x3d1f0a4e6bba4a09, 0xc0c2e289afa75181, 0x1e4fa2ad948978b7, 0x2a226a127a0bb26a, 0xe61738a70357ce76,
];

const WIDTH: usize = SPONGE_WIDTH;
pub trait Poseidon: PrimeField64 {
    // Total number of round constants required: width of the input
    // times number of rounds.
    const N_ROUND_CONSTANTS: usize = WIDTH * N_ROUNDS;

    // The MDS matrix we use is C + D, where C is the circulant matrix whose first row is given by
    // `MDS_MATRIX_CIRC`, and D is the diagonal matrix whose diagonal is given by `MDS_MATRIX_DIAG`.
    const MDS_MATRIX_DIAG: [u64; WIDTH];

    // Precomputed constants for the fast Poseidon calculation. See
    // the paper.
    const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS];

    fn mds_layer(state: &[Self; WIDTH]) -> [Self; WIDTH] {
        let mut result = [Self::ZERO; WIDTH];
        let mut result_u128 = [0u128; WIDTH];
        let mut stored = [0u128; 4];

        // Applying cheap 4x4 MDS matrix to each 4-element part of the state
        for i in 0..3 {
            let start_index = i * 4;
            let s0 = state[start_index].to_noncanonical_u64() as u128;
            let s1 = state[start_index + 1].to_noncanonical_u64() as u128;
            let s2 = state[start_index + 2].to_noncanonical_u64() as u128;
            let s3 = state[start_index + 3].to_noncanonical_u64() as u128;
            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = t1 + 2 * s1;
            let t3 = t0 + 2 * s3;
            let t4 = t3 + 4 * t1;
            let t5 = t2 + 4 * t0;

            result_u128[start_index] = t3 + t5;
            result_u128[start_index + 1] = t5;
            result_u128[start_index + 2] = t2 + t4;
            result_u128[start_index + 3] = t4;
        }

        // Applying second cheap matrix
        for i in 0..4 {
            stored[i] = result_u128[i] + result_u128[4 + i] + result_u128[8 + i];
        }
        for i in 0..12 {
            let sum = result_u128[i] + stored[i % 4];
            let sum_lo = sum as u64;
            let sum_hi = (sum >> 64) as u32;
            result[i] = Self::from_noncanonical_u96((sum_lo, sum_hi));
        }

        result
    }

    /// Same as `mds_layer` for field extensions of `Self`.
    fn mds_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; WIDTH],
    ) -> [F; WIDTH] {
        let mut result = [F::ZERO; WIDTH];
        let mut stored = [F::ZERO; 4];
        let four = F::from_canonical_u8(4);

        // Applying cheap 4x4 MDS matrix to each 4-element part of the state
        for i in 0..3 {
            let start_index = i * 4;
            let t0 = state[start_index] + state[start_index + 1];
            let t1 = state[start_index + 2] + state[start_index + 3];
            let t2 = t1.multiply_accumulate(state[start_index + 1], F::TWO);
            let t3 = t0.multiply_accumulate(state[start_index + 3], F::TWO);
            let t4 = t3.multiply_accumulate(t1, four);
            let t5 = t2.multiply_accumulate(t0, four);

            result[start_index] = t3 + t5;
            result[start_index + 1] = t5;
            result[start_index + 2] = t2 + t4;
            result[start_index + 3] = t4;
        }

        // Applying second cheap matrix
        for i in 0..4 {
            stored[i] = result[i] + result[4 + i] + result[8 + i];
        }
        for i in 0..12 {
            result[i] += stored[i % 4];
        }

        result
    }

    /// Recursive version of `mds_layer`.
    fn mds_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &[ExtensionTarget<D>; WIDTH],
    ) -> [ExtensionTarget<D>; WIDTH]
    where
        Self: RichField + Extendable<D>,
    {
        // If we have enough routed wires, we will use PoseidonMdsGate.
        let mds_gate = PoseidonMdsGate::<Self, D>::new();
        if builder.config.num_routed_wires >= mds_gate.num_wires() {
            let index = builder.add_gate(mds_gate, vec![]);
            for i in 0..WIDTH {
                let input_wire = PoseidonMdsGate::<Self, D>::wires_input(i);
                builder.connect_extension(state[i], ExtensionTarget::from_range(index, input_wire));
            }
            (0..WIDTH)
                .map(|i| {
                    let output_wire = PoseidonMdsGate::<Self, D>::wires_output(i);
                    ExtensionTarget::from_range(index, output_wire)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        } else {
            let mut result = [builder.zero_extension(); WIDTH];
            let mut stored = [builder.zero_extension(); 4];
            let two = builder.constant(Self::TWO);
            let four = builder.constant(Self::from_canonical_u8(4));

            // Applying cheap 4x4 MDS matrix to each 4-element part of the state
            for i in 0..3 {
                let start_index = i * 4;
                let t0 = builder.add_extension(state[start_index], state[start_index + 1]);
                let t1 = builder.add_extension(state[start_index + 2], state[start_index + 3]);
                let t2 = builder.scalar_mul_add_extension(two, state[start_index + 1], t1);
                let t3 = builder.scalar_mul_add_extension(two, state[start_index + 3], t0);
                let t4 = builder.scalar_mul_add_extension(four, t1, t3);
                let t5 = builder.scalar_mul_add_extension(four, t0, t2);

                result[start_index] = builder.add_extension(t3, t5);
                result[start_index + 1] = t5;
                result[start_index + 2] = builder.add_extension(t2, t4);
                result[start_index + 3] = t4;
            }

            // Applying second cheap matrix
            for i in 0..4 {
                stored[i] = builder.add_many_extension(&[result[i], result[4 + i], result[8 + i]]);
            }
            for i in 0..12 {
                result[i] = builder.add_extension(result[i], stored[i % 4]);
            }

            result
        }
    }

    #[inline]
    fn mds_partial_layer_fast(state: &[Self; WIDTH]) -> [Self; WIDTH] {
        let mut sum = state[0].to_noncanonical_u64() as u128;
        for i in 1..12 {
            if i < WIDTH {
                let si = state[i].to_noncanonical_u64() as u128;
                sum += si;
            }
        }
        let sum_lo = sum as u64;
        let sum_hi = (sum >> 64) as u32;
        let sum = Self::from_noncanonical_u96((sum_lo, sum_hi));

        let mut result = [Self::ZERO; WIDTH];
        for i in 0..12 {
            if i < WIDTH {
                let t = Self::from_canonical_u64(Self::MDS_MATRIX_DIAG[i]);
                result[i] = sum.multiply_accumulate(state[i], t);
            }
        }
        result
    }

    /// Same as `mds_partial_layer_fast` for field extensions of `Self`.
    fn mds_partial_layer_fast_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; WIDTH],
    ) -> [F; WIDTH] {
        let mut sum = state[0];
        for i in 1..12 {
            if i < WIDTH {
                sum += state[i];
            }
        }

        let mut result = [F::ZERO; WIDTH];
        for i in 0..12 {
            if i < WIDTH {
                let t = F::from_canonical_u64(Self::MDS_MATRIX_DIAG[i]);
                result[i] = sum.multiply_accumulate(state[i], t);
            }
        }
        result
    }

    /// Recursive version of `mds_partial_layer_fast`.
    fn mds_partial_layer_fast_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &[ExtensionTarget<D>; WIDTH],
    ) -> [ExtensionTarget<D>; WIDTH]
    where
        Self: RichField + Extendable<D>,
    {
        let mut sum = state[0];
        for i in 1..12 {
            if i < WIDTH {
                sum = builder.add_extension(sum, state[i]);
            }
        }

        let mut result = [builder.zero_extension(); WIDTH];
        for i in 0..12 {
            if i < WIDTH {
                let t = <Self as Poseidon>::MDS_MATRIX_DIAG[i];
                let t = Self::Extension::from_canonical_u64(t);
                let t = builder.constant_extension(t);
                result[i] = builder.mul_add_extension(state[i], t, sum);
            }
        }
        result
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn constant_layer(state: &mut [Self; WIDTH], round_ctr: usize) {
        for i in 0..12 {
            if i < WIDTH {
                let round_constant = ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr];
                unsafe {
                    state[i] = state[i].add_canonical_u64(round_constant);
                }
            }
        }
    }

    /// Same as `constant_layer` for field extensions of `Self`.
    fn constant_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; WIDTH],
        round_ctr: usize,
    ) {
        for i in 0..WIDTH {
            state[i] += F::from_canonical_u64(ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr]);
        }
    }

    /// Recursive version of `constant_layer`.
    fn constant_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; WIDTH],
        round_ctr: usize,
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..WIDTH {
            let c = ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr];
            let c = Self::Extension::from_canonical_u64(c);
            let c = builder.constant_extension(c);
            state[i] = builder.add_extension(state[i], c);
        }
    }

    #[inline(always)]
    fn sbox_monomial<F: FieldExtension<D, BaseField = Self>, const D: usize>(x: F) -> F {
        // x |--> x^7
        let x2 = x.square();
        let x4 = x2.square();
        let x3 = x * x2;
        x3 * x4
    }

    /// Recursive version of `sbox_monomial`.
    fn sbox_monomial_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        x: ExtensionTarget<D>,
    ) -> ExtensionTarget<D>
    where
        Self: RichField + Extendable<D>,
    {
        // x |--> x^7
        builder.exp_u64_extension(x, 7)
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn sbox_layer(state: &mut [Self; WIDTH]) {
        for i in 0..12 {
            if i < WIDTH {
                state[i] = Self::sbox_monomial(state[i]);
            }
        }
    }

    /// Same as `sbox_layer` for field extensions of `Self`.
    fn sbox_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; WIDTH],
    ) {
        for i in 0..WIDTH {
            state[i] = Self::sbox_monomial(state[i]);
        }
    }

    /// Recursive version of `sbox_layer`.
    fn sbox_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        for i in 0..WIDTH {
            state[i] = Self::sbox_monomial_circuit(builder, state[i]);
        }
    }

    #[inline]
    fn full_rounds(state: &mut [Self; WIDTH], round_ctr: &mut usize) {
        for _ in 0..HALF_N_FULL_ROUNDS {
            Self::constant_layer(state, *round_ctr);
            Self::sbox_layer(state);
            *state = Self::mds_layer(state);
            *round_ctr += 1;
        }
    }

    #[inline]
    fn partial_rounds(state: &mut [Self; WIDTH], round_ctr: &mut usize) {
        for i in 0..N_PARTIAL_ROUNDS {
            unsafe {
                state[0] = state[0].add_canonical_u64(Self::FAST_PARTIAL_ROUND_CONSTANTS[i]);
            }
            state[0] = Self::sbox_monomial(state[0]);
            *state = Self::mds_partial_layer_fast(state);
        }
        *round_ctr += N_PARTIAL_ROUNDS;
    }

    #[inline]
    fn poseidon(input: [Self; WIDTH]) -> [Self; WIDTH] {
        let mut state = input;
        let mut round_ctr = 0;

        // Linear layer at beginning
        state = Self::mds_layer(&state);

        Self::full_rounds(&mut state, &mut round_ctr);
        Self::partial_rounds(&mut state, &mut round_ctr);
        Self::full_rounds(&mut state, &mut round_ctr);
        debug_assert_eq!(round_ctr, N_ROUNDS);

        state
    }
}

pub struct PoseidonPermutation;
impl<F: RichField> PlonkyPermutation<F> for PoseidonPermutation {
    fn permute(input: [F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH] {
        F::poseidon(input)
    }
}

/// Poseidon hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PoseidonHash;
impl<F: RichField> Hasher<F> for PoseidonHash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = PoseidonPermutation;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }
}

impl<F: RichField> AlgebraicHasher<F> for PoseidonHash {
    fn permute_swapped<const D: usize>(
        inputs: [Target; SPONGE_WIDTH],
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> [Target; SPONGE_WIDTH]
    where
        F: RichField + Extendable<D>,
    {
        let gate_type = PoseidonGate::<F, D>::new();
        let gate = builder.add_gate(gate_type, vec![]);

        let swap_wire = PoseidonGate::<F, D>::WIRE_SWAP;
        let swap_wire = Target::wire(gate, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        for i in 0..SPONGE_WIDTH {
            let in_wire = PoseidonGate::<F, D>::wire_input(i);
            let in_wire = Target::wire(gate, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Collect output wires.
        (0..SPONGE_WIDTH)
            .map(|i| Target::wire(gate, PoseidonGate::<F, D>::wire_output(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::field::types::Field;
    use crate::hash::hashing::SPONGE_WIDTH;
    use crate::hash::poseidon::Poseidon;

    pub(crate) fn check_test_vectors<F: Field>(
        test_vectors: Vec<([u64; SPONGE_WIDTH], [u64; SPONGE_WIDTH])>,
    ) where
        F: Poseidon,
    {
        for (input_, expected_output_) in test_vectors.into_iter() {
            let mut input = [F::ZERO; SPONGE_WIDTH];
            for i in 0..SPONGE_WIDTH {
                input[i] = F::from_canonical_u64(input_[i]);
            }
            let output = F::poseidon(input);
            for i in 0..SPONGE_WIDTH {
                let ex_output = F::from_canonical_u64(expected_output_[i]);
                assert_eq!(output[i], ex_output);
            }
        }
    }
}
