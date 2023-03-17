//! Implementations for Poseidon2 over Goldilocks field of width 12.

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon::{Poseidon, N_PARTIAL_ROUNDS};

#[rustfmt::skip]
impl Poseidon for GoldilocksField {
    // The MDS matrix we use is C + D, where C is the circulant matrix whose first row is given by
    // `MDS_MATRIX_CIRC`, and D is the diagonal matrix whose diagonal is given by `MDS_MATRIX_DIAG`.
    //
    // WARNING: If the MDS matrix is changed, then the following
    // constants need to be updated accordingly:
    //  - FAST_PARTIAL_ROUND_CONSTANTS
    //  - FAST_PARTIAL_ROUND_VS
    //  - FAST_PARTIAL_ROUND_W_HATS
    //  - FAST_PARTIAL_ROUND_INITIAL_MATRIX
    const MDS_MATRIX_DIAG: [u64; 12] = [
        0xcf6f77ac16722af9,
        0x3fd4c0d74672aebc,
        0x9b72bf1c1c3d08a8,
        0xe4940f84b71e4ac2,
        0x61b27b077118bc72,
        0x2efd8379b8e661e2,
        0x858edcf353df0341,
        0x2d9c20affb5c4516,
        0x5120143f0695defb,
        0x62fc898ae34a5c5b,
        0xa3d9560c99123ed2,
        0x98fd739d8e7fc933,
    ];

    const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS]  = [
        0xe3ecbb6ba1e16211,
        0x70f5b3266792bbb6,
        0xe7560e690634757e,
        0xafd0202bc7eaf66e,
        0x349f4c5871f220fd,
        0x3697eb3e31529e0d,
        0x7735d5b0622d9900,
        0x5f5b58b9cf997668,
        0x645534b6548af9d9,
        0x4232d29d91a426a8,
        0xb987278aed485d35,
        0x6dabeef669bb406e,
        0x35ee78288b749d40,
        0x6dcd560f14af0fc3,
        0x71ed3dc007ea6383,
        0x8b6b51caab7f5b6f,
        0xcf2e8cc4181dbfa8,
        0xa01d3f1c306f825a,
        0xccee646a5d8ddb87,
        0x70df6f277cbaffeb,
        0x64ec0a6556b8f45c,
        0x6f68c9664fda6e37,
    ];

    // #[cfg(all(target_arch="x86_64", target_feature="avx2", target_feature="bmi2"))]
    // #[inline]
    // fn poseidon(input: [Self; 12]) -> [Self; 12] {
    //     unsafe {
    //         crate::hash::arch::x86_64::poseidon_goldilocks_avx2_bmi2::poseidon(&input)
    //     }
    // }

    // #[cfg(all(target_arch="x86_64", target_feature="avx2", target_feature="bmi2"))]
    // #[inline(always)]
    // fn constant_layer(state: &mut [Self; 12], round_ctr: usize) {
    //     unsafe {
    //         crate::hash::arch::x86_64::poseidon_goldilocks_avx2_bmi2::constant_layer(state, round_ctr);
    //     }
    // }

    // #[cfg(all(target_arch="x86_64", target_feature="avx2", target_feature="bmi2"))]
    // #[inline(always)]
    // fn sbox_layer(state: &mut [Self; 12]) {
    //     unsafe {
    //         crate::hash::arch::x86_64::poseidon_goldilocks_avx2_bmi2::sbox_layer(state);
    //     }
    // }

    // #[cfg(all(target_arch="x86_64", target_feature="avx2", target_feature="bmi2"))]
    // #[inline(always)]
    // fn mds_layer(state: &[Self; 12]) -> [Self; 12] {
    //     unsafe {
    //         crate::hash::arch::x86_64::poseidon_goldilocks_avx2_bmi2::mds_layer(state)
    //     }
    // }

    // #[cfg(all(target_arch="aarch64", target_feature="neon"))]
    // #[inline]
    // fn poseidon(input: [Self; 12]) -> [Self; 12] {
    //     unsafe {
    //         crate::hash::arch::aarch64::poseidon_goldilocks_neon::poseidon(input)
    //     }
    // }

    // #[cfg(all(target_arch="aarch64", target_feature="neon"))]
    // #[inline(always)]
    // fn sbox_layer(state: &mut [Self; 12]) {
    //     unsafe {
    //         crate::hash::arch::aarch64::poseidon_goldilocks_neon::sbox_layer(state);
    //     }
    // }

    // #[cfg(all(target_arch="aarch64", target_feature="neon"))]
    // #[inline(always)]
    // fn mds_layer(state: &[Self; 12]) -> [Self; 12] {
    //     unsafe {
    //         crate::hash::arch::aarch64::poseidon_goldilocks_neon::mds_layer(state)
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, PrimeField64};
    use crate::hash::poseidon::test_helpers::check_test_vectors;

    #[test]
    fn test_vectors() {
        // Test inputs are:
        // 1. all zeros
        // 2. range 0..WIDTH
        // 3. all -1's
        // 4. random elements of GoldilocksField.
        // expected output calculated with (modified) code from:
        // https://github.com/HorizenLabs/poseidon2

        let neg_one: u64 = F::NEG_ONE.to_canonical_u64();

        #[rustfmt::skip]
        let test_vectors12: Vec<([u64; 12], [u64; 12])> = vec![
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
             [0x40e4c449028440b0, 0x8d5233d4e1ff2fa3, 0xd8809d1ad0ac1e59, 0x5b9f29d2bf3a7634,
              0xb65233ad9dac1203, 0x7ad00f9950e4cac3, 0x98a0e39e8dac0fcf, 0xd307875ffd783c9b,
              0x52eedea15d5113c1, 0xdd0572185641c6cd, 0xf16bca8e9ac45377, 0xe8608627f86706ee, ]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ],
             [0xed3dbcc4ff1e8d33, 0xfb85eac6ac91a150, 0xd41e1e237ed3e2ef, 0x5e289bf0a4c11897,
              0x4398b20f93e3ba6b, 0x5659a48ffaf2901d, 0xe44d81e89a88f8ae, 0x08efdb285f8c3dbc,
              0x294ab7503297850e, 0xa11c61f4870b9904, 0xa6855c112cc08968, 0x17c6d53d2fb3e8c1, ]),
            ([neg_one, neg_one, neg_one, neg_one,
              neg_one, neg_one, neg_one, neg_one,
              neg_one, neg_one, neg_one, neg_one, ],
             [0x43440ca93949b6a3, 0xb5325069fc0c5be, 0x27f20a696c4a412f, 0xd3ef1fc535544e99, 
              0xa86f114930aa54e7, 0xe8de96658e80539a, 0xf491b996d437406e, 0x4a72b9ae397da707, 
              0x47e6721112b1948a, 0xd315f6e6a1f694c4, 0xeaef93714a28f7e9, 0x73a68e8d713b126, ]),
            ([0x1d7c737cc91e6483, 0x4feb55496ea6b6e6, 0x53423d77d4f30aa2, 0xe6138c885b0a5dbf, 
              0x6b483f95c6db532a, 0xe679be9527cdb0b5, 0xa17d03ae46d7a37, 0xe54703b5c2b20691, 
              0x335f1485f09e0343, 0x6dc544817c0912e, 0x3021fdd09dd3c146, 0x59886699bae1a62f, ],
             [0xdcad441ef7e65d47, 0x95c2bda147c5a79d, 0xc60fdabee2f2b228, 0x3a6fe78ed2edcfe2, 
              0x4cbe241c5329e4aa, 0x1fa87ef1df8f5a16, 0x3586e88a1da0e9d4, 0xf0a171763063ec4d, 
              0x53f4bd20938996a8, 0xc1d1ef0e1ab6a925, 0x3d20ed2799ffce06, 0x12357cf854fbc4a8, ]),
        ];

        check_test_vectors::<F>(test_vectors12);
    }
}
