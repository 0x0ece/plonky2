use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::marker::PhantomData;
use core::ops::Range;

use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::hashing::SPONGE_WIDTH;
use crate::hash::poseidon::Poseidon;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};

/// Poseidon MDS Gate
#[derive(Debug, Default)]
pub struct PoseidonMdsGate<F: RichField + Extendable<D> + Poseidon, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> PoseidonMdsGate<F, D> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    pub fn wires_input(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        i * D..(i + 1) * D
    }

    pub fn wires_output(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        (SPONGE_WIDTH + i) * D..(SPONGE_WIDTH + i + 1) * D
    }

    // Following are methods analogous to ones in `Poseidon`, but for extension algebras.

    /// Same as `mds_layer` for an extension algebra of `F`.
    fn mds_layer_algebra(
        state: &[ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH] {
        let mut result = [ExtensionAlgebra::ZERO; SPONGE_WIDTH];
        let mut stored = [ExtensionAlgebra::ZERO; 4];
        let four = F::Extension::from_canonical_u8(4);

        // Applying cheap 4x4 MDS matrix to each 4-element part of the state
        for i in 0..3 {
            let start_index = i * 4;
            let t0 = state[start_index] + state[start_index + 1];
            let t1 = state[start_index + 2] + state[start_index + 3];
            let t2 = t1 + state[start_index + 1] + state[start_index + 1];
            let t3 = t0 + state[start_index + 3] + state[start_index + 3];
            let t4 = t3 + t1.scalar_mul(four);
            let t5 = t2 + t0.scalar_mul(four);

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

    /// Same as `mds_layer_recursive` for an extension algebra of `F`.
    fn mds_layer_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        state: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebraTarget<D>; SPONGE_WIDTH] {
        let mut result = [builder.zero_ext_algebra(); SPONGE_WIDTH];
        let mut stored = [builder.zero_ext_algebra(); 4];
        let two = builder.constant_extension(F::Extension::TWO);
        let four = builder.constant_extension(F::Extension::from_canonical_u8(4));

        // Applying cheap 4x4 MDS matrix to each 4-element part of the state
        for i in 0..3 {
            let start_index = i * 4;
            let t0 = builder.add_ext_algebra(state[start_index], state[start_index + 1]);
            let t1 = builder.add_ext_algebra(state[start_index + 2], state[start_index + 3]);
            let t2 = builder.scalar_mul_add_ext_algebra(two, state[start_index + 1], t1);
            let t3 = builder.scalar_mul_add_ext_algebra(two, state[start_index + 3], t0);
            let t4 = builder.scalar_mul_add_ext_algebra(four, t1, t3);
            let t5 = builder.scalar_mul_add_ext_algebra(four, t0, t2);

            result[start_index] = builder.add_ext_algebra(t3, t5);
            result[start_index + 1] = t5;
            result[start_index + 2] = builder.add_ext_algebra(t2, t4);
            result[start_index + 3] = t4;
        }

        // Applying second cheap matrix
        for i in 0..4 {
            stored[i] = builder.add_ext_algebra(result[i], result[4 + i]);
            stored[i] = builder.add_ext_algebra(stored[i], result[8 + i]);
        }
        for i in 0..12 {
            result[i] = builder.add_ext_algebra(result[i], stored[i % 4]);
        }

        result
    }
}

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> Gate<F, D> for PoseidonMdsGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_layer_algebra(&inputs);

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = F::mds_layer_field(&inputs);

        yield_constr.many(
            (0..SPONGE_WIDTH)
                .map(|i| vars.get_local_ext(Self::wires_output(i)))
                .zip(computed_outputs)
                .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array()),
        )
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_layer_algebra_circuit(builder, &inputs);

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| {
                builder
                    .sub_ext_algebra(out, computed_out)
                    .to_ext_target_array()
            })
            .collect()
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        let gen = PoseidonMdsGenerator::<D> { row };
        vec![Box::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        2 * D * SPONGE_WIDTH
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        SPONGE_WIDTH * D
    }
}

#[derive(Clone, Debug)]
struct PoseidonMdsGenerator<const D: usize> {
    row: usize,
}

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> SimpleGenerator<F>
    for PoseidonMdsGenerator<D>
{
    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .flat_map(|i| {
                Target::wires_from_range(self.row, PoseidonMdsGate::<F, D>::wires_input(i))
            })
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_local_get_target = |wire_range| ExtensionTarget::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_get_target(wire_range));

        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| get_local_ext(PoseidonMdsGate::<F, D>::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let outputs = F::mds_layer_field(&inputs);

        for (i, &out) in outputs.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_get_target(PoseidonMdsGate::<F, D>::wires_output(i)),
                out,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon_mds::PoseidonMdsGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonMdsGate::<F, D>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonMdsGate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}
