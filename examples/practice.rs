use ark_bn254::{Bn254, Fr};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_groth16::Groth16;
use ark_relations::{
    gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError},
    lc,
};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use ark_std::UniformRand;
use std::time::Instant;

// Statement: "I know secret a, b such that a * b = c" (c is public)
struct MulCircuit {
    a: Option<Fr>,
    b: Option<Fr>,
}

impl ConstraintSynthesizer<Fr> for MulCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            Ok(self.a.ok_or(SynthesisError::AssignmentMissing)?
                * self.b.ok_or(SynthesisError::AssignmentMissing)?)
        })?;
        cs.enforce_r1cs_constraint(|| lc!() + a, || lc!() + b, || lc!() + c)?;
        Ok(())
    }
}

fn main() {
    // Fixed seed is fine for a demo; never for real keys.
    let mut rng = StdRng::seed_from_u64(0);

    // 1. Setup: keys depend only on the circuit's shape, not the values
    let start = Instant::now();
    let (pk, vk) =
        Groth16::<Bn254>::setup(MulCircuit { a: None, b: None }, &mut rng).unwrap();
    println!("setup:  {:?}", start.elapsed());

    // Field elements are ~77 digits long, so only print the start of each
    let short = |x: Fr| {
        let s = x.to_string();
        if s.len() > 10 { format!("{}...", &s[..10]) } else { s }
    };

    // 2–3. Prove and verify for random pairs (a, b), reusing the same keys
    for i in 1..=9 {
        let (a, b) = (Fr::rand(&mut rng), Fr::rand(&mut rng));
        let c = a * b;

        let start = Instant::now();
        let proof = Groth16::<Bn254>::prove(&pk, MulCircuit { a: Some(a), b: Some(b) }, &mut rng)
            .unwrap();
        let prove_time = start.elapsed();

        // What actually gets sent to the verifier: the compressed bytes
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).unwrap();
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();

        // The verifier sees only the public input c
        let start = Instant::now();
        let ok = Groth16::<Bn254>::verify(&vk, &[c], &proof).unwrap();
        let verify_time = start.elapsed();
        let bad = Groth16::<Bn254>::verify(&vk, &[c + Fr::from(1u64)], &proof).unwrap();

        println!(
            "#{i} a={} b={} c={}  prove {prove_time:?}  verify {verify_time:?}  \
             c ok: {ok}  c+1 ok: {bad}  proof {}...",
            short(a),
            short(b),
            short(c),
            &hex[..16],
        );
    }
}
