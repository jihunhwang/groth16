//! A distributed Groth16 prover in the Honest Majority Multi-Prover (HMMP) model,
//! following Theorem 1.1 of Block–Garman 2021 (Ref/BlockGarman21.pdf, Section 2.1).
//!
//! All M provers run in this one process; a `Vec` stands in for the authenticated
//! broadcast channel. Groth16 is non-interactive, so the prover is a single
//! "round circuit" C_0. We split the five MSMs of C_0 into M chunks C_{0,j}; chunk j
//! is computed by the τ-window of provers π(j), ..., π(j+τ-1).

use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_ec::{CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_groth16::{
    r1cs_to_qap::{LibsnarkReduction, R1CSToQAP},
    Groth16, Proof, ProvingKey,
};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::{
    gr1cs::{
        ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, OptimizationGoal,
        SynthesisError, SynthesisMode,
    },
    lc,
};
use ark_std::{
    ops::{Add, Range},
    rand::{rngs::StdRng, seq::SliceRandom, SeedableRng},
    UniformRand, Zero,
};
use std::time::Instant;

const M: usize = 16; // number of provers
const T: usize = 5; // number of corrupt provers, t < M/2
const TAU: usize = 4; // audit window size, τ <= t
const N: usize = 1 << 12; // number of secret factors in the circuit

// Statement: "I know secret x_1, ..., x_N whose product is c" (c is public)
#[derive(Clone)]
struct ProductCircuit {
    xs: Option<Vec<Fr>>,
}

impl ConstraintSynthesizer<Fr> for ProductCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let x = |i: usize| self.xs.as_ref().map(|xs| xs[i]);

        let mut prod = x(0);
        let mut prod_var = cs.new_witness_variable(|| prod.ok_or(SynthesisError::AssignmentMissing))?;
        for i in 1..N {
            let xi = x(i);
            let xi_var = cs.new_witness_variable(|| xi.ok_or(SynthesisError::AssignmentMissing))?;
            prod = prod.zip(xi).map(|(p, x)| p * x);
            // The final product is the public input; the running products are private
            let next_var = if i == N - 1 {
                cs.new_input_variable(|| prod.ok_or(SynthesisError::AssignmentMissing))?
            } else {
                cs.new_witness_variable(|| prod.ok_or(SynthesisError::AssignmentMissing))?
            };
            cs.enforce_r1cs_constraint(|| lc!() + prod_var, || lc!() + xi_var, || lc!() + next_var)?;
            prod_var = next_var;
        }
        Ok(())
    }
}

/// The input z every prover holds: the full assignment and the QAP quotient h.
/// HMMP provers all get the same witness and have no private inputs.
struct Witness {
    h: Vec<Fr>,
    assignment: Vec<Fr>, // public inputs then private witness, without the leading 1
    num_inputs: usize,
}

/// Synthesizes the circuit and computes h, as in `create_proof_with_reduction`
/// (src/prover.rs). Every prover could do this identically; it is not split up yet.
fn prepare(circuit: ProductCircuit) -> Witness {
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    cs.set_mode(SynthesisMode::Prove {
        construct_matrices: true,
        generate_lc_assignments: false,
    });
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();

    let h = LibsnarkReduction::witness_map::<Fr, GeneralEvaluationDomain<Fr>>(cs.clone()).unwrap();
    let prover = cs.borrow().unwrap();
    let inputs = &prover.instance_assignment().unwrap()[1..];
    let aux = prover.witness_assignment().unwrap();
    Witness {
        h,
        num_inputs: inputs.len(),
        assignment: [inputs, &aux[..]].concat(),
    }
}

/// y_{0,j}: one chunk's share of each of the five MSMs in the Groth16 prover.
#[derive(Clone, Copy, PartialEq, Debug)]
struct PartialSums {
    a: G1Projective,
    b_g1: G1Projective,
    b_g2: G2Projective,
    h: G1Projective,
    l: G1Projective,
}

impl PartialSums {
    fn zero() -> Self {
        Self {
            a: G1Projective::zero(),
            b_g1: G1Projective::zero(),
            b_g2: G2Projective::zero(),
            h: G1Projective::zero(),
            l: G1Projective::zero(),
        }
    }
}

impl Add for PartialSums {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            a: self.a + other.a,
            b_g1: self.b_g1 + other.b_g1,
            b_g2: self.b_g2 + other.b_g2,
            h: self.h + other.h,
            l: self.l + other.l,
        }
    }
}

/// The j-th of M equal slices of 0..len.
fn chunk_range(len: usize, j: usize) -> Range<usize> {
    len * j / M..len * (j + 1) / M
}

/// The MSM over chunk j only; like `msm_unchecked`, it uses the shorter length.
fn msm_chunk<G: VariableBaseMSM>(bases: &[G::MulBase], scalars: &[G::ScalarField], j: usize) -> G {
    let range = chunk_range(bases.len().min(scalars.len()), j);
    G::msm_unchecked(&bases[range.clone()], &scalars[range])
}

/// The sub-circuit C_{0,j}. The constant terms (query[0], α, β, r·δ, s·δ) are left
/// to `combine` so they are added exactly once.
fn compute_chunk(pk: &ProvingKey<Bn254>, w: &Witness, j: usize) -> PartialSums {
    PartialSums {
        a: msm_chunk(&pk.a_query[1..], &w.assignment, j),
        b_g1: msm_chunk(&pk.b_g1_query[1..], &w.assignment, j),
        b_g2: msm_chunk(&pk.b_g2_query[1..], &w.assignment, j),
        h: msm_chunk(&pk.h_query, &w.h, j),
        l: msm_chunk(&pk.l_query, &w.assignment[w.num_inputs..], j),
    }
}

/// Step 2(c): turn y_{0,1}, ..., y_{0,M} into the proof, mirroring
/// `create_proof_with_assignment` in src/prover.rs.
fn combine(pk: &ProvingKey<Bn254>, r: Fr, s: Fr, parts: &[PartialSums]) -> Proof<Bn254> {
    let sum = parts.iter().fold(PartialSums::zero(), |acc, y| acc + *y);

    let a = pk.delta_g1 * r + pk.a_query[0] + sum.a + pk.vk.alpha_g1;
    let b_g1 = if r.is_zero() {
        G1Projective::zero()
    } else {
        pk.delta_g1 * s + pk.b_g1_query[0] + sum.b_g1 + pk.beta_g1
    };
    let b_g2 = pk.vk.delta_g2 * s + pk.b_g2_query[0] + sum.b_g2 + pk.vk.beta_g2;
    let c = a * s + b_g1 * r - pk.delta_g1 * (r * s) + sum.l + sum.h;

    Proof {
        a: a.into_affine(),
        b: b_g2.into_affine(),
        c: c.into_affine(),
    }
}

struct Prover {
    corrupt: bool,
    kicked: bool,
    chunks_computed: usize,
}

impl Prover {
    /// What this prover broadcasts for chunk j. Corrupt provers collude: they all
    /// send the same wrong value, which is the worst case for the audit.
    fn compute(&mut self, pk: &ProvingKey<Bn254>, w: &Witness, j: usize) -> PartialSums {
        self.chunks_computed += 1;
        let mut y = compute_chunk(pk, w, j);
        if self.corrupt {
            y.h += G1Projective::generator();
        }
        y
    }

    /// What this prover sends to the verifier at the end.
    fn final_message(&self, proof: &Proof<Bn254>) -> Proof<Bn254> {
        let mut proof = proof.clone();
        if self.corrupt {
            proof.c = (G1Projective::generator() + proof.c).into_affine();
        }
        proof
    }
}

/// The value sent by more than half of the voters.
fn majority<V: Clone + PartialEq>(votes: &[V]) -> V {
    let (winner, count) = votes
        .iter()
        .map(|v| (v, votes.iter().filter(|u| *u == v).count()))
        .max_by_key(|(_, count)| *count)
        .unwrap();
    assert!(2 * count > votes.len(), "no honest majority");
    winner.clone()
}

fn main() {
    // Fixed seed is fine for a demo; never for real keys.
    let mut rng = StdRng::seed_from_u64(0);

    // Static corruption happens before setup. Which ids doesn't matter, since π is
    // sampled afterwards.
    let mut provers: Vec<Prover> = (0..M)
        .map(|id| Prover { corrupt: id < T, kicked: false, chunks_computed: 0 })
        .collect();

    // 1. Setup_hmp: Setup_arg plus a uniformly random permutation π of the provers
    let start = Instant::now();
    let (pk, vk) = Groth16::<Bn254>::setup(ProductCircuit { xs: None }, &mut rng).unwrap();
    let mut pi: Vec<usize> = (0..M).collect();
    pi.shuffle(&mut rng);
    println!("setup: {:?}  ({N} factors)", start.elapsed());
    println!("π = {pi:?}, corrupt = {:?}", (0..T).collect::<Vec<_>>());

    // Lemma 2.2: correctness fails only if some τ-window is all corrupt
    let bad_windows = (0..M)
        .filter(|&j| (0..TAU).all(|k| provers[pi[(j + k) % M]].corrupt))
        .count();
    let bound = M as f64 * (T as f64 / M as f64).powi(TAU as i32);
    println!("all-corrupt windows: {bad_windows}  (Pr[any] <= M·(t/M)^τ = {bound:.3})");

    // The statement, and the witness every prover holds
    let xs: Vec<Fr> = (0..N).map(|_| Fr::rand(&mut rng)).collect();
    let c = xs.iter().product::<Fr>();
    let circuit = ProductCircuit { xs: Some(xs) };
    let start = Instant::now();
    let w = prepare(circuit.clone());
    println!("prepare (synthesis + h): {:?}", start.elapsed());

    // Shared randomness for zero-knowledge (Section 2.1, "Defining and Obtaining
    // Zero-Knowledge"): every prover uses the same r, s, e.g. from a beacon.
    let (r, s) = (Fr::rand(&mut rng), Fr::rand(&mut rng));

    // 2. P_hmp: compute y_{0,j} chunk by chunk, auditing as necessary
    let start = Instant::now();
    let mut audits = 0;
    let mut transcript = Vec::with_capacity(M);
    for j in 0..M {
        // 2(a): the τ-window π(j), ..., π(j+τ-1) computes C_{0,j} and broadcasts it
        let committee: Vec<usize> = (0..TAU)
            .map(|k| pi[(j + k) % M])
            .filter(|&id| !provers[id].kicked)
            .collect();
        let broadcasts: Vec<(usize, PartialSums)> = committee
            .iter()
            .map(|&id| (id, provers[id].compute(&pk, &w, j)))
            .collect();

        // 2(b): if the window disagrees, run a global audit
        let agreed = !broadcasts.is_empty() && broadcasts.iter().all(|(_, y)| *y == broadcasts[0].1);
        let y = if agreed {
            broadcasts[0].1
        } else {
            audits += 1;
            let active: Vec<usize> = (0..M).filter(|&id| !provers[id].kicked).collect();
            let votes: Vec<(usize, PartialSums)> = active
                .into_iter()
                .map(|id| (id, provers[id].compute(&pk, &w, j)))
                .collect();
            let y = majority(&votes.iter().map(|(_, y)| *y).collect::<Vec<_>>());
            for (id, _) in broadcasts.iter().chain(&votes).filter(|(_, v)| *v != y) {
                if !provers[*id].kicked {
                    println!("chunk {j}: audit kicked out prover {id}");
                    provers[*id].kicked = true;
                }
            }
            y
        };
        transcript.push(y);
    }

    // 2(c): every remaining prover combines the chunks and sends its proof
    let proof = combine(&pk, r, s, &transcript);
    let messages: Vec<Proof<Bn254>> = provers
        .iter()
        .filter(|p| !p.kicked)
        .map(|p| p.final_message(&proof))
        .collect();
    println!("distributed prove: {:?}  ({audits} audits)", start.elapsed());

    // 3. V_hmp: take the majority of the received proofs and run V_arg
    let start = Instant::now();
    let received = majority(&messages);
    let ok = Groth16::<Bn254>::verify(&vk, &[c], &received).unwrap();
    println!("verify: {:?}  → {ok}", start.elapsed());

    // Sanity check: with the same r, s, a single prover gets the identical proof
    let start = Instant::now();
    let single = Groth16::<Bn254>::create_proof_with_reduction(circuit, &pk, r, s).unwrap();
    println!("single-prover prove: {:?}", start.elapsed());
    println!("distributed proof == single-prover proof: {}", received == single);

    // Per-prover work, in chunks (a single prover computes all M of them)
    for (id, p) in provers.iter().enumerate() {
        let role = if p.corrupt { "corrupt" } else { "honest " };
        let status = if p.kicked { "kicked" } else { "" };
        println!("prover {id:2} {role} chunks computed: {:2}  {status}", p.chunks_computed);
    }
}
