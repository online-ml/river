// Expected Mutual Information between two clusterings, following
// scikit-learn's `expected_mutual_information`.
//
// Inputs are the contingency table's row sums (`a`) and column sums (`b`)
// after dropping zero entries, plus the sample count `n_samples`. EMI is
// independent of the order of `a` and `b` because the formula sums over all
// (a[i], b[j]) pairs symmetrically.

use libm::lgamma;

pub fn expected_mutual_info(n_samples: f64, a: &[i64], b: &[i64]) -> f64 {
    let r = a.len();
    let c = b.len();

    // Any labelling with a single non-empty class implies EMI = 0.
    if r <= 1 || c <= 1 {
        return 0.0;
    }

    let n = n_samples;
    let n_int = n_samples as i64;

    let max_ab = a
        .iter()
        .chain(b.iter())
        .copied()
        .max()
        .expect("a and b non-empty by guard above") as usize;

    // Precompute lgamma(k + 1) for k in 0..=n_int. The inner loop's three
    // lgamma calls all take integer arguments in [0, n_int], so a single
    // shared table replaces them with array lookups — the same trick
    // scikit-learn uses in its EMI implementation.
    let n_table: usize = n_int as usize;
    let lgamma_tab: Vec<f64> = (0..=n_table).map(|k| lgamma(k as f64 + 1.0)).collect();

    let log_n = n.ln();
    let log_a: Vec<f64> = a.iter().map(|&v| (v as f64).ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|&v| (v as f64).ln()).collect();
    let log_nnij: Vec<f64> = (0..=max_ab)
        .map(|k| log_n + (k.max(1) as f64).ln())
        .collect();

    let gln_a: Vec<f64> = a.iter().map(|&v| lgamma_tab[v as usize]).collect();
    let gln_b: Vec<f64> = b.iter().map(|&v| lgamma_tab[v as usize]).collect();
    let gln_na: Vec<f64> = a
        .iter()
        .map(|&v| lgamma_tab[(n_int - v) as usize])
        .collect();
    let gln_nb: Vec<f64> = b
        .iter()
        .map(|&v| lgamma_tab[(n_int - v) as usize])
        .collect();
    let gln_n = lgamma_tab[n_table];

    let mut emi = 0.0_f64;
    for i in 0..r {
        let ai = a[i];
        let ai_idx = ai as usize;
        for j in 0..c {
            let bj = b[j];
            let bj_idx = bj as usize;

            let start_signed = ai + bj - n_int;
            let start = if start_signed < 1 {
                1_usize
            } else {
                start_signed as usize
            };
            let end = ai.min(bj) as usize + 1;

            let gln_const = gln_a[i] + gln_b[j] + gln_na[i] + gln_nb[j] - gln_n;
            let log_ab = log_a[i] + log_b[j];
            let n_minus_ab = n_int - ai - bj;
            let mode = ((((ai + 1) as i128 * (bj + 1) as i128) / (n_int + 2) as i128) as usize)
                .clamp(start, end - 1);
            let mode_i64 = mode as i64;
            let mode_probability = (gln_const
                - lgamma_tab[mode]
                - lgamma_tab[ai_idx - mode]
                - lgamma_tab[bj_idx - mode]
                - lgamma_tab[(n_minus_ab + mode_i64) as usize])
                .exp();
            let mut probability = mode_probability;

            emi += mode as f64 / n * (log_nnij[mode] - log_ab) * probability;

            for nij in mode + 1..end {
                let previous = nij as i64 - 1;
                probability *= (ai - previous) as f64 * (bj - previous) as f64
                    / (nij as f64 * (n_minus_ab + nij as i64) as f64);
                emi += nij as f64 / n * (log_nnij[nij] - log_ab) * probability;
            }

            probability = mode_probability;
            for nij in (start..mode).rev() {
                let next = nij as i64 + 1;
                probability *= next as f64 * (n_minus_ab + next) as f64
                    / ((ai - nij as i64) as f64 * (bj - nij as i64) as f64);
                emi += nij as f64 / n * (log_nnij[nij] - log_ab) * probability;
            }
        }
    }
    emi
}
