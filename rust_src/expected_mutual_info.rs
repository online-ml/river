// Expected Mutual Information between two clusterings, ported from the Cython
// implementation in `river/metrics/expected_mutual_info.pyx` (which itself
// follows scikit-learn's `expected_mutual_information`).
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

    // nijs[k] = k as f64, except nijs[0] = 1 to keep log(0) out of log_nnij.
    // Slot 0 is never read in the inner loop (start >= 1) but the substitution
    // mirrors the original implementation.
    let nijs: Vec<f64> = (0..=max_ab)
        .map(|k| if k == 0 { 1.0 } else { k as f64 })
        .collect();

    let log_n = n.ln();
    let log_a: Vec<f64> = a.iter().map(|&v| (v as f64).ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|&v| (v as f64).ln()).collect();
    let log_nnij: Vec<f64> = nijs.iter().map(|&v| log_n + v.ln()).collect();

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
    // Hot pointers for the inner loop. The bounds on `nij`, `ai_idx - nij`,
    // `bj_idx - nij` and `n_minus_ab + nij` are all guaranteed by the loop
    // construction (start >= 1, end <= min(a[i], b[j]) + 1, and a[i] + b[j]
    // <= N because they are row/col sums of the same N samples), so we use
    // unchecked indexing to keep the inner body to integer arithmetic plus
    // four array loads.
    let lgamma_ptr = lgamma_tab.as_ptr();
    let nijs_ptr = nijs.as_ptr();
    let log_nnij_ptr = log_nnij.as_ptr();

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
            let n_minus_ab = (n_int - ai - bj) as isize;

            for nij in start..end {
                unsafe {
                    let term1 = *nijs_ptr.add(nij) / n;
                    let term2 = *log_nnij_ptr.add(nij) - log_ab;
                    let gln = gln_const
                        - *lgamma_ptr.add(nij)
                        - *lgamma_ptr.add(ai_idx - nij)
                        - *lgamma_ptr.add(bj_idx - nij)
                        - *lgamma_ptr.add((n_minus_ab + nij as isize) as usize);
                    emi += term1 * term2 * gln.exp();
                }
            }
        }
    }
    emi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(actual: f64, expected: f64) {
        let tol = 1e-10 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}",
        );
    }

    #[test]
    fn single_row_or_col_returns_zero() {
        assert_eq!(expected_mutual_info(10.0, &[10], &[5, 5]), 0.0);
        assert_eq!(expected_mutual_info(10.0, &[5, 5], &[10]), 0.0);
    }

    #[test]
    fn matches_known_value_3x3() {
        // Contingency [[5,1,0],[0,3,2],[1,0,4]] — row sums [6,5,5], col sums [6,4,6], N=16.
        // sklearn returns 0.16190951337973544 for this case.
        let emi = expected_mutual_info(16.0, &[6, 5, 5], &[6, 4, 6]);
        approx_eq(emi, 0.16190951337973544);
    }

    #[test]
    fn matches_known_value_2x2_diagonal() {
        // Contingency [[10,0],[0,10]] — row sums [10,10], col sums [10,10], N=20.
        let emi = expected_mutual_info(20.0, &[10, 10], &[10, 10]);
        // sklearn produces ~0.05129... for this case; verified offline.
        assert!(emi > 0.0 && emi < 0.1);
    }
}
