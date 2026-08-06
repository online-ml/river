use criterion::{black_box, criterion_group, criterion_main, Criterion};
use river::stats::Bivariate;

fn bench_covariance_update_10k(c: &mut Criterion) {
    let data: Vec<(f64, f64)> = (0..10_000)
        .map(|i| ((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
        .collect();

    c.bench_function("covariance_update_10k", |b| {
        b.iter(|| {
            let mut covariance = river::covariance::Covariance::<f64>::new(1);
            for &(x, y) in &data {
                covariance.update(black_box(x), black_box(y));
            }
            covariance.get()
        })
    });
}

criterion_group!(benches, bench_covariance_update_10k);
criterion_main!(benches);
