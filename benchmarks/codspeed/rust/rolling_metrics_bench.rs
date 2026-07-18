use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_rolling_roc_auc_10k(c: &mut Criterion) {
    let data: Vec<(i32, f64)> = (0..10_000)
        .map(|i| (i32::from(i % 3 != 0), (i as f64 * 0.7).sin() / 2.0 + 0.5))
        .collect();

    c.bench_function("rolling_roc_auc_10k", |b| {
        b.iter(|| {
            let mut metric = river::rolling_roc_auc::RollingROCAUC::new(1, 1000);
            for &(label, score) in &data {
                metric.update(black_box(label), black_box(score));
            }
            metric.get()
        })
    });
}

fn bench_rolling_pr_auc_10k(c: &mut Criterion) {
    let data: Vec<(i32, f64)> = (0..10_000)
        .map(|i| (i32::from(i % 3 != 0), (i as f64 * 0.7).sin() / 2.0 + 0.5))
        .collect();

    c.bench_function("rolling_pr_auc_10k", |b| {
        b.iter(|| {
            let mut metric = river::rolling_pr_auc::RollingPRAUC::new(1, 1000);
            for &(label, score) in &data {
                metric.update(black_box(label), black_box(score));
            }
            metric.get()
        })
    });
}

criterion_group!(benches, bench_rolling_roc_auc_10k, bench_rolling_pr_auc_10k,);
criterion_main!(benches);
