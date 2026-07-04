use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_adwin_update_10k(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("adwin_update_10k", |b| {
        b.iter(|| {
            let mut adwin = river::adwin::AdaptiveWindowing::new(0.002, 32, 5, 5, 10);
            for &x in &data {
                adwin.update(black_box(x));
            }
            adwin.n_detections()
        })
    });
}

fn bench_adwin_update_drift(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000)
        .map(|i| {
            let shift = 2.0 * u8::from(i >= 5_000) as f64;
            (i as f64 * 0.7).sin() + shift
        })
        .collect();

    c.bench_function("adwin_update_drift", |b| {
        b.iter(|| {
            let mut adwin = river::adwin::AdaptiveWindowing::new(0.002, 32, 5, 5, 10);
            for &x in &data {
                adwin.update(black_box(x));
            }
            adwin.n_detections()
        })
    });
}

criterion_group!(benches, bench_adwin_update_10k, bench_adwin_update_drift);
criterion_main!(benches);
