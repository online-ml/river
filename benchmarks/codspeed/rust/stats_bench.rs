use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use river::stats::Univariate;

fn bench_quantile(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("quantile_p2_10k", |b| {
        b.iter(|| {
            let mut q = river::quantile::Quantile::new(0.5).unwrap();
            for &x in &data {
                q.update(black_box(x));
            }
            q.get()
        })
    });
}

fn bench_rolling_quantile(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    for window_size in [10, 100, 1000] {
        c.bench_with_input(
            BenchmarkId::new("rolling_quantile", window_size),
            &window_size,
            |b, &ws| {
                b.iter(|| {
                    let mut q = river::quantile::RollingQuantile::new(0.5, ws).unwrap();
                    for &x in &data {
                        q.update(black_box(x));
                    }
                    q.get()
                })
            },
        );
    }
}

fn bench_ewmean(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("ewmean_10k", |b| {
        b.iter(|| {
            let mut m = river::ewmean::EWMean::new(0.5);
            for &x in &data {
                m.update(black_box(x));
            }
            m.get()
        })
    });
}

fn bench_ewvariance(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("ewvariance_10k", |b| {
        b.iter(|| {
            let mut v = river::ewvariance::EWVariance::new(0.5);
            for &x in &data {
                v.update(black_box(x));
            }
            v.get()
        })
    });
}

fn bench_kurtosis(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("kurtosis_10k", |b| {
        b.iter(|| {
            let mut k = river::kurtosis::Kurtosis::new(false);
            for &x in &data {
                k.update(black_box(x));
            }
            k.get()
        })
    });
}

fn bench_skew(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("skew_10k", |b| {
        b.iter(|| {
            let mut s = river::skew::Skew::new(false);
            for &x in &data {
                s.update(black_box(x));
            }
            s.get()
        })
    });
}

fn bench_mean(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("mean_10k", |b| {
        b.iter(|| {
            let mut m = river::mean::Mean::new();
            for &x in &data {
                m.update(black_box(x));
            }
            m.get()
        })
    });
}

fn bench_variance(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("variance_10k", |b| {
        b.iter(|| {
            let mut v = river::variance::Variance::default();
            for &x in &data {
                v.update(black_box(x));
            }
            v.get()
        })
    });
}

fn bench_rolling_iqr(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    for window_size in [10, 100, 1000] {
        c.bench_with_input(
            BenchmarkId::new("rolling_iqr", window_size),
            &window_size,
            |b, &ws| {
                b.iter(|| {
                    let mut iq = river::iqr::RollingIQR::new(0.25, 0.75, ws).unwrap();
                    for &x in &data {
                        iq.update(black_box(x));
                    }
                    iq.get()
                })
            },
        );
    }
}

criterion_group!(
    benches,
    bench_mean,
    bench_variance,
    bench_ewmean,
    bench_ewvariance,
    bench_kurtosis,
    bench_skew,
    bench_quantile,
    bench_rolling_quantile,
    bench_rolling_iqr,
);
criterion_main!(benches);
