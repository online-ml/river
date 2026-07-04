use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_expected_mutual_info(c: &mut Criterion) {
    let labels_a: Vec<usize> = (0..1_000).map(|i| (i * 7) % 10).collect();
    let labels_b: Vec<usize> = (0..1_000).map(|i| (i * 11 + 3) % 10).collect();
    let mut counts_a = vec![0_i64; 10];
    let mut counts_b = vec![0_i64; 10];
    for (&a, &b) in labels_a.iter().zip(labels_b.iter()) {
        counts_a[a] += 1;
        counts_b[b] += 1;
    }

    c.bench_function("expected_mutual_info", |b| {
        b.iter(|| {
            river::expected_mutual_info::expected_mutual_info(
                black_box(1_000.0),
                black_box(&counts_a),
                black_box(&counts_b),
            )
        })
    });
}

criterion_group!(benches, bench_expected_mutual_info);
criterion_main!(benches);
