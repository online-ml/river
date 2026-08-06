use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_sorted_window_insert_1k_window(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.7).sin()).collect();

    c.bench_function("sorted_window_insert_1k_window", |b| {
        b.iter(|| {
            let mut window = river::sorted_window::SortedWindow::<f64>::new(1000);
            for &x in &data {
                window.push_back(black_box(x));
            }
            window.len()
        })
    });
}

criterion_group!(benches, bench_sorted_window_insert_1k_window);
criterion_main!(benches);
