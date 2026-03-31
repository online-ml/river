use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    ops::{AddAssign, Index, SubAssign},
};

#[doc(hidden)]
#[derive(Serialize, Deserialize)]
pub struct SortedWindow<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub(crate) sorted_window: VecDeque<F>,
    pub(crate) unsorted_window: VecDeque<F>,
    window_size: usize,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign> SortedWindow<F> {
    pub fn new(window_size: usize) -> Self {
        Self {
            sorted_window: VecDeque::with_capacity(window_size),
            unsorted_window: VecDeque::with_capacity(window_size),
            window_size,
        }
    }
    pub fn len(&self) -> usize {
        self.sorted_window.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sorted_window.len() == 0
    }

    pub fn front(&self) -> F {
        *self.sorted_window.front().expect("Window is empty")
    }

    pub fn back(&self) -> F {
        *self.sorted_window.back().expect("Window is empty")
    }

    pub fn push_back(&mut self, value: F) {
        // This will panic if `value` is NaN, which is the desired behavior
        // to maintain a sorted list of non-NaN floats.
        if value.is_nan() {
            panic!("Cannot push a NaN value into SortedWindow");
        }

        // Before add the newest value to the sorted window
        // we should remove the oldest value
        if self.sorted_window.len() == self.window_size {
            let oldest_unsorted = self
                .unsorted_window
                .pop_front()
                .expect("Unsorted window should not be empty when sorted window is full");

            // Find the position of the value to remove using a custom comparison.
            // `partial_cmp` returns None for NaN comparisons, so `expect` will panic,
            // which is consistent with the behavior of NotNan.
            let pos_to_remove = self
                .sorted_window
                .binary_search_by(|probe| {
                    probe
                        .partial_cmp(&oldest_unsorted)
                        .expect("Stored values should not be NaN")
                })
                .expect("The value to remove was not found in the sorted window");

            self.sorted_window.remove(pos_to_remove);
        }

        self.unsorted_window.push_back(value);

        let sorted_pos = self
            .sorted_window
            .binary_search_by(|probe| {
                probe
                    .partial_cmp(&value)
                    .expect("Stored values should not be NaN")
            })
            .unwrap_or_else(|e| e);
        self.sorted_window.insert(sorted_pos, value);
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Index<usize> for SortedWindow<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.sorted_window[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    #[test]
    fn test_new_and_empty() {
        let window: SortedWindow<f64> = SortedWindow::new(5);
        assert!(window.is_empty());
        assert_eq!(window.len(), 0);
        assert_eq!(window.window_size, 5);
    }

    #[test]
    fn test_push_and_sort() {
        let mut window = SortedWindow::new(5);
        window.push_back(10.0);
        window.push_back(5.0);
        window.push_back(15.0);

        assert!(!window.is_empty());
        assert_eq!(window.len(), 3);

        // Check sorted order
        assert_eq!(window[0], 5.0);
        assert_eq!(window[1], 10.0);
        assert_eq!(window[2], 15.0);

        // Check accessors
        assert_eq!(window.front(), 5.0);
        assert_eq!(window.back(), 15.0);

        // Check unsorted (insertion) order
        assert_eq!(
            window.unsorted_window.iter().copied().collect::<Vec<_>>(),
            vec![10.0, 5.0, 15.0]
        );
    }

    #[test]
    fn test_window_full_cycle() {
        let mut window = SortedWindow::new(3);

        // 1. Fill the window
        window.push_back(10.0); // unsorted: [10], sorted: [10]
        window.push_back(20.0); // unsorted: [10, 20], sorted: [10, 20]
        window.push_back(5.0);  // unsorted: [10, 20, 5], sorted: [5, 10, 20]

        assert_eq!(window.len(), 3);
        assert_eq!(window.front(), 5.0);
        assert_eq!(window.back(), 20.0);
        assert_eq!(
            window.sorted_window.iter().copied().collect::<Vec<_>>(),
            vec![5.0, 10.0, 20.0]
        );

        // 2. Push a new element, should remove the oldest (10.0)
        window.push_back(15.0); // oldest '10.0' is removed
        // unsorted: [20, 5, 15], sorted: [5, 15, 20]

        assert_eq!(window.len(), 3);
        assert_eq!(window.front(), 5.0);
        assert_eq!(window.back(), 20.0);
        assert_eq!(
            window.sorted_window.iter().copied().collect::<Vec<_>>(),
            vec![5.0, 15.0, 20.0]
        );
        assert_eq!(
            window.unsorted_window.iter().copied().collect::<Vec<_>>(),
            vec![20.0, 5.0, 15.0]
        );

        // 3. Push another new element, should remove the oldest (20.0)
        window.push_back(2.0); // oldest '20.0' is removed
        // unsorted: [5, 15, 2], sorted: [2, 5, 15]

        assert_eq!(window.len(), 3);
        assert_eq!(window.front(), 2.0);
        assert_eq!(window.back(), 15.0);
        assert_eq!(
            window.sorted_window.iter().copied().collect::<Vec<_>>(),
            vec![2.0, 5.0, 15.0]
        );
        assert_eq!(
            window.unsorted_window.iter().copied().collect::<Vec<_>>(),
            vec![5.0, 15.0, 2.0]
        );
    }

    #[test]
    fn test_with_duplicate_values() {
        let mut window = SortedWindow::new(4);
        window.push_back(10.0);
        window.push_back(5.0);
        window.push_back(10.0); // Duplicate value

        assert_eq!(window.len(), 3);
        assert_eq!(window.front(), 5.0);
        assert_eq!(window.back(), 10.0);
        assert_eq!(
            window.sorted_window.iter().copied().collect::<Vec<_>>(),
            vec![5.0, 10.0, 10.0]
        );

        // Fill window
        window.push_back(20.0);
        assert_eq!(
            window.sorted_window.iter().copied().collect::<Vec<_>>(),
            vec![5.0, 10.0, 10.0, 20.0]
        );

        // Push another value, oldest (the first 10.0) should be removed
        window.push_back(1.0);
        assert_eq!(
            window.sorted_window.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 5.0, 10.0, 20.0]
        );
        assert_eq!(
            window.unsorted_window.iter().copied().collect::<Vec<_>>(),
            vec![5.0, 10.0, 20.0, 1.0]
        );
    }

    #[test]
    fn test_window_size_one() {
        let mut window = SortedWindow::new(1);

        window.push_back(10.0);
        assert_eq!(window.len(), 1);
        assert_eq!(window.front(), 10.0);
        assert_eq!(window[0], 10.0);

        window.push_back(5.0);
        assert_eq!(window.len(), 1);
        assert_eq!(window.front(), 5.0);
        assert_eq!(window[0], 5.0);

        window.push_back(20.0);
        assert_eq!(window.len(), 1);
        assert_eq!(window.front(), 20.0);
        assert_eq!(window[0], 20.0);
    }

    #[test]
    #[should_panic]
    fn test_window_size_zero() {
        let mut window = SortedWindow::new(0);
        window.push_back(10.0);

        assert_eq!(window.len(), 0);
        assert!(window.is_empty());
        assert!(window.unsorted_window.is_empty());
    }

    #[test]
    #[should_panic(expected = "Cannot push a NaN value into SortedWindow")]
    fn test_panic_on_nan_push() {
        let mut window = SortedWindow::new(3);
        window.push_back(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "Window is empty")]
    fn test_panic_on_front_empty() {
        let window: SortedWindow<f64> = SortedWindow::new(3);
        window.front();
    }

    #[test]
    #[should_panic(expected = "Window is empty")]
    fn test_panic_on_back_empty() {
        let window: SortedWindow<f64> = SortedWindow::new(3);
        window.back();
    }

    #[test]
    #[should_panic]
    fn test_panic_on_index_out_of_bounds() {
        let mut window = SortedWindow::new(3);
        window.push_back(1.0);
        let _ = window[1]; // Should panic
    }
}