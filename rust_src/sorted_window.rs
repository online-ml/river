use num::{Float, FromPrimitive};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    ops::{AddAssign, Index, SubAssign},
};

#[doc(hidden)]
#[derive(Serialize, Deserialize)]
pub struct SortedWindow<F: Float + FromPrimitive + AddAssign + SubAssign> {
    pub(crate) sorted_window: VecDeque<NotNan<F>>,
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
        self.sorted_window
            .front()
            .expect("The value is Nan")
            .into_inner()
    }
    pub fn back(&self) -> F {
        self.sorted_window
            .back()
            .expect("The value is NaN")
            .into_inner()
    }
    pub fn push_back(&mut self, value: F) {
        let nn_value = NotNan::new(value).expect("Value is NaN");

        // Before adding the newest value to the sorted window
        // we should remove the oldest value
        if self.sorted_window.len() == self.window_size {
            let last_unsorted = self.unsorted_window.pop_front().unwrap();

            let last_unsorted_nn = NotNan::new(last_unsorted).expect("Value is NaN");
            let last_unsorted_pos = self
                .sorted_window
                .binary_search(&last_unsorted_nn)
                .expect("The value is Not in the sorted window");
            self.sorted_window.remove(last_unsorted_pos);
        }
        self.unsorted_window.push_back(value);

        let sorted_pos = self
            .sorted_window
            .binary_search(&nn_value)
            .unwrap_or_else(|e| e);
        self.sorted_window.insert(sorted_pos, nn_value);
    }
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign> Index<usize> for SortedWindow<F> {
    fn index(&self, index: usize) -> &Self::Output {
        &self.sorted_window[index]
    }
    type Output = F;
}
