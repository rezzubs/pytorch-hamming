use std::collections::HashMap;

/// An iterator which returns numbers from 0..n in a random order until all values are consumed.
///
/// Every returned value is unique.
///
/// This is based on the [Fisher-Yates
/// shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) but instead of shuffling
/// the whole sequence we just return the target index for the swap.
pub struct RandomPicker<R> {
    /// See [`Self::new`].
    size: usize,
    /// This map is used to keep track of the "current" values at each index. If there's no value
    /// at an index it's assumed to be the same as the index. Each iteration a random index from
    /// 0..size is picked and returned. The returned index will be swapped with `size` and size is
    /// decremented by 1.
    swaps: HashMap<usize, usize>,
    rng: R,
}

impl<R> RandomPicker<R>
where
    R: rand::Rng,
{
    /// The returned elements will be from a [`std::ops::Range`] of `0..size`.
    ///
    /// For example, `size` can be a length of a [`std::vec::Vec`] for getting indices of the `Vec`.
    pub fn new(size: usize, rng: R) -> Self {
        Self {
            size,
            swaps: HashMap::new(),
            rng,
        }
    }
}

impl<R> Iterator for RandomPicker<R>
where
    R: rand::Rng,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.size == 0 {
            return None;
        }
        let target_index_before_any_swaps = self.rng.random_range(0..self.size);
        let target_index_after_swaps = self
            .swaps
            .get(&target_index_before_any_swaps)
            .copied()
            .unwrap_or(target_index_before_any_swaps);

        let last_index_before_any_swaps = self.size - 1;
        let last_index_after_swaps = self
            .swaps
            .get(&last_index_before_any_swaps)
            .copied()
            .unwrap_or(last_index_before_any_swaps);

        self.swaps
            .insert(target_index_before_any_swaps, last_index_after_swaps);
        self.size -= 1;

        Some(target_index_after_swaps)
    }
}
