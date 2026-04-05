# Experiences

**Tensor indexing (strides)** — `TensorN` now keeps a row-major stride per axis (`m_strides`), recomputed whenever the shape changes. Linear index is `sum_i index[i] * stride[i]` instead of rebuilding tail products with `std::accumulate` on every access. Helpers like slice `assign` reuse cached strides from the source tensor; `linearToRowMajorIndices` takes strides directly. Profiling showed this removed most of the cost that looked like “slow `loopIndices`”: the visitor was fine; repeated \(O(\text{rank}^2)\) index work was not.
