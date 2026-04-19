# Sources

## Convolution & general

- [Blog post: CNNumpy — fast convolution (HackMD)](https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast#1-Convolutional-layer10)
- [Deep Learning in Modern C++](https://reference-global.com/book/9789365893519) — Luiz Carlos d'Oleron (BPB Publications)

## Optimization — momentum & accelerated methods

- **Polyak, B. T.** (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17. — classical **heavy-ball** method; foundation for momentum-style updates. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0041555364901375) / paywalled; often cited from secondary summaries.)
- **Nesterov, Y. E.** (1983). A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Doklady Akademii Nauk SSSR*, 269, 543–547. — **Nesterov accelerated gradient** (NAG). Widely reproduced in textbooks and surveys.
- **Sutskever, I., Martens, J., Dahl, G., & Hinton, G.** (2013). [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html). *Proceedings of the 30th International Conference on Machine Learning (ICML)*, 1139–1147. — momentum in deep learning; practical insight.
- **Recht, B.** (course notes). [Optimization for Modern Machine Learning — Momentum](https://people.eecs.berkeley.edu/~brecht/opt4ml_book/O4MD_04_Momentum.pdf) (PDF). Berkeley — continuous-time / heavy-ball intuition and discrete updates.

Related (unified views): e.g. [Nesterov’s Accelerated Gradient and Momentum as approximations to Regularised Update Descent](https://arxiv.org/abs/1607.01981) (arXiv:1607.01981).

## Data augmentation

- [A practical guide to data augmentation in PyTorch (with examples and visualizations)](https://medium.com/@BurtMcGurt/a-practical-guide-to-data-augmentation-in-pytorch-with-examples-and-visualizations-761ad5c2a903) — Medium (BurtMcGurt).
