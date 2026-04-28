# TEE-GNN Functional Simulation

This directory builds a normal-process simulation of the TEE-GNN inference flow. It does not train. It compares the simulated masked protocol against the plaintext GCN baseline and reports accuracy plus logits error.

The simulation includes:

- REE-side masked sparse-dense multiplication and dense multiplication.
- TEE-side restoration, low-rank correction, ReLU, argmax, and accuracy checking.
- Scaled permutations without materializing dense permutation matrices.
- Low-rank additive masks.
- SDIM masks `S = P + u v^T` with Sherman-Morrison inverse application.
- Confusion-edge graph support expansion.

## Build

```bash
cd nonTEE
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/teegnn_sim /dataset/cora --confusion-rate 0.2 --mask-rank 2 --seed 1234
```

For this repository layout you can also run:

```bash
./build/teegnn_sim ../dataset/cora --confusion-rate 0.2 --mask-rank 2 --seed 1234
/data/bin/teegnn_host /home/gjh/dataset/cora --confusion-rate 1 --mask-rank 2 --seed 114514
```

