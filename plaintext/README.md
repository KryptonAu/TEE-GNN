# Plaintext Two-Layer GCN Baseline

This directory builds a C++17 plaintext inference baseline. It reads:

- `edges.txt`
- `features.txt`
- `labels.txt`
- `w1.txt`
- `w2.txt`

`w1.txt` and `w2.txt` are transposed immediately after loading, then used as Eigen dense matrices. Graph aggregation uses the normalized adjacency
`D^{-1/2}(A + I)D^{-1/2}` through adjacency-list sparse-dense traversal.

## Build

```bash
cd plaintext
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/gcn_infer /dataset/cora
```

For this repository layout you can also run:

```bash
./build/gcn_infer ../dataset/cora
```

