MLX long-state serving, Apple M5 34 GB, macOS 26.6.2, mlx 0.32.2 / mlx-lm 0.31.3 (scripts/mlx_long_states.py)

| size | state tokens | cold (first / median of 3) | cached (median) | MLX peak (weights) | footprint peak | swap out | planted fact (p) | swapped / start | vs fp32 torch |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Kev-0.8B | 8,192 | 1.4 / 1.4 s | 74 ms | 2.7 GB (1.5) | 5.3 GB | 0.00 GB | ok (0.68) | ok (0.70) / ok (0.85) | max \|dp\| 0.0076, 0 flips |
| Kev-0.8B | 16,384 | 3.2 / 3.2 s | 94 ms | 3.0 GB (1.5) | 6.1 GB | 0.00 GB | ok (0.68) | ok (0.70) / ok (0.77) | max \|dp\| 0.0052, 0 flips |
| Kev-0.8B | 32,768 | 7.9 / 8.0 s | 134 ms | 3.1 GB (1.5) | 6.3 GB | 0.00 GB | ok (0.70) | ok (0.71) / ok (0.82) | - |
| Kev-0.8B | 65,000 | 21.6 / 21.2 s | 202 ms | 3.8 GB (1.5) | 5.4 GB | 0.00 GB | ok (0.62) | ok (0.66) / ok (0.83) | - |
| Kev-4B | 8,192 | 6.6 / 6.6 s | 354 ms | 10.2 GB (8.4) | 11.9 GB | 0.00 GB | ok (0.97) | ok (0.97) / ok (0.97) | - |
| Kev-4B | 16,384 | 14.0 / 14.0 s | 427 ms | 11.0 GB (8.4) | 12.8 GB | 0.00 GB | ok (0.95) | ok (0.95) / ok (0.96) | - |
| Kev-4B | 32,768 | 30.5 / 30.5 s | 533 ms | 11.9 GB (8.4) | 13.7 GB | 0.00 GB | ok (0.96) | ok (0.97) / ok (0.97) | - |
| Kev-4B | 65,000 | 80.8 / 84.5 s | 716 ms | 13.0 GB (8.4) | 14.1 GB | 0.05 GB | ok (0.94) | ok (0.94) / ok (0.97) | - |

- Kev-0.8B: the fp32 torch reference ran first in the same process, so its footprint and RSS include host memory torch freed but kept; the MLX peak does not
- Kev-0.8B: 9a45d25, bfloat16, load 1.3 s, admission of a 70,000-token state: 422
- Kev-4B fp32 torch reference skipped: the fp32 backbone (18.64 GB) + 1.5 GB margin exceeds the 18.62 GB available
- Kev-4B: 139fdd9, bfloat16, load 8.2 s, admission of a 70,000-token state: 422
- Kev-9B: not run: the 19.31 GB backbone + 1.5 GB working set + 1.5 GB margin exceeds the 19.82 GB available; it would swap
- Kev-0.8B, the state pass alone at 8,192 tokens (prefill-ab-0.8b.json): one pass 1.44 s / 2.74 GB above the weights, 1024-token chunks 1.38 s / 0.99 GB (max |dp| 0.00398), 2048-token chunks 1.38 s / 1.2 GB (max |dp| 0.0), 4096-token chunks 1.4 s / 1.71 GB (max |dp| 0.0)
- Kev-0.8B, the state pass alone at 16,384 tokens (prefill-ab-0.8b.json): one pass 3.39 s / 4.92 GB above the weights, 1024-token chunks 3.02 s / 1.06 GB (max |dp| 0.0055), 2048-token chunks 2.99 s / 1.27 GB (max |dp| 0.0), 4096-token chunks 3.04 s / 1.81 GB (max |dp| 0.0)
- Kev-4B, the state pass alone at 8,192 tokens (prefill-ab-4b.json): one pass 6.77 s / 4.6 GB above the weights, 1024-token chunks 6.5 s / 1.16 GB (max |dp| 0.00297), 2048-token chunks 6.58 s / 1.78 GB (max |dp| 0.00094), 4096-token chunks 6.85 s / 2.71 GB (max |dp| 0.0)
