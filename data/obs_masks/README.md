Observation masks used in the paper, one file per configuration
(`<strategy>_N<density>[_s<seed>].npy`), stored as (2, N) int32 arrays of
row and column indices.

These are the exact observation points behind every reported result, so the
experiments can be reproduced without re-running `src/placement.py`.
