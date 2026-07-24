# EP8 analysis fixture

`30b_ep8.csv` exercises the result summarizer, strict request-count protocol,
and plotting path without requiring a serving run.

It contains all 15 rows from the original parsed 30B EP8 result: three policy
modes at five concurrency levels. The retained columns are exactly those read
by `summarize_results.py` and `plot_main_res.py`; values were selected without
aggregation or recomputation.

Source file and digest:

```text
data/plot/1_main_res/30b_ep8.csv
SHA256 462da8b166ef54e2f3aaaa91eb31f3d00bad6adf73ff658a58036911503cd903
size   4654 bytes
```

The source measurements were collected on the paper's H20 system. Running the
analysis commands on this CSV validates the analysis code and data protocol.
