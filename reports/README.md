# reports/ — Per-phase engineering writeups

Each phase produces a report (ARCHITECTURE.md "Documentation Requirements"):
**Summary / Results / Problems / Fixes / Lessons Learned**. These are the most
valuable artifact of the project — the explanation of *why* each optimization
worked, backed by profiler evidence.

| Phase | Topic | Report |
|---|---|---|
| 0 | Setup & scaffolding | [phase0/](phase0/README.md) |
| 1 | GPU reduction | [phase1/](phase1/) |
| 2 | GEMM | [phase2/](phase2/) |
| 3 | Transformer components | [phase3/](phase3/) |
| 4 | Causal attention | [Attention report](phase4/attention_report.md) |
| 5 | GPT-2 forward and generation | [Phase 5 report](phase5/final_report.md) |
| 6 | KV cache and single-row GEMV | [KV cache design](phase6/kv_cache_design.md), [GEMV report](phase6/phase6b_report.md) |
| Release | Reproduction and measurements | [FP32 release report](final/fp32-release.md) |

Curated benchmark CSVs and Nsight exports for a phase live alongside its report.
