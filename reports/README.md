# reports/ — Per-phase engineering writeups

Each phase produces a report (ARCHITECTURE.md "Documentation Requirements"):
**Summary / Results / Problems / Fixes / Lessons Learned**. These are the most
valuable artifact of the project — the explanation of *why* each optimization
worked, backed by profiler evidence.

| Phase | Topic | Report |
|---|---|---|
| 0 | Setup & scaffolding | [phase0/](phase0/README.md) |
| 1 | GPU primitives (reduction, scan, transpose) | [phase1/](phase1/) |
| 2 | GEMM | [phase2/](phase2/) |
| 3 | Transformer components | [phase3/](phase3/) |
| final | End-to-end + publication | [final/](final/) |

Curated benchmark CSVs and Nsight exports for a phase live alongside its report.
