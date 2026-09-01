> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# ASC 718 Monte Carlo Valuation — Performance RSUs

Audit-ready Python engine for valuing performance-based RSUs with a **consecutive-day price barrier** under US GAAP ASC 718.

## What It Does

- **5 independent tranches**, each with its own price hurdle
- **Path-dependent barrier logic**: stock must close >= hurdle for 60 consecutive calendar days (reset on any miss)
- **100,000 GBM trials** at daily increments (dt = 1/252)
- Outputs per-tranche: Probability of Attainment, PV Fair Value per Share, Total Comp. Expense, Derived Service Period

## Architecture

```
ValuationInputs (dataclass)
    └── simulate_gbm_paths()        # Risk-neutral GBM: (n_sims × n_days)
            └── check_consecutive_barrier()  # Path-dependent counter per sim
                    └── value_single_tranche()  # P(attain), E[disc|attain], DSP
                            └── run_valuation()  # Aggregates all 5 tranches → DataFrame
```

## Quick Start

```bash
pip install -r requirements.txt

python main.py \
  --s0 25.00 \
  --hurdles 35 45 55 65 75 \
  --term 5.0 \
  --vol 0.55 \
  --rfr 0.045 \
  --div 0.00 \
  --shares 50000 50000 50000 50000 50000 \
  --sims 100000 \
  --output results.csv
```

## Sample Output

| Tranche | Hurdle | Shares | Prob. (%) | FV/Share ($) | Total Expense ($) | DSP (mo) |
|---------|--------|--------|-----------|--------------|-------------------|----------|
| 1 | $35 | 50,000 | — | — | — | — |
| 2 | $45 | 50,000 | — | — | — | — |
| 3 | $55 | 50,000 | — | — | — | — |
| 4 | $65 | 50,000 | — | — | — | — |
| 5 | $75 | 50,000 | — | — | — | — |

*(Run with your inputs to populate. Placeholder shown — client inputs confidential.)*

## ASC 718 Compliance Notes

- Market conditions are reflected in grant-date fair value; expense is recognized regardless of whether the condition is met (ASC 718-10-55-93 through 55-100)
- Derived service period drives amortization schedule; shorter periods are used when market condition expected to be met sooner than requisite service period
- Risk-neutral simulation under continuously compounded risk-free rate is industry-standard for path-dependent market conditions (consistent with FASB guidance and Big 4 audit practice)

## License

MIT — for valuation modeling and audit documentation purposes.
