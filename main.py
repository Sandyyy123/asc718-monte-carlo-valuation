#!/usr/bin/env python3
"""
ASC 718 Monte Carlo Valuation Engine — Performance RSUs with Consecutive-Day Barrier
====================================================================================================
Computes per-tranche: Probability of Attainment, Present-Value Fair Value per Share,
Total Compensation Expense, and Derived Service Period under US GAAP ASC 718-10-55.

Market Condition Logic:
  - 5 independent price hurdles, each treated as a separate market condition (ASC 718-10-55-93)
  - Barrier rule: stock price must close >= hurdle for EXACTLY 60 consecutive calendar days
  - Any daily close below hurdle resets consecutive counter to zero
  - Simulation: Geometric Brownian Motion, risk-neutral measure, 100,000 paths, daily step (dt=1/252)
  - Discount rate: continuous compounding at risk-free rate
  - Service period: derived from expected time-to-first-barrier-breach per tranche

Usage:
  python asc718_mc_valuation.py --s0 25.00 --hurdles 35 45 55 65 75 \
         --term 5.0 --vol 0.55 --rfr 0.045 --div 0.00 \
         --shares 50000 50000 50000 50000 50000 --sims 100000

Author: Dr. Sandeep Grover | Quantitative Analysis & Biostatistics
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ValuationInputs:
    s0: float                        # Grant-date stock price
    hurdles: List[float]             # Price hurdle for each tranche
    term: float                      # Valuation window in years
    vol: float                       # Annualized historical/implied volatility
    rfr: float                       # Continuously compounded risk-free rate
    div: float                       # Expected continuous dividend yield
    shares: List[float]              # Shares (or RSUs) per tranche
    n_sims: int = 100_000            # Number of Monte Carlo trials
    barrier_days: int = 60           # Consecutive-day requirement
    seed: int = 42                   # Reproducibility seed
    trading_days_per_year: int = 252 # Convention

    def __post_init__(self):
        assert len(self.hurdles) == len(self.shares), "hurdles and shares must match in length"
        assert self.vol > 0, "Volatility must be positive"
        assert self.term > 0, "Term must be positive"


def simulate_gbm_paths(inputs: ValuationInputs) -> np.ndarray:
    """
    Generate (n_sims x n_days) GBM price matrix under risk-neutral measure.
    dS = S * (r - q - 0.5*sigma^2)*dt + S*sigma*sqrt(dt)*Z
    Returns log-price paths converted back to levels.
    """
    rng = np.random.default_rng(inputs.seed)
    dt = 1.0 / inputs.trading_days_per_year
    n_days = int(inputs.term * inputs.trading_days_per_year)

    drift = (inputs.rfr - inputs.div - 0.5 * inputs.vol ** 2) * dt
    diffusion = inputs.vol * np.sqrt(dt)

    Z = rng.standard_normal((inputs.n_sims, n_days))
    log_increments = drift + diffusion * Z                  # (n_sims, n_days)
    log_paths = np.cumsum(log_increments, axis=1)           # cumulative log-return
    prices = inputs.s0 * np.exp(log_paths)                  # (n_sims, n_days)
    return prices


def check_consecutive_barrier(
    prices: np.ndarray,
    hurdle: float,
    barrier_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Path-dependent consecutive-day barrier check (vectorised outer loop, inner Numba-free).

    For each simulation:
      - Track consecutive days >= hurdle
      - Reset counter to 0 on any day < hurdle
      - Record first day when counter reaches barrier_days

    Returns
    -------
    attained   : bool ndarray (n_sims,)  — True if hurdle ever attained
    first_day  : int ndarray (n_sims,)   — 0-indexed trading day of attainment (n_days if not)
    """
    n_sims, n_days = prices.shape
    above = (prices >= hurdle)                              # boolean mask
    attained = np.zeros(n_sims, dtype=bool)
    first_day = np.full(n_sims, n_days, dtype=np.int32)

    for i in range(n_sims):
        run = 0
        for d in range(n_days):
            if above[i, d]:
                run += 1
                if run >= barrier_days:
                    attained[i] = True
                    first_day[i] = d - barrier_days + 1    # day the window opened
                    break
            else:
                run = 0

    return attained, first_day


def value_single_tranche(
    prices: np.ndarray,
    hurdle: float,
    shares: float,
    inputs: ValuationInputs,
    tranche_id: int,
) -> dict:
    """
    Compute ASC 718 fair-value metrics for one tranche.

    Per-share fair value = E_Q[ e^{-r * T_attain} ] * S0  (if attained, else 0)
    where the expectation is over paths that attain the barrier, weighted by
    the risk-neutral attainment probability.

    Derived Service Period = E[ T_attain | attained ] expressed in months.
    If no path attains the hurdle, DSP = full term.
    """
    dt = 1.0 / inputs.trading_days_per_year

    attained, first_day_idx = check_consecutive_barrier(prices, hurdle, inputs.barrier_days)

    prob = float(attained.mean())

    if prob > 1e-8:
        t_attain = first_day_idx[attained].astype(float) * dt   # years
        disc_factors = np.exp(-inputs.rfr * t_attain)
        # Fair value per share under risk-neutral expectation
        # RSU pays one share; FV = prob * E[discount | attained] * S0
        pv_per_share = prob * disc_factors.mean() * inputs.s0
        dsp_trading_days = float(first_day_idx[attained].mean())
    else:
        pv_per_share = 0.0
        dsp_trading_days = float(inputs.term * inputs.trading_days_per_year)

    dsp_months = dsp_trading_days / (inputs.trading_days_per_year / 12.0)
    total_expense = pv_per_share * shares

    return {
        "Tranche": tranche_id,
        "Price Hurdle ($)": hurdle,
        "Shares": int(shares),
        "Prob. of Attainment": round(prob, 4),
        "Prob. of Attainment (%)": round(prob * 100, 2),
        "PV Fair Value per Share ($)": round(pv_per_share, 4),
        "Total Comp. Expense ($)": round(total_expense, 2),
        "Derived Service Period (months)": round(dsp_months, 2),
        "Derived Service Period (years)": round(dsp_months / 12, 3),
    }


def run_valuation(inputs: ValuationInputs) -> pd.DataFrame:
    """
    Full valuation pipeline:
      1. Simulate GBM paths (shared across all tranches)
      2. Apply barrier check per tranche
      3. Aggregate into summary DataFrame
    """
    print(f"\n{'='*70}")
    print("ASC 718 Monte Carlo Valuation | Performance RSU | Consecutive-Day Barrier")
    print(f"{'='*70}")
    print(f"  Trials       : {inputs.n_sims:,}")
    print(f"  S0           : ${inputs.s0:.2f}")
    print(f"  Term         : {inputs.term} yrs")
    print(f"  Volatility   : {inputs.vol*100:.1f}%")
    print(f"  Risk-Free    : {inputs.rfr*100:.2f}%")
    print(f"  Dividend     : {inputs.div*100:.2f}%")
    print(f"  Barrier      : {inputs.barrier_days} consecutive days")
    print(f"  Tranches     : {len(inputs.hurdles)}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    print("Simulating GBM paths ...", end=" ", flush=True)
    prices = simulate_gbm_paths(inputs)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    results = []
    for idx, (hurdle, shares) in enumerate(zip(inputs.hurdles, inputs.shares), start=1):
        t1 = time.perf_counter()
        print(f"  Tranche {idx}: hurdle=${hurdle:.0f} ...", end=" ", flush=True)
        row = value_single_tranche(prices, hurdle, shares, inputs, idx)
        results.append(row)
        print(f"done ({time.perf_counter()-t1:.1f}s) | P={row['Prob. of Attainment (%)']:.1f}%  FV/sh=${row['PV Fair Value per Share ($)']:.4f}")

    df = pd.DataFrame(results)
    return df


def print_report(df: pd.DataFrame, inputs: ValuationInputs) -> None:
    """Print formatted valuation summary to stdout."""
    print(f"\n{'='*70}")
    print("VALUATION SUMMARY TABLE")
    print(f"{'='*70}")
    cols = [
        "Tranche", "Price Hurdle ($)", "Shares",
        "Prob. of Attainment (%)", "PV Fair Value per Share ($)",
        "Total Comp. Expense ($)", "Derived Service Period (months)"
    ]
    print(df[cols].to_string(index=False))
    print(f"\n  Total Aggregate Compensation Expense: ${df['Total Comp. Expense ($)'].sum():,.2f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ASC 718 GBM Monte Carlo Valuation — Performance RSUs with Consecutive-Day Barrier"
    )
    parser.add_argument("--s0",      type=float, required=True,  help="Grant-date stock price")
    parser.add_argument("--hurdles", type=float, nargs="+",      help="Space-separated price hurdles (5 values)")
    parser.add_argument("--term",    type=float, required=True,  help="Valuation term in years (e.g. 5.0)")
    parser.add_argument("--vol",     type=float, required=True,  help="Annualized volatility (e.g. 0.55)")
    parser.add_argument("--rfr",     type=float, required=True,  help="Risk-free rate continuous (e.g. 0.045)")
    parser.add_argument("--div",     type=float, default=0.0,    help="Dividend yield continuous (e.g. 0.00)")
    parser.add_argument("--shares",  type=float, nargs="+",      help="Shares per tranche (5 values)")
    parser.add_argument("--sims",    type=int,   default=100_000,help="Number of Monte Carlo trials")
    parser.add_argument("--seed",    type=int,   default=42,     help="RNG seed for reproducibility")
    parser.add_argument("--output",  type=str,   default=None,   help="Optional CSV output path")

    args = parser.parse_args()

    hurdles = args.hurdles or [35, 45, 55, 65, 75]
    shares = args.shares or [50000] * 5

    if len(hurdles) != 5 or len(shares) != 5:
        print("Error: exactly 5 hurdles and 5 share counts required.", file=sys.stderr)
        sys.exit(1)

    inputs = ValuationInputs(
        s0=args.s0,
        hurdles=hurdles,
        term=args.term,
        vol=args.vol,
        rfr=args.rfr,
        div=args.div,
        shares=shares,
        n_sims=args.sims,
        seed=args.seed,
    )

    df = run_valuation(inputs)
    print_report(df, inputs)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
