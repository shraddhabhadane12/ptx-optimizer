"""
benchmarks/run_benchmark.py — Performance benchmarking script

Runs the optimizer on all sample PTX files and produces a summary table
showing instruction counts and optimization rates.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --verbose
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.parser import parse_source
from src.emitter import count_instructions
from src.optimizer import optimize, DEFAULT_PASS_ORDER


def benchmark_file(ptx_path: Path, rounds: int = 2, verbose: bool = False):
    """
    Run the full optimizer on a single PTX file and return stats.

    Returns:
        dict with benchmark results
    """
    source = ptx_path.read_text(encoding='utf-8')

    # Count original instructions
    original_nodes = parse_source(source)
    original_count = count_instructions(original_nodes)

    # Time the optimization
    t0 = time.perf_counter()
    result = optimize(source, pass_order=DEFAULT_PASS_ORDER,
                      rounds=rounds, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    stats = result['stats']
    opt_count = stats['optimized_count']
    total_removed = (stats['dead'] + stats['folded']
                     + stats['simplified'] + stats['copies'])
    reduction_pct = (total_removed / original_count * 100
                     if original_count > 0 else 0.0)

    return {
        'file':          ptx_path.name,
        'original':      original_count,
        'optimized':     opt_count,
        'removed':       total_removed,
        'reduction_pct': reduction_pct,
        'dead':          stats['dead'],
        'folded':        stats['folded'],
        'simplified':    stats['simplified'],
        'copies':        stats['copies'],
        'time_ms':       elapsed_ms,
    }


def print_results_table(results: list) -> None:
    """Pretty-print the benchmark results as a table."""
    print()
    print("=" * 90)
    print("  PTX OPTIMIZER — BENCHMARK RESULTS")
    print("=" * 90)
    print(f"  {'File':<25} {'Orig':>6} {'Opt':>6} {'Removed':>8} "
          f"{'Reduction':>10}  {'DIE':>5} {'CF':>5} {'AS':>5} {'CE':>5}  {'Time':>8}")
    print("-" * 90)

    for r in results:
        print(f"  {r['file']:<25} {r['original']:>6} {r['optimized']:>6} "
              f"{r['removed']:>8} {r['reduction_pct']:>9.1f}%  "
              f"{r['dead']:>5} {r['folded']:>5} {r['simplified']:>5} "
              f"{r['copies']:>5}  {r['time_ms']:>6.2f}ms")

    print("-" * 90)

    # Totals row
    total_orig    = sum(r['original']  for r in results)
    total_opt     = sum(r['optimized'] for r in results)
    total_removed = sum(r['removed']   for r in results)
    total_die     = sum(r['dead']      for r in results)
    total_cf      = sum(r['folded']    for r in results)
    total_as      = sum(r['simplified']for r in results)
    total_ce      = sum(r['copies']    for r in results)
    total_time    = sum(r['time_ms']   for r in results)
    avg_reduction = (total_removed / total_orig * 100) if total_orig > 0 else 0

    print(f"  {'TOTAL':<25} {total_orig:>6} {total_opt:>6} "
          f"{total_removed:>8} {avg_reduction:>9.1f}%  "
          f"{total_die:>5} {total_cf:>5} {total_as:>5} "
          f"{total_ce:>5}  {total_time:>6.2f}ms")
    print("=" * 90)
    print()
    print("  Column key:")
    print("    DIE = Dead Instruction Elimination")
    print("    CF  = Constant Folding")
    print("    AS  = Algebraic Simplification")
    print("    CE  = Copy Elimination")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PTX Optimizer Benchmark — runs all sample PTX files"
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-file detailed stats')
    parser.add_argument('--rounds', type=int, default=2,
                        help='Optimization rounds per file (default: 2)')
    parser.add_argument('--ptx-dir',
                        default=str(ROOT / 'samples' / 'ptx_files'),
                        help='Directory containing .ptx files to benchmark')
    args = parser.parse_args()

    ptx_dir = Path(args.ptx_dir)
    ptx_files = sorted(ptx_dir.glob('*.ptx'))

    if not ptx_files:
        print(f"No .ptx files found in {ptx_dir}")
        print("Generate PTX files first:")
        print("  nvcc -ptx samples/cuda_kernels/vector_add.cu "
              "-o samples/ptx_files/vector_add.ptx")
        return 1

    print(f"\nBenchmarking {len(ptx_files)} PTX file(s) "
          f"({args.rounds} optimization rounds each)...")

    results = []
    for ptx_file in ptx_files:
        if args.verbose:
            print(f"\n  Processing: {ptx_file.name}")
        r = benchmark_file(ptx_file, rounds=args.rounds, verbose=args.verbose)
        results.append(r)
        if args.verbose:
            print(f"    Original: {r['original']} instructions")
            print(f"    Optimized: {r['optimized']} instructions "
                  f"(-{r['reduction_pct']:.1f}%)")

    print_results_table(results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
