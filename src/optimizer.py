"""
optimizer.py — Main Driver & CLI

Entry point for the PTX Optimizer tool.

Usage:
    python -m src.optimizer input.ptx [options]

Options:
    -o OUTPUT       Output file path (default: <input>_opt.ptx)
    --passes PASSES Comma-separated list of passes to run
                    Available: die,cf,as,ce  (default: all, order matters)
    --no-header     Skip the stats header comment in output
    --stats         Print stats table to stdout even without --verbose
    -v, --verbose   Verbose output
    -n, --dry-run   Parse and analyse only, do not write output file

Pass order (default):
    1. ce  — Copy Elimination       (exposes opportunities for others)
    2. cf  — Constant Folding       (may create new mov candidates)
    3. as  — Algebraic Simplification
    4. die — Dead Instruction Elimination  (cleans up after all above)

The passes are run in two rounds by default, because one pass can expose
new opportunities for another (e.g. copy elimination can expose constant
folding, which exposes dead code).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from .parser import parse_source
from .emitter import emit_with_header, count_instructions
from .ir import PTXNode

from .passes import copy_elim, const_fold, algebraic, dead_code


# ── Pass registry ────────────────────────────────────────────────────────────

PASS_REGISTRY = {
    "ce":  ("Copy Elimination",          copy_elim.run),
    "cf":  ("Constant Folding",          const_fold.run),
    "as":  ("Algebraic Simplification",  algebraic.run),
    "die": ("Dead Instruction Elimination", dead_code.run),
}

DEFAULT_PASS_ORDER = ["ce", "cf", "as", "die"]
NUM_ROUNDS = 2   # run all passes this many times to catch cascading opts


# ── Core optimization pipeline ───────────────────────────────────────────────

def optimize(source: str,
             pass_order: Optional[List[str]] = None,
             rounds: int = NUM_ROUNDS,
             verbose: bool = False) -> dict:
    """
    Run the full optimization pipeline on PTX source text.

    Args:
        source:     PTX source as a string
        pass_order: list of pass keys to run (default: DEFAULT_PASS_ORDER)
        rounds:     number of times to repeat the pass sequence
        verbose:    print per-pass stats

    Returns:
        dict with keys:
            optimized_source  — final PTX text (with header)
            stats             — breakdown of optimizations performed
            nodes             — final PTXNode list
    """
    if pass_order is None:
        pass_order = DEFAULT_PASS_ORDER

    nodes = parse_source(source)
    original_count = count_instructions(nodes)

    totals = {key: 0 for key in PASS_REGISTRY}
    passes_run_log = []

    for round_num in range(1, rounds + 1):
        for key in pass_order:
            name, run_fn = PASS_REGISTRY[key]
            nodes, count = run_fn(nodes)
            totals[key] += count
            passes_run_log.append(f"Round {round_num} / {name}: {count}")
            if verbose:
                print(f"  [{round_num}] {name:<35} → {count:3d} optimized")

    optimized_count = count_instructions(nodes)

    stats = {
        "dead":              totals["die"],
        "folded":            totals["cf"],
        "simplified":        totals["as"],
        "copies":            totals["ce"],
        "original_count":    original_count,
        "optimized_count":   optimized_count,
        "passes_run":        ", ".join(passes_run_log),
    }

    return {
        "nodes":  nodes,
        "stats":  stats,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptx-optimizer",
        description="Mini PTX Peephole Optimizer — compiler pass demo for NVIDIA PTX assembly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.optimizer samples/ptx_files/vector_add.ptx
  python -m src.optimizer input.ptx -o output.ptx --passes die,cf,as
  python -m src.optimizer input.ptx --stats --no-header
  python -m src.optimizer input.ptx --dry-run -v
        """
    )
    p.add_argument("input",
                   help="Path to input .ptx file")
    p.add_argument("-o", "--output",
                   help="Output .ptx file path (default: <input>_opt.ptx)")
    p.add_argument("--passes",
                   default=",".join(DEFAULT_PASS_ORDER),
                   help=f"Comma-separated pass list. Available: {', '.join(PASS_REGISTRY)}"
                        f"  (default: {','.join(DEFAULT_PASS_ORDER)})")
    p.add_argument("--rounds", type=int, default=NUM_ROUNDS,
                   help=f"Number of optimization rounds (default: {NUM_ROUNDS})")
    p.add_argument("--no-header", action="store_true",
                   help="Do not prepend stats header to output file")
    p.add_argument("--stats", action="store_true",
                   help="Always print stats table to stdout")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-pass counts during optimization")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Analyse only — do not write output file")
    return p


def _print_stats_table(stats: dict, input_path: str) -> None:
    """Pretty-print the optimization stats to stdout."""
    orig = stats["original_count"]
    opt  = stats["optimized_count"]
    removed = stats["dead"] + stats["folded"] + stats["simplified"] + stats["copies"]
    pct  = (removed / orig * 100) if orig > 0 else 0.0

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║            PTX OPTIMIZER — STATS REPORT             ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Input file       : {input_path:<32}║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Dead instructions removed  : {stats['dead']:<23}║")
    print(f"║  Constants folded           : {stats['folded']:<23}║")
    print(f"║  Algebraic simplifications  : {stats['simplified']:<23}║")
    print(f"║  Copy eliminations          : {stats['copies']:<23}║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Original instruction count : {orig:<23}║")
    print(f"║  Optimized instruction count: {opt:<23}║")
    print(f"║  Total removed              : {removed} ({pct:.1f}% reduction){' ' * (13 - len(str(removed)))}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── Read input ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    source = input_path.read_text(encoding="utf-8")

    # ── Validate pass list ────────────────────────────────────────────────
    pass_keys = [k.strip() for k in args.passes.split(",")]
    invalid = [k for k in pass_keys if k not in PASS_REGISTRY]
    if invalid:
        print(f"Error: unknown pass(es): {invalid}. "
              f"Available: {list(PASS_REGISTRY)}", file=sys.stderr)
        return 1

    # ── Run optimizer ─────────────────────────────────────────────────────
    if args.verbose:
        print(f"\nOptimizing: {input_path}")
        print(f"Passes    : {pass_keys}  ×{args.rounds} rounds\n")

    t0 = time.perf_counter()
    result = optimize(source,
                      pass_order=pass_keys,
                      rounds=args.rounds,
                      verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    stats = result["stats"]
    nodes = result["nodes"]

    # ── Print stats ───────────────────────────────────────────────────────
    if args.stats or args.verbose:
        _print_stats_table(stats, str(input_path))
        print(f"  Optimization time: {elapsed*1000:.2f} ms\n")

    # ── Write output ──────────────────────────────────────────────────────
    if args.dry_run:
        print("Dry run — no output file written.")
        return 0

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_opt")

    from .emitter import emit_with_header, emit
    if args.no_header:
        output_text = emit(nodes)
    else:
        output_text = emit_with_header(nodes, stats, str(input_path))

    output_path.write_text(output_text, encoding="utf-8")

    if not args.stats and not args.verbose:
        # Minimal one-line summary when no verbose flags
        removed = (stats["dead"] + stats["folded"]
                   + stats["simplified"] + stats["copies"])
        print(f"  {input_path.name} → {output_path.name}"
              f"  ({removed} instructions optimized)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
