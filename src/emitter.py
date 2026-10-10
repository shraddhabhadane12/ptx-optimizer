"""
emitter.py — PTX Code Emitter

Converts the optimized list of PTXNodes back into a valid PTX text file.
Handles:
  - Reconstructing instruction lines from parsed Instruction objects
  - Passing through RawLine objects unchanged
  - Injecting a header comment with optimization statistics
"""

from typing import List, Tuple

from .ir import Instruction, RawLine, PTXNode


def emit(nodes: List[PTXNode]) -> str:
    """
    Serialize a list of PTXNodes back to PTX source text.

    Args:
        nodes: optimized PTX node list

    Returns:
        complete PTX source as a string
    """
    lines: List[str] = []

    for node in nodes:
        if isinstance(node, RawLine):
            lines.append(node.text)
        elif isinstance(node, Instruction):
            lines.append(_emit_instruction(node))

    return "\n".join(lines)


def _emit_instruction(instr: Instruction) -> str:
    """
    Rebuild a single instruction line.

    If the instruction was modified by a pass (raw was updated in-place),
    use the stored raw text.  Otherwise reconstruct from fields.
    """
    # Passes update instr.raw when they rewrite an instruction,
    # so we can safely emit raw directly.
    return instr.raw


def emit_with_header(nodes: List[PTXNode], stats: dict,
                     original_file: str) -> str:
    """
    Emit optimized PTX with a stats header block prepended.

    Args:
        nodes:         optimized PTX node list
        stats:         dict with keys: dead, folded, simplified, copies,
                       original_count, optimized_count, passes_run
        original_file: path to input file (for the header comment)

    Returns:
        complete PTX source string with header
    """
    header = _build_header(stats, original_file)
    body   = emit(nodes)
    return header + "\n" + body


def _build_header(stats: dict, original_file: str) -> str:
    """Build a PTX block comment summarising what the optimizer did."""
    total_removed = stats.get("dead", 0) + stats.get("folded", 0) + stats.get("simplified", 0)
    orig  = stats.get("original_count", 0)
    opt   = stats.get("optimized_count", 0)
    pct   = (total_removed / orig * 100) if orig > 0 else 0.0

    lines = [
        "//",
        "// ╔══════════════════════════════════════════════════════════════╗",
        "// ║              PTX OPTIMIZER — OUTPUT REPORT                  ║",
        "// ╚══════════════════════════════════════════════════════════════╝",
        f"//  Input file      : {original_file}",
        f"//  Passes run       : {stats.get('passes_run', 'N/A')}",
        "//  ─────────────────────────────────────────────────────────────",
        f"//  Dead instructions removed  : {stats.get('dead', 0)}",
        f"//  Constants folded           : {stats.get('folded', 0)}",
        f"//  Algebraic simplifications  : {stats.get('simplified', 0)}",
        f"//  Copy eliminations          : {stats.get('copies', 0)}",
        "//  ─────────────────────────────────────────────────────────────",
        f"//  Original instruction count : {orig}",
        f"//  Optimized instruction count: {opt}",
        f"//  Total instructions removed : {total_removed}  ({pct:.1f}% reduction)",
        "//",
    ]
    return "\n".join(lines)


def count_instructions(nodes: List[PTXNode]) -> int:
    """Count only Instruction nodes (excludes directives, blanks, comments)."""
    return sum(1 for n in nodes if isinstance(n, Instruction))
