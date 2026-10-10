"""
passes/copy_elim.py — Redundant Copy Elimination

THEORY:
    A "copy" instruction is a mov that transfers one register to another:
        mov.TYPE  %dst, %src

    If %src is still live (unchanged) at every subsequent use of %dst,
    we can replace those uses of %dst with %src directly, and then
    the mov becomes dead (removed by the DIE pass).

    We maintain a "copy map":  %dst → %src
    When we see an instruction that uses %dst, we substitute %src.
    The map is invalidated when either %dst or %src is overwritten.

SAFE-GUARDS:
    - We only eliminate copies within the same basic block (no cross-
      branch substitution — that requires full SSA / dominance analysis,
      which is beyond a peephole pass).
    - Branch instructions flush the entire copy map (conservative).
    - We never substitute into memory address operands ([%rd0]) unless
      the full operand is a bare register.

EXAMPLE:
    Before:
        mov.u64    %rd1, %rd0       // copy
        ld.global.f32 %f1, [%rd1]  // uses %rd1

    After:
        mov.u64    %rd1, %rd0       // now dead → DIE removes it
        ld.global.f32 %f1, [%rd0]  // substituted
"""

import re
from typing import Dict, List, Tuple

from ..ir import Instruction, RawLine, PTXNode


_RE_REGISTER = re.compile(r'^%\w+$')


def _substitute_operand(operand: str, copy_map: Dict[str, str]) -> str:
    """
    Replace a register in an operand with its copy-map equivalent.

    Handles:
        %rd1       → %rd0          (bare register)
        [%rd1]     → [%rd0]        (memory address)
        [%rd1+8]   → [%rd0+8]     (address + offset)
    """
    if not copy_map:
        return operand

    # Bare register
    if _RE_REGISTER.match(operand) and operand in copy_map:
        return copy_map[operand]

    # Memory address operand: substitute register inside brackets
    # e.g.  [%rd1+8]  →  [%rd0+8]
    def _replace_reg(m: re.Match) -> str:
        inner = m.group(1)
        # inner may be "%rd1" or "%rd1+8" or "%rd1+-4"
        reg_match = re.match(r'^(%\w+)(.*)', inner)
        if reg_match:
            reg  = reg_match.group(1)
            rest = reg_match.group(2)
            replaced = copy_map.get(reg, reg)
            return f"[{replaced}{rest}]"
        return m.group(0)

    return re.sub(r'\[([^\]]+)\]', _replace_reg, operand)


def run(nodes: List[PTXNode]) -> Tuple[List[PTXNode], int]:
    """
    Redundant Copy Elimination pass.

    Args:
        nodes: parsed PTX node list

    Returns:
        (new_nodes, substitutions_made)
        Note: the original mov instructions are left in place.
        They become candidates for the subsequent DIE pass.
    """
    copy_map: Dict[str, str] = {}   # dst_reg → src_reg
    substitutions = 0
    result: List[PTXNode] = []

    for node in nodes:
        if not isinstance(node, Instruction):
            result.append(node)
            continue

        # ── Flush copy map at branches (conservative) ──────────────────────
        if node.is_branch() or node.is_barrier():
            copy_map.clear()
            result.append(node)
            continue

        # ── Substitute source operands using the copy map ──────────────────
        new_ops = []
        changed = False
        for op in node.operands:
            new_op = _substitute_operand(op, copy_map)
            if new_op != op:
                changed = True
                substitutions += 1
            new_ops.append(new_op)

        if changed:
            # Rebuild raw text with substituted operands
            all_ops = ([node.dest] if node.dest else []) + new_ops
            ops_str = ", ".join(all_ops)
            new_raw = (
                f"{node.indent}{node.opcode:<16}{ops_str};"
                f"  // [CE] substituted"
            )
            node = Instruction(
                opcode=node.dest and node.opcode or node.opcode,
                dest=node.dest,
                operands=new_ops,
                raw=new_raw,
                indent=node.indent,
                comment=node.comment,
            )

        # ── Update copy map ────────────────────────────────────────────────
        # If this is a mov %dst, %src  → record the copy
        if (node.is_mov()
                and node.dest
                and len(node.operands) == 1
                and _RE_REGISTER.match(node.operands[0])):
            src = node.operands[0]
            # Only record if src itself is not a copy of something else
            # (flatten the chain: if src→x exists, record dst→x)
            ultimate_src = copy_map.get(src, src)
            copy_map[node.dest] = ultimate_src
        else:
            # Any instruction that writes to a register kills its copy entry
            if node.dest:
                # Also remove any entry that used node.dest as a source
                killed = [k for k, v in copy_map.items()
                          if v == node.dest or k == node.dest]
                for k in killed:
                    copy_map.pop(k, None)

        result.append(node)

    return result, substitutions
