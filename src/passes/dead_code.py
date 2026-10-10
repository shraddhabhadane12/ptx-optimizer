"""
passes/dead_code.py — Dead Instruction Elimination (DIE)

THEORY:
    An instruction is "dead" if its destination register is never READ
    by any subsequent instruction.  We perform a single backward scan
    (liveness analysis) to collect every register that is ever used as
    a source operand.  Any instruction that WRITES to a register not in
    that live set is marked dead and removed.

SAFE-GUARDS (never delete):
    - Store instructions  (st.*)  — they have side effects
    - Branch / return     (bra, ret, exit, call, …)
    - Barrier             (bar.*)
    - Instructions with a predicate register as source (@%p0 bra …)
    - Special registers   (%tid, %ntid, %ctaid, %nctaid, %laneid, …)

EXAMPLE:
    Before:
        add.f32   %f3, %f1, %f2   // %f3 never read  → DEAD
        add.f32   %f4, %f1, %f2   // %f4 used below
        st.global.f32 [%rd0], %f4

    After:
        add.f32   %f4, %f1, %f2
        st.global.f32 [%rd0], %f4
"""

import re
from typing import List, Set, Tuple

from ..ir import Instruction, RawLine, PTXNode


# Special / built-in PTX registers that must never be treated as dead writes
_SPECIAL_REGS = re.compile(
    r'^%('
    r'tid|ntid|ctaid|nctaid|gridid|laneid|warpid|nwarpid|smid|nsmid'
    r'|clock|clock64|pm\d|envreg\d+|globaltimer'
    r')(\.\w+)?$'
)

# Register name pattern
_RE_REG = re.compile(r'%\w+')


def _extract_registers(operand: str) -> List[str]:
    """Extract all register names from an operand string (handles [%rd0+4])."""
    return _RE_REG.findall(operand)


def _build_live_set(nodes: List[PTXNode]) -> Set[str]:
    """
    Collect every register that appears as a SOURCE operand anywhere
    in the function body.  This is our approximation of 'liveness'.

    A single backward pass is conservative but sound: it may keep some
    instructions alive that a full dataflow analysis would remove, but
    it will NEVER incorrectly remove a live instruction.
    """
    live: Set[str] = set()

    for node in nodes:
        if not isinstance(node, Instruction):
            continue

        # All source operands keep their registers live
        for op in node.operands:
            for reg in _extract_registers(op):
                live.add(reg)

        # Predicate register used by a predicated instruction
        if node.comment.startswith("pred:"):
            pred_reg = node.comment.split("pred:")[-1].lstrip("!")
            live.add(pred_reg)

        # Stores: the "dest" slot in our IR is actually the address,
        # which is a source.  Make sure we capture it.
        if node.is_store() and node.dest:
            for reg in _extract_registers(node.dest):
                live.add(reg)

    return live


def run(nodes: List[PTXNode]) -> Tuple[List[PTXNode], int]:
    """
    Dead Instruction Elimination pass.

    Args:
        nodes: parsed PTX node list

    Returns:
        (new_nodes, eliminated_count)
        new_nodes has dead instructions replaced by annotated RawLines
        so the output PTX still shows what was removed (as a comment).
    """
    live_set = _build_live_set(nodes)
    eliminated = 0
    result: List[PTXNode] = []

    for node in nodes:
        if not isinstance(node, Instruction):
            result.append(node)
            continue

        # Never touch side-effecting instructions
        if node.is_store() or node.is_branch() or node.is_barrier():
            result.append(node)
            continue

        # Never touch instructions without a destination
        if not node.dest:
            result.append(node)
            continue

        # Never touch special registers
        if _SPECIAL_REGS.match(node.dest):
            result.append(node)
            continue

        # If the destination register is never read → dead
        if node.dest not in live_set:
            node.is_dead = True
            eliminated += 1
            # Emit as a PTX comment so the output shows the removal
            result.append(RawLine(
                f"{node.indent}// [DIE] removed dead: {node.raw.strip()}"
            ))
        else:
            result.append(node)

    return result, eliminated
