"""
passes/const_fold.py — Constant Folding + Constant Propagation

THEORY:
    1. CONSTANT PROPAGATION
       Track which registers hold a known constant value (set by mov
       instructions with an immediate operand).  Substitute those
       constants into subsequent uses of that register.

    2. CONSTANT FOLDING
       After substitution, if ALL source operands of an arithmetic /
       logic instruction are now literals, evaluate the expression at
       compile time and replace the whole instruction with a single
       mov of the computed result.

PTX LITERAL FORMATS supported:
    Float  : 0f3F800000  (IEEE-754 hex, big-endian)
    Double : 0d3FF0000000000000
    Integer: decimal (42) or hex (0x2A)

EXAMPLE:
    Before:
        mov.f32   %f1, 0f40000000    // %f1 = 2.0
        mov.f32   %f2, 0f40400000    // %f2 = 3.0
        add.f32   %f3, %f1, %f2      // both sources are constants

    After:
        mov.f32   %f3, 0f40A00000    // %f3 = 5.0  (folded)
        // %f1 and %f2 movs become dead → removed by DIE pass next round
"""

import re
import struct
from typing import Dict, List, Optional, Tuple, Union

from ..ir import Instruction, RawLine, PTXNode


# ── Literal parsing helpers ──────────────────────────────────────────────────

_RE_FLOAT_IMM  = re.compile(r'^0[fF]([0-9A-Fa-f]{8})$')
_RE_DOUBLE_IMM = re.compile(r'^0[dD]([0-9A-Fa-f]{16})$')
_RE_INT_IMM    = re.compile(r'^(-?(?:0[xX][0-9A-Fa-f]+|\d+))$')
_RE_REGISTER   = re.compile(r'^%\w+$')

# Arithmetic opcodes that can be folded (type suffix after first dot)
_FOLDABLE_OPS = {
    "add", "sub", "mul", "div", "rem",
    "min", "max", "and", "or", "xor",
    "shl", "shr",
}


def _parse_literal(s: str) -> Optional[Union[float, int]]:
    """Return Python numeric value of a PTX literal, or None."""
    m = _RE_FLOAT_IMM.match(s)
    if m:
        raw = bytes.fromhex(m.group(1))
        return struct.unpack('>f', raw)[0]

    m = _RE_DOUBLE_IMM.match(s)
    if m:
        raw = bytes.fromhex(m.group(1))
        return struct.unpack('>d', raw)[0]

    m = _RE_INT_IMM.match(s)
    if m:
        txt = m.group(1)
        return int(txt, 16) if txt.lower().startswith(('0x', '-0x')) else int(txt)

    return None


def _encode_float(value: float, double: bool = False) -> str:
    """Encode a Python float back to PTX literal form."""
    if double:
        raw = struct.pack('>d', value)
        return "0d" + raw.hex().upper()
    else:
        raw = struct.pack('>f', value)
        return "0f" + raw.hex().upper()


def _encode_int(value: int) -> str:
    """Encode a Python int back to PTX literal (decimal)."""
    return str(value)


def _get_type(opcode: str) -> str:
    """Extract type suffix: 'add.f32' → 'f32', 'mul.lo.u32' → 'u32'."""
    parts = opcode.split('.')
    return parts[-1] if len(parts) > 1 else ""


def _fold_binary(opcode: str, a: Union[float, int],
                 b: Union[float, int]) -> Optional[Union[float, int]]:
    """Evaluate a binary PTX operation on two literals."""
    base_op = opcode.split('.')[0]
    try:
        if base_op == "add": return a + b
        if base_op == "sub": return a - b
        if base_op == "mul": return a * b
        if base_op == "div":
            if b == 0:
                return None  # don't fold division by zero
            return a / b if isinstance(a, float) else a // b
        if base_op == "rem": return int(a) % int(b)
        if base_op == "min": return min(a, b)
        if base_op == "max": return max(a, b)
        if base_op == "and": return int(a) & int(b)
        if base_op == "or":  return int(a) | int(b)
        if base_op == "xor": return int(a) ^ int(b)
        if base_op == "shl": return int(a) << int(b)
        if base_op == "shr": return int(a) >> int(b)
    except (OverflowError, ValueError):
        return None
    return None


# ── Pass implementation ──────────────────────────────────────────────────────

def run(nodes: List[PTXNode]) -> Tuple[List[PTXNode], int]:
    """
    Constant Folding + Constant Propagation pass.

    Args:
        nodes: parsed PTX node list

    Returns:
        (new_nodes, folded_count)
    """
    # Maps register → known constant value (Python int or float)
    const_map: Dict[str, Union[float, int]] = {}
    folded = 0
    result: List[PTXNode] = []

    for node in nodes:
        if not isinstance(node, Instruction):
            result.append(node)
            continue

        # ── Step 1: Resolve source operands via const_map ──────────────────
        resolved_ops: List[Optional[Union[float, int]]] = []
        for op in node.operands:
            if _RE_REGISTER.match(op) and op in const_map:
                resolved_ops.append(const_map[op])
            else:
                resolved_ops.append(_parse_literal(op))

        # ── Step 2: Track mov with immediate → update const_map ────────────
        if node.is_mov() and node.dest and len(node.operands) == 1:
            lit = _parse_literal(node.operands[0])
            if lit is not None:
                const_map[node.dest] = lit
            elif _RE_REGISTER.match(node.operands[0]):
                src = node.operands[0]
                if src in const_map:
                    const_map[node.dest] = const_map[src]
                else:
                    # propagated constant is no longer valid if src changes
                    const_map.pop(node.dest, None)
            else:
                const_map.pop(node.dest, None)
            result.append(node)
            continue

        # ── Step 3: Any instruction that writes a register invalidates it ──
        if node.dest:
            const_map.pop(node.dest, None)

        # ── Step 4: Try to fold binary op with two constant operands ───────
        base_op = node.opcode.split('.')[0]
        if (base_op in _FOLDABLE_OPS
                and node.dest
                and len(resolved_ops) == 2
                and all(v is not None for v in resolved_ops)):

            a, b = resolved_ops[0], resolved_ops[1]
            folded_val = _fold_binary(node.opcode, a, b)

            if folded_val is not None:
                type_suffix = _get_type(node.opcode)
                is_double = "f64" in type_suffix or "d64" in type_suffix

                if isinstance(folded_val, float):
                    lit_str = _encode_float(folded_val, double=is_double)
                else:
                    lit_str = _encode_int(int(folded_val))

                new_opcode = f"mov.{type_suffix}" if type_suffix else "mov.b32"
                new_raw = (
                    f"{node.indent}{new_opcode:<16}{node.dest}, {lit_str};"
                    f"  // [CF] folded: {node.raw.strip()}"
                )
                new_instr = Instruction(
                    opcode=new_opcode,
                    dest=node.dest,
                    operands=[lit_str],
                    raw=new_raw,
                    indent=node.indent,
                )
                const_map[node.dest] = folded_val
                result.append(new_instr)
                folded += 1
                continue

        result.append(node)

    return result, folded
