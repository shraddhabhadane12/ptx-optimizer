"""
passes/algebraic.py — Algebraic Simplification (Peephole Optimizer)

THEORY:
    A peephole optimizer looks at a small window of instructions and
    replaces known-inefficient patterns with cheaper equivalents using
    algebraic identities.  No dataflow analysis required — just pattern
    matching on opcode + operands.

PATTERNS HANDLED:
    Identity / annihilator rules:
        add  x, 0      →  mov x          (additive identity)
        sub  x, 0      →  mov x
        mul  x, 1      →  mov x          (multiplicative identity)
        mul  x, 0      →  mov 0          (annihilator)
        div  x, 1      →  mov x
        or   x, 0      →  mov x          (bitwise identity)
        and  x, -1/0xFFFFFFFF → mov x
        and  x, 0      →  mov 0
        xor  x, 0      →  mov x
        shl  x, 0      →  mov x
        shr  x, 0      →  mov x

    Self-cancellation:
        sub  x, x      →  mov 0
        xor  x, x      →  mov 0
        and  x, x      →  mov x
        or   x, x      →  mov x

    Strength reduction:
        mul  x, 2      →  add x, x       (cheaper on GPU)
        mul.u32 x, POW2 → shl x, log2   (shift instead of multiply)

EXAMPLE:
    Before:
        add.f32   %f1, %f2, 0f00000000   // add 0.0

    After:
        mov.f32   %f1, %f2               // [AS] add identity
"""

import math
import re
from typing import List, Optional, Tuple

from ..ir import Instruction, RawLine, PTXNode


_RE_FLOAT_ZERO = re.compile(r'^0[fF]0{8}$')
_RE_FLOAT_ONE  = re.compile(r'^0[fF]3[Ff]800000$', re.IGNORECASE)
_RE_INT_ZERO   = re.compile(r'^0+$|^0[xX]0+$')
_RE_INT_ONE    = re.compile(r'^1$|^0[xX]1$')
_RE_INT_NEG1   = re.compile(r'^-1$|^0[xX][Ff]{8}$|^0[xX][Ff]{4}$', re.IGNORECASE)
_RE_REGISTER   = re.compile(r'^%\w+$')


def _is_zero(operand: str) -> bool:
    return bool(_RE_FLOAT_ZERO.match(operand) or _RE_INT_ZERO.match(operand))


def _is_one(operand: str) -> bool:
    return bool(_RE_FLOAT_ONE.match(operand) or _RE_INT_ONE.match(operand))


def _is_neg_one(operand: str) -> bool:
    return bool(_RE_INT_NEG1.match(operand))


def _is_register(operand: str) -> bool:
    return bool(_RE_REGISTER.match(operand))


def _zero_literal(opcode: str) -> str:
    """Return the zero literal appropriate for this opcode's type."""
    if '.f64' in opcode:
        return '0d0000000000000000'
    if '.f32' in opcode:
        return '0f00000000'
    return '0'


def _power_of_two_log(n: int) -> Optional[int]:
    """If n is a power of two > 0, return log2(n), else None."""
    if n > 0 and (n & (n - 1)) == 0:
        return int(math.log2(n))
    return None


def _int_val(s: str) -> Optional[int]:
    try:
        return int(s, 16) if s.lower().startswith('0x') else int(s)
    except ValueError:
        return None


def _make_mov(node: Instruction, src: str, reason: str) -> Instruction:
    """Create a replacement mov instruction."""
    type_suffix = node.opcode.split('.', 1)[1] if '.' in node.opcode else 'b32'
    new_opcode = f"mov.{type_suffix}"
    new_raw = (
        f"{node.indent}{new_opcode:<16}{node.dest}, {src};"
        f"  // [AS] {reason}: {node.raw.strip()}"
    )
    return Instruction(
        opcode=new_opcode,
        dest=node.dest,
        operands=[src],
        raw=new_raw,
        indent=node.indent,
    )


def _try_simplify(node: Instruction) -> Optional[Instruction]:
    """
    Try all algebraic rules on a single instruction.
    Returns a simplified Instruction, or None if no rule applies.
    """
    if not node.dest or len(node.operands) < 1:
        return None

    base = node.opcode.split('.')[0]
    ops  = node.operands

    # ── Unary / single-operand (shouldn't simplify further) ────────────────
    if len(ops) == 1:
        return None

    # ── Binary operations ───────────────────────────────────────────────────
    if len(ops) == 2:
        a, b = ops[0], ops[1]

        # --- ADD / SUB ---
        if base == 'add':
            if _is_zero(b): return _make_mov(node, a, "add 0")
            if _is_zero(a): return _make_mov(node, b, "0 + x")

        if base == 'sub':
            if _is_zero(b): return _make_mov(node, a, "sub 0")
            if _is_register(a) and a == b:  # x - x = 0
                return _make_mov(node, _zero_literal(node.opcode), "x-x=0")

        # --- MUL ---
        if base == 'mul':
            if _is_zero(b) or _is_zero(a):
                return _make_mov(node, _zero_literal(node.opcode), "mul 0")
            if _is_one(b):  return _make_mov(node, a, "mul 1")
            if _is_one(a):  return _make_mov(node, b, "1*x")

            # Strength reduction: mul x, 2 → add x, x  (integer only)
            if '.f' not in node.opcode:
                n = _int_val(b)
                if n == 2 and _is_register(a):
                    type_suffix = node.opcode.split('.', 1)[1] if '.' in node.opcode else 'b32'
                    new_opcode = f"add.{type_suffix}"
                    new_raw = (
                        f"{node.indent}{new_opcode:<16}{node.dest}, {a}, {a};"
                        f"  // [AS] mul2→add: {node.raw.strip()}"
                    )
                    return Instruction(new_opcode, node.dest, [a, a], new_raw, node.indent)

                # Strength reduction: mul x, POW2 → shl x, log2
                if n is not None and n > 0:
                    shift = _power_of_two_log(n)
                    if shift and shift > 1:  # mul 2 already handled above
                        # Use matching integer type for shl
                        parts = node.opcode.split('.')
                        int_type = parts[-1] if parts[-1] in ('u32','s32','u64','s64') else 'u32'
                        new_opcode = f"shl.b{int_type[-2:]}"
                        new_raw = (
                            f"{node.indent}{new_opcode:<16}{node.dest}, {a}, {shift};"
                            f"  // [AS] mul→shl: {node.raw.strip()}"
                        )
                        return Instruction(new_opcode, node.dest, [a, str(shift)], new_raw, node.indent)

        # --- DIV ---
        if base == 'div':
            if _is_one(b): return _make_mov(node, a, "div 1")

        # --- AND ---
        if base == 'and':
            if _is_zero(b) or _is_zero(a):
                return _make_mov(node, _zero_literal(node.opcode), "and 0")
            if _is_neg_one(b): return _make_mov(node, a, "and -1")
            if _is_neg_one(a): return _make_mov(node, b, "-1 and x")
            if _is_register(a) and _is_register(b) and a == b:
                return _make_mov(node, a, "x&x=x")

        # --- OR ---
        if base == 'or':
            if _is_zero(b): return _make_mov(node, a, "or 0")
            if _is_zero(a): return _make_mov(node, b, "0|x")
            if _is_register(a) and _is_register(b) and a == b:
                return _make_mov(node, a, "x|x=x")

        # --- XOR ---
        if base == 'xor':
            if _is_zero(b): return _make_mov(node, a, "xor 0")
            if _is_zero(a): return _make_mov(node, b, "0^x")
            if _is_register(a) and _is_register(b) and a == b:
                return _make_mov(node, _zero_literal(node.opcode), "x^x=0")

        # --- SHL / SHR ---
        if base in ('shl', 'shr'):
            if _is_zero(b): return _make_mov(node, a, f"{base} 0")

    return None


def run(nodes: List[PTXNode]) -> Tuple[List[PTXNode], int]:
    """
    Algebraic Simplification pass.

    Args:
        nodes: parsed PTX node list

    Returns:
        (new_nodes, simplified_count)
    """
    simplified = 0
    result: List[PTXNode] = []

    for node in nodes:
        if not isinstance(node, Instruction):
            result.append(node)
            continue

        replacement = _try_simplify(node)
        if replacement is not None:
            result.append(replacement)
            simplified += 1
        else:
            result.append(node)

    return result, simplified
