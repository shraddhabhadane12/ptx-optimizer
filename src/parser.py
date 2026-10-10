"""
parser.py — PTX Parser

Converts a list of Tokens into a list of PTXNodes (Instruction | RawLine).
Instructions get fully parsed into opcode + dest + operands.
Everything else becomes a RawLine (passthrough).
"""

import re
from typing import List

from .ir import Instruction, RawLine, PTXNode
from .lexer import Token, TokenKind, split_operands


# ── Regex for parsing an instruction token ───────────────────────────────────

# Matches optional predicate, opcode, and operand list:
#   @%p0  add.f32   %f3, %f1, %f2;
_RE_INSTR_FULL = re.compile(
    r'^\s*'
    r'(?:@(!?)(%\w+)\s+)?'         # optional predicate: @%p0 or @!%p0
    r'([\w][\w\.]*)'               # opcode  (group 3)
    r'(?:\s+(.+))?'                # operands (group 4)
    r'\s*;?\s*(?://.*)?$'
)

# Registers look like %f0, %rd1, %p2, %r3
_RE_REGISTER = re.compile(r'^%\w+$')


def parse(tokens: List[Token]) -> List[PTXNode]:
    """
    Parse a list of tokens into PTXNodes.

    Args:
        tokens: output of lexer.tokenize()

    Returns:
        list of Instruction and RawLine objects
    """
    nodes: List[PTXNode] = []

    for token in tokens:
        if token.kind != TokenKind.INSTRUCTION:
            # Directives, labels, blanks, comments → passthrough
            nodes.append(RawLine(token.text))
            continue

        node = _parse_instruction(token)
        nodes.append(node)

    return nodes


def _parse_instruction(token: Token) -> PTXNode:
    """
    Parse a single INSTRUCTION token into an Instruction object.
    Falls back to RawLine if the line cannot be parsed.
    """
    m = _RE_INSTR_FULL.match(token.text)
    if not m:
        return RawLine(token.text)

    pred_neg   = m.group(1)    # "!" if negated predicate
    pred_reg   = m.group(2)    # e.g. "%p0" or None
    opcode     = m.group(3)    # e.g. "add.f32"
    operand_str = m.group(4) or ""

    operands = split_operands(operand_str)

    # Determine leading whitespace
    indent = len(token.text) - len(token.text.lstrip())
    indent_str = token.text[:indent]

    # For most instructions: first operand is the destination (written reg)
    # Exceptions: stores, branches, barriers — no destination register
    dest = None
    src_operands = operands

    if _has_destination(opcode) and operands:
        candidate = operands[0]
        # Destination must be a plain register (not a memory address)
        if _RE_REGISTER.match(candidate):
            dest = candidate
            src_operands = operands[1:]

    instr = Instruction(
        opcode=opcode,
        dest=dest,
        operands=src_operands,
        raw=token.text,
        indent=indent_str,
    )

    # Attach predicate info as a comment annotation (preserved in emit)
    if pred_reg:
        neg = "!" if pred_neg else ""
        instr.comment = f"pred:{neg}{pred_reg}"

    return instr


def _has_destination(opcode: str) -> bool:
    """
    Return True if this opcode writes a result to its first operand.
    Store, branch, barrier, and ret instructions do NOT have a dest.
    """
    NO_DEST_PREFIXES = (
        "st.",          # st.global.f32, st.shared.b32, …
        "ret",
        "bra",
        "exit",
        "bar",
        "call",
        "trap",
        "membar",
        "prefetch",
        "prefetchu",
        "discard",
    )
    for prefix in NO_DEST_PREFIXES:
        if opcode == prefix or opcode.startswith(prefix):
            return False
    return True


def parse_source(source: str) -> List[PTXNode]:
    """
    Convenience: parse PTX source text directly.

    Args:
        source: full PTX file content as string

    Returns:
        list of PTXNodes
    """
    from .lexer import tokenize
    tokens = tokenize(source)
    return parse(tokens)
