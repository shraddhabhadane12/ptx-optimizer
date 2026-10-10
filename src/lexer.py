"""
lexer.py — PTX Tokenizer

Classifies each raw text line as one of:
  - INSTRUCTION  : an executable PTX opcode line
  - DIRECTIVE    : .reg, .param, .visible, .entry, .func, etc.
  - LABEL        : a branch target like "BB0_1:"
  - COMMENT      : line starting with //
  - BLANK        : empty / whitespace only

The parser uses these classifications to decide what to parse deeply
vs. what to pass through as a RawLine.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenKind(Enum):
    INSTRUCTION = auto()
    DIRECTIVE   = auto()
    LABEL       = auto()
    COMMENT     = auto()
    BLANK       = auto()


@dataclass
class Token:
    kind: TokenKind
    text: str          # original line text (with newline stripped)
    lineno: int        # 1-based line number (useful for error messages)

    def stripped(self) -> str:
        return self.text.strip()


# ── Regex patterns ──────────────────────────────────────────────────────────

# PTX directives always start with a dot
_RE_DIRECTIVE = re.compile(r'^\s*\.')

# Labels end with a colon  (e.g.  "BB0_1:" or "$L__BB0_3:")
_RE_LABEL = re.compile(r'^\s*[\$\w]+\s*:\s*$')

# Comments start with //
_RE_COMMENT = re.compile(r'^\s*//')

# PTX register names: %f0, %rd1, %p2, %r3, %fd4, etc.
REG_PATTERN = re.compile(r'%\w+')

# PTX immediate float literal: 0f3F800000
FLOAT_IMM_PATTERN = re.compile(r'0[fF]([0-9A-Fa-f]{8})')

# PTX immediate integer literal (hex or decimal)
INT_IMM_PATTERN = re.compile(r'\b(0x[0-9A-Fa-f]+|\d+)\b')

# Memory address operand: [%rd0+4] or [%rd0]
MEM_OPERAND_PATTERN = re.compile(r'\[([^\]]+)\]')

# Full instruction line (very permissive — parser does the real work)
# Matches:  <opcode>  <operand>, <operand>, ...
# opcode can have dots: add.f32, ld.global.f32, st.shared.b64
_RE_INSTRUCTION = re.compile(
    r'^\s+'                        # must be indented (PTX convention)
    r'([\w][\w\.]*)'               # opcode
    r'(?:\s+(.+))?'                # optional operand list
    r'\s*(?://.*)?$'               # optional trailing comment
)


# ── Public API ───────────────────────────────────────────────────────────────

def tokenize(source: str) -> List[Token]:
    """
    Tokenize a full PTX source string into a list of Token objects.

    Args:
        source: complete PTX file contents as a string

    Returns:
        list of Token, one per line
    """
    tokens: List[Token] = []
    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line  # keep original for the token text
        stripped = raw_line.strip()

        if not stripped:
            tokens.append(Token(TokenKind.BLANK, raw_line, lineno))

        elif _RE_COMMENT.match(stripped):
            tokens.append(Token(TokenKind.COMMENT, raw_line, lineno))

        elif _RE_DIRECTIVE.match(stripped):
            tokens.append(Token(TokenKind.DIRECTIVE, raw_line, lineno))

        elif _RE_LABEL.match(stripped):
            tokens.append(Token(TokenKind.LABEL, raw_line, lineno))

        elif _RE_INSTRUCTION.match(raw_line):
            tokens.append(Token(TokenKind.INSTRUCTION, raw_line, lineno))

        else:
            # Fallback — treat as directive/passthrough (e.g. closing braces)
            tokens.append(Token(TokenKind.DIRECTIVE, raw_line, lineno))

    return tokens


def split_operands(operand_str: str) -> List[str]:
    """
    Split the operand portion of a PTX instruction into individual operands.

    Handles:
      - Simple:     "%f1, %f2, %f3"  →  ["%f1", "%f2", "%f3"]
      - Memory:     "[%rd0+4], %f1"  →  ["[%rd0+4]", "%f1"]
      - Predicate:  "@%p0 bra BB0;"  (predicate stripped by parser)

    Args:
        operand_str: raw operand string after the opcode

    Returns:
        list of trimmed operand strings
    """
    # Strip trailing comment
    operand_str = re.sub(r'//.*$', '', operand_str).strip()
    # Strip trailing semicolon
    operand_str = operand_str.rstrip(';').strip()

    if not operand_str:
        return []

    # Split on commas NOT inside brackets
    parts = []
    depth = 0
    current = []
    for ch in operand_str:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    return [p for p in parts if p]
