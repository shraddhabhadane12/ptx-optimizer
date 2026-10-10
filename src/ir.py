"""
ir.py — Intermediate Representation for PTX instructions.

Each line of PTX is parsed into an Instruction object.
Non-instruction lines (directives, labels, blank lines) are
stored as RawLine objects and passed through unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Instruction:
    """
    Represents a single PTX instruction.

    Example PTX line:
        add.f32   %f3, %f1, %f2

    Parsed as:
        opcode   = "add.f32"
        dest     = "%f3"
        operands = ["%f1", "%f2"]
        raw      = "    add.f32   %f3, %f1, %f2"
    """
    opcode: str                        # e.g. "add.f32", "ld.global.f32"
    dest: Optional[str]                # destination register or None
    operands: List[str]                # source operands
    raw: str                           # original text line (for fallback emit)
    indent: str = "    "               # leading whitespace (preserved)

    # Flags set by optimization passes
    is_dead: bool = False              # marked for deletion by dead-code pass
    comment: str = ""                  # optional annotation added by a pass

    def is_store(self) -> bool:
        """Store instructions must never be deleted."""
        return self.opcode.startswith("st.")

    def is_load(self) -> bool:
        return self.opcode.startswith("ld.")

    def is_branch(self) -> bool:
        return self.opcode in ("bra", "bra.uni", "ret", "exit",
                               "call", "call.uni", "brx.idx")

    def is_barrier(self) -> bool:
        return self.opcode.startswith("bar.") or self.opcode == "bar"

    def is_mov(self) -> bool:
        return self.opcode.startswith("mov.")

    def writes_register(self) -> bool:
        """True if this instruction produces a value into dest."""
        return (self.dest is not None
                and not self.is_store()
                and not self.is_branch()
                and not self.is_barrier())

    def all_operands(self) -> List[str]:
        """All register operands (dest + sources), for analysis."""
        ops = []
        if self.dest:
            ops.append(self.dest)
        ops.extend(self.operands)
        return ops

    def __repr__(self) -> str:
        dest_str = f"{self.dest}, " if self.dest else ""
        ops_str = ", ".join(self.operands)
        return f"Instruction({self.opcode}  {dest_str}{ops_str})"


@dataclass
class RawLine:
    """
    A PTX line that is NOT an instruction — directives, labels,
    blank lines, comments. Passed through the pipeline unchanged.
    """
    text: str                          # original text including newline

    def __repr__(self) -> str:
        return f"RawLine({self.text.strip()!r})"


# Type alias used throughout the project
PTXNode = Instruction | RawLine
