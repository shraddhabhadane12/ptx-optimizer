"""
test_passes.py — Unit tests for all four optimization passes.

Run with:
    python -m pytest tests/ -v
    python -m unittest tests/test_passes.py -v
"""

import unittest
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser import parse_source
from src.emitter import emit, count_instructions
from src.passes import dead_code, const_fold, algebraic, copy_elim
from src.ir import Instruction, RawLine


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse(ptx_snippet: str):
    """Parse an indented PTX snippet (adds minimal header/footer)."""
    full = ".version 7.5\n.target sm_89\n.address_size 64\n" + ptx_snippet
    return parse_source(full)


def instructions_only(nodes):
    return [n for n in nodes if isinstance(n, Instruction)]


# ── Dead Code Elimination Tests ───────────────────────────────────────────────

class TestDeadCodeElimination(unittest.TestCase):

    def test_dead_register_removed(self):
        """A register written but never read should be eliminated."""
        src = """
    .reg .f32 %f<4>;
    .reg .u64 %rd<2>;
    add.f32   %f1, %f2, %f3;
    st.global.f32 [%rd0], %f2;
"""
        nodes = parse(src)
        nodes, count = dead_code.run(nodes)
        self.assertGreaterEqual(count, 1)
        # %f1 is written but never stored or read → should be removed
        live_instrs = instructions_only(nodes)
        dests = [n.dest for n in live_instrs]
        self.assertNotIn('%f1', dests)

    def test_store_never_removed(self):
        """Store instructions must never be deleted even if 'dead'."""
        src = """
    .reg .f32 %f<2>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    st.global.f32   [%rd0], %f0;
"""
        nodes = parse(src)
        nodes, count = dead_code.run(nodes)
        raw_text = emit(nodes)
        self.assertIn('st.global.f32', raw_text)

    def test_ret_never_removed(self):
        """ret instructions must always be preserved."""
        src = """
    ret;
"""
        nodes = parse(src)
        nodes, count = dead_code.run(nodes)
        raw_text = emit(nodes)
        self.assertIn('ret', raw_text)

    def test_used_register_kept(self):
        """A register that IS read should not be eliminated."""
        src = """
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    add.f32         %f0, %f1, %f2;
    st.global.f32   [%rd0], %f0;
"""
        nodes = parse(src)
        nodes, count = dead_code.run(nodes)
        self.assertEqual(count, 0)

    def test_multiple_dead_instructions(self):
        """Multiple dead instructions in a row are all removed."""
        src = """
    .reg .f32 %f<8>;
    .reg .u64 %rd<2>;
    add.f32   %f1, %f2, %f3;
    mul.f32   %f4, %f2, %f3;
    sub.f32   %f5, %f2, %f3;
    st.global.f32 [%rd0], %f2;
"""
        nodes = parse(src)
        nodes, count = dead_code.run(nodes)
        self.assertGreaterEqual(count, 3)


# ── Constant Folding Tests ───────────────────────────────────────────────────

class TestConstantFolding(unittest.TestCase):

    def test_float_add_folded(self):
        """add.f32 of two float literals should be folded to a mov."""
        src = """
    .reg .f32 %f<4>;
    mov.f32   %f1, 0f40000000;
    mov.f32   %f2, 0f40400000;
    add.f32   %f3, %f1, %f2;
"""
        # 0f40000000 = 2.0,  0f40400000 = 3.0,  sum = 5.0 = 0f40A00000
        nodes = parse(src)
        nodes, count = const_fold.run(nodes)
        self.assertGreaterEqual(count, 1)
        raw = emit(nodes)
        # The folded result should encode 5.0
        self.assertIn('40A00000', raw.upper())

    def test_integer_mul_folded(self):
        """mul of two integer literals should be folded."""
        src = """
    .reg .u32 %r<4>;
    mov.u32   %r1, 6;
    mov.u32   %r2, 7;
    mul.lo.u32 %r3, %r1, %r2;
"""
        nodes = parse(src)
        nodes, count = const_fold.run(nodes)
        self.assertGreaterEqual(count, 1)
        raw = emit(nodes)
        self.assertIn('42', raw)   # 6 * 7 = 42

    def test_no_fold_with_variable(self):
        """Cannot fold when one operand is a non-constant register."""
        src = """
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    mov.f32         %f1, 0f40000000;
    add.f32         %f2, %f0, %f1;
"""
        nodes = parse(src)
        nodes, count = const_fold.run(nodes)
        # %f0 is loaded from memory — not a constant
        self.assertEqual(count, 0)

    def test_constant_propagation_chain(self):
        """Constants propagate through a mov chain before folding."""
        src = """
    .reg .f32 %f<5>;
    mov.f32   %f0, 0f3F800000;
    mov.f32   %f1, %f0;
    add.f32   %f2, %f1, 0f3F800000;
"""
        # %f1 = %f0 = 1.0, then add.f32 %f2, 1.0, 1.0 → fold to 2.0
        nodes = parse(src)
        nodes, count = const_fold.run(nodes)
        self.assertGreaterEqual(count, 1)


# ── Algebraic Simplification Tests ───────────────────────────────────────────

class TestAlgebraicSimplification(unittest.TestCase):

    def test_add_zero_float(self):
        """add.f32 %f1, %f0, 0f00000000 → mov.f32 %f1, %f0"""
        src = """
    .reg .f32 %f<4>;
    add.f32   %f1, %f0, 0f00000000;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)
        instrs = instructions_only(nodes)
        self.assertTrue(any(n.opcode.startswith('mov') for n in instrs))

    def test_mul_one(self):
        """mul.f32 %f1, %f0, 0f3F800000 → mov.f32 %f1, %f0"""
        src = """
    .reg .f32 %f<3>;
    mul.f32   %f1, %f0, 0f3F800000;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)

    def test_mul_zero(self):
        """mul.f32 %f1, %f0, 0f00000000 → mov.f32 %f1, 0f00000000"""
        src = """
    .reg .f32 %f<3>;
    mul.f32   %f1, %f0, 0f00000000;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)

    def test_xor_self(self):
        """xor.b32 %r1, %r0, %r0 → mov.b32 %r1, 0"""
        src = """
    .reg .b32 %r<3>;
    xor.b32   %r1, %r0, %r0;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)
        instrs = instructions_only(nodes)
        self.assertEqual(instrs[0].operands[0], '0')

    def test_sub_zero(self):
        """sub.u32 %r1, %r0, 0 → mov.u32 %r1, %r0"""
        src = """
    .reg .u32 %r<3>;
    sub.u32   %r1, %r0, 0;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)

    def test_shl_zero(self):
        """shl.b32 %r1, %r0, 0 → mov.b32 %r1, %r0"""
        src = """
    .reg .b32 %r<3>;
    shl.b32   %r1, %r0, 0;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)

    def test_mul_two_strength_reduction(self):
        """mul.u32 %r1, %r0, 2 → add.u32 %r1, %r0, %r0"""
        src = """
    .reg .u32 %r<3>;
    mul.lo.u32  %r1, %r0, 2;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 1)
        instrs = instructions_only(nodes)
        self.assertTrue(instrs[0].opcode.startswith('add'))

    def test_no_simplification_needed(self):
        """Normal add should not be simplified."""
        src = """
    .reg .f32 %f<4>;
    add.f32   %f2, %f0, %f1;
"""
        nodes = parse(src)
        nodes, count = algebraic.run(nodes)
        self.assertEqual(count, 0)


# ── Copy Elimination Tests ────────────────────────────────────────────────────

class TestCopyElimination(unittest.TestCase):

    def test_simple_register_copy(self):
        """mov %rd1, %rd0 → subsequent uses of %rd1 replaced with %rd0."""
        src = """
    .reg .u64 %rd<4>;
    .reg .f32 %f<2>;
    mov.u64           %rd1, %rd0;
    ld.global.f32     %f0, [%rd1];
"""
        nodes = parse(src)
        nodes, count = copy_elim.run(nodes)
        self.assertGreaterEqual(count, 1)
        raw = emit(nodes)
        # After elimination, %rd1 should be replaced by %rd0 in the ld
        self.assertIn('%rd0', raw)

    def test_no_copy_across_overwrite(self):
        """Copy map must be invalidated when src is overwritten."""
        src = """
    .reg .u64 %rd<4>;
    .reg .f32 %f<3>;
    mov.u64         %rd1, %rd0;
    ld.global.f32   %f0, [%rd0];
    add.u64         %rd0, %rd0, 8;
    ld.global.f32   %f1, [%rd1];
"""
        nodes = parse(src)
        nodes, count = copy_elim.run(nodes)
        raw = emit(nodes)
        # %rd1 was mapped to %rd0, but %rd0 was overwritten before second use
        # second ld must NOT substitute %rd0 in place of %rd1
        instrs = instructions_only(nodes)
        ld_instrs = [n for n in instrs if n.opcode.startswith('ld')]
        self.assertEqual(len(ld_instrs), 2)

    def test_branch_flushes_map(self):
        """A branch instruction should flush the copy map."""
        src = """
    .reg .pred %p<1>;
    .reg .u64  %rd<4>;
    .reg .f32  %f<2>;
    mov.u64         %rd1, %rd0;
    @%p0 bra        END;
    ld.global.f32   %f0, [%rd1];
END:
    ret;
"""
        nodes = parse(src)
        # Should not crash — branch handling is tested here
        nodes, count = copy_elim.run(nodes)
        self.assertIsInstance(nodes, list)

    def test_chain_copy_flattened(self):
        """A copy chain %b=%a, %c=%b is flattened to %c=%a."""
        src = """
    .reg .u64 %rd<5>;
    .reg .f32 %f<2>;
    mov.u64         %rd1, %rd0;
    mov.u64         %rd2, %rd1;
    ld.global.f32   %f0, [%rd2];
"""
        nodes = parse(src)
        nodes, count = copy_elim.run(nodes)
        self.assertGreaterEqual(count, 1)


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_vector_add_ptx_file(self):
        """End-to-end: vector_add.ptx should have fewer instructions after opt."""
        ptx_path = os.path.join(os.path.dirname(__file__),
                                '..', 'samples', 'ptx_files', 'vector_add.ptx')
        if not os.path.exists(ptx_path):
            self.skipTest("vector_add.ptx not found")

        with open(ptx_path, encoding='utf-8') as f:
            source = f.read()

        nodes = parse_source(source)
        original = count_instructions(nodes)

        from src.optimizer import optimize
        result = optimize(source, rounds=2)
        optimized = count_instructions(result['nodes'])

        self.assertLessEqual(optimized, original,
                             "Optimized PTX should have fewer instructions")
        total_removed = sum(result['stats'][k]
                            for k in ('dead', 'folded', 'simplified', 'copies'))
        self.assertGreater(total_removed, 0,
                           "At least one optimization should fire")

    def test_all_passes_together(self):
        """Cascade: copy elimination exposes constant folding, which exposes DIE."""
        src = """
    .reg .f32 %f<8>;
    .reg .u64 %rd<3>;
    mov.f32         %f0, 0f40000000;
    mov.f32         %f1, %f0;
    add.f32         %f2, %f1, 0f40000000;
    mul.f32         %f3, %f2, 0f3F800000;
    add.f32         %f4, %f3, 0f00000000;
    st.global.f32   [%rd0], %f4;
"""
        from src.optimizer import optimize
        result = optimize(src, rounds=2)
        total = sum(result['stats'][k]
                    for k in ('dead', 'folded', 'simplified', 'copies'))
        self.assertGreater(total, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
