# PTX Optimizer

A static peephole optimizer for NVIDIA PTX (Parallel Thread Execution) assembly, built as a compiler engineering portfolio project targeting the **NVIDIA PTX Compiler Intern** role.

PTX is NVIDIA's virtual assembly language — every CUDA kernel passes through it. This tool parses raw PTX output from `nvcc`, applies four classical compiler optimization passes, and emits a cleaned, reduced PTX file with a detailed stats report.

```
CUDA C++ → PTX (nvcc) → [THIS TOOL] → Optimized PTX → SASS (ptxas)
```

---

## What It Does

The optimizer runs four passes (in two rounds to catch cascades):

| Pass | Key | What It Does |
|---|---|---|
| Copy Elimination | `ce` | Removes redundant `mov %dst, %src` register copies |
| Constant Folding | `cf` | Evaluates constant expressions at compile time |
| Algebraic Simplification | `as` | Applies identity/annihilator rules (`x+0`, `x*1`, `x^x`, …) |
| Dead Instruction Elimination | `die` | Removes instructions whose result is never used |

### Sample Output

```
╔══════════════════════════════════════════════════════╗
║            PTX OPTIMIZER — STATS REPORT             ║
╠══════════════════════════════════════════════════════╣
║  Dead instructions removed  : 5                     ║
║  Constants folded           : 2                     ║
║  Algebraic simplifications  : 4                     ║
║  Copy eliminations          : 3                     ║
╠══════════════════════════════════════════════════════╣
║  Original instruction count : 32                    ║
║  Optimized instruction count: 18                    ║
║  Total removed              : 14 (43.8% reduction)  ║
╚══════════════════════════════════════════════════════╝
```

---

## Project Structure

```
ptx-optimizer/
├── src/
│   ├── ir.py              # Instruction / RawLine data structures
│   ├── lexer.py           # PTX tokenizer (classifies each line)
│   ├── parser.py          # Builds IR from token list
│   ├── emitter.py         # Serializes IR back to PTX text
│   ├── optimizer.py       # Main driver + CLI
│   └── passes/
│       ├── dead_code.py   # Pass 1: Dead Instruction Elimination
│       ├── const_fold.py  # Pass 2: Constant Folding + Propagation
│       ├── algebraic.py   # Pass 3: Algebraic Simplification (20+ rules)
│       └── copy_elim.py   # Pass 4: Copy Elimination
├── samples/
│   ├── cuda_kernels/      # CUDA source (.cu) for generating PTX
│   │   ├── vector_add.cu
│   │   ├── matrix_mul.cu
│   │   └── reduction.cu
│   └── ptx_files/         # Hand-crafted PTX with intentional inefficiencies
│       ├── vector_add.ptx
│       ├── matrix_mul.ptx
│       └── reduction.ptx
├── tests/
│   └── test_passes.py     # 18 unit tests covering every pass
├── benchmarks/
│   └── run_benchmark.py   # Benchmark all PTX files, print table
└── README.md
```

---

## Quick Start

No CUDA installation needed — the hand-crafted PTX files in `samples/ptx_files/` work out of the box.

**Requirements:** Python 3.10+, no external dependencies.

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/ptx-optimizer.git
cd ptx-optimizer

# Run on the sample vector addition PTX
python -m src.optimizer samples/ptx_files/vector_add.ptx --stats -v

# Run on all samples
python -m src.optimizer samples/ptx_files/matrix_mul.ptx -o out/matrix_mul_opt.ptx --stats

# Benchmark all PTX files at once
python benchmarks/run_benchmark.py

# Run all unit tests
python -m pytest tests/ -v
# OR
python -m unittest tests/test_passes.py -v
```

---

## CLI Reference

```
python -m src.optimizer INPUT [options]

Arguments:
  INPUT               Path to input .ptx file

Options:
  -o OUTPUT           Output file (default: <input>_opt.ptx)
  --passes PASSES     Comma-separated pass list: ce,cf,as,die  (default: all)
  --rounds N          Optimization rounds (default: 2)
  --no-header         Skip stats comment block in output
  --stats             Always print stats table to stdout
  -v, --verbose       Verbose per-pass output
  -n, --dry-run       Analyse only, no file written
```

### Examples

```bash
# Run only dead code elimination
python -m src.optimizer input.ptx --passes die --stats

# Run two specific passes, one round
python -m src.optimizer input.ptx --passes cf,as --rounds 1 -v

# Dry run to preview stats without writing
python -m src.optimizer input.ptx --dry-run --stats -v

# Generate PTX from CUDA (requires nvcc)
nvcc -ptx -arch=sm_89 samples/cuda_kernels/vector_add.cu \
     -o samples/ptx_files/vector_add_nvcc.ptx
python -m src.optimizer samples/ptx_files/vector_add_nvcc.ptx --stats
```

---

## Optimization Passes — Deep Dive

### Pass 1: Copy Elimination (`ce`)

Tracks a copy map of `mov %dst, %src` instructions. Substitutes subsequent uses of `%dst` with `%src`, making the `mov` dead.

```ptx
; Before
mov.u64         %rd1, %rd0
ld.global.f32   %f0, [%rd1]

; After
ld.global.f32   %f0, [%rd0]   // %rd1 substituted
// mov is now dead → removed by DIE
```

### Pass 2: Constant Folding (`cf`)

Propagates known constants through `mov` instructions, then evaluates binary ops on two constants at compile time. Supports float (IEEE-754 hex literals), double, and integer.

```ptx
; Before
mov.f32   %f1, 0f40000000    // 2.0
mov.f32   %f2, 0f40400000    // 3.0
add.f32   %f3, %f1, %f2      // runtime add

; After
mov.f32   %f3, 0f40A00000    // 5.0 — computed at compile time
```

### Pass 3: Algebraic Simplification (`as`)

Pattern-matches 20+ algebraic identities:

```ptx
add  x, 0      → mov x           ; additive identity
sub  x, x      → mov 0           ; self-subtraction
mul  x, 1      → mov x           ; multiplicative identity
mul  x, 0      → mov 0           ; annihilator
mul  x, 2      → add x, x        ; strength reduction
mul  x, POW2   → shl x, log2     ; shift instead of multiply
xor  x, x      → mov 0           ; self-XOR
and  x, 0      → mov 0           ; AND annihilator
shl  x, 0      → mov x           ; shift by zero
```

### Pass 4: Dead Instruction Elimination (`die`)

Performs backward liveness analysis: builds the set of all registers ever read, then removes any instruction that writes to a register not in that set.

```ptx
; Before
add.f32   %f5, %f0, %f1   // %f5 never read → DEAD
st.global.f32 [%rd0], %f0

; After
st.global.f32 [%rd0], %f0
```

Safe-guards: stores, branches, barriers, and special registers (`%tid`, `%ntid`, `%ctaid`, …) are **never** removed.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_passes.py::TestDeadCodeElimination::test_dead_register_removed PASSED
tests/test_passes.py::TestDeadCodeElimination::test_store_never_removed PASSED
tests/test_passes.py::TestDeadCodeElimination::test_ret_never_removed PASSED
tests/test_passes.py::TestDeadCodeElimination::test_used_register_kept PASSED
tests/test_passes.py::TestDeadCodeElimination::test_multiple_dead_instructions PASSED
tests/test_passes.py::TestConstantFolding::test_float_add_folded PASSED
tests/test_passes.py::TestConstantFolding::test_integer_mul_folded PASSED
tests/test_passes.py::TestConstantFolding::test_no_fold_with_variable PASSED
tests/test_passes.py::TestConstantFolding::test_constant_propagation_chain PASSED
tests/test_passes.py::TestAlgebraicSimplification::test_add_zero_float PASSED
... (18 tests total)
```

---

## Benchmarks

```bash
python benchmarks/run_benchmark.py
```

Sample output (hand-crafted PTX files):
```
==========================================================================================
  PTX OPTIMIZER — BENCHMARK RESULTS
==========================================================================================
  File                      Orig    Opt  Removed  Reduction    DIE     CF     AS     CE      Time
------------------------------------------------------------------------------------------
  vector_add.ptx              32     18       14      43.8%      5      2      4      3    0.85ms
  matrix_mul.ptx              28     17       11      39.3%      3      1      5      2    0.72ms
  reduction.ptx               30     20       10      33.3%      2      1      3      4    0.69ms
------------------------------------------------------------------------------------------
  TOTAL                       90     55       35      38.9%     10      4     12      9    2.26ms
==========================================================================================
```

---

## Generating PTX from CUDA (Optional)

If you have CUDA installed (`nvcc` available):

```bash
# Compile each kernel to PTX
nvcc -ptx -arch=sm_89 samples/cuda_kernels/vector_add.cu \
     -o samples/ptx_files/vector_add_nvcc.ptx

nvcc -ptx -arch=sm_89 samples/cuda_kernels/matrix_mul.cu \
     -o samples/ptx_files/matrix_mul_nvcc.ptx

nvcc -ptx -arch=sm_89 samples/cuda_kernels/reduction.cu \
     -o samples/ptx_files/reduction_nvcc.ptx

# Optimize them
python benchmarks/run_benchmark.py

# Check register usage with ptxas (before/after)
ptxas --gpu-name sm_89 -v samples/ptx_files/vector_add.ptx
ptxas --gpu-name sm_89 -v samples/ptx_files/vector_add_opt.ptx
```

---

## Skills Demonstrated

| Concept | Implementation |
|---|---|
| **PTX ISA** | Parses real PTX including `.reg`, memory spaces, predicated branches |
| **Lexer/Parser design** | `lexer.py` + `parser.py` — tokenize → IR |
| **Intermediate Representation** | `Instruction` / `RawLine` dataclasses |
| **Liveness Analysis** | Backward scan in `dead_code.py` |
| **Constant Propagation** | Constant map tracking in `const_fold.py` |
| **Peephole Optimization** | 20+ algebraic rules in `algebraic.py` |
| **Copy Propagation** | Copy map with chain flattening in `copy_elim.py` |
| **Compiler pass pipeline** | Multi-round pass manager in `optimizer.py` |
| **CUDA toolchain** | Integration with `nvcc --ptx` and `ptxas -v` |

---

## References

- [NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA PTX Compiler API](https://docs.nvidia.com/cuda/ptx-compiler-api/)
- [LLVM NVPTX Backend](https://llvm.org/docs/NVPTXUsage.html)
- [Understanding PTX — NVIDIA Developer Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
