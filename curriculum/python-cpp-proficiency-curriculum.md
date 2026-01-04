---
title: "2-Week Python & C++ Proficiency for Statisticians and Data Scientists"
source: Notion
notion_url: https://www.notion.so/2-Week-Python-C-Proficiency-for-Statisticians-and-Data-Scientists-2dc342cf7cc8807b9122dc79f2781d1e
last_synced: 2026-01-04T06:13:24.106Z
last_edited_in_notion: 2026-01-03T19:03:00.000Z
---


# 2-Week Python & C++ Proficiency for Statisticians and Data Scientists


**Version**: 1.0.0-beta | **Last Updated**: January 2, 2026


This curriculum assumes fluency in probability, statistics, linear algebra, and optimization.


**The goal is working proficiency in statistical computing**: the ability to implement numerically stable algorithms in Python (NumPy/pandas ecosystem) and foundational C++ skills for performance-critical numerical code with Eigen. Graduates can read production statistical code, write tested implementations of common algorithms (MCMC, optimization, linear algebra), debug numerical issues, and understand when to use Python vs C++ for statistical computing tasks. _(Note: Full Python-C++ interoperability via pybind11 is covered in optional Week 3, Day 17.)_


**This is NOT general-purpose software engineering training.** The curriculum focuses exclusively on computational statistics and data science workflows. You will gain depth in numerical computing patterns, not breadth in web development, databases, or systems programming.


Week 1 covers Python for statistical computing. Week 2 covers foundational C++ for numerical work. Capstones require cross-language comparison and demonstrate cumulative mastery.


---


# Getting Started


## Setup (Day 0)

## Day 0: Environment Setup


**Goal**: Verify you have working Python and C++ development environments. This should take 15-20 minutes.


## Python Setup


### Installation

- **Required version**: Python 3.10 or newer
- Download from [python.org](http://python.org/) or use your system package manager
- Verify installation:

    ```bash
    python3 --version  # Should show 3.10+
    ```


### Virtual Environment


Create an isolated environment for this course:


```bash
python3 -m venv stats_env
source stats_env/bin/activate  # On Windows: stats_env\Scripts\activate

```


### Required Packages


Install core dependencies with version pins:


```bash
pip install --upgrade pip
pip install numpy>=1.20,<2.0 pandas>=1.3,<3.0 scipy>=1.7 matplotlib>=3.3 seaborn>=0.11 pytest>=7.0 hypothesis>=6.0 ipython>=7.0

```


**Why version pins?** NumPy 2.0+ and Pandas 3.0+ may introduce breaking API changes. These pins ensure compatibility with all exercises in this curriculum.


### Verification Script


Save as `verify_`[`python.py`](http://python.py/) and run:


```python
import sys
import numpy as np
import pandas as pd

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

# Test modern RNG
rng = np.random.default_rng(42)
data = rng.normal(0, 1, 100)
print(f"Generated {len(data)} random values")
print("✓ Python environment ready")

```


Expected output: All imports succeed, versions displayed, no errors.


## C++ Setup


### Compiler Installation


**Linux (Ubuntu/Debian)**:


```bash
sudo apt update
sudo apt install build-essential cmake gdb
g++ --version  # Should show 9.0+

```


**macOS**:


```bash
xcode-select --install  # Installs clang
clang++ --version  # Should show 12.0+

```


**Windows**:

- Option 1: Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/) with "Desktop development with C++"
- Option 2: Install [MSYS2](https://msys2.org/) and run: `pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake`

### Build Tools

- **CMake**: Version 3.16+ ([cmake.org](http://cmake.org/))
- Verify: `cmake --version`

### Sanitizers Support


Verify AddressSanitizer works:


```bash
g++ -fsanitize=address -o test_asan test.cpp

# On macOS with clang:
clang++ -fsanitize=address -o test_asan test.cpp

```


### Verification Program


Save as `verify_cpp.cpp`:


```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::vector<double> data(100);
    std::iota(data.begin(), data.end(), 0.0);
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    std::cout << "Sum: " << sum << std::endl;
    
    auto mean = sum / data.size();
    std::cout << "Mean: " << mean << std::endl;
    
    std::cout << "✓ C++ environment ready" << std::endl;
    return 0;
}

```


Compile and run:


```bash

# Verify C++17 support
g++ -std=c++17 -Wall -Wextra -o verify_cpp verify_cpp.cpp
./verify_cpp

# Verify sanitizers
g++ -std=c++17 -fsanitize=address -g -o verify_cpp_asan verify_cpp.cpp
./verify_cpp_asan

```


Expected output: Program compiles with no warnings, prints sum and mean, no sanitizer errors.


## Troubleshooting


**Python: "numpy not found"**

- Ensure virtual environment is activated
- Try: `pip install --force-reinstall numpy`

**C++: "g++: command not found"**

- Linux: Install build-essential package
- macOS: Run `xcode-select --install`
- Windows: Ensure compiler is in PATH

**C++: Sanitizer not available**

- Some older compilers lack sanitizer support
- Minimum versions: g++ 9.0, clang 12.0
- On Windows, sanitizers may require clang or recent MSVC

**CMake: "cmake: command not found"**

- Download from [cmake.org](http://cmake.org/) or use package manager
- macOS: `brew install cmake`
- Linux: `sudo apt install cmake`

## Quick Reference


**Python commands you'll use daily:**


```bash
source stats_env/bin/activate  # Activate environment
python script.py              # Run a script
python -m pytest tests/       # Run tests
ipython                       # Interactive shell

```


**C++ commands you'll use daily:**


```bash
g++ -std=c++17 -Wall -Wextra -O2 -o program program.cpp  # Compile optimized
g++ -std=c++17 -g -fsanitize=address -o program program.cpp  # Debug build
gdb ./program                  # Debug with gdb
make                          # Build with Makefile
cmake --build build/          # Build with CMake

```


## Ready to Start


Once both verification scripts run successfully, you're ready for Week 1. If you encounter issues not covered in troubleshooting, check:

- Python: [docs.python.org/3/using](http://docs.python.org/3/using)
- C++ Compiler: Your platform's documentation
- CMake: [cmake.org/getting-started](http://cmake.org/getting-started)

**Note on Algorithmic Thinking**: Throughout Weeks 1 and 2, you will encounter exercises that require translating mathematical concepts into performant code, reasoning about numerical stability, and choosing appropriate data structures. These skills are developed progressively across the daily exercises and capstones. See the "Algorithmic Thinking for Statistical Code" section for mental models that guide these choices.


# Foundations


## Algorithmic Thinking for Statistical Code

Statistical computing demands a particular kind of algorithmic reasoning—distinct from general software engineering and from pure mathematics. This section makes that reasoning explicit.


**Scope**: We focus exclusively on algorithms that arise in statistical and data science work: estimation, simulation, optimization, linear algebra, and streaming computation. This is not a course in classical algorithms (graphs, dynamic programming, or competitive programming patterns).


**Why this matters**: Proficiency in statistical computing requires translating mathematical procedures into numerically stable, performant code. You must reason about correctness (does this compute what I intend?), efficiency (will this finish in reasonable time?), and robustness (does this handle edge cases and numerical limits?).


## Core Mental Models


### Math → Computation Translation


Mathematical notation is declarative; code is procedural. The translation requires explicit choices:

- **Summations**: $\sum_{i=1}^{n} x_i$ becomes `np.sum(x)` (vectorized) or `sum(x)` (sequential). Choice depends on data structure and size.
- **Products**: $\prod_{i=1}^{n} x_i$ is numerically unstable. Use log-space: $\exp(\sum \log x_i)$ to prevent overflow/underflow.
- **Argmax**: $\arg\max_i f(x_i)$ becomes `np.argmax(f(x))` if vectorizable, or iterative search with early stopping if $f$ is expensive.
- **Conditionals**: $y = \begin{cases} a & x > 0 \\ b & x \leq 0 \end{cases}$ becomes `np.where(x > 0, a, b)` (vectorized) not `if x > 0: y = a` (scalar).

**Key principle**: The "obvious" translation is often wrong for performance or stability.


### Estimators as Reductions


Statistical estimators are reduction operations: data → summary statistic.

- **Mean**: Reduction via sum. Numerically stable: use Welford's online algorithm for streaming data to avoid catastrophic cancellation.
- **Variance**: Reduction via sum of squared deviations. Naive formula $\text{Var} = E[X^2] - E[X]^2$ is numerically unstable. Use two-pass or Welford's algorithm.
- **Quantiles**: Reduction via sorting (O(n log n)) or selection (O(n) with quickselect). For approximate quantiles in streaming, use t-digest or P² algorithm.
- **MLE**: Reduction via optimization of log-likelihood. Often requires gradient-based methods (Newton-Raphson, BFGS) or EM algorithm for latent variables.

**Pattern**: Understand the reduction structure, then choose algorithm based on: (1) single-pass vs multi-pass, (2) exact vs approximate, (3) memory footprint.


### Simulation as Pipelines


Monte Carlo and bootstrap algorithms are data transformation pipelines:

1. **Generate**: Draw samples from distribution (requires RNG)
2. **Transform**: Apply statistic or model to each sample
3. **Aggregate**: Collect results (mean, quantiles, histogram)
4. **Decide**: Compute intervals, p-values, convergence criteria

Each stage has algorithmic choices:

- Generate: Inverse CDF, rejection sampling, Metropolis-Hastings for complex distributions
- Transform: Vectorize when possible; parallelize across samples when not
- Aggregate: Streaming statistics to avoid storing all samples
- Decide: Early stopping rules to avoid unnecessary computation

**Performance bottleneck**: Usually stage 2 (transform). Profile to confirm before optimizing.


### Data Representation and Memory Intuition


Choosing data structures determines performance:

- **Contiguous arrays** (NumPy, C++ `std::vector`): Fast iteration, cache-friendly, enable vectorization. Use for: numerical computation, linear algebra, time series.
- **Hash tables** (Python `dict`, C++ `std::unordered_map`): O(1) lookup, irregular access. Use for: counting, grouping, sparse data.
- **Trees** (C++ `std::map`, `std::set`): O(log n) operations, sorted iteration. Use for: ordered data, range queries.
- **DataFrames** (pandas): Row-oriented abstraction over columnar storage. Convenient but slow for row-wise operations.

**Memory costs**:

- Python int: 28 bytes overhead
- NumPy int64: 8 bytes
- Python list of 1M ints: ~28 MB
- NumPy array of 1M int64: ~8 MB (3.5× smaller)

**Lesson**: Use NumPy for numerical data; Python native types for small heterogeneous data.


### Invariants and Correctness Conditions


Every algorithm has invariants—properties that must hold throughout execution:

- **Loop invariants**: At iteration $i$, `result` contains correct answer for first $i$ elements
- **Probabilistic invariants**: Posterior always sums to 1; probabilities in [0,1]
- **Numerical invariants**: Covariance matrix must be positive semi-definite; normalized vectors have unit length
- **Data invariants**: Sorted arrays remain sorted; no NaN in input to optimizer

**Use assertions** to verify invariants in development; disable in production for performance.


```python

# Example: invariants in standardization
def standardize(X):
    assert X.ndim == 2, "Input must be 2D"
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    assert np.all(stds > 0), "Cannot standardize constant columns"
    result = (X - means) / stds
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10), "Mean not zero"
    assert np.allclose(result.std(axis=0, ddof=1), 1, atol=1e-10), "Std not one"
    return result

```


### Numerical Robustness and Stability


Floating-point arithmetic is not exact. Algorithms must handle:

- **Overflow/underflow**: Use log-space for products of small numbers (likelihood computations)
- **Catastrophic cancellation**: $(a + b) - a \neq b$ when $a \gg b$ in floating point. Use compensated summation (Kahan) or reorganize computation.
- **Ill-conditioning**: Matrix inversion amplifies errors when condition number is large. Use SVD or regularization.
- **Loss of significance**: $\sqrt{1 + x} - 1$ loses precision for small $x$. Use Taylor expansion: $x/2 - x^2/8 + ...$

**Stability checks**:

- Condition number for linear systems
- Residuals for iterative methods
- Comparing with higher-precision arithmetic (mpmath)

### Performance Workflow


Never optimize without measuring:

1. **Hypothesis**: "Function X is slow because of Y"
2. **Profile**: Use `cProfile` (Python) or `perf` (C++) to measure actual bottleneck
3. **Optimize**: Change algorithm, vectorize, or rewrite in compiled language
4. **Validate**: Verify output matches original (up to tolerance)
5. **Re-profile**: Confirm speedup and identify next bottleneck

**Amdahl's Law**: If bottleneck is 20% of runtime, infinite speedup there gives only 1.25× total speedup. Focus on largest contributors.


**When to stop**: When code is "fast enough" for your use case. Premature optimization wastes time.


_(Optional: Week 3, Days 15-16)_ Advanced profiling, microbenchmarking, and performance tuning under realistic pressure.


## Statistical Algorithm Patterns


Common statistical tasks have characteristic algorithmic patterns. The table below maps tasks to patterns, typical pitfalls, and language-specific recommendations.


| **Statistical Task**                              | **Algorithmic Pattern**                                                                            | **Common Pitfalls**                                                                                          | **Python Approach**                                                                                           | **C++ Approach**                                                                                                   |

| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |

| **Monte Carlo / Bootstrap**                       | Embarrassingly parallel; generate → transform → aggregate                                          | Global RNG state breaks reproducibility; Python loop over samples is slow                                    | Vectorize inner loop with NumPy; use multiprocessing.Pool for parallelism; pass explicit RNG                  | Use std::vector preallocated; parallelize with OpenMP; spawn independent RNG streams                               |

| **Gradient-based optimization (MLE)**             | Iterative: gradient → step → check convergence; may require Hessian for Newton                     | Finite-difference gradients are slow and unstable; line search can fail on non-convex problems               | Use scipy.optimize with analytical gradients; BFGS for medium-scale; trust-region for robustness              | Eigen library for linear algebra; implement BFGS or use Ceres/NLopt; careful with memory in Hessian                |

| **Iterative algorithms (EM, coordinate descent)** | Fixed-point iteration: E-step → M-step → convergence check; maintain sufficient statistics         | Slow convergence; premature stopping; oscillation near optimum                                               | Vectorize E-step and M-step; use relative change in log-likelihood for stopping; add momentum or acceleration | Separate data structures for sufficient statistics; in-place updates; use convergence tolerance tied to data scale |

| **Linear algebra (covariance, least squares)**    | Matrix decomposition: QR for least squares; SVD for rank-deficient; Cholesky for positive definite | Explicit matrix inversion is slow and unstable; dense algebra on sparse matrices wastes memory               | Never invert; use np.linalg.lstsq or solve; check condition number; use scipy.sparse for sparse matrices      | Eigen library; use decompositions (A.ldlt().solve(b)); avoid temporaries; consider BLAS/LAPACK for large problems  |

| **Streaming updates (running mean, variance)**    | Online algorithm: update statistic incrementally with O(1) memory                                  | Naive formula E[X²] - E[X]² loses precision; overflow in sum for large n                                     | Welford's algorithm for mean/variance; exponential moving average for weighted; use numba for tight loop      | Separate sum and count; use Kahan summation for precision; store sufficient statistics not raw data                |

| **Sampling / resampling pipelines**               | Transform samples through chain: generate → filter → weight → resample                             | Systematic resampling introduces correlation; poor RNG breaks independence; naive rejection sampling is slow | Use np.random.Generator; vectorize weights; stratified/systematic resampling for variance reduction           | Use std::discrete_distribution for weighted sampling; preallocate buffers; parallel RNG streams per thread         |


**Pattern recognition**: When you see a statistical task, ask:

1. Can I vectorize this? (Python: NumPy; C++: Eigen, expression templates)
2. Is this memory-bound or compute-bound? (Profile to determine)
3. What numerical issues might arise? (Overflow, cancellation, conditioning)
4. Can I stream this or must I store all data? (Determines memory footprint)

_(Optional: Week 3, Days 15-17)_ Implement several of these patterns with cross-language comparison and numerical validation.


## Tradeoff Playbook


Every implementation choice involves tradeoffs. Here are rules of thumb and counterexamples:


### Vectorize vs Loop


**Rule**: Vectorize whenever possible—10-100× faster in Python.


**When to break**:

- Early stopping (can't vectorize conditionals that depend on iteration)
- State-dependent loops (Markov chains, recursive filters)
- Memory-constrained (vectorization creates large temporaries)

**Example**:


```python

# Vectorized: fast but memory-intensive
result = X @ W + b  # Allocates (n, m) temporary

# Loop: slower but O(1) memory if output is streaming
for i in range(n):
    yield X[i] @ W + b

```


### Pandas vs NumPy Arrays


**Rule**: Use pandas for heterogeneous, labeled data with missing values. Use NumPy for numerical computation.


**Why**: Pandas has overhead for indexing, type checking, and handling NaN. NumPy is thin wrapper over C arrays.


**Counterexample**: For groupby-aggregation on categorical data, pandas is faster than manual NumPy loops.


**Pattern**: Use pandas for I/O and transformations; extract NumPy arrays for numerical work; convert back to pandas for output.


### Python vs Numba vs C++


**Rule**: Start with NumPy. If bottleneck is unavoidable Python loop, try Numba. If Numba insufficient, C++.


**Speedup expectations**:

- NumPy vs Python loop: 10-100×
- Numba (JIT) vs Python loop: 10-50×
- C++ vs Python loop: 20-100×
- C++ vs NumPy (already vectorized): 1-3× (not worth it unless critical path)

**Numba sweet spot**: Tight numerical loops with simple control flow. No classes, limited Python features.


**C++ when**:

- Complex data structures (trees, graphs)
- Fine-grained memory control (RAII, move semantics)
- Integration with existing C++ libraries
- Numba fails or performance still inadequate

### Memory vs Speed


**Rule**: Optimize for speed first; then reduce memory if needed.


**Why**: Memory is cheap; developer time is expensive. Exception: Big data that doesn't fit in RAM.


**Techniques**:

- **Chunking**: Process data in blocks (pandas `read_csv` with `chunksize`)
- **Memory mapping**: Access disk as if RAM (NumPy `memmap`)
- **Compression**: Store compressed, decompress on access (HDF5, Parquet)
- **Out-of-core**: Dask, Vaex for larger-than-memory DataFrames

**Counterexample**: In-place operations save memory but lose pipeline clarity. Only use when profiling shows memory is bottleneck.


### Precompute vs Compute-on-the-Fly


**Rule**: Precompute if used repeatedly; compute on-the-fly if used once or space-constrained.


**Example**: Distance matrix

- **Precompute**: Store (n, n) matrix. Fast access O(1), but O(n²) memory.
- **On-the-fly**: Compute distance when needed. O(d) memory, but slower.

**Heuristic**: If data is reused k times and precomputation cost is C, it's worth it if k × (cost per query) > C + storage cost.


_(Optional: Week 3, Day 16)_ Benchmarks these tradeoffs systematically with profiling and memory instrumentation.


## Micro-Drills


These short drills test your ability to choose correct algorithmic approaches. Focus on **correctness invariants** and **avoiding common pitfalls**.


**Note on numbering**: These exercises are labeled "Foundational 1-3", "Proficiency 1-3", and "Mastery 1-4" within this Algorithmic Thinking section. Daily exercises (Days 1-14) use separate numbering within each day.


**Foundational 1**: Compute log-sum-exp: $\log(\sum_i \exp(x_i))$ stably for $x \in \mathbb{R}^n$ with potentially large $|x_i|$.


**Expected approach**: Subtract max before exponentiation: $m = max(x)$; return $m + log(sum exp(x - m))$.


**Correctness invariant**: Result equals naive formula when no overflow, but remains finite for large $x_i$.


---


**Foundational 2**: Maintain running mean and variance for a stream of numbers without storing all data.


**Expected approach**: Welford's online algorithm. Track count $n$, mean $mu$, and $M_2 = sum (x - mu)^2$. Update incrementally.


**Correctness invariant**: Variance matches batch formula $\text{Var} = M_2 / (n-1)$ at every step.


---


**Foundational 3**: Generate $n$ samples from a discrete distribution with probabilities $p_1, ldots, p_k$.


**Expected approach**: Cumulative sum of probabilities; binary search for each sample. Or use `np.random.choice` with probabilities.


**Correctness invariant**: Frequency of outcome $i$ in large sample converges to $p_i$.


---


**Proficiency 1**: Compute pairwise Euclidean distances between two sets of points $X in mathbb{R}^{m times d}$, $Y \in \mathbb{R}^{n \times d}$ without explicit loops.


**Expected approach**: Use broadcasting or matrix formula: $D^2 = |X|^2 mathbf{1}^T + mathbf{1} |Y|^2 - 2XY^T$.


**Correctness invariant**: $D_{ij}^2 = sum_k (X_{ik} - Y_{jk})^2$. Check with small example.


---


**Proficiency 2**: Implement reservoir sampling: maintain uniform random sample of $k$ items from stream of unknown length $n$.


**Expected approach**: Keep first $k$ items; for item $i > k$, include with probability $k/i$, replacing random existing item.


**Correctness invariant**: Each of first $n$ items has probability $k/n$ of being in final sample.


---


**Proficiency 3**: Compute $A^{-1}b$ for symmetric positive definite matrix $A$ without explicit inversion.


**Expected approach**: Cholesky decomposition $A = LL^T$, then solve $Ly = b$ and $L^T x = y$ via forward/back substitution.


**Correctness invariant**: $\|Ax - b\| < \epsilon$ for small $epsilon$. Check residual, not by comparing with `np.linalg.inv`.


---


**Mastery 1**: Implement Kahan summation to reduce rounding error when summing many floating-point numbers.


**Expected approach**: Maintain compensation term $c$ that accumulates lost low-order bits. Update: $y = x_i - c$; $t = s + y$; $c = (t - s) - y$; $s = t$.


**Correctness invariant**: Error is O(ε) instead of O(nε) for naive summation.


---


**Mastery 2**: Implement stable computation of softmax: $sigma(x)_i = exp(x_i) / sum_j exp(x_j)$.


**Expected approach**: Subtract max before exponentiation. Combine with log-sum-exp.


**Correctness invariant**: $\sum_i \sigma(x)_i = 1$ exactly (up to machine precision). No overflow for large $x_i$.

<details>
<summary>Solution Sketch</summary>

```python
import numpy as np

def softmax_stable(x):
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# Verification
x = np.array([1000, 1001, 1002])  # Would overflow without stability trick
probs = softmax_stable(x)
assert np.isclose(np.sum(probs), 1.0)  # Sums to 1
assert np.all(np.isfinite(probs))  # No inf/nan

```


</details>


---


**Mastery 3**: Implement iterative refinement for solving $Ax = b$ to recover accuracy lost in ill-conditioned system.


**Expected approach**: Solve $Ax_0 = b$ with standard method. Compute residual $r = b - Ax_0$ in higher precision. Solve $A delta = r$. Update $x = x_0 + delta$. Repeat.


**Correctness invariant**: Residual $\|b - Ax\|$ decreases geometrically until machine precision limit.

<details>
<summary>Solution Sketch</summary>

```python
import numpy as np

def iterative_refinement(A, b, max_iter=5, tol=1e-12):
    """Iterative refinement for Ax = b."""
    # Initial solution
    x = np.linalg.solve(A, b)
    
    for i in range(max_iter):
        # Compute residual (use higher precision if available)
        r = b - A @ x
        residual_norm = np.linalg.norm(r)
        
        if residual_norm < tol:
            break
        
        # Solve for correction
        delta = np.linalg.solve(A, r)
        
        # Update solution
        x = x + delta
    
    return x, residual_norm

# Example with ill-conditioned matrix
A = np.array([[1e10, 1], [1, 1]])
b = np.array([1e10, 2])
x_refined, res = iterative_refinement(A, b)
print(f"Residual: {res:.2e}")  # Should be very small

```


</details>


---


**Mastery 4**: Implement alias method for O(1) sampling from discrete distribution after O(k) preprocessing.


**Expected approach**: Construct alias table: partition probabilities into $k$ bins, each with at most two outcomes. Sample by: choose bin uniformly, then choose one of two outcomes in that bin.


**Correctness invariant**: After preprocessing, each sample takes O(1) time. Frequency matches target distribution.

<details>
<summary>Solution Sketch</summary>

```python
import numpy as np

class AliasMethod:
    """O(1) sampling from discrete distribution."""
    def __init__(self, probs):
        n = len(probs)
        self.prob = np.zeros(n)
        self.alias = np.zeros(n, dtype=int)
        
        # Normalize probabilities
        probs = np.array(probs) * n
        
        # Partition into small and large
        small = [i for i, p in enumerate(probs) if p < 1]
        large = [i for i, p in enumerate(probs) if p >= 1]
        
        # Build alias table
        while small and large:
            s, l = small.pop(), large.pop()
            self.prob[s] = probs[s]
            self.alias[s] = l
            probs[l] = probs[l] - (1 - probs[s])
            if probs[l] < 1:
                small.append(l)
            else:
                large.append(l)
        
        # Remaining probabilities are 1
        for i in small + large:
            self.prob[i] = 1
    
    def sample(self, rng):
        """O(1) sample."""
        i = rng.integers(0, len(self.prob))
        return i if rng.random() < self.prob[i] else self.alias[i]

# Example usage
probs = [0.1, 0.2, 0.3, 0.4]
sampler = AliasMethod(probs)
rng = np.random.default_rng(42)
samples = [sampler.sample(rng) for _ in range(10000)]

# Verify frequencies match target distribution

```


</details>


_(Optional: Week 3, Days 15-18)_ Revisit these patterns under time pressure, with explicit profiling and numerical validation requirements.


## Oral Defense Add-On


These questions test your ability to reason about algorithmic choices in statistical code. Practice articulating your thought process.


### Python-Centric


**Question 1**: Why does `df.apply(func, axis=1)` tend to be slow? What alternatives exist?


**Strong answer must include**: Row-wise apply invokes Python function per row, losing vectorization. Alternatives: (1) Vectorize with NumPy on `df.values`, (2) Use pandas vectorized methods (`df['col1'] + df['col2']`), (3) Use `df.apply` with axis=0 if possible, (4) Extract to NumPy, compute, assign back.


**Weak pattern to avoid**: "Because Python is slow" (not specific enough—doesn't explain why or what to do).


---


**Question 2**: When should you use `np.random.Generator` instead of `np.random.seed()`?


**Strong answer**: Always in new code. `Generator` provides explicit RNG instances (no global state), enabling reproducibility in parallel code and test isolation. `seed()` modifies global state, causing test interference and non-reproducible parallel execution.


**Weak pattern**: "Because it's newer" (doesn't explain the problem with global state).


---


**Question 3**: You have a bottleneck in a tight loop over 1M items. NumPy doesn't help. What are your options?


**Strong answer**: (1) Numba JIT with `@njit`, expecting 10-50× speedup for numerical loops. (2) Cython for mixed Python/C. (3) Rewrite critical section in C++ with pybind11. (4) Check if loop can be reformulated as vectorized operation. Order: profile to confirm bottleneck, try Numba (easiest), then C++ if insufficient.


**Weak pattern**: "Use multiprocessing" (doesn't help for inherently sequential loop; misdiagnoses problem).


---


**Question 4**: What's wrong with `def f(x, cache={}): ...` and how do you fix it?


**Strong answer**: Mutable default argument is evaluated once at function definition, shared across all calls. Cache persists across invocations. Fix: Use `cache=None` and `if cache is None: cache = {}` inside function. Or use `@functools.lru_cache` for memoization.


**Weak pattern**: "It's not thread-safe" (true, but not the primary issue in single-threaded code; misses the shared state problem).


---


**Question 5**: You computed a bootstrap CI as `[np.percentile(samples, 2.5), np.percentile(samples, 97.5)]`. Your colleague says this is biased. Why?


**Strong answer**: Percentile bootstrap (quantile method) has poor coverage for small samples or skewed distributions. Better: (1) BCa (bias-corrected and accelerated) bootstrap, (2) Studentized bootstrap, (3) Use more samples (10,000+). However, percentile method is still acceptable for large samples and symmetric distributions.


**Weak pattern**: "Should use 95% not 97.5%" (confuses percentiles with confidence level; percentile bootstrap uses 2.5/97.5 for 95% CI).


---


**Question 6**: Why might `pd.merge` be slower than a manual dictionary-based join?


**Strong answer**: Pandas merge has overhead for: index alignment, handling multiple join keys, type checking, preserving dtypes, and choosing join algorithm. For simple single-key joins on small data, a `dict` lookup can be faster. However, pandas handles edge cases (duplicate keys, missing values, multiple columns) that manual code often misses.


**Weak pattern**: "Because pandas is slow in general" (too vague; doesn't explain when/why or trade-offs).


### C++ / Performance-Centric


**Question 7**: You pass a `std::vector<double>` to a function by value. What's the performance implication?


**Strong answer**: Copies the entire vector (O(n) time and memory). Should pass by `const std::vector<double>&` for read-only access (zero cost) or `std::vector<double>&` for modification. Pass by value only when you need a local copy or are using move semantics.


**Weak pattern**: "It's slower" (doesn't quantify or explain when it matters).


---


**Question 8**: When should you use `std::unique_ptr` vs `std::shared_ptr`?


**Strong answer**: Use `unique_ptr` for exclusive ownership (single owner, no shared references). Use `shared_ptr` only when multiple owners with unclear lifetimes. `unique_ptr` has zero overhead; `shared_ptr` has reference counting cost (atomic operations). Default to `unique_ptr`; switch to `shared_ptr` only when needed.


**Weak pattern**: "Always use `shared_ptr` to be safe" (adds unnecessary overhead and obscures ownership).


---


**Question 9**: Your C++ code computes a covariance matrix. Should you use `Eigen::MatrixXd` or `std::vector<std::vector<double>>`?


**Strong answer**: `Eigen::MatrixXd`. Provides: (1) Contiguous storage (cache-friendly), (2) Vectorized operations (SIMD), (3) Expression templates (avoid temporaries), (4) Integration with BLAS/LAPACK. `std::vector<std::vector<double>>` has indirection overhead and no vectorization. Only use nested vectors for ragged arrays.


**Weak pattern**: "Eigen is faster" (true, but doesn't explain why or when the difference matters).


---


**Question 10**: What does `-O2` optimization do and when might it cause problems?


**Strong answer**: Enables mid-level optimizations: loop unrolling, function inlining, dead code elimination, etc. Can cause problems: (1) Harder to debug (variables optimized away, control flow reordered), (2) Undefined behavior becomes visible (code relying on UB may "work" at `-O0`), (3) Longer compile times. Use `-O0 -g` for development, `-O2` or `-O3` for production.


**Weak pattern**: "Makes code faster" (doesn't address when or what trade-offs exist).


### Numerical Stability-Centric


**Question 11**: You implement variance as `np.mean(x**2) - np.mean(x)**2`. When does this fail?


**Strong answer**: Fails when $\text{Var}(X) \ll E[X]^2$ due to catastrophic cancellation. Example: $x = [10^6, 10^6 + 1, 10^6 + 2]$ gives negative variance in floating point. Fix: Use two-pass formula $\frac{1}{n}\sum (x_i - \bar{x})^2$ or Welford's online algorithm. NumPy's `np.var` uses stable formula internally.


**Weak pattern**: "Floating point error" (too vague; doesn't explain which operation or how to fix).


---


**Question 12**: You compute $\exp(x)$ for $x$ from a neural network logit. When might this overflow, and how do you fix it?


**Strong answer**: Overflows when $x > 709$ (for float64). Common in softmax. Fix: Subtract max before exponentiation: $exp(x - max(x))$. This shifts dynamic range without changing relative probabilities. For log-softmax, use log-sum-exp trick. Libraries (PyTorch, NumPy) implement this internally.


**Weak pattern**: "Use log space" (correct direction, but doesn't explain the specific trick or when it's needed).


---


_(Optional: Week 3, Day 18)_ Includes mock oral defense sessions where you must answer similar questions under time pressure, with follow-up probing.


# Core Curriculum (Required for Proficiency)


## Week 1: Python (Days 1-7)

## Overview


Week 1 focuses on **Python proficiency** for statisticians and data scientists transitioning from R, MATLAB, or similar environments. By the end of this week, you will understand Python's unique behaviors, write production-grade code, and complete a capstone project demonstrating mastery.


---


## Week Structure


This week is organized into 7 daily modules, each covering a critical Python concept:


## Day 1: Functions, Modules, and Idiomatic Python

## Overview


**Focus**: Transition from quick-and-dirty scripting to production-grade Python. Topics include function best practices, module organization, and idiomatic patterns that prevent common bugs.


**Why it matters**: R and MATLAB users often write "notebook-style" code. Learning to write modular, reusable functions is essential for collaboration and debugging.


---


## Learning Objectives


By the end of Day 1, you will:

- Write functions with explicit type hints and docstrings
- Organize code into importable modules
- Use context managers (`with` statements) for resource management
- Apply list/dict comprehensions and generator expressions idiomatically
- Understand when to use `*args`, `**kwargs`, and default arguments

---


## Core Concepts


### 1. Function Design


**Type Hints and Docstrings**


```python
from typing import List, Optional

def compute_loss(predictions: List[float], targets: List[float], 
                 regularization: Optional[float] = None) -> float:
    """
    Compute mean squared error with optional L2 regularization.
    
    Args:
        predictions: Model outputs (length n)
        targets: Ground truth values (length n)
        regularization: L2 penalty coefficient (default: None)
    
    Returns:
        Scalar loss value
    
    Raises:
        ValueError: If predictions and targets have different lengths
    """
    if len(predictions) != len(targets):
        raise ValueError("Length mismatch")
    
    mse = sum((p - t)**2 for p, t in zip(predictions, targets)) / len(predictions)
    
    if regularization:
        penalty = regularization * sum(p**2 for p in predictions)
        return mse + penalty
    return mse

```


**Default Arguments (Mutable Trap)**


```python

# ❌ WRONG: Default list is shared across calls
def append_result(value, results=[]):
    results.append(value)
    return results

# First call
append_result(1)  # [1] ✓

# Second call reuses the SAME list!
append_result(2)  # [1, 2] ✗

# ✅ CORRECT: Use None as sentinel
def append_result(value, results=None):
    if results is None:
        results = []
    results.append(value)
    return results

```


### 2. Module Organization


**File Structure**


```javascript
my_stats_lib/
├── __init__.py          # Package initialization
├── preprocessing.py     # Data cleaning functions
├── models.py            # Statistical models
├── utils.py             # Helper functions
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_models.py

```


**`__init__.py`** **Example**


```python

# my_stats_lib/__init__.py
from .preprocessing import standardize, handle_missing
from .models import linear_regression

__all__ = ['standardize', 'handle_missing', 'linear_regression']
__version__ = '0.1.0'

```


### 3. Context Managers


**Why Use** **`with`****?**


```python

# ❌ Manual resource management (risky)
f = open('data.csv')
data = f.read()
f.close()  # Easy to forget!

# What if an exception occurs before close()?

# ✅ Automatic resource management (safe)
with open('data.csv') as f:
    data = f.read()

# File automatically closed, even if exception occurs

```


```javascript

```


**Custom Context Manager**


```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

# Usage
with timer("Model training"):
    # ... training code ...
    pass

```


### 4. Comprehensions and Generators


**List Comprehensions**


```python

# Convert R-style loop to Pythonic comprehension

# R: sapply(data, function(x) x^2)

# ❌ Verbose
squares = []
for x in data:
    squares.append(x**2)

# ✅ Pythonic
squares = [x**2 for x in data]

# ✅ With filtering
positive_squares = [x**2 for x in data if x > 0]

```


**Generator Expressions (Memory Efficient)**


```python

# Process large dataset without loading everything into memory

# ( ) creates generator, [ ] creates list

# List comprehension: loads all into memory
sum([x**2 for x in range(10_000_000)])  # ~400 MB

# Generator: processes one at a time
sum(x**2 for x in range(10_000_000))    # ~100 bytes

```


---


## Hands-On Exercises


### Exercise 1.1: Refactor Function


Convert this "notebook-style" function to production-grade code:


```python
def process(data, threshold):
    result = []
    for i in range(len(data)):
        if data[i] > threshold:
            result.append(data[i] * 2)
    return result

```


**Your Task**: Add type hints, docstring, use comprehension, handle edge cases.

<details>
<summary>Solution</summary>

```python
from typing import List

def amplify_above_threshold(data: List[float], threshold: float) -> List[float]:
    """
    Double all values exceeding a threshold.
    
    Args:
        data: Input numeric values
        threshold: Cutoff value (inclusive)
    
    Returns:
        List of doubled values where data[i] > threshold
    
    Example:
        >>> amplify_above_threshold([1, 5, 3], threshold=2)
        [10, 6]
    """
    if not data:
        return []
    return [x * 2 for x in data if x > threshold]

```


</details>


### Exercise 1.2: Module Design


Create a `stats_`[`utils.py`](http://utils.py/) module with these functions:

1. `mean(values: List[float]) -> float`
2. `std(values: List[float], ddof: int = 1) -> float`
3. `z_score(values: List[float]) -> List[float]`

Include proper docstrings and a `__main__` block for testing.

<details>
<summary>Solution</summary>

```python

# stats_utils.py
"""Statistical utility functions."""
from typing import List
import math

def mean(values: List[float]) -> float:
    """Compute arithmetic mean.
    
    Args:
        values: List of numeric values
    
    Returns:
        Arithmetic mean
    
    Raises:
        ValueError: If values is empty
    """
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)

def std(values: List[float], ddof: int = 1) -> float:
    """Compute standard deviation.
    
    Args:
        values: List of numeric values
        ddof: Delta degrees of freedom (default: 1 for sample std)
    
    Returns:
        Standard deviation
    
    Raises:
        ValueError: If values has fewer than ddof+1 elements
    """
    n = len(values)
    if n <= ddof:
        raise ValueError(f"Need at least {ddof + 1} values")
    mu = mean(values)
    variance = sum((x - mu)**2 for x in values) / (n - ddof)
    return math.sqrt(variance)

def z_score(values: List[float]) -> List[float]:
    """Standardize values to z-scores (mean=0, std=1).
    
    Args:
        values: List of numeric values
    
    Returns:
        List of z-scores
    """
    mu = mean(values)
    sigma = std(values)
    return [(x - mu) / sigma for x in values]

if __name__ == "__main__":
    # Quick test
    test_data = [1, 2, 3, 4, 5]
    print(f"Mean: {mean(test_data)}")
    print(f"Std:  {std(test_data)}")
    print(f"Z:    {z_score(test_data)}")

```


</details>


**Scoring Rubric - Exercise 1.1** (10 points)


| Criterion                                   | Points |

| ------------------------------------------- | ------ |

| Type hints on all parameters and return     | 2      |

| Complete docstring (Args, Returns, Example) | 3      |

| List comprehension instead of loop          | 2      |

| Edge case handling (empty list)             | 2      |

| Descriptive function name                   | 1      |


**Scoring Rubric - Exercise 1.2** (15 points)


| Criterion                                      | Points |

| ---------------------------------------------- | ------ |

| All three functions implemented correctly      | 6      |

| Type hints on all functions                    | 3      |

| Docstrings with Args/Returns/Raises            | 3      |

| Error handling (empty list, insufficient data) | 2      |

| `__main__` block with test cases               | 1      |


## Common Pitfalls


| Pitfall                     | Why It's Bad                         | Fix                                     |

| --------------------------- | ------------------------------------ | --------------------------------------- |

| `def func(x=[])`            | Mutable default shared across calls  | Use `x=None`, then `x = x or []`        |

| No type hints               | Hard to catch bugs, poor IDE support | Add `-> ReturnType` annotations         |

| `for i in range(len(data))` | Unpythonic, error-prone              | Use `for item in data` or `enumerate()` |

| Deeply nested `if`/`for`    | Hard to read and debug               | Extract to helper functions             |


---


## Self-Assessment Checklist


Before moving to Day 2, verify you can:

- [ ] Write a function with type hints, docstring, and proper error handling
- [ ] Explain why `def f(x=[])` is dangerous
- [ ] Use `with` statements for file I/O
- [ ] Convert a `for` loop to a list comprehension
- [ ] Organize code into an importable module with `__init__.py`

---


## Resources

- [PEP 8 – Style Guide](https://peps.python.org/pep-0008/)
- [Real Python: Type Hints](https://realpython.com/python-type-checking/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Day 2: Memory Model, Views, and Copies

## Overview


**Focus**: Understand how Python and NumPy manage memory. Topics include references vs. copies, views vs. base arrays, and when mutations propagate unexpectedly.


**Why it matters**: R and MATLAB often copy-on-write by default. Python's reference semantics cause subtle bugs if you assume copies happen automatically.


---


## Learning Objectives


By the end of Day 2, you will:

- Distinguish between shallow copies, deep copies, and references
- Predict when NumPy operations return views vs. copies
- Debug unintended mutations in nested data structures
- Use `is` vs. `==` correctly
- Understand object mutability and its implications

---


## Core Concepts


### 1. References vs. Copies


**The Fundamental Difference**


```python

# Lists are mutable objects
a = [1, 2, 3]
b = a          # b is a REFERENCE to the same list
b.append(4)
print(a)       # [1, 2, 3, 4] ← a changed too!

# To get a copy:
c = a.copy()   # or a[:] or list(a)
c.append(5)
print(a)       # [1, 2, 3, 4] ← a unchanged

```


**Identity vs. Equality**


```python
a = [1, 2]
b = a
c = [1, 2]

a == b  # True (same contents)
a is b  # True (same object in memory)

a == c  # True (same contents)
a is c  # False (different objects)

# Use `is` for None, True, False; use `==` for value comparison

```


### 2. Shallow vs. Deep Copy


**Nested Structures**


```python
import copy

# Shallow copy: copies top level only
matrix = [[1, 2], [3, 4]]
shallow = matrix.copy()
shallow[0][0] = 99

print(matrix)  # [[99, 2], [3, 4]] ← nested list still shared!

# Deep copy: recursively copies everything
matrix = [[1, 2], [3, 4]]
deep = copy.deepcopy(matrix)
deep[0][0] = 99

print(matrix)  # [[1, 2], [3, 4]] ← unchanged

```


**When to Use Which**


| Scenario                 | Use                |

| ------------------------ | ------------------ |

| Flat list/dict           | `.copy()` or `[:]` |

| Nested structures        | `copy.deepcopy()`  |

| NumPy arrays (see below) | `.copy()`          |


### 3. NumPy Views vs. Copies


**The View Mechanism**


```python
import numpy as np

# Most slicing operations return VIEWS (not copies)
x = np.array([1, 2, 3, 4, 5])
y = x[1:4]      # y is a view into x

y[0] = 99
print(x)        # [1, 99, 3, 4, 5] ← x changed!

# Force a copy:
z = x[1:4].copy()
z[0] = 77
print(x)        # [1, 99, 3, 4, 5] ← x unchanged

```


**Detecting Views**


```python
x = np.array([1, 2, 3, 4])
y = x[::2]      # every other element

# Check if y is a view of x:
y.base is x     # True ← y is a view

# Check if array owns its data:
y.flags['OWNDATA']  # False ← doesn't own data
x.flags['OWNDATA']  # True  ← owns data

```


**Operations That Return Views vs. Copies**


| Operation                         | Returns            |

| --------------------------------- | ------------------ |

| `arr[start:stop]`                 | View               |

| `arr[[0, 2, 4]]` (fancy indexing) | Copy               |

| `arr[arr > 0]` (boolean mask)     | Copy               |

| `arr.reshape(...)`                | View (if possible) |

| `arr.T` (transpose)               | View               |

| `arr.flatten()`                   | Copy               |

| `arr.ravel()`                     | View (if possible) |


### 4. Common Pitfalls


**Pitfall 1: Loop Variable Aliasing**


```python

# ❌ All functions reference the SAME variable
functions = []
for i in range(3):
    functions.append(lambda: i**2)

[f() for f in functions]  # [4, 4, 4] ← all return 2^2!

# ✅ Capture i's value explicitly
functions = []
for i in range(3):
    functions.append(lambda x=i: x**2)  # default arg captures value

[f() for f in functions]  # [0, 1, 4] ✓

```


**Pitfall 2: DataFrame Column Assignment**


```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3]})

# ❌ May create a view or copy (depends on pandas version)
col = df['A']
col[0] = 99  # Might trigger SettingWithCopyWarning

# ✅ Explicit copy
col = df['A'].copy()
col[0] = 99  # Safe

# ✅ Or modify in place
df.loc[0, 'A'] = 99

```


---


## Hands-On Exercises


### Exercise 2.1: Debug the Mutation


Find and fix the bug:


```python
def reset_matrix(mat):
    """Set all elements to zero."""
    for row in mat:
        row = [0] * len(row)
    return mat

data = [[1, 2], [3, 4]]
reset_matrix(data)
print(data)  # [[1, 2], [3, 4]] ← Why didn't it reset?

```

<details>
<summary>Solution</summary>

**Problem**: Reassigning `row` creates a new local variable; it doesn't modify the original list.


```python

# ✅ Fix 1: Modify in place
def reset_matrix(mat):
    for row in mat:
        for i in range(len(row)):
            row[i] = 0
    return mat

# ✅ Fix 2: Rebuild matrix
def reset_matrix(mat):
    return [[0] * len(row) for row in mat]

# ✅ Fix 3: NumPy (best for numeric data)
import numpy as np
def reset_matrix(mat):
    arr = np.array(mat)
    arr[:] = 0  # In-place assignment
    return arr.tolist()

```


</details>


### Exercise 2.2: NumPy View Detective


Predict whether each operation creates a view or copy:


```python
import numpy as np
x = np.arange(12).reshape(3, 4)

a = x[0, :]          # ?
b = x[[0, 2], :]     # ?
c = x[x > 5]         # ?
d = x.T              # ?
e = x.reshape(4, 3)  # ?

```

<details>
<summary>Solution</summary>

| Variable | View or Copy? | Reason                           |

| -------- | ------------- | -------------------------------- |

| `a`      | View          | Basic slicing                    |

| `b`      | Copy          | Fancy indexing (list of indices) |

| `c`      | Copy          | Boolean mask                     |

| `d`      | View          | Transpose is metadata-only       |

| `e`      | View          | Reshape without data copy        |


**Verification**:


```python
print(a.base is x)  # True
print(b.base is x)  # False
print(c.base is x)  # False
print(d.base is x)  # True
print(e.base is x)  # True

```


</details>


**Scoring Rubric - Exercise 2.1** (10 points)


| Criterion                                            | Points |

| ---------------------------------------------------- | ------ |

| Correctly identifies the bug (rebinding vs mutation) | 3      |

| Provides working fix that modifies in-place          | 4      |

| Explains why original code fails                     | 2      |

| Code is clean and follows Python conventions         | 1      |


**Scoring Rubric - Exercise 2.2** (10 points)


| Criterion                                  | Points |

| ------------------------------------------ | ------ |

| Correctly identifies all 5 as view or copy | 5      |

| Provides correct reasoning for each        | 3      |

| Includes verification code using `.base`   | 2      |


## Common Pitfalls


| Pitfall                                     | Why It's Bad                          | Fix                            |

| ------------------------------------------- | ------------------------------------- | ------------------------------ |

| `b = a` without `.copy()`                   | Unintended mutations propagate        | Use `.copy()` explicitly       |

| Modifying view assuming it's a copy         | Changes affect original array         | Check `.base` or use `.copy()` |

| Using `==` instead of `is` for `None`       | Less efficient, wrong semantics       | Use `if x is None:`            |

| Forgetting nested structures need deep copy | Shallow copy still shares nested refs | Use `copy.deepcopy()`          |


---


## Self-Assessment Checklist


Before moving to Day 3, verify you can:

- [ ] Explain the difference between `a = b` and `a = b.copy()`
- [ ] Predict when `arr[slice]` returns a view vs. copy
- [ ] Use `.base` to check if an array is a view
- [ ] Fix a bug caused by unintended list aliasing
- [ ] Use `copy.deepcopy()` for nested structures

---


## Resources

- [Python Memory Management](https://realpython.com/python-memory-management/)
- [NumPy Views and Copies](https://numpy.org/doc/stable/user/basics.copies.html)
- [Ned Batchelder: Facts and Myths about Python Names and Values](https://nedbatchelder.com/text/names.html)

## Day 3: Broadcasting and Vectorization

## Overview


**Focus**: Learn how NumPy's broadcasting rules enable vectorized operations without explicit loops. Topics include shape compatibility, performance optimization, and common pitfalls.


**Why it matters**: R and MATLAB vectorize automatically in many cases. Understanding Python's broadcasting rules prevents shape-mismatch errors and unlocks massive speedups.


---


## Learning Objectives


By the end of Day 3, you will:

- Apply broadcasting rules to combine arrays of different shapes
- Vectorize computations to replace Python loops
- Use `np.newaxis` and `reshape` to control broadcasting
- Avoid common broadcasting pitfalls (e.g., unintended dimension expansion)
- Benchmark vectorized vs. loop-based code

---


## Core Concepts


### 1. Broadcasting Rules


**The Three Rules**


NumPy compares array shapes element-wise from right to left. Two dimensions are compatible when:

1. They are equal, OR
2. One of them is 1

```python
import numpy as np

# Rule 1: Same shape
a = np.array([1, 2, 3])      # shape (3,)
b = np.array([10, 20, 30])   # shape (3,)
a + b  # [11, 22, 33] ✓

# Rule 2: Trailing dimensions match
a = np.array([[1], [2], [3]])  # shape (3, 1)
b = np.array([10, 20, 30])     # shape (3,)
a + b  # Broadcasting expands to (3, 3)

# [[11, 21, 31],

#  [12, 22, 32],

#  [13, 23, 33]]

# Rule 3: Dimension of 1 stretches
a = np.ones((3, 1))   # shape (3, 1)
b = np.ones((1, 4))   # shape (1, 4)
(a + b).shape         # (3, 4) ✓

```


**Visual Example: Mean Centering**


```python

# Center each column of a matrix (subtract column means)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # shape (3, 3)

col_means = X.mean(axis=0)  # shape (3,) → [4, 5, 6]

# Broadcasting: (3, 3) - (3,) → (3, 3)
X_centered = X - col_means

# [[-3, -3, -3],

#  [ 0,  0,  0],

#  [ 3,  3,  3]]

```


### 2. Controlling Broadcasting with `newaxis`


**Adding Dimensions**


```python
a = np.array([1, 2, 3])  # shape (3,)

# Add dimension at end
a[:, np.newaxis]  # shape (3, 1)

# [[1],

#  [2],

#  [3]]

# Add dimension at start
a[np.newaxis, :]  # shape (1, 3)

# [[1, 2, 3]]

# Equivalent to reshape
a.reshape(-1, 1)  # shape (3, 1)
a.reshape(1, -1)  # shape (1, 3)

```


**Use Case: Distance Matrix**


```python

# Compute pairwise distances between points
points = np.array([[0, 0], [1, 1], [2, 2]])  # shape (3, 2)

# Expand to (3, 1, 2) and (1, 3, 2)
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]

# Compute Euclidean distance
distances = np.sqrt((diff ** 2).sum(axis=2))

# [[0.        , 1.41421356, 2.82842712],

#  [1.41421356, 0.        , 1.41421356],

#  [2.82842712, 1.41421356, 0.        ]]

```


### 3. Vectorization


**Loop-to-Vector Conversion**


```python

# ❌ Python loop (slow)
def standardize_loop(X):
    result = np.empty_like(X)
    for i in range(X.shape[1]):
        col = X[:, i]
        result[:, i] = (col - col.mean()) / col.std()
    return result

# ✅ Vectorized (fast)
def standardize_vectorized(X):
    means = X.mean(axis=0)  # shape (n_features,)
    stds = X.std(axis=0)    # shape (n_features,)
    return (X - means) / stds  # Broadcasting handles everything

```


**Performance Comparison**


```python
import time

X = np.random.randn(1000, 100)

# Loop version
start = time.perf_counter()
standardize_loop(X)
loop_time = time.perf_counter() - start

# Vectorized version
start = time.perf_counter()
standardize_vectorized(X)
vec_time = time.perf_counter() - start

print(f"Speedup: {loop_time / vec_time:.1f}x")

# Typical output: Speedup: 50-100x

```


### 4. Common Pitfalls


**Pitfall 1: Unexpected Broadcasting**


```python
a = np.array([[1], [2], [3]])  # shape (3, 1)
b = np.array([[10, 20, 30]])   # shape (1, 3)

# Intended: element-wise multiplication

# Actual: outer product due to broadcasting!
result = a * b  # shape (3, 3)

# [[ 10,  20,  30],

#  [ 20,  40,  60],

#  [ 30,  60,  90]]

# Fix: Ensure matching shapes
a_flat = a.ravel()  # shape (3,)
b_flat = b.ravel()  # shape (3,)
result = a_flat * b_flat  # shape (3,)

```


**Pitfall 2: Ambiguous 1D Arrays**


```python

# 1D array: shape (3,)
a = np.array([1, 2, 3])

# Does this become (3, 1) or (1, 3)?

# Answer: Neither! It stays (3,) and broadcasts flexibly
a + np.ones((4, 3))  # ✓ Works (broadcasts as row)
a + np.ones((3, 4))  # ✗ ValueError (incompatible)

# Explicit reshaping prevents confusion
a.reshape(-1, 1) + np.ones((3, 4))  # ✓ Clear intent

```


---


## Hands-On Exercises


### Exercise 3.1: Normalize by Row and Column


Given a matrix, compute both row-normalized and column-normalized versions.


```python
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# TODO: Compute row_normed and col_normed

```


<details>


<summary>Solution</summary>


```python

# Row normalization (each row sums to 1)
row_sums = X.sum(axis=1, keepdims=True)  # shape (3, 1)
row_normed = X / row_sums

# Column normalization (each column sums to 1)
col_sums = X.sum(axis=0, keepdims=True)  # shape (1, 3)
col_normed = X / col_sums

print(row_normed)

# [[0.16666667, 0.33333333, 0.5       ],

#  [0.26666667, 0.33333333, 0.4       ],

#  [0.29166667, 0.33333333, 0.375     ]]

print(col_normed)

# [[0.08333333, 0.13333333, 0.16666667],

#  [0.33333333, 0.33333333, 0.33333333],

#  [0.58333333, 0.53333333, 0.5       ]]

```


**Key insight**: `keepdims=True` preserves the reduced dimension as size 1, enabling automatic broadcasting.


</details>


### Exercise 3.2: Vectorize Sigmoid Function


Implement the sigmoid activation function and its derivative without loops.


```python
def sigmoid(z):
    # TODO: Implement using NumPy
    pass

def sigmoid_derivative(z):
    # TODO: Implement using sigmoid(z)
    pass

```


<details>


<summary>Solution</summary>


```python
def sigmoid(z):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative: σ'(z) = σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

# Test on array
z = np.array([-2, -1, 0, 1, 2])
print(sigmoid(z))

# [0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708]

print(sigmoid_derivative(z))

# [0.10499359, 0.19661193, 0.25, 0.19661193, 0.10499359]

```


</details>


**Scoring Rubric - Exercise 3.1** (10 points)


| Criterion                                    | Points |

| -------------------------------------------- | ------ |

| Correct row normalization formula            | 3      |

| Correct column normalization formula         | 3      |

| Uses `keepdims=True` for proper broadcasting | 2      |

| Verifies output (rows/columns sum to 1)      | 2      |


**Scoring Rubric - Exercise 3.2** (10 points)


| Criterion                                 | Points |

| ----------------------------------------- | ------ |

| Correct sigmoid implementation            | 3      |

| Correct derivative using σ(z)·(1-σ(z))    | 3      |

| Works on arrays of any shape (vectorized) | 2      |

| Includes test cases with expected output  | 2      |


## Common Pitfalls


| Pitfall                                | Why It's Bad                               | Fix                                          |

| -------------------------------------- | ------------------------------------------ | -------------------------------------------- |

| Not using `keepdims=True`              | Forces manual reshaping                    | Use `keepdims=True` in `sum`, `mean`, `std`  |

| Assuming 1D arrays have orientation    | Broadcasting behavior is ambiguous         | Explicitly reshape to (n, 1) or (1, n)       |

| Forgetting to benchmark                | May vectorize code that's not a bottleneck | Use `%timeit` or `perf_counter`              |

| Broadcasting large intermediate arrays | Memory explosion                           | Check shapes; use `einsum` for complex cases |


---


## Self-Assessment Checklist


Before moving to Day 4, verify you can:

- [ ] Predict the output shape of a broadcasted operation
- [ ] Use `np.newaxis` to add dimensions for broadcasting
- [ ] Vectorize a function that processes matrix rows/columns
- [ ] Compute a distance matrix without loops
- [ ] Benchmark vectorized vs. loop-based code

---


## Resources

- [NumPy Broadcasting](62)
- [From Python to NumPy (Nicolas Rougier)](63)
- [Vectorization Tricks (Jake VanderPlas)](64)

## Day 4: Pandas Pitfalls and Alternatives

## Overview


**Focus**: Learn to use Pandas efficiently and recognize when alternatives (Polars, DuckDB) are better. Topics include avoiding common performance traps and writing maintainable data transformations.


**Why it matters**: Pandas is ubiquitous but has subtle pitfalls that cause bugs and slowdowns. Understanding its limitations prevents frustration.


---


## Learning Objectives


By the end of Day 4, you will:

- Avoid chained indexing and `SettingWithCopyWarning`
- Use `.loc`, `.iloc`, and `.query()` correctly
- Recognize when Pandas is slow (e.g., row iteration, string operations)
- Apply Polars or DuckDB for large datasets
- Write transformation pipelines using method chaining

---


## Core Concepts


### 1. Indexing Done Right


**The Golden Rules**


```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# ✅ CORRECT: Single assignment with .loc
df.loc[df['A'] > 1, 'B'] = 99

# ❌ WRONG: Chained indexing (may fail silently!)
df[df['A'] > 1]['B'] = 99  # SettingWithCopyWarning

# ✅ CORRECT: Use .loc for clarity
df.loc[0, 'A'] = 10  # Set scalar value

# ❌ WRONG: Ambiguous
df['A'][0] = 10  # Works but discouraged

```


**Boolean Indexing**


```python

# Filter rows
high_values = df[df['A'] > 2]

# Multiple conditions: Use & (and), | (or), ~ (not)

# Parentheses are REQUIRED
filtered = df[(df['A'] > 1) & (df['B'] < 6)]

# Query syntax (alternative)
filtered = df.query('A > 1 and B < 6')

```


### 2. Performance Pitfalls


**Pitfall 1: Row Iteration**


```python

# ❌ SLOW: Python loop over rows
total = 0
for idx, row in df.iterrows():
    total += row['A'] * row['B']

# ✅ FAST: Vectorized operation
total = (df['A'] * df['B']).sum()

# Speedup: 100-1000x

```


**Pitfall 2: String Operations**


```python

# ❌ SLOW: Python loop
df['upper'] = df['text'].apply(lambda x: x.upper())

# ✅ FAST: Vectorized string method
df['upper'] = df['text'].str.upper()

# For complex regex or parsing, consider Polars

```


**Pitfall 3: Appending in Loop**


```python

# ❌ VERY SLOW: Repeated concatenation
result = pd.DataFrame()
for chunk in chunks:
    result = pd.concat([result, chunk])  # O(n²) complexity!

# ✅ FAST: Collect then concatenate once
result = pd.concat(chunks, ignore_index=True)

```


### 3. Method Chaining


**Readable Pipelines**


```python

# ❌ Hard to read: Intermediate variables everywhere
df = pd.read_csv('data.csv')
df = df.dropna()
df = df[df['age'] > 18]
df['income_log'] = np.log(df['income'])
df = df.groupby('region')['income_log'].mean()
df = df.reset_index()

# ✅ Clean: Method chaining
result = (
    pd.read_csv('data.csv')
    .dropna()
    .query('age > 18')
    .assign(income_log=lambda x: np.log(x['income']))
    .groupby('region')['income_log']
    .mean()
    .reset_index()
)

```


**Using** **`.pipe()`** **for Custom Functions**


```python
def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    return df[df[column].between(mean - n_std*std, mean + n_std*std)]

# Use in pipeline
result = (
    df
    .pipe(remove_outliers, 'income', n_std=2)
    .groupby('region')['income']
    .median()
)

```


### 4. Alternatives to Pandas


**When to Use Polars**


[Polars](https://www.pola.rs/) is faster and more memory-efficient than Pandas for large datasets.


```python
import polars as pl

# Polars: eager execution
df = pl.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.filter(pl.col('A') > 1).select(pl.col('B') * 2)

# Polars: lazy execution (optimizes query plan)
result = (
    pl.scan_csv('large_file.csv')
    .filter(pl.col('age') > 18)
    .groupby('region')
    .agg(pl.col('income').mean())
    .collect()  # Executes optimized plan
)

```


**When to Use DuckDB**


[DuckDB](https://duckdb.org/) is ideal for SQL-style analytics on DataFrames.


```python
import duckdb

# Query Pandas DataFrame using SQL
result = duckdb.query("""
    SELECT region, AVG(income) as avg_income
    FROM df
    WHERE age > 18
    GROUP BY region
    ORDER BY avg_income DESC
""").to_df()

# Or register multiple tables
duckdb.register('customers', customers_df)
duckdb.register('orders', orders_df)

result = duckdb.query("""
    SELECT 
        c.region,
        COUNT(o.order_id) as num_orders,
        SUM(o.amount) as total_amount
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.region
    ORDER BY total_amount DESC
""").to_df()

```


**Comparison Table**


| Tool   | Best For                                | Speed          | SQL Support  |

| ------ | --------------------------------------- | -------------- | ------------ |

| Pandas | Small data (<1GB), interactive analysis | Baseline       | No           |

| Polars | Large data, parallel processing         | 5-10x faster   | Query syntax |

| DuckDB | Complex joins, SQL users                | 10-100x faster | Native SQL   |


---


## Hands-On Exercises


### Exercise 4.1: Fix the Chained Indexing


Find and fix the bug:


```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Intended: Set B=99 where A > 1
df[df['A'] > 1]['B'] = 99

print(df)

# Expected: [[1, 4], [2, 99], [3, 99]]

# Actual:   [[1, 4], [2, 5], [3, 6]] ← Didn't work!

```


<details>


<summary>Solution</summary>


**Problem**: `df[df['A'] > 1]` creates a copy (sometimes), so the assignment doesn't affect the original.


```python

# ✅ Fix: Use .loc
df.loc[df['A'] > 1, 'B'] = 99

```


</details>


### Exercise 4.2: Optimize the Loop


Vectorize this code:


```python
df = pd.DataFrame({'x': range(10000), 'y': range(10000)})

# Compute z = sqrt(x^2 + y^2)
z_values = []
for idx, row in df.iterrows():
    z_values.append((row['x']**2 + row['y']**2)**0.5)

df['z'] = z_values

```


<details>


<summary>Solution</summary>


```python

# ✅ Vectorized
df['z'] = (df['x']**2 + df['y']**2)**0.5

# Speedup: ~100x

```


</details>


### Exercise 4.3: Convert to Polars


Rewrite this Pandas code in Polars:


```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'region': ['North', 'South', 'North', 'South', 'North'],
    'sales': [100, 200, 150, 250, 300],
    'active': [True, True, False, True, True]
})

result = (
    df[df['active']]
    .groupby('region')['sales']
    .agg(['mean', 'sum'])
    .reset_index()
)

```


<details>


<summary>Solution</summary>


</summary>


```python
import polars as pl

df = pl.DataFrame({
    'region': ['North', 'South', 'North', 'South', 'North'],
    'sales': [100, 200, 150, 250, 300],
    'active': [True, True, False, True, True]
})

result = (
    df
    .filter(pl.col('active'))
    .group_by('region')
    .agg([
        pl.col('sales').mean().alias('mean'),
        pl.col('sales').sum().alias('sum')
    ])
)

print(result)

# shape: (2, 3)

# ┌────────┬────────┬──────┐

# │ region ┆ mean   ┆ sum  │

# │ ---    ┆ ---    ┆ ---  │

# │ str    ┆ f64    ┆ i64  │

# ╞════════╪════════╪══════╡

# │ North  ┆ 200.0  ┆ 400  │

# │ South  ┆ 225.0  ┆ 450  │

# └────────┴────────┴──────┘

```


**Key differences**:

- Polars uses `filter` instead of boolean indexing
- No need for `reset_index()` (Polars doesn't use indexes)
- Column references use `pl.col()` instead of strings

</details>


**Scoring Rubric - Exercise 4.1** (8 points)


| Criterion                                  | Points |

| ------------------------------------------ | ------ |

| Identifies chained indexing as the problem | 3      |

| Provides correct `.loc` fix                | 3      |

| Explains why original creates a copy       | 2      |


**Scoring Rubric - Exercise 4.2** (8 points)


| Criterion                             | Points |

| ------------------------------------- | ------ |

| Removes loop entirely                 | 3      |

| Uses vectorized operations on columns | 3      |

| Benchmarks speedup (optional bonus)   | 2      |


**Scoring Rubric - Exercise 4.3** (9 points)


| Criterion                                      | Points |

| ---------------------------------------------- | ------ |

| Correct `filter()` instead of boolean indexing | 2      |

| Correct `group_by()` syntax                    | 2      |

| Correct `agg()` with `pl.col()` expressions    | 3      |

| Output matches expected format                 | 2      |


## Common Pitfalls


| Pitfall                         | Why It's Bad                     | Fix                                  |

| ------------------------------- | -------------------------------- | ------------------------------------ |

| `df[condition][col] = value`    | Chained indexing fails silently  | Use `df.loc[condition, col] = value` |

| `for idx, row in df.iterrows()` | 100x slower than vectorization   | Use `.apply()` or vectorized ops     |

| `pd.concat()` in loop           | O(n²) complexity                 | Collect list, concat once            |

| Not considering Polars/DuckDB   | Pandas struggles with large data | Benchmark alternatives               |


---


## Self-Assessment Checklist


Before moving to Day 5, verify you can:

- [ ] Use `.loc` and `.iloc` correctly
- [ ] Avoid `SettingWithCopyWarning`
- [ ] Write a method-chained transformation pipeline
- [ ] Identify when row iteration is a performance bottleneck
- [ ] Convert a simple Pandas operation to Polars

---


## Resources

- [Pandas Best Practices](65)
- [Modern Polars](66)
- [DuckDB Python API](67)
- [Why You Should Use Polars](68)

## Day 5: Testable, Reusable Code

## Overview


**Focus**: Write modular, testable code that can be reused across projects. Topics include function design, unit testing, and packaging.


**Why it matters**: Data science code often starts as throwaway scripts. Learning to write reusable functions accelerates future work and enables collaboration.


---


## Learning Objectives


By the end of Day 5, you will:

- Design functions with single responsibilities
- Write unit tests with `pytest`
- Use fixtures and parametrize tests
- Organize code into packages
- Apply test-driven development (TDD) basics

---


## Core Concepts


### 1. Function Design Principles


**Single Responsibility Principle**


```python

# ❌ BAD: Function does too much
def analyze_data(filepath):
    data = pd.read_csv(filepath)        # I/O
    data = data.dropna()                 # Cleaning
    mean = data['value'].mean()          # Analysis
    plt.hist(data['value'])              # Visualization
    plt.savefig('output.png')            # More I/O
    return mean

# ✅ GOOD: Single responsibility each
def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    return df.dropna()

def compute_mean(df, column):
    return df[column].mean()

def plot_histogram(df, column, output_path):
    plt.hist(df[column])
    plt.savefig(output_path)

```


**Pure Functions (When Possible)**


```python

# ❌ IMPURE: Modifies global state
total = 0

def add_to_total(x):
    global total
    total += x
    return total

# ✅ PURE: No side effects
def add_to_total(x, current_total):
    return current_total + x

```


### 2. Unit Testing with pytest


**Basic Test Structure**


```python

# my_math.py
def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

```


```python

# test_my_math.py
import pytest
from my_math import add, divide

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -1) == -2

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero_raises():
    with pytest.raises(ValueError):
        divide(10, 0)

```


**Run tests**:


```bash
pytest test_my_math.py -v

```


### 3. Fixtures and Parametrize


**Fixtures for Test Data**


```python
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """Reusable test data."""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

def test_column_sum(sample_data):
    assert sample_data['A'].sum() == 6

def test_column_mean(sample_data):
    assert sample_data['B'].mean() == 5

```


**Parametrized Tests**


```python
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (10, -5, 5),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected

```


### 4. Package Organization


**Recommended Structure**


```javascript
my_project/
├── my_package/
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│   └── io.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── setup.py
├── README.md
└── requirements.txt

```


[**`setup.py`**](http://setup.py/) **Example**


```python
from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'pandas>=1.3',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'black', 'flake8'],
    },
)

```


**Install in development mode**:


```bash
pip install -e .

# Now you can import my_package anywhere

```


### 5. Test-Driven Development (TDD)


**The Red-Green-Refactor Cycle**

1. **Red**: Write a failing test
2. **Green**: Write minimal code to pass
3. **Refactor**: Improve code while tests still pass

**Example**


```python

# 1. RED: Write test first (it will fail)
def test_standardize():
    data = np.array([1, 2, 3, 4, 5])
    result = standardize(data)
    assert np.isclose(result.mean(), 0)
    assert np.isclose(result.std(), 1)

# 2. GREEN: Implement to pass test
def standardize(data):
    return (data - data.mean()) / data.std()

# 3. REFACTOR: Improve implementation
def standardize(data):
    """Standardize array to mean=0, std=1."""
    data = np.asarray(data)
    return (data - data.mean()) / data.std(ddof=1)  # Use sample std

```


---


## Hands-On Exercises


### Exercise 5.1: Write Tests


Write tests for this function:


```python
def clip_outliers(data, n_std=3):
    """Remove values beyond n standard deviations from mean."""
    mean = np.mean(data)
    std = np.std(data)
    lower = mean - n_std * std
    upper = mean + n_std * std
    return data[(data >= lower) & (data <= upper)]

```


<details>


<summary>Solution</summary>


```python
import pytest
import numpy as np

def test_clip_outliers_removes_extremes():
    data = np.array([1, 2, 3, 4, 100])  # 100 is outlier
    result = clip_outliers(data, n_std=2)
    assert 100 not in result
    assert len(result) == 4

def test_clip_outliers_keeps_normal_values():
    data = np.array([1, 2, 3, 4, 5])
    result = clip_outliers(data, n_std=3)
    assert len(result) == 5  # All values within 3 std

@pytest.mark.parametrize("n_std", [1, 2, 3])
def test_clip_outliers_different_thresholds(n_std):
    data = np.array([1, 2, 3, 4, 100])
    result = clip_outliers(data, n_std=n_std)
    assert len(result) <= len(data)

```


</details>


### Exercise 5.2: Refactor for Testability


Make this function easier to test:


```python
def process_file(filepath):
    """Load CSV, clean, and save result."""
    data = pd.read_csv(filepath)
    data = data.dropna()
    data['zscore'] = (data['value'] - data['value'].mean()) / data['value'].std()
    data.to_csv('output.csv', index=False)
    return data

```


<details>


<summary>Solution</summary>


</summary>


**Problem**: Hard to test because it mixes I/O with logic.


```python

# ✅ Separate concerns
def clean_data(data):
    """Remove missing values."""
    return data.dropna()

def add_zscore(data, column):
    """Add z-score column."""
    data = data.copy()
    data['zscore'] = (data[column] - data[column].mean()) / data[column].std()
    return data

def process_file(filepath, output_path):
    """Orchestrate data processing."""
    data = pd.read_csv(filepath)
    data = clean_data(data)
    data = add_zscore(data, 'value')
    data.to_csv(output_path, index=False)
    return data

# Now we can test the logic without file I/O:
def test_clean_data():
    df = pd.DataFrame({'A': [1, None, 3]})
    result = clean_data(df)
    assert len(result) == 2
    assert result['A'].isna().sum() == 0

def test_add_zscore():
    df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    result = add_zscore(df, 'value')
    assert 'zscore' in result.columns
    assert abs(result['zscore'].mean()) < 1e-10  # Mean should be ~0

```


</details>


**Scoring Rubric - Exercise 5.1** (10 points)


| Criterion                                    | Points |

| -------------------------------------------- | ------ |

| Tests outlier removal behavior               | 3      |

| Tests normal value preservation              | 2      |

| Uses parametrize for multiple thresholds     | 2      |

| Tests edge cases (empty array, all outliers) | 2      |

| Clear test function names                    | 1      |


**Scoring Rubric - Exercise 5.2** (10 points)


| Criterion                                         | Points |

| ------------------------------------------------- | ------ |

| Separates I/O from transformation logic           | 4      |

| Each extracted function has single responsibility | 3      |

| Provides tests for pure functions                 | 2      |

| Output path made configurable                     | 1      |


## Common Pitfalls


| Pitfall                              | Why It's Bad                 | Fix                                       |

| ------------------------------------ | ---------------------------- | ----------------------------------------- |

| Testing implementation, not behavior | Tests break when refactoring | Test inputs/outputs, not internals        |

| Not using fixtures                   | Duplicate test data setup    | Use `@pytest.fixture`                     |

| Skipping edge cases                  | Bugs in corner cases         | Test empty inputs, None, negatives        |

| Mixing I/O with logic                | Hard to test                 | Separate pure functions from side effects |


---


## Self-Assessment Checklist


Before moving to Day 6, verify you can:

- [ ] Write a pure function with single responsibility
- [ ] Write unit tests with `pytest`
- [ ] Use fixtures to share test data
- [ ] Parametrize tests for multiple inputs
- [ ] Organize code into an installable package

---


## Resources

- [pytest Documentation](69)
- [Testing Best Practices](70)
- [Python Packaging Guide](71)
- [Clean Code in Python](72)

## Day 6: Reproducibility, Randomness, and State

## Overview


**Focus**: Understand how randomness, state, and side effects impact reproducibility. Topics include random seeds, stateful generators, and best practices for reproducible research.


**Why it matters**: Statistical simulations must be reproducible. Understanding how Python manages randomness prevents "works on my machine" bugs.


---


## Learning Objectives


By the end of Day 6, you will:

- Set random seeds for NumPy and Python's `random` module
- Use `np.random.Generator` for modern random number generation
- Understand when state persists across function calls
- Write reproducible simulation code
- Manage randomness in parallel computations

---


## Core Concepts


### 1. Random Seed Basics


**Why Seeds Matter**


```python
import numpy as np

# Without seed: different results each run
print(np.random.rand(3))  # [0.374 0.950 0.731]
print(np.random.rand(3))  # [0.598 0.156 0.155]

# With seed: reproducible results
np.random.seed(42)
print(np.random.rand(3))  # [0.374 0.950 0.731]

np.random.seed(42)
print(np.random.rand(3))  # [0.374 0.950 0.731] ← same!

```


**Global State Problem**


```python

# ❌ BAD: Modifies global state
def simulate_data(n, seed=42):
    np.random.seed(seed)
    return np.random.randn(n)

# This will give IDENTICAL results!
data1 = simulate_data(100, seed=42)
data2 = simulate_data(100, seed=42)  # Resets seed again

```


### 2. Modern Random Number Generation


**Using** **`np.random.Generator`**


```python

# ✅ GOOD: Isolated random state
rng = np.random.default_rng(42)
data = rng.normal(0, 1, size=100)

# Each generator has independent state
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(99)

print(rng1.random(3))  # [0.77395605 0.43887844 0.85859792]
print(rng2.random(3))  # [0.77513282 0.93949894 0.89482735]
print(rng1.random(3))  # [0.69736803 0.09417735 0.97562235]

```


**Passing Generators to Functions**


```python
def bootstrap_sample(data, rng):
    """Generate bootstrap sample using provided RNG."""
    indices = rng.integers(0, len(data), size=len(data))
    return data[indices]

# Usage
rng = np.random.default_rng(42)
sample1 = bootstrap_sample(data, rng)
sample2 = bootstrap_sample(data, rng)  # Different sample (state advanced)

```


### 3. Python's `random` Module


**Separate State from NumPy**


```python
import random

# Python's random module has its own state
random.seed(42)
print([random.random() for _ in range(3)])

# [0.639, 0.025, 0.275]

# NumPy's seed doesn't affect Python's random
np.random.seed(42)
print([random.random() for _ in range(3)])

# [0.220, 0.796, 0.118] ← different!

```


**Seeding Both**


```python
def set_all_seeds(seed):
    """Set seeds for all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    # For PyTorch: torch.manual_seed(seed)
    # For TensorFlow: tf.random.set_seed(seed)

```


### 4. Stateful Functions


**When State Persists**


```python

# Example: Generator function maintains state
def counter():
    count = 0
    while True:
        count += 1
        yield count

gen = counter()
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3

# State persists across calls!

```


**Avoiding Unintended State**


```python

# ❌ BAD: Hidden state in default argument
def add_to_cache(item, cache=[]):
    cache.append(item)
    return cache

print(add_to_cache(1))  # [1]
print(add_to_cache(2))  # [1, 2] ← unexpected!

# ✅ GOOD: Explicit state management
def add_to_cache(item, cache=None):
    if cache is None:
        cache = []
    cache.append(item)
    return cache

```


### 5. Reproducible Simulations


**Complete Example**


```python
import numpy as np
from typing import Optional

def monte_carlo_pi(n_samples: int, rng: Optional[np.random.Generator] = None) -> float:
    """
    Estimate π using Monte Carlo simulation.
    
    Args:
        n_samples: Number of random points
        rng: Random number generator (if None, creates new one)
    
    Returns:
        Estimate of π
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random points in [0, 1] × [0, 1]
    x = rng.random(n_samples)
    y = rng.random(n_samples)
    
    # Count points inside unit circle
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside.mean()
    
    return pi_estimate

# Reproducible usage
rng = np.random.default_rng(42)
print(monte_carlo_pi(100000, rng))  # 3.14392

# Reset for exact reproduction
rng = np.random.default_rng(42)
print(monte_carlo_pi(100000, rng))  # 3.14392 ← identical

```


### 6. Parallel Randomness


**Problem: Shared State in Parallel Code**


```python
from multiprocessing import Pool
import numpy as np

def worker(seed):
    np.random.seed(seed)  # Each worker needs different seed
    return np.random.rand(3)

# ❌ BAD: All workers get same seed
with Pool(4) as pool:
    results = pool.map(worker, [42, 42, 42, 42])  # All identical!

# ✅ GOOD: Different seeds per worker
with Pool(4) as pool:
    results = pool.map(worker, [42, 43, 44, 45])  # Different results

```


**Better: Use SeedSequence**


```python
from numpy.random import SeedSequence, default_rng

def worker_with_rng(seed):
    rng = default_rng(seed)
    return rng.random(3).tolist()

# Generate child seeds from parent
parent_seed = 42
ss = SeedSequence(parent_seed)
child_seeds = ss.spawn(4)  # Create 4 independent seeds

with Pool(4) as pool:
    results = pool.map(worker_with_rng, child_seeds)

# Each worker gets independent, reproducible random stream

```


---


## Hands-On Exercises


### Exercise 6.1: Debug Reproducibility


Why doesn't this give the same result twice?


```python
import numpy as np

def simulate():
    np.random.seed(42)
    data = np.random.randn(100)
    noise = np.random.randn(100)  # Second call!
    return data + noise

result1 = simulate()
result2 = simulate()

print(np.allclose(result1, result2))  # True ✓

# But if we call it like this:
result1 = simulate()
_ = np.random.rand(10)  # Some other code uses random
result2 = simulate()

print(np.allclose(result1, result2))  # False ✗ Why?

```


<details>


<summary>Solution</summary>


**Problem**: The global state is affected by any code that uses `np.random` between calls.


```python

# ✅ Fix: Use isolated generators
def simulate(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    
    data = rng.normal(0, 1, size=100)
    noise = rng.normal(0, 1, size=100)
    return data + noise

# Now reproducible regardless of external code
result1 = simulate()
_ = np.random.rand(10)  # External randomness
result2 = simulate()    # Still uses fresh RNG with seed 42

print(np.allclose(result1, result2))  # True ✓

```


</details>


### Exercise 6.2: Implement Reproducible Bootstrap


Write a function that performs bootstrap confidence intervals with reproducible results.


```python
def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95, rng=None):
    """
    Compute bootstrap confidence interval for the mean.
    
    Args:
        data: Input array
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0-1)
        rng: Random number generator
    
    Returns:
        (lower, upper) confidence interval
    """
    # TODO: Implement
    pass

```


<details>


<summary>Solution</summary>


```python
import numpy as np

def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95, rng=None):
    """Compute bootstrap confidence interval for the mean."""
    if rng is None:
        rng = np.random.default_rng()
    
    data = np.asarray(data)
    n = len(data)
    
    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(sample.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute percentile confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper

# Test reproducibility
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

rng1 = np.random.default_rng(42)
ci1 = bootstrap_ci(data, rng=rng1)

rng2 = np.random.default_rng(42)
ci2 = bootstrap_ci(data, rng=rng2)

print(ci1)  # (3.5, 7.5)
print(ci2)  # (3.5, 7.5) ← identical!

```


</details>


**Scoring Rubric - Exercise 6.1** (10 points)


| Criterion                                | Points |

| ---------------------------------------- | ------ |

| Identifies global state as the problem   | 3      |

| Uses `np.random.default_rng()` with seed | 3      |

| RNG parameter with None default          | 2      |

| Demonstrates reproducibility with test   | 2      |


**Scoring Rubric - Exercise 6.2** (12 points)


| Criterion                                   | Points |

| ------------------------------------------- | ------ |

| Correct bootstrap sampling with replacement | 3      |

| Proper percentile calculation for CI        | 3      |

| RNG parameter for reproducibility           | 3      |

| Handles array conversion with `np.asarray`  | 1      |

| Tests demonstrate reproducibility           | 2      |


## Common Pitfalls


| Pitfall                               | Why It's Bad                    | Fix                            |

| ------------------------------------- | ------------------------------- | ------------------------------ |

| Using `np.random.seed()` in functions | Modifies global state           | Pass `rng` parameter           |

| Forgetting Python's `random` module   | Separate state from NumPy       | Seed both or use only one      |

| Same seed for parallel workers        | Workers generate identical data | Use `SeedSequence.spawn()`     |

| Not documenting seed values           | Can't reproduce results         | Log seeds in notebooks/scripts |


---


## Self-Assessment Checklist


Before moving to Day 7, verify you can:

- [ ] Explain why global `np.random.seed()` is problematic
- [ ] Use `np.random.Generator` for isolated random state
- [ ] Write a function that accepts an `rng` parameter
- [ ] Set up reproducible parallel computations
- [ ] Debug a reproducibility issue in simulation code

---


## Resources

- [NumPy Random Sampling](73)
- [NEP 19: Random Number Generator Policy](74)
- [Reproducible Research Best Practices](75)

## Day 7: Debugging, Profiling, and Python Capstone

## Overview


**Focus**: Learn to debug and profile Python code efficiently. Topics include debuggers, logging, profiling tools, and the Week 1 Python capstone project.


**Why it matters**: Debugging and performance optimization are critical skills. The capstone integrates all Week 1 concepts.


---


## Learning Objectives


By the end of Day 7, you will:

- Use `pdb` and IDE debuggers effectively
- Instrument code with logging instead of print statements
- Profile code to identify bottlenecks
- Optimize slow code using profiling insights
- Complete the Week 1 Python capstone project

---


## Core Concepts


### 1. Debugging with pdb


**Basic Usage**


```python
import pdb

def buggy_function(x):
    result = 0
    for i in range(len(x)):
        pdb.set_trace()  # Execution pauses here
        result += x[i] ** 2
    return result

# When execution hits pdb.set_trace():

# (Pdb) p i          # Print variable

# (Pdb) p result     # Print variable

# (Pdb) n            # Next line

# (Pdb) c            # Continue

# (Pdb) q            # Quit

```


**Post-Mortem Debugging**


```python
import pdb

def divide(a, b):
    return a / b

try:
    divide(10, 0)
except:
    pdb.post_mortem()  # Debug the exception after it occurred

```


**Breakpoint() (Python 3.7+)**


```python
def process_data(data):
    cleaned = data.dropna()
    breakpoint()  # Modern alternative to pdb.set_trace()
    scaled = (cleaned - cleaned.mean()) / cleaned.std()
    return scaled

```


### 2. Logging


**Why Logging > Print**


```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug(f"Input shape: {data.shape}")
    logger.info("Starting data processing")
    
    if data.shape[0] == 0:
        logger.warning("Empty dataset received")
        return None
    
    try:
        result = data.mean()
        logger.info(f"Processing complete, result: {result}")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

```


**Log Levels**


| Level    | Use Case                                        |

| -------- | ----------------------------------------------- |

| DEBUG    | Detailed diagnostic info                        |

| INFO     | Confirmation that things work as expected       |

| WARNING  | Something unexpected, but code still works      |

| ERROR    | Serious problem, code couldn't perform function |

| CRITICAL | Program may crash                               |


### 3. Profiling


**Time Profiling with cProfile**


```python
import cProfile
import pstats
import io

def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

# Profile the function
profiler = cProfile.Profile()
profiler.enable()
result = slow_function()
profiler.disable()

# Print stats sorted by cumulative time
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Alternative: profile with context manager
with cProfile.Profile() as pr:
    slow_function()
    

# Save stats to file for later analysis
pr.dump_stats('profile_output.prof')

# Visualize with: snakeviz profile_output.prof

```


**Line Profiling with line_profiler**


```bash

# Install: pip install line_profiler

```


```python

# Add @profile decorator
@profile
def process_data(data):
    result = []                           # Line 1
    for i in range(len(data)):            # Line 2
        if data[i] > 0:                   # Line 3
            result.append(data[i] ** 2)   # Line 4
    return result                         # Line 5

# Run: kernprof -l -v script.py

# Output shows time spent on each line:

# Line #  Hits     Time  Per Hit  % Time  Line Contents

#      2  1000    500.0      0.5    10.0  for i in range(len(data)):

#      3  1000    200.0      0.2     4.0      if data[i] > 0:

#      4   800   4000.0      5.0    80.0          result.append(data[i] ** 2)

```


**Memory Profiling with memory_profiler**


```bash

# Install: pip install memory_profiler

```


```python
from memory_profiler import profile

@profile
def memory_hog():
    a = [1] * (10 ** 6)        # ~8 MB
    b = [2] * (2 * 10 ** 7)    # ~160 MB
    del b                       # Free memory
    return a

# Run: python -m memory_profiler script.py

# Output shows memory usage per line:

# Line #    Mem usage    Increment  Line Contents

#      4     50.0 MiB     0.0 MiB   a = [1] * (10 ** 6)

#      5    210.0 MiB   160.0 MiB   b = [2] * (2 * 10 ** 7)

#      6     50.0 MiB  -160.0 MiB   del b

```


### 4. Optimization Strategies


**Strategy 1: Vectorize Loops**


```python
import numpy as np
import time

# ❌ Slow: Python loop
def sum_of_squares_loop(data):
    result = 0
    for x in data:
        result += x ** 2
    return result

# ✅ Fast: Vectorized
def sum_of_squares_vectorized(data):
    return np.sum(data ** 2)

data = np.random.randn(1000000)

start = time.perf_counter()
sum_of_squares_loop(data)
loop_time = time.perf_counter() - start

start = time.perf_counter()
sum_of_squares_vectorized(data)
vec_time = time.perf_counter() - start

print(f"Speedup: {loop_time / vec_time:.0f}x")  # ~100x

```


**Strategy 2: Use Appropriate Data Structures**


```python

# ❌ Slow: List lookup
def count_occurrences_list(data, value):
    return data.count(value)  # O(n) every time

# ✅ Fast: Dictionary/Counter
from collections import Counter

def count_occurrences_dict(data):
    return Counter(data)  # O(n) once, then O(1) lookups

data = [1, 2, 3, 1, 2, 1] * 10000
counts = count_occurrences_dict(data)
print(counts[1])  # Instant lookup

```


**Strategy 3: Cache Expensive Computations**


```python
from functools import lru_cache

# ❌ Slow: Recomputes every time
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# ✅ Fast: Caches results
@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n < 2:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

# fibonacci(35) takes ~5 seconds

# fibonacci_cached(35) takes ~0.0001 seconds

```


---


## Week 1 Python Capstone


### Project: Statistical Analysis Pipeline


**Expected Time (Proficient): 90-120 minutes**


Build a complete statistical analysis pipeline that integrates all Week 1 concepts.


**Validation Requirement**: Your implementation must produce results that match reference implementations (NumPy/SciPy functions) within numerical tolerance of **±1e-6** (relative error) for all test cases. Include validation tests in your test suite.


**Requirements**:

1. **Functions and Modules** (Day 1)
    - Create a reusable module with type-hinted functions
    - Use context managers for file I/O
2. **Memory Management** (Day 2)
    - Correctly handle NumPy views and copies
    - Avoid unintended mutations
3. **Vectorization** (Day 3)
    - Vectorize all computations (no Python loops over data)
    - Use broadcasting for matrix operations
4. **Data Processing** (Day 4)
    - Use Pandas with `.loc` (no chained indexing)
    - Implement method chaining
5. **Testing** (Day 5)
    - Write unit tests with pytest
    - Achieve >80% code coverage
6. **Reproducibility** (Day 6)
    - Accept `rng` parameter for all random operations
    - Document all random seeds
7. **Debugging & Profiling** (Day 7)
    - Use logging instead of print statements
    - Profile code and optimize bottlenecks

**Assessment Structure**: The checklist below ensures you've applied all Week 1 concepts. After verifying all checklist items, score your work using the rubric that follows. **You must score ≥1.5/2.0 on EACH rubric dimension to claim proficiency.** The checklist is for self-assessment of completeness; the rubric is the proficiency gate.


### Implementation Checklist

- [ ] **Module structure**: Code organized into importable module with `__init__.py`
- [ ] **Type hints**: All functions have type annotations
- [ ] **Docstrings**: All functions have numpy-style docstrings
- [ ] **Context managers**: File I/O uses `with` statements
- [ ] **No mutations**: Functions don't unexpectedly modify inputs
- [ ] **Vectorized**: No Python loops over data arrays
- [ ] **Broadcasting**: Uses NumPy broadcasting correctly
- [ ] **Pandas best practices**: Uses `.loc`, method chaining, no `SettingWithCopyWarning`
- [ ] **Unit tests**: Tests cover main functions with pytest
- [ ] **Reproducible**: Random operations use `rng` parameter
- [ ] **Logging**: Uses logging module (not print)
- [ ] **Profiled**: Code has been profiled and optimized

### Proficiency Rubric


| Dimension          | 0 (Not Met)                                                              | 1 (Partial)                                                 | 2 (Proficient)                                                        |

| ------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------------------------------------------------------------- |

| **Code Quality**   | Multiple style violations, no type hints or docstrings                   | Some type hints/docstrings, inconsistent style              | Comprehensive type hints, docstrings, PEP 8 compliant                 |

| **Functionality**  | Core functionality missing or broken                                     | Basic functionality works with minor bugs                   | All features work correctly, handles edge cases                       |

| **Best Practices** | Ignores Week 1 patterns (mutations, loops over arrays, print statements) | Applies some patterns (vectorization OR testing OR logging) | Consistently applies patterns (vectorization AND testing AND logging) |

| **Testing**        | No tests or <50% coverage                                                | Some tests, 50-80% coverage                                 | Comprehensive tests, >80% coverage, uses fixtures                     |


**Scoring**: You must score ≥1.5/2.0 on EACH dimension. Scoring 2.0 on some and 0 on others does NOT constitute proficiency.


### Starter Template


```python
"""stats_pipeline.py - Statistical Analysis Pipeline

This module provides a complete statistical analysis pipeline
integrating all Week 1 Python concepts.

Example:
    >>> from stats_pipeline import StatsPipeline
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> pipeline = StatsPipeline(rng=rng)
    >>> result = pipeline.analyze('data.csv')
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsPipeline:
    """Statistical analysis pipeline with reproducible random operations.
    
    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        If None, creates new Generator with random seed.
    
    Attributes
    ----------
    rng : np.random.Generator
        The random number generator used for all random operations.
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        logger.info("StatsPipeline initialized")
    
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.
            
        Returns
        -------
        pd.DataFrame
            Loaded data.
        """
        logger.debug(f"Loading data from {filepath}")
        with open(filepath, 'r') as f:
            df = pd.read_csv(f)
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def compute_statistics(self, data: NDArray[np.float64]) -> dict:
        """Compute summary statistics using vectorized operations.
        
        Parameters
        ----------
        data : ndarray
            Input data array (not modified).
            
        Returns
        -------
        dict
            Dictionary containing mean, std, median, and percentiles.
        """
        # Work on a copy to avoid mutation
        arr = np.asarray(data).copy()
        
        # Vectorized computations (no loops)
        stats = {
            'mean': np.mean(arr),
            'std': np.std(arr, ddof=1),
            'median': np.median(arr),
            'p25': np.percentile(arr, 25),
            'p75': np.percentile(arr, 75),
        }
        
        logger.debug(f"Computed statistics: {stats}")
        return stats
    
    def bootstrap_ci(
        self,
        data: NDArray[np.float64],
        statistic: str = 'mean',
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval.
        
        Parameters
        ----------
        data : ndarray
            Input data array.
        statistic : str
            Statistic to compute ('mean' or 'median').
        n_bootstrap : int
            Number of bootstrap samples.
        confidence : float
            Confidence level (e.g., 0.95 for 95% CI).
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound) of confidence interval.
        """
        arr = np.asarray(data)
        n = len(arr)
        
        # Vectorized bootstrap: generate all indices at once
        indices = self.rng.integers(0, n, size=(n_bootstrap, n))
        samples = arr[indices]  # Shape: (n_bootstrap, n)
        
        # Compute statistic for all samples at once (vectorized)
        if statistic == 'mean':
            bootstrap_stats = np.mean(samples, axis=1)
        elif statistic == 'median':
            bootstrap_stats = np.median(samples, axis=1)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Percentile CI
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        logger.info(f"Bootstrap {confidence*100}% CI for {statistic}: ({lower:.4f}, {upper:.4f})")
        return (lower, upper)


# Tests (run with: pytest stats_pipeline.py -v)
def test_compute_statistics():
    """Test that statistics match NumPy reference."""
    rng = np.random.default_rng(42)
    pipeline = StatsPipeline(rng=rng)
    
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = pipeline.compute_statistics(data)
    
    assert np.isclose(stats['mean'], np.mean(data), rtol=1e-6)
    assert np.isclose(stats['std'], np.std(data, ddof=1), rtol=1e-6)
    assert np.isclose(stats['median'], np.median(data), rtol=1e-6)


def test_bootstrap_reproducibility():
    """Test that bootstrap is reproducible with same RNG."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    rng1 = np.random.default_rng(42)
    pipeline1 = StatsPipeline(rng=rng1)
    ci1 = pipeline1.bootstrap_ci(data, n_bootstrap=1000)
    
    rng2 = np.random.default_rng(42)
    pipeline2 = StatsPipeline(rng=rng2)
    ci2 = pipeline2.bootstrap_ci(data, n_bootstrap=1000)
    
    assert ci1 == ci2, "Bootstrap should be reproducible with same seed"


def test_no_mutation():
    """Test that input data is not modified."""
    rng = np.random.default_rng(42)
    pipeline = StatsPipeline(rng=rng)
    
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original = data.copy()
    
    pipeline.compute_statistics(data)
    pipeline.bootstrap_ci(data)
    
    assert np.array_equal(data, original), "Input data should not be modified"

```


---


## Common Pitfalls


| Pitfall                      | Why It's Bad                     | Fix                                      |

| ---------------------------- | -------------------------------- | ---------------------------------------- |

| Using print() for debugging  | Output persists in production    | Use logging with appropriate levels      |

| Profiling without hypothesis | Waste time optimizing wrong code | Profile first, then optimize bottlenecks |

| Over-optimization            | Premature optimization is evil   | Optimize only proven bottlenecks         |

| Not removing debug code      | `pdb.set_trace()` in production  | Use breakpoint() with env variable       |


---


## Self-Assessment Checklist


After completing Day 7, verify you can:

- [ ] Use `pdb` or IDE debugger to step through code
- [ ] Configure and use Python's logging module
- [ ] Profile code with cProfile and identify bottlenecks
- [ ] Optimize slow code using appropriate strategies
- [ ] **Complete the Week 1 Python capstone with proficiency score ≥1.5/2.0 on each dimension**

---


## Resources

- [Python Debugging with pdb](%7B%7B76%7D%7D)
- [Logging HOWTO](%7B%7B77%7D%7D)
- [Python Profilers](%7B%7B78%7D%7D)
- [Optimization Tips](%7B%7B79%7D%7D) in production</td>

<td>Use breakpoint() with env variable</td>


</tr>


</table>


---


## Self-Assessment Checklist


After completing Day 7, verify you can:

- [ ] Use `pdb` or IDE debugger to step through code
- [ ] Configure and use Python's logging module
- [ ] Profile code with cProfile and identify bottlenecks
- [ ] Optimize slow code using appropriate strategies
- [ ] **Complete the Week 1 Python capstone with proficiency score ≥1.5/2.0 on each dimension**

---


## Resources

- [Python Debugging with pdb](76)
- [Logging HOWTO](77)
- [Python Profilers](78)
- [Optimization Tips](79)

---


## Learning Path


**Sequential progression**: Each day builds on concepts from previous days. Complete them in order.


**Time commitment**: Plan 3-4 hours per day (includes reading, exercises, and practice).


**Proficiency gate**: Day 7 includes a capstone project. You must score ≥1.5/2.0 on each rubric dimension to claim Week 1 proficiency.


---


## Key Themes


### For R Users

- **Memory semantics**: Python uses references, not copy-on-write
- **Vectorization**: NumPy requires explicit broadcasting rules
- **Pandas quirks**: Avoid chained indexing and `SettingWithCopyWarning`

### For MATLAB Users

- **Module system**: Code organization differs from MATLAB's path-based approach
- **Zero-indexing**: Arrays start at 0, not 1
- **Random state**: Must explicitly manage `rng` objects

### For All Backgrounds

- **Type hints**: Modern Python uses optional type annotations
- **Testing culture**: Unit tests are expected, not optional
- **Reproducibility**: Statistical code must be deterministic

---


## Prerequisites


Before starting Week 1, ensure you have:

- [ ] Python 3.9+ installed
- [ ] NumPy, Pandas, Pytest installed (`pip install numpy pandas pytest`)
- [ ] Basic Python syntax familiarity (variables, loops, conditionals)
- [ ] Experience with statistical computing in R, MATLAB, or similar

---


## Success Criteria


By the end of Week 1, you should be able to:

- Write type-hinted, well-documented Python functions
- Explain when NumPy operations create views vs. copies
- Vectorize computations without Python loops
- Use Pandas correctly (no chained indexing)
- Write unit tests with pytest
- Manage random state for reproducible simulations
- Debug and profile code to identify bottlenecks
- **Complete the capstone project with proficiency scores ≥1.5/2.0**

---


## Daily Self-Assessment


Each day includes a self-assessment checklist. Before moving to the next day, verify you can complete all items. If not, review the exercises and resources.


---


## Getting Help


If you encounter difficulties:

1. **Re-read the relevant section**: Concepts build on each other
2. **Work through exercises**: Hands-on practice is essential
3. **Check resources**: Each day links to official documentation
4. **Review error messages**: Python errors are usually informative

---


## Next Steps


Ready to begin? Start with <page url="[https://www.notion.so/ff5ab5e0e9e747e5b66b030e825a109d">Day](https://www.notion.so/ff5ab5e0e9e747e5b66b030e825a109d%22%3EDay) 1: Functions, Modules, and Idiomatic Python</page>.


## Week 2: C++ (Days 8-14)

## Overview


Week 2 focuses on **C++ proficiency** for statisticians and data scientists who need high-performance computing. By the end of this week, you will understand C++'s memory model, write efficient code using the STL, and integrate C++ with Python.


---


## Week Structure


This week is organized into 7 daily modules, each covering a critical C++ concept:


## Day 8: References, Pointers, and Ownership

## Overview


**Focus**: Understand C++'s memory model, including pointers, references, and ownership semantics. Learn when to use stack vs. heap allocation and how to reason about resource lifetime.


**Why it matters**: Memory management is C++'s superpower and greatest pitfall. Understanding references and pointers is fundamental to writing correct C++ code.


---


## Learning Objectives


By the end of Day 8, you will:

- Distinguish between references and pointers
- Use pass-by-reference correctly for function parameters
- Understand stack vs. heap allocation
- Reason about object lifetime and ownership
- Avoid dangling pointers and references
- Use smart pointers for automatic memory management

---


## Core Concepts


### 1. Pointers vs References


**Both allow you to refer to objects without copying, but with different semantics.**


**References** are aliases for existing objects:


```c++
int x = 10;
int& ref = x;  // ref is an alias for x
ref = 20;      // Modifies x through the reference
std::cout << x;  // Prints 20

```


**Key properties of references**:

- Must be initialized when declared (cannot be null)
- Cannot be rebound to refer to a different object
- No separate storage (just an alias)
- Syntax is transparent (use like the original variable)

**Pointers** store memory addresses:


```c++
int x = 10;
int* ptr = &x;  // ptr stores the address of x
*ptr = 20;      // Dereference: modifies x through the pointer
std::cout << x;   // Prints 20

```


**Key properties of pointers**:

- Can be null (`nullptr`)
- Can be rebound to point to different objects
- Have their own storage (store an address)
- Require explicit dereferencing with `*`
- Can do pointer arithmetic

**When to use which**:

- **Use references** for function parameters (avoiding copies) and return values when the object definitely exists
- **Use pointers** when you need to represent "optional" (can be null), rebinding, or dynamic allocation

### 2. Pass by Value vs Reference vs Pointer


```c++
// Pass by value: copies the vector (expensive!)
void process_copy(std::vector<int> v) {
    v[0] = 999;  // Modifies the copy, not the original
}

// Pass by reference: no copy, can modify original
void process_ref(std::vector<int>& v) {
    v[0] = 999;  // Modifies the original
}

// Pass by const reference: no copy, cannot modify (most common for read-only)
void process_const_ref(const std::vector<int>& v) {
    // v[0] = 999;  // Error: cannot modify const reference
    std::cout << v[0];  // Can read
}

// Pass by pointer: can be null, explicit about indirection
void process_ptr(std::vector<int>* v) {
    if (v == nullptr) return;  // Can check for null
    (*v)[0] = 999;  // Dereference to modify
}

```


**Golden rule**: For function parameters, prefer `const T&` for input, `T&` for output/inout, and `T*` only when null is valid.


### 3. Ownership: Who is Responsible for Cleanup?


Ownership determines who is responsible for freeing memory. C++ requires explicit reasoning about ownership to avoid leaks and use-after-free bugs.


**Raw pointers don't express ownership**:


```c++
int* ptr = new int(42);  // Who owns this?
// Should the caller delete it?
// Should the callee delete it?
// Is it even heap-allocated?
// Unclear!

```


**Smart pointers express ownership explicitly**:


**`std::unique_ptr`****: Exclusive ownership**


```c++
#include <memory>

// Create unique_ptr - owns the object exclusively
std::unique_ptr<int> ptr = std::make_unique<int>(42);

// Automatic cleanup when ptr goes out of scope
// No need for delete!

// Cannot copy (ownership is exclusive)
// std::unique_ptr<int> ptr2 = ptr;  // Error!

// But can move (transfer ownership)
std::unique_ptr<int> ptr2 = std::move(ptr);
// Now ptr is null, ptr2 owns the object

```


**When to use** **`unique_ptr`**:

- Single owner is clear
- Object lifetime matches the owning scope
- Factory functions returning heap-allocated objects
- Replacing raw `new`/`delete` patterns

**`std::shared_ptr`****: Shared ownership**


```c++
#include <memory>

// Create shared_ptr - multiple owners allowed
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
std::shared_ptr<int> ptr2 = ptr1;  // Both own the object

std::cout << ptr1.use_count();  // 2 (reference count)

// Object deleted when last shared_ptr is destroyed
ptr1.reset();  // ptr1 no longer owns it
std::cout << ptr2.use_count();  // 1
// ptr2 goes out of scope -> object is deleted

```


**When to use** **`shared_ptr`**:

- Multiple owners with unclear lifetimes
- Shared resources (caches, pools)
- Tree/graph structures with back-references

**Cost**: Reference counting has overhead (atomic operations for thread safety)


### 4. Dangling Pointers and References


A **dangling pointer** refers to memory that has been freed:


```c++
int* ptr = new int(42);
delete ptr;
int x = *ptr;  // UNDEFINED BEHAVIOR! Use-after-free

```


A **dangling reference** refers to an object that no longer exists:


```c++
int& get_local_ref() {
    int x = 42;
    return x;  // BUG! Returns reference to local variable
}  // x is destroyed here

int& ref = get_local_ref();
int val = ref;  // UNDEFINED BEHAVIOR! x no longer exists

```


**Common dangling reference bugs**:


```c++
// BAD: Returning reference to temporary
const std::string& get_name() {
    return std::string("Alice");  // Temporary destroyed at end of statement!
}

// BAD: Returning reference to local
int& increment(int x) {
    x += 1;
    return x;  // x dies when function returns!
}

// GOOD: Return by value (copy elision makes this efficient)
std::string get_name() {
    return std::string("Alice");  // Returned efficiently
}

```


**Preventing dangling pointers**:

- Use smart pointers (automatic cleanup)
- Set pointers to `nullptr` after delete
- Never return references to local variables
- Use linters (clang-tidy warns about these)

### 5. Stack vs Heap Allocation


**Stack allocation**:


```c++
int x = 10;  // Allocated on stack
double arr[100];  // 800 bytes on stack

```

- **Fast**: Just pointer arithmetic
- **Limited size**: Typically 1-8 MB total
- **Automatic cleanup**: Destroyed when scope ends
- **Use for**: Small objects with known lifetime

**Heap allocation**:


```c++
int* ptr = new int(10);  // Allocated on heap
double* arr = new double[100];  // Heap array

delete ptr;      // Manual cleanup required
delete[] arr;    // Array delete

```

- **Slower**: Allocator overhead
- **Large**: Limited only by system RAM
- **Manual cleanup**: Must call delete
- **Use for**: Large objects, dynamic sizes, unclear lifetime

**Modern C++ avoids raw** **`new`****/****`delete`**. Use smart pointers or containers:


```c++
// GOOD: Smart pointer handles cleanup
auto data = std::make_unique<std::vector<int>>(1000);

// EVEN BETTER: Container handles everything
std::vector<int> data(1000);

```


### 6. Build Systems and Debugging


**Compilation and Linking**:

- Compilation translates source to object files (`.o`)
- Linking combines object files into an executable
- Common flags: `-O2` (optimization), `-g` (debug symbols), `-Wall -Wextra` (warnings), `-std=c++17` (standard version)

**Debuggers**: `gdb` and `lldb` enable breakpoints, stepping, and variable inspection.


**Sanitizers** catch memory errors at runtime:


```bash
g++ -fsanitize=address,undefined -g -o program program.cpp
./program

```


**Build Systems**:

- **Makefile**: Traditional Unix build tool, simple for small projects
- **CMake**: Modern standard, generates platform-specific build files

**Basic CMakeLists.txt**:


```javascript
cmake_minimum_required(VERSION 3.16)
project(MyProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)
add_executable(main main.cpp utils.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE -fsanitize=address,undefined)
    target_link_options(main PRIVATE -fsanitize=address,undefined)
endif()

```


---


## Hands-On Exercises


The exercises from the original curriculum provide hands-on practice with these concepts. Work through them to solidify your understanding of stack/heap allocation, debugging with gdb, memory sanitizers, and building multi-file projects with Makefiles.


---


## Common Pitfalls


| Pitfall                        | Why It's Bad                               | Fix                                                    |

| ------------------------------ | ------------------------------------------ | ------------------------------------------------------ |

| Using raw `new`/`delete`       | Easy to forget cleanup, not exception-safe | Use smart pointers or containers                       |

| Returning reference to local   | Undefined behavior                         | Return by value or use smart pointers                  |

| Not checking for `nullptr`     | Crashes on dereference                     | Always check pointers from uncertain sources           |

| Mixing `delete` and `delete[]` | Undefined behavior                         | Use `delete[]` for arrays, `delete` for single objects |


---


## Self-Assessment Checklist


Before moving to Day 9, verify you can:

- [ ] Explain the difference between a reference and a pointer
- [ ] Write a function that takes `const T&` parameters correctly
- [ ] Use `std::unique_ptr` to manage heap memory
- [ ] Identify a dangling pointer or reference in code
- [ ] Compile a program with debug symbols and use gdb to inspect variables
- [ ] Run AddressSanitizer and interpret its output

---


## Resources

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [cppreference.com](http://cppreference.com/)
- [GDB Tutorial](https://www.cs.cmu.edu/~gilpin/tutorial/)
- [Sanitizers Documentation](https://github.com/google/sanitizers)

## Day 9: Smart Pointers and Ownership Semantics

## Overview


**Focus**: Master smart pointers and understand ownership semantics in depth. Learn when to use `unique_ptr`, `shared_ptr`, and `weak_ptr`.


**Why it matters**: Smart pointers are C++'s solution to manual memory management. They encode ownership in the type system, making memory bugs impossible at compile time.


---


## Learning Objectives


By the end of Day 9, you will:

- Use `std::unique_ptr` for exclusive ownership
- Use `std::shared_ptr` for shared ownership
- Use `std::weak_ptr` to break reference cycles
- Understand when each smart pointer is appropriate
- Implement the Rule of Three (or Rule of Zero)
- Avoid memory leaks through RAII

---


## Core Concepts


### 1. Why Smart Pointers?


**The problem with raw pointers**:


```c++
void dangling_pointer_bug() {
    int* ptr = new int(42);
    delete ptr;          // Free the memory
    std::cout << *ptr;   // BUG: use-after-free (undefined behavior)
}

void double_free_bug() {
    int* ptr = new int(42);
    delete ptr;
    delete ptr;          // BUG: freeing already-freed memory (crash)
}

void leak_bug() {
    int* ptr = new int(42);
    // Forgot to delete!
    return;              // BUG: memory leaked
}

```


**Solution: Smart pointers encode ownership**:


```c++
void safe_with_unique_ptr() {
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    std::cout << *ptr << std::endl;
    // ptr goes out of scope here → destructor automatically calls delete
    // No leak, no double-free, no use-after-free possible
}

```


### 2. `std::unique_ptr`: Exclusive Ownership


**Basic usage**:


```c++
#include <memory>

// Create unique_ptr
std::unique_ptr<int> ptr = std::make_unique<int>(42);

// Access value
std::cout << *ptr << std::endl;  // Dereference
std::cout << ptr.get() << std::endl;  // Get raw pointer

// Transfer ownership
std::unique_ptr<int> ptr2 = std::move(ptr);
// Now ptr is nullptr, ptr2 owns the object

// This would crash (accessing nullptr):
// std::cout << *ptr;

// This is safe:
if (ptr2) {
    std::cout << *ptr2 << std::endl;
}

```


**Cannot copy (ownership is exclusive)**:


```c++
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
// std::unique_ptr<int> ptr2 = ptr1;  // ERROR! Cannot copy

// But can move (transfer ownership):
std::unique_ptr<int> ptr2 = std::move(ptr1);  // OK

```


**Factory pattern**:


```c++
class Widget {
public:
    Widget(int value) : value_(value) {}
    int value() const { return value_; }
private:
    int value_;
};

// Factory function returning unique_ptr
std::unique_ptr<Widget> create_widget(int value) {
    return std::make_unique<Widget>(value);
}

// Usage
auto widget = create_widget(42);
std::cout << widget->value() << std::endl;

```


### 3. `std::shared_ptr`: Shared Ownership


**Basic usage**:


```c++
// Create shared_ptr - multiple owners allowed
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
std::shared_ptr<int> ptr2 = ptr1;  // Both own the object

std::cout << ptr1.use_count();  // 2 (reference count)

// Object deleted when last shared_ptr is destroyed
ptr1.reset();  // ptr1 no longer owns it
std::cout << ptr2.use_count();  // 1

// When ptr2 goes out of scope, object is deleted

```


**Use case: Shared resources**:


```c++
class Image {
public:
    Image(const std::string& path) {
        std::cout << "Loading " << path << std::endl;
        // ... expensive loading ...
    }
    ~Image() {
        std::cout << "Unloading image" << std::endl;
    }
};

class Sprite {
private:
    std::shared_ptr<Image> image_;
public:
    Sprite(std::shared_ptr<Image> img) : image_(img) {}
};

int main() {
    // Load image once, share among sprites
    auto image = std::make_shared<Image>("texture.png");
    
    Sprite sprite1(image);
    Sprite sprite2(image);
    Sprite sprite3(image);
    
    // Image stays loaded while any sprite exists
    // Automatically unloaded when last sprite is destroyed
}

```


**Cost of shared_ptr**:

- Reference counting requires atomic operations (thread-safe)
- Slightly slower than unique_ptr
- Two allocations: object + control block (use `make_shared` to reduce to one)

### 4. `std::weak_ptr`: Breaking Cycles


**The problem: Reference cycles cause leaks**:


```c++
struct Node {
    std::string name;
    std::vector<std::shared_ptr<Node>> children;
    std::shared_ptr<Node> parent;  // Problem!
    
    Node(const std::string& n) : name(n) {
        std::cout << "Node " << name << " created\n";
    }
    ~Node() {
        std::cout << "Node " << name << " destroyed\n";
    }
};

void create_cycle_leak() {
    auto parent = std::make_shared<Node>("Parent");
    auto child = std::make_shared<Node>("Child");
    
    parent->children.push_back(child);  // Parent -> Child (strong)
    child->parent = parent;              // Child -> Parent (strong) - CYCLE!
    
    // ref counts: parent=2, child=2
    // When function ends: parent=1, child=1
    // Neither reaches 0 → LEAK!
}

```


**Solution: Use** **`weak_ptr`** **for back-references**:


```c++
struct Node {
    std::string name;
    std::vector<std::shared_ptr<Node>> children;
    std::weak_ptr<Node> parent;  // Weak reference doesn't prevent destruction
    
    Node(const std::string& n) : name(n) {}
    ~Node() { std::cout << "Node " << name << " destroyed\n"; }
};

void create_cycle_fixed() {
    auto parent = std::make_shared<Node>("Parent");
    auto child = std::make_shared<Node>("Child");
    
    parent->children.push_back(child);  // Parent -> Child (strong)
    child->parent = parent;              // Child -> Parent (weak) - No cycle!
    
    // When function ends: parent ref count 1→0, destroyed
    // Then child ref count 1→0, destroyed
    // No leak!
}

```


**Using weak_ptr**:


```c++
std::weak_ptr<Node> weak = shared;  // Create weak_ptr from shared_ptr

// To access, must convert to shared_ptr:
if (auto strong = weak.lock()) {  // Returns shared_ptr or nullptr
    std::cout << strong->name << std::endl;  // Safe to use
} else {
    std::cout << "Object has been destroyed" << std::endl;
}

```


### 5. Rule of Three (or Rule of Zero)


**Rule of Three**: If your class needs a custom destructor, it likely needs custom copy constructor and copy assignment operator.


**Example: Managing a resource**:


```c++
class DynamicArray {
private:
    int* data_;
    size_t size_;

public:
    // Constructor
    explicit DynamicArray(size_t size) 
        : data_(new int[size]), size_(size) {}
    
    // Destructor
    ~DynamicArray() {
        delete[] data_;
    }
    
    // Copy constructor (deep copy)
    DynamicArray(const DynamicArray& other) 
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }
    
    // Copy assignment operator
    DynamicArray& operator=(const DynamicArray& other) {
        if (this != &other) {  // Self-assignment check
            delete[] data_;  // Free existing resource
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }
    
    // Accessors
    size_t size() const { return size_; }
    int& operator[](size_t i) { return data_[i]; }
    const int& operator[](size_t i) const { return data_[i]; }
};

// Usage showing Rule of Three in action:
void demonstrate_rule_of_three() {
    DynamicArray arr1(5);        // Constructor
    arr1[0] = 42;
    
    DynamicArray arr2 = arr1;    // Copy constructor (deep copy)
    arr2[0] = 100;               // Doesn't affect arr1
    
    DynamicArray arr3(10);
    arr3 = arr1;                 // Copy assignment (deep copy)
    
    // When function ends, all three are safely destroyed
    // Without Rule of Three: double-free crash!
}

```


**Rule of Zero**: Prefer using existing RAII types instead of manual resource management:


```c++
class BetterDynamicArray {
private:
    std::vector<int> data_;  // std::vector handles everything!

public:
    explicit BetterDynamicArray(size_t size) : data_(size) {}
    
    // No destructor, copy constructor, or copy assignment needed!
    // Compiler-generated versions use vector's implementations
    
    size_t size() const { return data_.size(); }
    int& operator[](size_t i) { return data_[i]; }
};

```


### 6. Passing Smart Pointers


**Best practices**:


```c++
// GOOD: Pass by reference if not transferring ownership
void process(const std::vector<int>& data) { /* read-only */ }
void modify(std::vector<int>& data) { /* can modify */ }

// BAD: Copying shared_ptr when you just need to read
void process_bad(std::shared_ptr<std::vector<int>> data) {
    // Copies the shared_ptr, increments ref count unnecessarily
}

// GOOD: Pass raw pointer or reference when not transferring ownership
void process_good(const std::vector<int>* data) {
    if (!data) return;
    // Use data
}

// Transfer unique ownership: take by value, caller uses std::move
void take_ownership(std::unique_ptr<Data> data) {
    // Function now owns data
}

// Usage:
auto data = std::make_unique<Data>();
take_ownership(std::move(data));  // Transfers ownership
// data is now nullptr

```


---


## Hands-On Exercises


The original curriculum includes exercises for:

- Implementing swap functions with references vs pointers
- Creating dangling references and observing undefined behavior
- Implementing the Rule of Three for a Matrix class
- Using smart pointers to break reference cycles
- Refactoring code to use move-only types

---


## Common Pitfalls


| Pitfall                       | Why It's Bad                     | Fix                                     |

| ----------------------------- | -------------------------------- | --------------------------------------- |

| Forgetting Rule of Three      | Double-free or shallow copy bugs | Implement all three or use Rule of Zero |

| Circular `shared_ptr`         | Memory leak from reference cycle | Use `weak_ptr` for back-references      |

| Passing `shared_ptr` by value | Unnecessary ref count overhead   | Pass by `const&` or raw pointer         |

| Copying when you should move  | Performance loss                 | Use `std::move` for transfer            |


---


## Self-Assessment Checklist


Before moving to Day 10, verify you can:

- [ ] Use `std::unique_ptr` for exclusive ownership
- [ ] Use `std::shared_ptr` for shared ownership
- [ ] Use `std::weak_ptr` to break reference cycles
- [ ] Implement the Rule of Three correctly
- [ ] Explain when to use Rule of Zero instead
- [ ] Pass smart pointers efficiently to functions

---


## Resources

- [C++ Core Guidelines: Smart Pointers](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-smartptrparam)
- [cppreference: unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr)
- [cppreference: shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr)
- [Understanding Smart Pointers](https://www.internalpointers.com/post/beginner-s-look-smart-pointers-modern-c)

## Day 10: RAII, Move Semantics, and Destructors

## Overview


**Focus**: Understand RAII (Resource Acquisition Is Initialization) fundamentals and basic `unique_ptr` usage. Learn how C++ guarantees deterministic cleanup.


**Scope Note**: This day focuses on RAII philosophy and using smart pointers effectively. Advanced move semantics topics (Rule of Five, move constructors/assignment, `std::move` internals, lvalue/rvalue distinctions) are deferred to Week 3 for the 2-week timeline.


**Why it matters**: RAII is C++'s fundamental idiom for resource management. It converts manual cleanup into compiler-enforced guarantees, eliminating an entire class of bugs.


---


## Learning Objectives


By the end of Day 10, you will:

- Understand the RAII principle and philosophy
- Use `unique_ptr` for automatic cleanup
- Recognize when destructors run
- Write RAII wrappers for custom resources
- Understand exception safety through RAII

---


## Core Concepts


### 1. RAII: Resource Acquisition Is Initialization


**The core principle**: Tie resource lifetime to object lifetime.

- **Acquire resources** (memory, files, locks, sockets) in the constructor
- **Release resources** in the destructor
- **The compiler guarantees** destructors run when objects go out of scope

**Why this matters**: No manual cleanup code. No forgotten cleanup. Exception-safe by design.


**Example: File handling**


```c++
// BAD: Manual resource management (C-style)
void process_file_bad(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return;  // Error handling
    
    // ... process file ...
    
    fclose(f);  // Easy to forget!
    // What if an exception is thrown? File never closed!
}

// GOOD: RAII (C++-style)
void process_file_good(const std::string& filename) {
    std::ifstream f(filename);  // Opens in constructor
    if (!f) return;
    
    // ... process file ...
    
    // Destructor automatically closes file when f goes out of scope
    // Works even if exception is thrown!
}

```


**RAII guarantees**:

1. **Automatic cleanup**: Destructor runs when scope ends
2. **Exception safety**: Resources released during stack unwinding
3. **No leaks**: Impossible to forget cleanup

### 2. Common RAII Types


**Standard library RAII types**:

- `std::unique_ptr`, `std::shared_ptr`: Manage heap memory
- `std::vector`, `std::string`: Manage dynamic arrays
- `std::ifstream`, `std::ofstream`: Manage files
- `std::lock_guard`, `std::unique_lock`: Manage mutexes

**Example: RAII ensures cleanup during exceptions**


```c++
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource acquired\n"; }
    ~Resource() { std::cout << "Resource released\n"; }
};

void might_throw(bool should_throw) {
    Resource r;  // RAII: acquired here
    
    std::cout << "Using resource...\n";
    
    if (should_throw) {
        throw std::runtime_error("Error!");
        // Destructor still called during stack unwinding
    }
    
    std::cout << "Finished normally\n";
}  // Destructor called here if no exception

int main() {
    try {
        might_throw(true);
    } catch (...) {
        std::cout << "Exception caught\n";
    }
    // Output shows resource was released despite exception!
    
    return 0;
}

```


### 3. Writing RAII Wrappers


**Pattern for custom resources**:


```c++
class FileHandle {
private:
    FILE* file;
    
public:
    FileHandle(const std::string& path, const std::string& mode) {
        file = std::fopen(path.c_str(), mode.c_str());
        if (!file) {
            throw std::runtime_error("Failed to open: " + path);
        }
    }
    
    ~FileHandle() {
        if (file) {
            std::fclose(file);
        }
    }
    
    // Non-copyable
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    
    // Movable
    FileHandle(FileHandle&& other) noexcept : file(other.file) {
        other.file = nullptr;
    }
    
    size_t read(void* buffer, size_t size, size_t count) {
        return std::fread(buffer, size, count, file);
    }
};

void use_file() {
    FileHandle f("data.txt", "r");
    // Use f...
    // Automatic cleanup when f goes out of scope
}

```


### 4. Destructors: Cleanup Guarantees


**When destructors run**:

1. **Automatic objects**: End of scope
2. **Dynamic objects**: When `delete` is called
3. **During exception unwinding**: For all constructed objects on the stack

**Destructor best practices**:

1. **Never throw**: Destructors should be `noexcept` (it's the default)
2. **Idempotent cleanup**: Safe to call even if resource already released
3. **No complex logic**: Keep destructors simple
4. **Virtual if inherited**: Base class destructors should be virtual

```c++
class Base {
public:
    virtual ~Base() = default;  // Virtual destructor for polymorphism
};

class Derived : public Base {
private:
    int* data_;
public:
    Derived() : data_(new int[100]) {}
    ~Derived() override { delete[] data_; }
};

// Safe polymorphic deletion:
Base* ptr = new Derived();
delete ptr;  // Calls Derived::~Derived() then Base::~Base()

```


### 5. Exception Safety and RAII


**The problem**: Exceptions can skip cleanup code


```c++
void unsafe() {
    int* p = new int[100];
    risky_operation();  // Might throw!
    delete[] p;  // Never reached if exception thrown → LEAK
}

```


**The solution**: RAII guarantees cleanup


```c++
void safe() {
    std::unique_ptr<int[]> p(new int[100]);
    risky_operation();  // Might throw!
    // p's destructor runs even during unwinding → NO LEAK
}

```


**The three exception safety guarantees**:

1. **Basic guarantee**: Invariants preserved, no leaks
2. **Strong guarantee**: Operation succeeds completely or has no effect
3. **Nothrow guarantee**: Operation never throws (marked `noexcept`)

**RAII provides at least the basic guarantee automatically**.


### 6. The Rule of Zero


**Prefer using existing RAII types**:


If you can use standard library types, you don't need custom destructors:


```c++
class DataProcessor {
private:
    std::vector<int> data_;           // RAII: automatic cleanup
    std::unique_ptr<Cache> cache_;    // RAII: automatic cleanup
    std::string name_;                 // RAII: automatic cleanup

public:
    DataProcessor(size_t size, const std::string& name) 
        : data_(size), cache_(std::make_unique<Cache>()), name_(name) {}
    
    // No destructor needed! Compiler-generated one works perfectly
    // No copy constructor needed! Compiler handles it correctly
    // No copy assignment needed! Compiler handles it correctly
};

```


**When you need custom**: Only when managing resources not wrapped by existing RAII types (rare in modern C++).


---


## Hands-On Exercises


The original curriculum includes exercises for:

- Writing RAII wrappers for FILE*
- Using `std::lock_guard` to protect shared data
- Implementing move-only database connection wrappers
- Creating a `ScopedTimer` class
- Building a memory pool allocator with RAII

---


## Common Pitfalls


| Pitfall                                | Why It's Bad                            | Fix                                |

| -------------------------------------- | --------------------------------------- | ---------------------------------- |

| Manual cleanup in destructors throwing | Undefined behavior if destructor throws | Make destructors `noexcept`        |

| Forgetting virtual destructor          | Derived destructors don't run           | Mark base destructor `virtual`     |

| Mixing RAII and manual management      | Easy to double-delete or leak           | Choose one: all RAII or all manual |

| Complex logic in destructors           | Hard to reason about exceptions         | Keep destructors simple            |


---


## Self-Assessment Checklist


Before moving to Day 11, verify you can:

- [ ] Explain the RAII principle in your own words
- [ ] Write a simple RAII wrapper for a custom resource
- [ ] Use `std::unique_ptr` to manage heap memory automatically
- [ ] Explain why RAII provides exception safety
- [ ] Recognize when Rule of Zero applies

---


## Resources

- [C++ Core Guidelines: RAII](91)
- [RAII in C++](92)
- [Exception Safety](93)
- [Herb Sutter: GotW on RAII](94)

## Day 11: STL Containers, Algorithms, and Iterators

## Overview


**Focus**: Master the Standard Template Library's containers, algorithms, and iterators. Learn to write expressive, efficient code using STL building blocks.


**Why it matters**: The STL is C++'s superpower for generic programming. Mastering it means writing less code with fewer bugs and better performance.


---


## Learning Objectives


By the end of Day 11, you will:

- Choose appropriate containers for different use cases
- Use STL algorithms to replace raw loops
- Understand iterator categories and invalidation
- Write lambdas for use with algorithms
- Compose operations using algorithm chaining

---


## Core Concepts


### 1. Containers: Choosing the Right Data Structure


**Sequence containers** maintain element order:


**`std::vector`**: Dynamic array, contiguous storage

- **Use when**: Default choice for sequences, need random access
- **Cost**: Insertion/deletion in middle is O(n)

**`std::deque`**: Double-ended queue

- **Use when**: Need fast insertion at both ends
- **Cost**: Slightly slower random access than vector

**`std::list`**: Doubly-linked list

- **Use when**: Frequent insertion/deletion in middle, don't need random access
- **Cost**: No random access, poor cache locality

**Associative containers** for fast lookup:


**`std::map`**: Ordered key-value pairs (Red-Black tree)

- **Use when**: Need sorted keys or guaranteed iteration order
- **Cost**: O(log n) operations

**`std::unordered_map`**: Hash table

- **Use when**: Fast lookup is priority, don't need sorted order
- **Cost**: O(1) average, O(n) worst case

**Container comparison**:


| Operation     | vector | deque | list  | map      | unordered_map |

| ------------- | ------ | ----- | ----- | -------- | ------------- |

| Random access | O(1)   | O(1)  | O(n)  | -        | -             |

| Insert front  | O(n)   | O(1)  | O(1)  | -        | -             |

| Insert back   | O(1)*  | O(1)  | O(1)  | -        | -             |

| Insert middle | O(n)   | O(n)  | O(1)† | -        | -             |

| Lookup by key | -      | -     | -     | O(log n) | O(1) avg      |


*Amortized  †If you have the iterator


### 2. Iterators: The Universal Pointer Abstraction


**Iterator categories** (weakest to strongest):

1. **Input/Output**: Single-pass, read or write
2. **Forward**: Multi-pass, can read/write multiple times
3. **Bidirectional**: Can go forward and backward (`++`, `--`)
4. **Random access**: Can jump to any position (`+`, `-`, `[]`)

**Basic usage**:


```c++
std::vector<int> v = {1, 2, 3, 4, 5};

// begin() and end() return iterators
auto it = v.begin();   // Points to first element
auto end = v.end();    // Points PAST last element

while (it != end) {
    std::cout << *it << " ";  // Dereference to access value
    ++it;  // Move to next element
}

// Range-based for loop uses iterators internally
for (int x : v) {
    std::cout << x << " ";
}

```


**Iterator invalidation**: When containers reallocate, iterators may become invalid


```c++
std::vector<int> v = {1, 2, 3};
auto it = v.begin();
v.push_back(4);  // May reallocate!
// *it is now UNDEFINED BEHAVIOR if vector reallocated

```


**Invalidation rules**:

- **vector/deque**: Insertion/deletion may invalidate all iterators
- **list/map/set**: Only erased elements' iterators are invalidated

### 3. Algorithms: Generic, Composable, Efficient


**Key algorithm categories**:


**Non-modifying**:


```c++
// Find element
auto it = std::find(v.begin(), v.end(), 3);

// Count occurrences
int count = std::count(v.begin(), v.end(), 2);

// Check conditions
bool all_positive = std::all_of(v.begin(), v.end(), 
                                [](int x) { return x > 0; });

```


**Modifying**:


```c++
// Transform (apply function to each element)
std::transform(v.begin(), v.end(), squares.begin(),
               [](int x) { return x * x; });

// Remove element (erase-remove idiom)
v.erase(std::remove(v.begin(), v.end(), 3), v.end());

```


**Erase-remove idiom** (important!):


```c++
std::vector<int> v = {1, 2, 3, 2, 4};

// CORRECT: Removes all occurrences
v.erase(std::remove(v.begin(), v.end(), 2), v.end());

// Why: std::remove shifts elements, returns new logical end
// You must erase from that point to the old end

```


**Sorting**:


```c++
// Sort ascending
std::sort(v.begin(), v.end());

// Sort with custom comparator
std::sort(v.begin(), v.end(), std::greater<int>());

// Binary search (requires sorted range)
if (std::binary_search(v.begin(), v.end(), 5)) {
    std::cout << "Found 5";
}

```


**Numeric algorithms**:


```c++
#include <numeric>

// Sum
int sum = std::accumulate(v.begin(), v.end(), 0);

// Product
int product = std::accumulate(v.begin(), v.end(), 1, 
                              std::multiplies<int>());

// Inner product (dot product)
int dot = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0);

```


### 4. Lambdas: Inline Anonymous Functions


**Basic syntax**:


```c++
// [capture](parameters) -> return_type { body }
auto add = [](int a, int b) -> int { return a + b; };

// Return type deduction
auto square = [](int x) { return x * x; };

```


**Capture modes**:


```c++
int factor = 10;

// Capture by value (copy)
auto f1 = [factor](int x) { return x * factor; };

// Capture by reference
auto f2 = [&factor](int x) { factor += x; return factor; };

// Capture all by value
auto f3 = [=](int x) { return x * factor; };

// Capture all by reference
auto f4 = [&](int x) { factor += x; };

// Mixed capture
int a = 1, b = 2;
auto f5 = [a, &b](int x) { b = a + x; return b; };

```


**Common usage with algorithms**:


```c++
// Count even numbers
int even_count = std::count_if(v.begin(), v.end(),
                               [](int x) { return x % 2 == 0; });

// Transform with lambda
std::transform(v.begin(), v.end(), v.begin(),
               [](int x) { return x * 2; });

// Sort by custom criteria
struct Person { std::string name; int age; };
std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}};

std::sort(people.begin(), people.end(),
          [](const Person& a, const Person& b) {
              return a.age < b.age;
          });

```


### 5. Algorithm Composition


**Chaining operations**:


```c++
std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Sum of squares of even numbers
// Step 1: Filter to evens
std::vector<int> evens;
std::copy_if(v.begin(), v.end(), std::back_inserter(evens),
             [](int x) { return x % 2 == 0; });

// Step 2: Transform to squares
std::transform(evens.begin(), evens.end(), evens.begin(),
               [](int x) { return x * x; });

// Step 3: Sum
int sum = std::accumulate(evens.begin(), evens.end(), 0);

```


**Modern C++ ranges (C++20)**:


```c++
#include <ranges>

auto result = v 
    | std::views::filter([](int x) { return x % 2 == 0; })
    | std::views::transform([](int x) { return x * x; });

int sum = std::accumulate(result.begin(), result.end(), 0);

```


### 6. Performance Considerations


**STL algorithms are highly optimized**:

- Often use SIMD instructions
- Inlining removes lambda overhead
- Implementations are battle-tested

**Best practices**:

1. **Prefer algorithms over raw loops** (more expressive, less error-prone)
2. **Use range-based for when you don't need index** (cleaner syntax)
3. **Reserve capacity for vectors** (`v.reserve(n)`)
4. **Use const references in range-based for**:

    ```c++
    for (const auto& item : large_vector) { ... }  // No copy
    ```

5. **Use emplace over push**:

    ```c++
    v.emplace_back(args...);  // Constructs in-place
    ```


---


## Common Pitfalls


| Pitfall                                   | Why It's Bad                         | Fix                                                |

| ----------------------------------------- | ------------------------------------ | -------------------------------------------------- |

| Forgetting iterator validity check        | Undefined behavior                   | Check `it != end()` before dereference             |

| Using `operator[]` on map                 | Inserts default value if key missing | Use `find()` if you don't want insertion           |

| Wrong type for `accumulate` initial value | Integer overflow or precision loss   | Use correct type: `0L` for long, `0.0` for double  |

| Modifying container while iterating       | Iterator invalidation                | Use algorithm or save modifications for after loop |


---


## Self-Assessment Checklist


Before moving to Day 12, verify you can:

- [ ] Choose appropriate container for a given use case
- [ ] Use `std::find`, `std::sort`, `std::transform`, `std::accumulate`
- [ ] Write lambdas with explicit captures
- [ ] Apply the erase-remove idiom correctly
- [ ] Understand iterator invalidation rules

---


## Resources

- [C++ Reference: Containers](95)
- [C++ Reference: Algorithms](96)
- [STL Tutorial](97)
- [C++20 Ranges](98)

## Day 12: Numerical Stability and Floating-Point

## Overview


**Focus**: Understand numerical stability and floating-point arithmetic. Learn to write numerically robust code that avoids catastrophic cancellation, overflow, and underflow.


**Why it matters**: Statistical computations involve floating-point arithmetic. Naive implementations can produce completely wrong results due to numerical instability.


---


## Learning Objectives


By the end of Day 12, you will:

- Understand IEEE 754 floating-point representation
- Avoid catastrophic cancellation in computations
- Implement stable variance and summation algorithms
- Use log-space computation to avoid overflow
- Compare floating-point numbers correctly

---


## Core Concepts


### 1. Floating-Point Representation


**IEEE 754** represents numbers with finite precision:

- **Machine epsilon**: Smallest number ε such that 1 + ε ≠ 1
- **Denormalized numbers**: Very small numbers near zero
- **Special values**: Infinity (`inf`), Not-a-Number (`NaN`)

**Rounding errors** accumulate in computations:


```c++
double x = 0.1 + 0.2;  // Not exactly 0.3!
std::cout << (x == 0.3);  // false
std::cout << std::abs(x - 0.3) < 1e-10;  // true (use tolerance)

```


### 2. Catastrophic Cancellation


**The problem**: Subtracting nearly equal numbers loses precision.


**Example: Naive variance**:


```c++
// BAD: E[X²] - E[X]² is numerically unstable
double variance_naive(const std::vector<double>& v) {
    double sum = 0, sum_sq = 0;
    for (double x : v) {
        sum += x;
        sum_sq += x * x;
    }
    double mean = sum / v.size();
    double mean_sq = sum_sq / v.size();
    return mean_sq - mean * mean;  // Catastrophic cancellation!
}

// Test: {1e10, 1e10 + 1, 1e10 + 2}
// True variance: 1.0
// Naive result: 0.0  (WRONG! Lost all precision)

```


**Solution: Welford's algorithm**:


```c++
// GOOD: Stable online algorithm
double variance_welford(const std::vector<double>& v) {
    double mean = 0, M2 = 0;
    size_t n = 0;
    for (double x : v) {
        n++;
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        M2 += delta * delta2;
    }
    return M2 / (n - 1);  // Sample variance
}

// Welford result: 1.0 (CORRECT)

```


### 3. Summation Stability


**Naive summation** loses precision when summing many small numbers:


```c++
// BAD: Naive summation
double sum = 0.0;
for (int i = 0; i < 1000000; i++) {
    sum += 1e-10;  // Each addition loses precision
}
// Result: 0.000100000000169 (wrong!)
// True:   0.0001

```


**Kahan summation** compensates for lost low-order bits:


```c++
// GOOD: Compensated summation
double kahan_sum(const std::vector<double>& v) {
    double sum = 0.0;
    double c = 0.0;  // Compensation for lost low-order bits
    
    for (double x : v) {
        double y = x - c;      // Compensated value
        double t = sum + y;    // Tentative sum
        c = (t - sum) - y;     // Recover lost bits
        sum = t;
    }
    return sum;
}

```


### 4. Log-Space Computation


**The problem**: exp(large_number) overflows


```c++
// BAD: Naive log-sum-exp
double logsumexp_naive(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += std::exp(x);  // exp(1000) = inf!
    }
    return std::log(sum);
}

```


**Solution**: Subtract max before exponentiating


```c++
// GOOD: Numerically stable log-sum-exp
double logsumexp_stable(const std::vector<double>& v) {
    if (v.empty()) return -std::numeric_limits<double>::infinity();
    
    double max_val = *std::max_element(v.begin(), v.end());
    
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return max_val;
    }
    
    double sum = 0.0;
    for (double x : v) {
        sum += std::exp(x - max_val);  // exp(x - max) ≤ 1, no overflow
    }
    
    return max_val + std::log(sum);
}

// Math: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

```


**Application: Softmax**


```c++
std::vector<double> softmax_stable(const std::vector<double>& v) {
    if (v.empty()) return {};
    
    // Find max for numerical stability
    double max_val = *std::max_element(v.begin(), v.end());
    
    // Compute exp(x - max) and sum
    std::vector<double> result(v.size());
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = std::exp(v[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (double& x : result) {
        x /= sum;
    }
    
    return result;
}

// Works for extreme values: {-1000, 0, 1000}
// Without stability trick: produces NaN/inf

```


### 5. Comparing Floating-Point Numbers


**Never use** **`==`** **for floating-point**:


```c++
// BAD
if (x == y) { ... }  // Almost always wrong

// GOOD: Absolute tolerance
if (std::abs(x - y) < 1e-10) { ... }

// BETTER: Relative tolerance
double relative_error = std::abs(x - y) / std::max(std::abs(x), std::abs(y));
if (relative_error < 1e-10) { ... }

// BEST: Use both absolute and relative
bool approx_equal(double x, double y, double abs_tol = 1e-10, double rel_tol = 1e-10) {
    double diff = std::abs(x - y);
    return diff < abs_tol || diff < rel_tol * std::max(std::abs(x), std::abs(y));
}

```


### 6. Overflow and Underflow


**Detecting and handling**:


```c++
#include <limits>
#include <cmath>

// Check for special values
bool is_finite(double x) {
    return std::isfinite(x);  // Not inf or NaN
}

bool has_overflow(double x) {
    return std::isinf(x);
}

bool has_nan(double x) {
    return std::isnan(x);
}

// Safe multiplication
double safe_multiply(double a, double b) {
    double result = a * b;
    if (!std::isfinite(result)) {
        throw std::overflow_error("Multiplication overflow");
    }
    return result;
}

```


---


## Common Pitfalls


| Pitfall                      | Why It's Bad                       | Fix                            |

| ---------------------------- | ---------------------------------- | ------------------------------ |

| `E[X²] - E[X]²` for variance | Catastrophic cancellation          | Use Welford's algorithm        |

| Summing in arbitrary order   | Accumulated rounding errors        | Use Kahan summation            |

| `exp(large_value)`           | Overflow to infinity               | Use log-space computation      |

| `x == y` for floats          | Rounding makes exact equality rare | Use tolerance-based comparison |


---


## Self-Assessment Checklist


Before moving to Day 13, verify you can:

- [ ] Explain catastrophic cancellation with an example
- [ ] Implement Welford's algorithm for variance
- [ ] Implement Kahan summation
- [ ] Implement log-sum-exp without overflow
- [ ] Compare floating-point numbers with appropriate tolerance

---


## Resources

- [What Every Computer Scientist Should Know About Floating-Point](99)
- [Numerically Stable Algorithms](100)
- [Kahan Summation](101)
- [Log-Sum-Exp Trick](102)

## Day 13: Performance Reasoning and Optimization

## Overview


**Focus**: Learn to reason about performance, measure it accurately, and optimize where it matters. Understand cache behavior, data layout, and compiler optimizations.


**Why it matters**: Performance optimization is often the reason for using C++. Knowing when and how to optimize separates proficient C++ programmers from novices.


---


## Learning Objectives


By the end of Day 13, you will:

- Understand the memory hierarchy (cache, RAM, disk)
- Reason about cache-friendly vs cache-hostile code
- Use profiling tools to identify bottlenecks
- Apply Amdahl's law to predict speedups
- Benchmark code correctly

---


## Core Concepts


### 1. Memory Hierarchy


**The reality**: Memory speed varies by 1000x


| Level    | Latency | Size    |

| -------- | ------- | ------- |

| L1 cache | ~1 ns   | ~32 KB  |

| L2 cache | ~3 ns   | ~256 KB |

| L3 cache | ~10 ns  | ~8 MB   |

| RAM      | ~100 ns | ~16 GB  |

| Disk     | ~10 ms  | ~1 TB   |


**Cache misses dominate** performance for memory-bound code.


### 2. Data Layout Matters


**Row-major vs column-major access**:


```c++
// Row-major (C/C++ default): data[row][col]
// Accessing data[i][j+1] is cache-friendly (adjacent in memory)
// Accessing data[i+1][j] may miss cache (stride-n access)

// Matrix multiplication: ijk vs ikj order
void matmul_ijk(const Matrix& A, const Matrix& B, Matrix& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];  // B strides by row (cache miss)
            }
            C[i][j] = sum;
        }
    }
}

// Better: ikj order
void matmul_ikj(const Matrix& A, const Matrix& B, Matrix& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += a_ik * B[k][j];  // B[k] row accessed sequentially!
            }
        }
    }
}

// Typical speedup: 2-5x faster due to better cache usage

```


**Contiguous vs scattered memory**:


```c++
// vector: contiguous memory → prefetcher works, cache lines reused
std::vector<int> vec(1000000);
for (int x : vec) sum += x;  // Fast: ~1ms

// list: nodes scattered → every access is cache miss
std::list<int> lst;
for (int i = 0; i < 1000000; i++) lst.push_back(i);
for (int x : lst) sum += x;  // Slow: ~15ms (15x slower)

```


### 3. Compiler Optimizations


**Common flags**:

- `-O0`: No optimization (default, for debugging)
- `-O1`: Basic optimizations
- `-O2`: Recommended for production (enables most optimizations)
- `-O3`: Aggressive (loop unrolling, vectorization, may increase code size)
- `-march=native`: Enable CPU-specific instructions (AVX, etc.)

**What the compiler does**:

- **Inlining**: Eliminates function call overhead
- **Loop unrolling**: Reduces loop overhead, increases instruction-level parallelism
- **Vectorization**: Uses SIMD instructions (AVX, SSE)
- **Constant propagation**: Computes constants at compile time
- **Dead code elimination**: Removes unused code

**Example: Manual vs compiler optimization**:


```c++
// Naive sum-of-squares
double sum_squares_naive(const std::vector<double>& v) {
    double sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

// Manual unrolling (4-way)
double sum_squares_unrolled(const std::vector<double>& v) {
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    size_t i = 0;
    for (; i + 3 < v.size(); i += 4) {
        sum0 += v[i] * v[i];
        sum1 += v[i+1] * v[i+1];
        sum2 += v[i+2] * v[i+2];
        sum3 += v[i+3] * v[i+3];
    }
    for (; i < v.size(); i++) sum0 += v[i] * v[i];
    return sum0 + sum1 + sum2 + sum3;
}

// With -O0: unrolled ~2x faster
// With -O3: both nearly identical (compiler auto-unrolls and vectorizes)

```


### 4. Profiling


**Measure before optimizing**: Profile to find bottlenecks.


**Using perf (Linux)**:


```bash

# Compile with debug symbols
g++ -O2 -g -o program program.cpp

# Profile with perf
perf stat -e cache-misses,cache-references ./program

# Profile with sampling
perf record ./program
perf report

```


**Using gprof**:


```bash

# Compile with profiling
g++ -O2 -pg -o program program.cpp

# Run program (generates gmon.out)
./program

# View report
gprof program gmon.out

```


**Cache miss analysis**:


```bash
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./program

# Example output:

# 25,000,000 L1-dcache-loads

#  5,000,000 L1-dcache-load-misses  (20% miss rate - high!)

```


### 5. Benchmarking


**Pitfalls to avoid**:

1. **Cold vs warm caches**: Run warm-up iterations first
2. **Dead code elimination**: Use `volatile` or `benchmark::DoNotOptimize`
3. **Measuring transients**: Measure steady state, not startup
4. **Insufficient repetitions**: Run many iterations for statistical significance

**Proper benchmarking**:


```c++
#include <chrono>
#include <iostream>

template<typename Func>
double benchmark(Func f, int iterations = 1000) {
    // Warm-up
    for (int i = 0; i < 10; i++) f();
    
    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    return duration.count() / iterations;  // Average per iteration
}

// Prevent dead code elimination
volatile double sink;

void use_result(double x) {
    sink = x;
}

// Usage
double avg_time = benchmark([&]() {
    double result = compute_something();
    use_result(result);  // Prevent optimization
});

std::cout << "Average time: " << avg_time << " ms" << std::endl;

```


### 6. Amdahl's Law


**Predicting speedup**: Optimizing fraction `f` by factor `s` yields overall speedup:


```javascript
Speedup = 1 / ((1 - f) + f/s)

```


**Example**: If 90% of time is in one function, and you make it 10x faster:


```javascript
Speedup = 1 / ((1 - 0.9) + 0.9/10) = 1 / 0.19 ≈ 5.3x

```


**Implication**: Optimize hottest functions first. Profile before optimizing!


### 7. Optimization Strategies


**When to optimize**:

1. **Profile first**: Find the actual bottleneck
2. **Optimize the right thing**: 80% of time in 20% of code
3. **Measure impact**: Verify optimization actually helps

**Common optimizations**:

1. **Vectorize loops**: Replace Python-style loops with operations on containers
2. **Use appropriate data structures**: vector vs list vs map
3. **Reserve capacity**: `v.reserve(n)` before pushing n elements
4. **Move instead of copy**: Use `std::move` for large objects
5. **Inline small functions**: Compiler usually does this at -O2
6. **Loop tiling/blocking**: Keep working set in cache
7. **SIMD intrinsics**: Manual vectorization for critical loops

---


## Common Pitfalls


| Pitfall                                | Why It's Bad                           | Fix                                      |

| -------------------------------------- | -------------------------------------- | ---------------------------------------- |

| Premature optimization                 | Wastes time optimizing non-bottlenecks | Profile first                            |

| Optimizing without measuring           | Don't know if it helped                | Benchmark before and after               |

| Cache-hostile data structures          | 10-100x slower than cache-friendly     | Use contiguous memory when possible      |

| Forgetting compiler optimization flags | Leaving 5-10x performance on table     | Always use `-O2` or `-O3` for production |


---


## Self-Assessment Checklist


Before moving to Day 14, verify you can:

- [ ] Explain the memory hierarchy and why cache matters
- [ ] Identify cache-friendly vs cache-hostile code patterns
- [ ] Use a profiler to find the slowest function
- [ ] Benchmark code correctly (warm-up, prevent dead code elimination)
- [ ] Apply Amdahl's law to predict optimization impact

---


## Resources

- [What Every Programmer Should Know About Memory](103)
- [Compiler Explorer](104)
- [perf Tutorial](105)
- [Optimization Manual](106)

## Day 14: C++ Capstone and Cross-Language Integration

## Overview


**Focus**: Complete the C++ capstone project and learn to integrate C++ with Python using pybind11. Understand when to use C++ vs Python.


**Why it matters**: The capstone integrates all Week 2 concepts. Cross-language integration enables using C++ for performance-critical code while keeping Python for productivity.


---


## Learning Objectives


By the end of Day 14, you will:

- Complete a comprehensive C++ capstone project
- Use pybind11 to expose C++ functions to Python
- Pass NumPy arrays between Python and C++
- Understand when to use C++ vs Python
- Demonstrate proficiency in all Week 2 skills

---


## Core Concepts


### 1. Python vs C++: When to Use Each


**Python strengths**:

- Fast development (prototyping, exploration)
- Rich ecosystem (libraries, notebooks)
- I/O-bound tasks
- Already fast enough (NumPy is compiled!)

**C++ strengths**:

- Compute-bound inner loops
- Real-time constraints
- Custom data structures
- Memory-constrained environments

**Typical speedups**: 10-100x for compute-bound code, but <2x if NumPy already vectorized.


### 2. Cross-Language Integration with pybind11


**Installing pybind11**:


```bash
pip install pybind11

```


**Basic example**:


```c++
// example.cpp
#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "Example module";
    m.def("add", &add, "Add two numbers");
}

```


**Build**:


```bash
c++ -O3 -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    example.cpp -o example$(python3-config --extension-suffix)

```


**Use from Python**:


```python
import example
print(example.add(1, 2))  # 3

```


### 3. Passing NumPy Arrays


**C++ side**:


```c++
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double sum_array(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.size;
    
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += ptr[i];
    }
    return sum;
}

PYBIND11_MODULE(nparray, m) {
    m.def("sum_array", &sum_array);
}

```


**Python side**:


```python
import numpy as np
import nparray

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = nparray.sum_array(arr)  # 15.0

```


**Zero-copy**: NumPy arrays are passed without copying data (shared memory).


### 4. Performance Comparison


**Example: Pairwise distances**


```python

# Python with loops (slow)
def pairwise_loop(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
    return D

# NumPy vectorized (fast)
def pairwise_numpy(X):
    X_sq = (X**2).sum(axis=1, keepdims=True)
    D_sq = X_sq + X_sq.T - 2 * X @ X.T
    return np.sqrt(np.maximum(D_sq, 0))

# C++ implementation (fastest)
// C++ code (omitted for brevity - see curriculum)

```


**Benchmark** (n=10000, d=100):


| Version      | Time | Speedup |

| ------------ | ---- | ------- |

| Python loops | 180s | 1x      |

| NumPy        | 1.2s | 150x    |

| C++ (-O3)    | 0.8s | 225x    |


**Lesson**: NumPy gets you 90% of the way. C++ gives another 1.5x for compute-bound code.


### 5. When C++ is Worth It


**Use C++ when**:

- Tight loops that can't vectorize
- Complex per-sample logic
- Memory-constrained (can't allocate n_samples array)
- Stateful simulations (Markov chains, particle systems)

**Stay in Python when**:

- Code is already fast enough
- NumPy can vectorize the operation
- Development time matters more than runtime
- I/O is the bottleneck

---


## Week 2 C++ Capstone


### Project: Gaussian Mixture Model (GMM) Fitting Library


**Task**: Implement a Gaussian Mixture Model fitting library in C++ that demonstrates all skills from Days 8-14.


**Expected Time (Proficient): 4–6 hours**


**Note on Timeline**: This capstone integrates 7 days of material across 7 rubric dimensions. The extended timeline reflects realistic implementation complexity. Consider breaking into two sessions:

- **Session 1 (2-3 hrs)**: Data structures (Matrix/Vector), core EM algorithm, unit tests
- **Session 2 (2-3 hrs)**: Numerical stability, performance optimization, pybind11 binding, sklearn validation

**Requirements**:

1. **Data structures** (Day 8-9): Implement `Matrix` and `Vector` classes with appropriate constructors, destructors, and copy/move semantics. Use RAII for memory management.
2. **Core algorithms** (Day 10-11): Implement E-step (compute responsibilities), M-step (update parameters), and log-likelihood computation using STL containers and algorithms.
3. **Numerical stability** (Day 12): Handle log-sum-exp for likelihoods, ensure positive-definite covariances.
4. **Performance** (Day 13): Cache-friendly memory layout, profile E-step and M-step separately.
5. **Python binding** (Day 14): Expose via pybind11, accept/return NumPy arrays.
6. **Testing**: Unit tests for each component, verify against sklearn.

**Success Criteria**:

- All tests pass
- No memory errors detected by AddressSanitizer
- Fits 10,000 points in 20 dimensions with 5 components in <1 second
- **Results match** **`sklearn.mixture.GaussianMixture`** **within tolerance of ±1e-4 (absolute error) for means and covariances, ±1e-3 for log-likelihood**
- Python binding works and accepts/returns NumPy arrays
- A written document (500+ words) compares C++ vs Python implementation

**Validation Test Example**:


```python
import numpy as np
from sklearn.mixture import GaussianMixture
import your_gmm_module

# Generate test data with known structure
rng = np.random.default_rng(42)
X = rng.normal(0, 1, (1000, 10))

# Fit sklearn reference implementation
sklearn_gmm = GaussianMixture(n_components=3, random_state=42, max_iter=100)
sklearn_gmm.fit(X)

# Fit your C++ implementation
your_gmm = your_gmm_module.GMM(n_components=3, seed=42, max_iter=100)
your_gmm.fit(X)

# Validate means match
for i in range(3):
    np.testing.assert_allclose(
        your_gmm.means[i], sklearn_gmm.means_[i],
        atol=1e-4, err_msg=f"Mean {i} mismatch"
    )

# Validate log-likelihood matches
sklearn_ll = sklearn_gmm.score(X) * len(X)  # score returns per-sample avg
your_ll = your_gmm.log_likelihood(X)
np.testing.assert_allclose(
    your_ll, sklearn_ll, atol=1e-3,
    err_msg="Log-likelihood mismatch"
)

print("✓ All validations passed!")

```


**Cumulative Skills Checklist**:


This capstone must demonstrate ALL skills from Days 8-14:


**From Days 8-9**:

- [ ] Correct use of references for read-only parameters
- [ ] Pointers used only where necessary
- [ ] No raw pointer ownership (use smart pointers)
- [ ] Understanding of stack vs heap allocation

**From Day 10**:

- [ ] All resource-owning classes use smart pointers
- [ ] No memory leaks (verify with AddressSanitizer)

**From Day 11**:

- [ ] `std::vector` used for dynamic arrays
- [ ] STL algorithms used where appropriate
- [ ] Iterators used correctly

**From Day 12**:

- [ ] Log-sum-exp trick implemented
- [ ] Covariances enforced positive semi-definite
- [ ] No numerical overflow/underflow

**From Day 13**:

- [ ] Cache-friendly memory access
- [ ] Profiled with timing measurements
- [ ] Benchmark shows <1s for target dataset

**From Day 14**:

- [ ] pybind11 binding exposes `fit()`, `predict()`, `score()`
- [ ] Accepts NumPy arrays as input
- [ ] Returns NumPy arrays as output
- [ ] Validated against sklearn

**Passing Threshold**: Complete ≥85% of checklist items AND score ≥1.5/2.0 on each rubric dimension.


**Proficiency Rubric**:


| Dimension           | 0                            | 1                                        | 2                                            |

| ------------------- | ---------------------------- | ---------------------------------------- | -------------------------------------------- |

| **Data Structures** | Raw pointers, memory leaks   | RAII but missing move semantics          | Clean Matrix/Vector with copy/move, RAII     |

| **EM Algorithm**    | E-step or M-step incorrect   | Both steps work but numerically unstable | Correct E/M steps with log-sum-exp stability |

| **Testing**         | No tests                     | Some tests but not comprehensive         | Unit tests for each component, all passing   |

| **Performance**     | >10s for 10K points          | <10s but memory errors                   | <1s, no ASan errors                          |

| **Python Binding**  | Not implemented              | Binds but crashes or wrong types         | Clean pybind11, accepts/returns NumPy arrays |

| **sklearn Match**   | Results differ significantly | Close but not within tolerance           | Matches sklearn within 1e-4                  |

| **Documentation**   | No document                  | <500 words or superficial                | 500+ words comparing C++ vs Python           |


**Scoring**: You must score ≥1.5/2.0 on EACH dimension.


---


## Common Pitfalls


| Pitfall                    | Why It's Bad                          | Fix                                               |

| -------------------------- | ------------------------------------- | ------------------------------------------------- |

| Using C++ when NumPy works | Wasted development time               | Profile first, optimize only if needed            |

| Not verifying results      | C++ bug may go unnoticed              | Always validate against known-good implementation |

| Memory leaks in binding    | Python can't detect C++ leaks         | Run AddressSanitizer on C++ tests                 |

| Ignoring exceptions        | C++ exceptions must convert to Python | Use pybind11's automatic exception translation    |


---


## Self-Assessment Checklist


After completing Day 14, verify you can:

- [ ] Use pybind11 to expose C++ functions to Python
- [ ] Pass NumPy arrays to C++ without copying
- [ ] Explain when C++ is worth the effort vs staying in Python
- [ ] Profile and benchmark C++ vs Python implementations
- [ ] **Complete the Week 2 C++ capstone with proficiency score ≥1.5/2.0 on each dimension**

---


## Resources

- [pybind11 Documentation](107)
- [NumPy C API](108)
- [C++ Performance Tips](109)
- [sklearn GMM Documentation](110)

---


## Learning Path


**Sequential progression**: Each day builds on concepts from previous days. Complete them in order.


**Time commitment**: Plan 4-5 hours per day (includes reading, exercises, and practice).


**Proficiency gate**: Day 14 includes a capstone project. You must score ≥1.5/2.0 on each rubric dimension to claim Week 2 proficiency.


---


## Key Themes


### Memory Management

- **References vs Pointers**: Understand when to use each
- **Smart Pointers**: Use `unique_ptr` and `shared_ptr` for automatic cleanup
- **RAII**: Tie resource lifetime to object lifetime
- **No raw** **`new`****/****`delete`**: Modern C++ uses smart pointers and containers

### Standard Template Library

- **Containers**: Choose vector, map, unordered_map appropriately
- **Algorithms**: Replace loops with `std::sort`, `std::transform`, `std::accumulate`
- **Iterators**: Understand categories and invalidation
- **Lambdas**: Write inline functions for algorithms

### Numerical Computing

- **Stability**: Avoid catastrophic cancellation (Welford's algorithm)
- **Overflow**: Use log-space computation (log-sum-exp)
- **Performance**: Understand cache behavior and data layout
- **Profiling**: Measure before optimizing

### Python Integration

- **pybind11**: Expose C++ functions to Python
- **NumPy arrays**: Pass without copying
- **When to use C++**: 10-100x speedup for compute-bound code
- **When to stay in Python**: NumPy already fast enough

---


## Prerequisites


Before starting Week 2, ensure you have:

- [ ] Completed Week 1 (Python proficiency)
- [ ] C++ compiler installed (g++ 9+ or clang 10+)
- [ ] CMake 3.16+ (optional but recommended)
- [ ] Basic understanding of C++ syntax (variables, loops, functions)

---


## Success Criteria


By the end of Week 2, you should be able to:

- Explain the difference between references and pointers
- Use smart pointers to manage heap memory automatically
- Implement RAII wrappers for custom resources
- Use STL containers and algorithms correctly
- Write numerically stable code (Welford, Kahan, log-sum-exp)
- Profile and optimize C++ code
- Integrate C++ with Python via pybind11
- **Complete the capstone project with proficiency scores ≥1.5/2.0**

---


## Daily Self-Assessment


Each day includes a self-assessment checklist. Before moving to the next day, verify you can complete all items. If not, review the exercises and resources.


---


## Getting Help


If you encounter difficulties:

1. **Re-read the relevant section**: Concepts build on each other
2. **Work through exercises**: Hands-on practice is essential
3. **Check resources**: Each day links to official documentation
4. **Use compiler warnings**: Compile with `-Wall -Wextra` to catch common mistakes
5. **Use sanitizers**: Run with `-fsanitize=address,undefined` to catch memory bugs

---


## Next Steps


Ready to begin? Start with <page url="[https://www.notion.so/48da4fa9d9b4495698ebe27a66a92de5">Day](https://www.notion.so/48da4fa9d9b4495698ebe27a66a92de5%22%3EDay) 8: References, Pointers, and Ownership</page>.


# Optional Extensions


## Week 3: Optional Extensions (Days 15-20)

**Framing**: Week 3 is optional. The 2-week curriculum achieves credible proficiency. Week 3 is for those who want to deepen judgment, consolidate under time pressure, and polish a portfolio artifact.


## How Week 3 Relates to Algorithmic Thinking


Algorithmic thinking is **already required** in Weeks 1 and 2. The exercises in Days 1–7 (Python) and Days 8–14 (C++) demand that you:

- Translate mathematical concepts into numerically stable code
- Choose appropriate data structures and algorithms
- Reason about performance, memory, and correctness
- Profile before optimizing and validate after changes

Week 3 does **not** introduce new algorithmic content. Instead, it consolidates these skills under **realistic pressure**:

- Tighter time constraints (simulating production deadlines)
- Integrated multi-language tasks (Python ↔ C++ boundaries)
- Portfolio-quality deliverables (code that you would defend in a technical interview)

Week 3 is **not required for proficiency**. It is for deepening judgment and building confidence through repetition and polish.


---


## Day 15: Advanced Python Engineering for Data Science


### Algorithmic Anchors


This day builds on:

- **Performance workflow** (Algorithmic Thinking: hypothesis → profile → optimize → validate)
- **Python vs Numba vs C++ tradeoffs** (when to stay in Python, when to drop to compiled code)
- **Memory vs speed tradeoffs** (choosing data structures based on access patterns)
- **Pure functions and dependency injection** (from Day 5: testable, reusable code)

### Objectives

- Structure medium-scale Python projects for data science (not research notebooks, not production services—something in between)
- Use type hints and static analysis (`mypy`) to catch errors before runtime
- Implement logging and configuration management for reproducible experiments
- Design clean APIs for statistical functions that others can use

### Topics


**Python Project Structure**:

- Separating source code (`src/`), tests (`tests/`), scripts (`scripts/`), and notebooks (`notebooks/`)
- `pyproject.toml` for dependencies and metadata
- Entry points and command-line interfaces with `argparse` or `click`

**Type Hints and Static Analysis**:

- Beyond basic types: `Union`, `Optional`, `Literal`, `TypeVar` for generic functions
- `numpy.typing` for array shapes and dtypes
- Running `mypy --strict` and interpreting errors
- When to use `# type: ignore` and when it's a code smell

**Logging for Experiments**:

- Structured logging with `logging` module (not print statements)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Logging RNG seeds, hyperparameters, and provenance information
- Rotating log files and log aggregation

**Configuration Management**:

- Separating code from configuration (YAML, TOML, or dataclasses)
- Validation with `pydantic` or `dataclasses` with type checking
- Handling environment-specific configs (dev, test, prod)

### Exercises


**Foundational 1**: Take an existing analysis script (provided) and refactor it into a Python package with `src/`, `tests/`, and `pyproject.toml`. Add type hints to all public functions. Run `mypy --strict` and fix all errors.


**Expected Time (Proficient): 25–35 minutes**


---


**Proficiency 1**: Implement a configurable Monte Carlo simulation where all parameters (n_samples, seed, output_path) come from a YAML config file. Add structured logging that records: config hash, start time, end time, and summary statistics. Write a test that verifies the config is validated correctly.


**Expected Time (Proficient): 30–40 minutes**


**Algorithmic focus**: Reproducibility via configuration; dependency injection for RNG


---


**Mastery**: Design a "statistical function registry" where users can register custom estimators by decorating them with `@estimator.register`. The registry should validate that functions have the correct signature (take `data` and `rng`, return a scalar or array). Implement type-checking with `Protocol` and write tests using Hypothesis to verify the registry works with arbitrary valid functions.


**Expected Time (Proficient): 40–60 minutes**


**Algorithmic focus**: Pure functions as reusable components; protocol-based polymorphism


### Deliverable


A refactored Python package (from Foundational 1) that passes `mypy --strict`, has ≥80% test coverage, includes a working CLI, and has a README with usage examples.


---


## Day 16: Profiling, Benchmarking, and Performance Tuning


### Algorithmic Anchors


This day builds on:

- **Performance workflow** (Algorithmic Thinking: always profile before optimizing)
- **Vectorize vs loop** and **pandas vs NumPy** tradeoffs (quantifying with measurements)
- **Amdahl's Law reasoning** (focus on largest bottlenecks first)
- **Memory vs speed** (profiling memory alongside time)

### Objectives

- Conduct systematic performance profiling (CPU and memory)
- Interpret profiling output to identify true bottlenecks (not guesses)
- Apply targeted optimizations and validate speedups
- Write microbenchmarks that measure specific operations in isolation
- Understand when optimization is premature vs necessary

### Topics


**Profiling Tools Deep Dive**:

- `cProfile` + `pstats` for function-level profiling
- `line_profiler` for line-by-line CPU profiling
- `memory_profiler` for memory usage over time
- `py-spy` for sampling profiler (low overhead, can attach to running process)
- `viztracer` for timeline visualization

**Microbenchmarking**:

- Using `timeit` correctly (warming up, sufficient iterations)
- `pyperf` for robust benchmarking (handles system noise)
- Comparing alternatives with statistical confidence
- Avoiding common pitfalls (measuring setup time, optimizer interference)

**Optimization Patterns**:

- Loop fusion (combining multiple passes into one)
- Vectorization (replacing Python loops with NumPy operations)
- Caching/memoization (when to precompute vs compute-on-the-fly)
- Algorithmic improvements (O(n²) → O(n log n) is better than micro-optimizations)

**Memory Optimization**:

- Identifying memory leaks (retained references)
- Using generators instead of lists for streaming data
- `__slots__` for memory-efficient classes
- Memory mapping for large datasets (`np.memmap`)

### Exercises


**Foundational 1**: Profile a provided data processing pipeline with `cProfile` and `memory_profiler`. Generate a report identifying the top 3 bottlenecks by time and the top 2 by memory. For each, state whether optimization is worthwhile (Amdahl's Law).


**Expected Time (Proficient): 20–30 minutes**


---


**Proficiency 1**: Implement three versions of pairwise distance computation: (1) Python loops, (2) NumPy broadcasting, (3) `scipy.spatial.distance.cdist`. Benchmark all three using `pyperf` with confidence intervals. Write a report explaining when each approach is appropriate.


**Expected Time (Proficient): 30–40 minutes**


**Algorithmic focus**: Measuring the vectorization vs loop tradeoff empirically


---


**Mastery**: Take a grouped aggregation operation in pandas that uses `.apply()` (provided). Profile it, identify the bottleneck, rewrite using vectorized pandas operations or NumPy on `.values`. Achieve ≥10x speedup. Write a before/after profiling report with `line_profiler` output showing the eliminated bottleneck.


**Expected Time (Proficient): 45–60 minutes**


**Algorithmic focus**: Escaping to NumPy for tight loops; pandas overhead quantified


### Deliverable


A profiling report (Markdown or PDF) with annotated profiler output, optimization decisions justified by Amdahl's Law, and before/after benchmarks showing measured speedups.


---


## Day 17: Python ↔ C++ Boundaries and API Design


### Algorithmic Anchors


This day builds on:

- **Python vs Numba vs C++ tradeoffs** (Algorithmic Thinking: when to cross the language boundary)
- **Data representation and memory intuition** (understanding memory layout across languages)
- **Pure functions** (easier to wrap across languages than stateful code)

### Objectives

- Wrap C++ functions for use in Python via `pybind11`
- Design APIs that minimize data copying across the language boundary
- Handle NumPy arrays in C++ using `Eigen` or raw pointers
- Understand when Python ↔ C++ overhead dominates vs when it's negligible
- Write hybrid codebases where Python orchestrates and C++ computes

### Topics


**pybind11 Basics**:

- Wrapping simple C++ functions
- Handling argument conversion (Python types ↔ C++ types)
- Error handling: C++ exceptions → Python exceptions
- Building with `CMake` or `setuptools`

**NumPy ↔ C++ Integration**:

- Using `py::array_t<double>` to accept NumPy arrays
- Zero-copy access via `.data()` and `.mutable_data()`
- Shape and stride validation
- Returning NumPy arrays from C++ without copying

**API Design for Hybrid Code**:

- Keep Python for orchestration, configuration, and I/O
- Drop to C++ for tight numerical loops
- Batch operations to amortize call overhead
- Avoid chatty interfaces (many small calls vs few large calls)

**Performance Considerations**:

- Call overhead: ~1–10 µs per Python → C++ call
- When overhead matters: tight loops, small arrays
- When it doesn't: batch processing, large arrays

### Exercises


**Foundational 1**: Wrap a simple C++ function `double compute_mean(const double* data, size_t n)` using `pybind11`. Make it accept NumPy arrays from Python. Write a Python test that verifies it produces the same result as `np.mean()`.


**Expected Time (Proficient): 25–35 minutes**


---


**Proficiency 1**: Implement a rolling window operation in C++ that accepts a NumPy array and window size, returns a 2D NumPy array of windows (zero-copy view via stride manipulation). Wrap with `pybind11`. Benchmark against pure Python implementation.


**Expected Time (Proficient): 35–50 minutes**


**Algorithmic focus**: Memory layout and stride manipulation across languages


---


**Mastery**: Design a hybrid bootstrap implementation: Python generates seeds and aggregates results; C++ performs the resampling and statistic computation. Minimize data transfer. Benchmark against pure Python and pure C++ versions. Write a short design doc explaining the API choices.


**Expected Time (Proficient): 50–70 minutes**


**Algorithmic focus**: Batching to amortize boundary-crossing overhead


### Deliverable


A working Python package with C++ extension (via `pybind11`) that includes: compiled `.so`/`.pyd`, Python wrapper, tests comparing Python and C++ implementations, and a benchmark report.


---


## Day 18: Numerical Robustness, Validation, and Stress Testing


### Algorithmic Anchors


This day builds on:

- **Numerical robustness and stability** (Algorithmic Thinking: overflow, cancellation, conditioning)
- **Invariants and correctness conditions** (what must always be true)
- **Property-based testing** (from Day 5: Hypothesis for stress testing)

### Objectives

- Identify numerical instabilities in statistical algorithms
- Implement numerically stable alternatives
- Write tests that probe edge cases (overflow, underflow, cancellation)
- Use property-based testing to find numerical bugs
- Validate implementations against high-precision arithmetic or analytical solutions

### Topics


**Common Numerical Issues**:

- **Overflow/underflow**: log-space arithmetic for products of small numbers
- **Catastrophic cancellation**: $(a + b) - a \neq b$ in floating point
- **Loss of significance**: $\sqrt{1 + x} - 1$ for small $x$
- **Ill-conditioning**: matrix inversion, condition numbers

**Stable Implementations**:

- Variance: two-pass or Welford's algorithm (never $E[X^2] - E[X]^2$)
- Softmax: subtract max before exp
- Log-sum-exp: $m + \log(\sum \exp(x - m))$ where $m = \max(x)$
- Compensated summation: Kahan's algorithm

**Validation Strategies**:

- Compare with high-precision arithmetic (`mpmath`)
- Analytical solutions for simple cases
- Check invariants: probabilities sum to 1, covariance matrices are PSD
- Residual checks: $\|Ax - b\|$ for linear systems

**Stress Testing with Hypothesis**:

- Generate extreme inputs: very large, very small, nearly-equal values
- Test invariants across wide input ranges
- Shrinking: Hypothesis finds minimal failing examples

### Exercises


**Foundational 1**: Implement naive and stable versions of sample variance. Test both with `[1e9, 1e9 + 1, 1e9 + 2]`. Show that naive version fails (negative variance or large error). Verify stable version with Hypothesis over wide input ranges.


**Expected Time (Proficient): 20–30 minutes**


---


**Proficiency 1**: Implement log-sum-exp and softmax with numerical stability. Write Hypothesis tests that verify: (1) no overflow for inputs up to 1000, (2) softmax output sums to 1.0 within machine precision, (3) softmax preserves relative ordering.


**Expected Time (Proficient): 30–40 minutes**


**Algorithmic focus**: Numerical stability via log-space arithmetic


---


**Mastery**: Implement a numerically stable version of the multivariate normal log-likelihood using Cholesky decomposition (avoid explicit matrix inversion). Validate against `scipy.stats.multivariate_normal` for well-conditioned cases. Write Hypothesis tests that check invariants for ill-conditioned covariance matrices (condition number up to 1e10).


**Expected Time (Proficient): 50–70 minutes**


**Algorithmic focus**: Ill-conditioning and stable decompositions


### Deliverable


A test suite with Hypothesis-based stress tests for a statistical function library. Include a report documenting at least one numerical bug found by Hypothesis and how it was fixed.


---


## Day 19: C++ Numerical Patterns and RAII


### Algorithmic Anchors


This day builds on:

- **Data representation and memory intuition** (C++ memory model, stack vs heap)
- **Ownership** (from Week 2 Day 9: who is responsible for cleanup)
- **Linear algebra routines** (Algorithmic Thinking: using decompositions, avoiding temporaries)

### Objectives

- Use RAII (Resource Acquisition Is Initialization) for automatic memory management
- Apply `const` correctness to prevent bugs and enable optimizations
- Use `Eigen` expression templates to avoid temporary allocations
- Implement move semantics for efficient data structures
- Write modern C++ (C++17/20) for numerical computing

### Topics


**RAII and Smart Pointers**:

- `std::unique_ptr` for exclusive ownership
- `std::shared_ptr` only when needed (reference counting has overhead)
- Custom deleters for non-memory resources (file handles, mutexes)
- Avoiding `new`/`delete` in user code

**Const Correctness**:

- `const` member functions (don't modify state)
- `const` references for read-only parameters
- `const` enables compiler optimizations and prevents bugs
- `mutable` for logically-const but physically-mutable state (caching)

**Eigen Best Practices**:

- Expression templates: `auto` vs explicit types
- Avoiding aliasing: `.noalias()` for assignment
- Block operations for cache efficiency
- Choosing decompositions: QR, LU, Cholesky, SVD

**Move Semantics**:

- Rvalue references and `std::move`
- Moving large objects instead of copying
- Rule of Five (or Rule of Zero with smart pointers)

### Exercises


**Foundational 1**: Refactor a provided C++ class that uses raw pointers (`new`/`delete`) to use `std::unique_ptr`. Verify with AddressSanitizer that there are no leaks. Add `const` correctness to all member functions.


**Expected Time (Proficient): 25–35 minutes**


---


**Proficiency 1**: Implement a matrix class wrapper around `Eigen::MatrixXd` that uses move semantics for efficient temporaries. Write benchmarks showing that moving is O(1) while copying is O(n²). Add `const` member functions for read-only operations.


**Expected Time (Proficient): 35–50 minutes**


**Algorithmic focus**: Ownership and move semantics as performance constraint


---


**Mastery**: Implement QR decomposition-based least squares solver using Eigen. Compare against naive $(X^T X)^{-1} X^T y$ for ill-conditioned matrices (condition number up to 1e10). Show that QR is stable while naive method fails. Write unit tests with known analytical solutions.


**Expected Time (Proficient): 50–70 minutes**


**Algorithmic focus**: Numerical stability via decompositions; avoiding explicit inversion


### Deliverable


A C++ library with RAII-based resource management, `const` correctness, and Eigen-based numerical routines. Include benchmarks comparing move vs copy, and numerical tests comparing QR vs naive least squares.


---


## Day 20: Compilation, Optimization Flags, and Microbenchmarking


### Algorithmic Anchors


This day builds on:

- **Performance workflow** (Algorithmic Thinking: measure, optimize, validate)
- **C++ compilation** (from Week 2: understanding `-O2`, `-O3`, sanitizers)
- **Profiling** (connecting source code changes to assembly and hardware counters)

### Objectives

- Understand what `-O2` and `-O3` optimizations do (inlining, loop unrolling, vectorization)
- Use compiler flags to enable/disable specific optimizations
- Write microbenchmarks in C++ that measure specific operations
- Interpret `perf` output (cache misses, branch mispredictions)
- Prevent compiler from optimizing away benchmark code

### Topics


**Compiler Optimizations**:

- `-O0`: No optimization (debugging)
- `-O1`: Basic optimizations
- `-O2`: Standard production optimizations
- `-O3`: Aggressive optimizations (may increase code size)
- `-march=native`: CPU-specific instructions (SIMD)
- `-flto`: Link-time optimization

**Vectorization**:

- Auto-vectorization: compiler converts loops to SIMD
- Intrinsics: manual SIMD programming
- Alignment requirements for SIMD (`alignas`, `__attribute__((aligned))`)
- Checking vectorization: `-fopt-info-vec` or Compiler Explorer

**Microbenchmarking in C++**:

- Google Benchmark library
- Preventing optimization with `DoNotOptimize()` and `ClobberMemory()`
- Measuring throughput vs latency
- Handling setup/teardown costs

**Hardware Performance Counters**:

- `perf stat` for hardware metrics
- Cache misses, branch mispredictions, IPC
- Profiling with `perf record` and flamegraphs

### Exercises


**Foundational 1**: Compile a simple dot product function with `-O0`, `-O2`, `-O3`, and `-O3 -march=native`. Benchmark all four versions. Explain the speedup (or lack thereof) based on compiler output or assembly inspection.


**Expected Time (Proficient): 25–35 minutes**


---


**Proficiency 1**: Write a microbenchmark for matrix multiplication comparing: (1) naive triple loop, (2) loop reordering for cache efficiency, (3) Eigen. Measure with Google Benchmark. Use `perf stat` to measure cache misses. Explain the results.


**Expected Time (Proficient): 40–55 minutes**


**Algorithmic focus**: Memory layout and cache efficiency measured empirically


---


**Mastery**: Implement a SIMD-optimized sum function using compiler auto-vectorization (pragmas or intrinsics). Verify vectorization with compiler output. Benchmark against naive loop. Achieve ≥2x speedup on floating-point arrays. Write a report explaining when SIMD helps and when it doesn't.


**Expected Time (Proficient): 60–90 minutes**


**Algorithmic focus**: SIMD vectorization at the hardware level


### Deliverable


A microbenchmark suite with Google Benchmark, showing performance across optimization levels. Include `perf` output explaining cache behavior. Add a README with optimization flag recommendations.


---


## Optional Capstone Extension


**This extension is optional.** The integrated capstone from Weeks 1–2 already demonstrates proficiency. This extension is for those who want a portfolio piece.


### Task


Extend the Week 1 Python capstone (Bayesian A/B testing pipeline) OR the Week 2 C++ capstone with ONE of the following:


**Option A: Performance Upgrade**

- Identify bottleneck in Python capstone via profiling
- Rewrite critical section in C++ and wrap with `pybind11`
- Achieve ≥10x speedup on the bottleneck
- Maintain bitwise reproducibility
- Add benchmarks comparing Python-only vs hybrid

**Option B: Robustness Upgrade**

- Add Hypothesis-based stress tests that probe numerical edge cases
- Identify and fix at least one numerical stability issue
- Add validation against high-precision arithmetic or analytical solutions
- Document the failure modes and fixes

**Option C: API & Polish Upgrade**

- Refactor into a clean package structure with `pyproject.toml`
- Add CLI with `argparse` or `click`
- Full type hints + `mypy --strict` compliance
- README with usage examples, benchmarks, and design decisions
- ≥90% test coverage

### Success Criteria (choose based on option)


**Option A**: Profiling report showing 10x+ speedup on bottleneck; benchmarks; reproducibility tests pass.


**Option B**: Hypothesis finds and reproduces a numerical bug; fix documented; stress tests pass across wide input ranges.


**Option C**: Package installable with `pip install -e .`; CLI functional; `mypy --strict` passes; README publication-quality.


### Rubric (Optional—only for self-assessment)


| Dimension           | Good                            | Excellent                                                          |

| ------------------- | ------------------------------- | ------------------------------------------------------------------ |

| Technical Execution | Meets success criteria          | Exceeds criteria; additional insights or optimizations             |

| Documentation       | README explains what and how    | README explains why; design tradeoffs articulated                  |

| Code Quality        | Clean, readable, passes linters | Idiomatic; could be merged into a production codebase              |

| Rigor               | Tests pass, benchmarks present  | Stress-tested with edge cases; performance validated across inputs |


**Expected Time (Proficient): 3–6 hours across multiple sessions**


---


## Supplementary: Concurrency and Visualization


These topics are not covered in the daily exercises but are essential for production Python work. Review these concepts and complete at least one exercise from each section.


### Concurrency Concepts


Python's Global Interpreter Lock (GIL) prevents true parallel execution of Python bytecode. For CPU-bound work, use `multiprocessing` to spawn separate processes. For I/O-bound work (network, disk), use `asyncio` or `threading`.


`multiprocessing.Pool` provides a simple interface for parallel map operations. Each worker is a separate process with its own memory space and Python interpreter.


`concurrent.futures` provides a unified interface: `ThreadPoolExecutor` for I/O-bound work, `ProcessPoolExecutor` for CPU-bound work.


`asyncio` enables cooperative multitasking. Functions declared with `async def` can `await` other async functions. The event loop schedules coroutines without OS thread overhead.


When to use each:

- `multiprocessing`: CPU-bound work that can be embarrassingly parallel (bootstrap, cross-validation folds, Monte Carlo simulations)
- `threading`: I/O-bound work with shared state (careful with race conditions)
- `asyncio`: High-concurrency I/O (many network requests, database queries)

### Concurrency Exercise


**Exercise**: Implement a function `parallel_bootstrap(data, stat_func, n_bootstrap, n_workers)` that computes bootstrap confidence intervals using `multiprocessing.Pool`. Compare runtime against a sequential implementation for 10,000 bootstrap samples on 100,000 data points. Ensure reproducibility by passing different seeds to each worker.


**Expected Time (Proficient): 20–30 minutes**

<details>
<summary>Solution Sketch</summary>

```python
from multiprocessing import Pool
import numpy as np

def _bootstrap_worker(args):
    data, stat_func, n_samples, seed = args
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_samples):
        sample = rng.choice(data, size=len(data), replace=True)
        results.append(stat_func(sample))
    return results

def parallel_bootstrap(data, stat_func, n_bootstrap, n_workers, base_seed=42):
    samples_per_worker = n_bootstrap // n_workers
    seeds = [base_seed + i for i in range(n_workers)]
    args = [(data, stat_func, samples_per_worker, seed) for seed in seeds]
    
    with Pool(n_workers) as pool:
        results = pool.map(_bootstrap_worker, args)
    
    return np.concatenate(results)

```


</details>


### Visualization Concepts


Matplotlib is the foundation for Python visualization. Understand the object-oriented API: `fig, ax = plt.subplots()` creates Figure and Axes objects. Use `ax.plot()`, `ax.scatter()`, `ax.hist()` methods. Avoid `plt.plot()` in production code—it uses implicit global state.


For statistical visualization, seaborn provides higher-level functions: `sns.histplot()`, `sns.kdeplot()`, `sns.boxplot()`, `sns.pairplot()`. It integrates well with pandas DataFrames and handles grouping automatically.


**Matplotlib architecture**:

- **Figure**: The entire window/page. Contains one or more Axes.
- **Axes**: A single plot with its own coordinate system, labels, title.
- **Artist**: Everything drawn on the figure (lines, text, patches).

**Common statistical plot types**:

- Distribution: `histplot`, `kdeplot`, `ecdfplot`, `rugplot`
- Relationship: `scatterplot`, `lineplot`, `regplot`
- Categorical: `boxplot`, `violinplot`, `stripplot`, `swarmplot`
- Matrix: `heatmap`, `clustermap`

**Publication-quality checklist**:

- [ ] Font sizes readable at target print/display size
- [ ] Axis labels with units
- [ ] Legend positioned to not obscure data
- [ ] Colorblind-accessible palette (use `sns.color_palette('colorblind')`)
- [ ] Figure dimensions match journal/venue requirements
- [ ] Vector format (PDF, SVG) for print; raster (PNG) for web
- [ ] DPI ≥ 300 for print

**Seaborn statistical plots**:


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set publication style
sns.set_theme(style='whitegrid', font_scale=1.2)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Distribution with confidence interval
sns.histplot(data=df, x='value', hue='group', kde=True, stat='density')

# Regression with confidence band
sns.regplot(data=df, x='x', y='y', ci=95, scatter_kws={'alpha': 0.5})

# Faceted plots
g = sns.FacetGrid(df, col='category', row='treatment')
g.map_dataframe(sns.histplot, x='value')

```


**Interactive visualization** (for exploration, not publication):

- Plotly: `import` [`plotly.express`](http://plotly.express/) `as px; px.scatter(df, x='x', y='y', color='group')`
- Altair: Declarative grammar of graphics, good for complex interactions

### Visualization Exercises


**Exercise 1**: Create a function `plot_posterior_comparison(samples_a, samples_b, credible_level=0.95)` that produces a publication-quality figure with: (a) KDE plots of both posteriors on the same axes, (b) shaded credible intervals, (c) a vertical line at zero, (d) proper labels and legend. Save as both PNG and PDF.


**Expected Time (Proficient): 15–25 minutes**

<details>
<summary>Solution Sketch</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_posterior_comparison(samples_a, samples_b, credible_level=0.95):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    alpha = 1 - credible_level
    
    for samples, label, color in [(samples_a, 'Group A', 'C0'), 
                                   (samples_b, 'Group B', 'C1')]:
        kde = stats.gaussian_kde(samples)
        x = np.linspace(samples.min(), samples.max(), 200)
        y = kde(x)
        ax.plot(x, y, color=color, label=label)
        
        # Credible interval
        lo, hi = np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])
        mask = (x >= lo) & (x <= hi)
        ax.fill_between(x[mask], y[mask], alpha=0.3, color=color)
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero')
    ax.set_xlabel('Effect Size', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f'Posterior Distributions ({credible_level*100:.0f}% CI)', fontsize=14)
    
    fig.tight_layout()
    fig.savefig('posterior_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig('posterior_comparison.pdf', bbox_inches='tight')
    return fig, ax

```


</details>


**Exercise 2**: Create a function `plot_regression_diagnostics(y_true, y_pred, feature_names=None)` that produces a 2×2 subplot figure with: (a) predicted vs actual scatter with identity line, (b) residual histogram with normal curve overlay, (c) residuals vs predicted (check for heteroscedasticity), (d) Q-Q plot of residuals. All subplots must have proper labels.


**Expected Time (Proficient): 20–30 minutes**

<details>
<summary>Solution Sketch</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_regression_diagnostics(y_true, y_pred, figsize=(10, 10)):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # (a) Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_pred, y_true, alpha=0.5, s=20)
    lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
    ax.plot(lims, lims, 'r--', label='Perfect fit')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    
    # (b) Residual Histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
            'r-', lw=2, label='Normal')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    
    # (c) Residuals vs Predicted
    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted')
    
    # (d) Q-Q Plot
    ax = axes[1, 1]
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title('Q-Q Plot')
    
    fig.tight_layout()
    return fig, axes

```


</details>


**Exercise 3**: Create a `style_for_journal(journal='nature')` context manager that temporarily sets matplotlib rcParams appropriate for the specified journal. Support at least 'nature' (single column: 89mm, double: 183mm) and 'ieee' (single: 3.5in, double: 7in). Reset to defaults on exit.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Solution Sketch</summary>

```python
import matplotlib.pyplot as plt
from contextlib import contextmanager

JOURNAL_STYLES = {
    'nature': {
        'figure.figsize': (3.5, 2.625),  # 89mm single column
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'font.family': 'sans-serif',
        'savefig.dpi': 300,
    },
    'ieee': {
        'figure.figsize': (3.5, 2.625),  # Single column
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'font.family': 'serif',
        'savefig.dpi': 300,
    }
}

@contextmanager
def style_for_journal(journal='nature'):
    if journal not in JOURNAL_STYLES:
        raise ValueError(f"Unknown journal: {journal}")
    
    # Save current settings
    original = {k: plt.rcParams[k] for k in JOURNAL_STYLES[journal]}
    
    try:
        plt.rcParams.update(JOURNAL_STYLES[journal])
        yield
    finally:
        plt.rcParams.update(original)

# Usage:
with style_for_journal('nature'):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    fig.savefig('figure_for_nature.pdf')

```


</details>


## Day 15: Advanced Python Engineering for Data Science

## Overview


**Focus**: Structure medium-scale Python projects for data science with type hints, static analysis, logging, and configuration management.


**Why it matters**: Production-quality data science code requires more than working notebooks. Learn to build maintainable, testable, and deployable Python packages.


---


## Learning Objectives


By the end of Day 15, you will:

- Structure Python projects with proper directory layout
- Use type hints and `mypy` for static analysis
- Implement structured logging for reproducibility
- Design configuration management for experiments
- Create clean APIs for statistical functions

---


## Core Concepts


### 1. Python Project Structure


**Directory layout**:


```javascript
project/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── scripts/
│   └── run_analysis.py
├── notebooks/
│   └── exploration.ipynb
├── pyproject.toml
├── README.md
└── .gitignore

```


**Why this structure**:

- `src/`: Prevents import confusion, ensures tests use installed package
- `tests/`: Separate from source code
- `scripts/`: Entry points and command-line tools
- `notebooks/`: Exploration only, not production code

**pyproject.toml** (modern Python packaging):


```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "Statistical analysis package"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "scipy>=1.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "mypy>=0.950",
    "black>=22.0",
]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

```


### 2. Type Hints and Static Analysis


**Beyond basic types**:


```python
from typing import Union, Optional, Literal, TypeVar, Protocol
import numpy as np
import numpy.typing as npt

# Generic functions
T = TypeVar('T')
def first(items: list[T]) -> Optional[T]:
    return items[0] if items else None

# NumPy arrays with shape hints
def normalize(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (arr - arr.mean()) / arr.std()

# Literal types for fixed options
def compute_statistic(
    data: npt.NDArray[np.float64],
    method: Literal["mean", "median", "trimmed_mean"]
) -> float:
    if method == "mean":
        return float(np.mean(data))
    elif method == "median":
        return float(np.median(data))
    else:
        return float(np.mean(np.sort(data)[10:-10]))

# Protocol for duck typing
class Estimator(Protocol):
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None: ...
    def predict(self, X: npt.NDArray) -> npt.NDArray: ...

```


**Running mypy**:


```bash

# Install
pip install mypy

# Run strict checks
mypy --strict src/

# Common options
mypy --ignore-missing-imports src/  # Skip import errors
mypy --show-error-codes src/        # Show error codes for ignoring

```


**When to use** **`# type: ignore`**:

- Third-party libraries without type stubs
- Complex NumPy operations mypy can't infer
- **Code smell**: If you need many type ignores, redesign the code

### 3. Logging for Experiments


**Structured logging** (not print statements):


```python
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_dir: Path = Path("logs")) -> logging.Logger:
    """Configure structured logger for experiments."""
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler (less verbose)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format: timestamp - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Usage
logger = setup_logger("experiment")
logger.info("Starting analysis")
logger.debug(f"RNG seed: {seed}")
logger.warning("Missing values detected")
logger.error("Convergence failed")

```


**What to log**:

- Configuration parameters and their source
- Random seeds for reproducibility
- Data provenance (file paths, URLs)
- Performance metrics (time, memory)
- Warnings and errors with context

### 4. Configuration Management


**Separating code from configuration**:


```python
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ExperimentConfig:
    """Configuration for Monte Carlo experiment."""
    n_samples: int
    n_iterations: int
    seed: int
    output_dir: Path
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(
            n_samples=config_dict["n_samples"],
            n_iterations=config_dict["n_iterations"],
            seed=config_dict["seed"],
            output_dir=Path(config_dict["output_dir"]),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

# config.yaml
"""
n_samples: 10000
n_iterations: 1000
seed: 42
output_dir: "results/"
"""

# Usage
config = ExperimentConfig.from_yaml(Path("config.yaml"))
config.validate()
logger.info(f"Config: {config}")

```


**Environment-specific configs**:


```python
import os

def load_config(env: str = "dev") -> ExperimentConfig:
    """Load environment-specific configuration."""
    config_path = Path(f"configs/{env}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return ExperimentConfig.from_yaml(config_path)

# Usage
env = os.getenv("ENV", "dev")  # dev, test, or prod
config = load_config(env)

```


### 5. Command-Line Interfaces


**Using argparse**:


```python
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load config and run
    config = ExperimentConfig.from_yaml(args.config)
    config.output_dir = args.output
    
    logger = setup_logger("analysis")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    run_analysis(config, logger)

if __name__ == "__main__":
    main()

```


---


## Hands-On Exercises


The original curriculum includes exercises for:

- **Foundational 1**: Refactoring a script into a package with proper structure, type hints, and mypy compliance
- **Proficiency 1**: Implementing configurable Monte Carlo simulation with YAML configuration and structured logging
- **Mastery**: Designing a statistical function registry with type-checked protocols

---


## Common Pitfalls


| Pitfall                              | Why It's Bad                  | Fix                                |

| ------------------------------------ | ----------------------------- | ---------------------------------- |

| Mixing notebooks and production code | Hard to test, version control | Use notebooks for exploration only |

| Print statements for logging         | Lost after execution          | Use logging module with levels     |

| Hard-coded parameters                | Not reproducible              | Use configuration files            |

| No type hints                        | Runtime errors                | Add type hints, run mypy           |


---


## Self-Assessment Checklist


Before moving to Day 16, verify you can:

- [ ] Structure a Python project with `src/`, `tests/`, `pyproject.toml`
- [ ] Add type hints to functions and run `mypy --strict`
- [ ] Set up structured logging with appropriate levels
- [ ] Use YAML or dataclasses for configuration management
- [ ] Create a CLI with argparse

---


## Deliverable


A refactored Python package that:

- Passes `mypy --strict`
- Has ≥80% test coverage
- Includes a working CLI
- Has a README with usage examples

---


## Resources

- [Python Packaging Guide](111)
- [mypy Documentation](112)
- [Python Logging HOWTO](113)
- [Hydra (Advanced Config)](114)

## Day 16: Profiling, Benchmarking, and Performance Tuning

## Overview


**Focus**: Conduct systematic performance profiling, interpret profiling output, apply targeted optimizations, and validate speedups with robust benchmarking.


**Why it matters**: "Premature optimization is the root of all evil." Learn to measure first, optimize bottlenecks, and validate improvements.


---


## Learning Objectives


By the end of Day 16, you will:

- Profile CPU and memory usage systematically
- Identify true bottlenecks (not guesses)
- Apply targeted optimizations based on profiling
- Write robust microbenchmarks
- Understand when optimization is necessary vs premature

---


## Core Concepts


### 1. Profiling Tools


**cProfile + pstats** (function-level profiling):


```python
import cProfile
import pstats
from pstats import SortKey

def analyze_with_profiler() -> None:
    # Profile code
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    result = expensive_computation()
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10)  # Top 10 functions

```


**line_profiler** (line-by-line profiling):


```bash

# Install
pip install line_profiler

# Decorate function
@profile
def slow_function():
    # ... code ...
    pass

# Run
kernprof -l -v script.py

```


**memory_profiler** (memory usage over time):


```bash
pip install memory_profiler

# Decorate function
@profile
def memory_hog():
    data = []
    for i in range(1000000):
        data.append([i] * 100)
    return data

# Run
python -m memory_profiler script.py

```


**py-spy** (sampling profiler, low overhead):


```bash
pip install py-spy

# Profile running process
py-spy record -o profile.svg -- python script.py

# Attach to running process
py-spy top --pid 12345

```


### 2. Interpreting Profiler Output


**cProfile output**:


```javascript
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1    0.000    0.000   10.500   10.500 script.py:10(main)
100000    8.500    0.000    8.500    0.000 script.py:20(slow_func)
100000    1.500    0.000    2.000    0.000 numpy:1234(array_op)

```


**Key metrics**:

- **ncalls**: Number of times function called
- **tottime**: Total time in function (excluding subcalls)
- **cumtime**: Total time including subcalls
- **percall**: Average time per call

**What to optimize**: Functions with high `cumtime` and high `ncalls`.


### 3. Amdahl's Law


**Predicting speedup**:


```python
def amdahl_speedup(fraction_optimized: float, speedup_factor: float) -> float:
    """
    Calculate overall speedup from Amdahl's Law.
    
    Args:
        fraction_optimized: Fraction of runtime being optimized (0-1)
        speedup_factor: How much faster the optimized part is
    
    Returns:
        Overall speedup
    """
    return 1.0 / ((1 - fraction_optimized) + fraction_optimized / speedup_factor)

# Example: If 80% of time is in one function, making it 5x faster gives:
overall_speedup = amdahl_speedup(0.8, 5)  # 2.78x
print(f"Overall speedup: {overall_speedup:.2f}x")

# Implication: Optimize the hottest functions first!

```


### 4. Microbenchmarking


**Using timeit correctly**:


```python
import timeit
import numpy as np

# Setup code (run once)
setup = """
import numpy as np
data = np.random.randn(10000)
"""

# Code to benchmark
stmt = "data.sum()"

# Run benchmark
time = timeit.timeit(stmt, setup=setup, number=10000)
print(f"Time per iteration: {time/10000 * 1e6:.2f} µs")

```


**Using pyperf** (handles system noise):


```bash
pip install pyperf

# Run benchmark
python -m pyperf timeit -s "import numpy as np; data = np.random.randn(10000)" "data.sum()"

```


**Preventing optimizer interference**:


```python

# BAD: Compiler may optimize away
result = expensive_function()

# GOOD: Force use of result
import sys
result = expensive_function()
sys.stdout.write(str(result))  # Or use volatile in C++

```


### 5. Optimization Patterns


**Loop fusion** (combining multiple passes):


```python

# BEFORE: Two passes
squared = [x**2 for x in data]
result = [x + 1 for x in squared]

# AFTER: One pass (loop fusion)
result = [x**2 + 1 for x in data]

# Even better: NumPy vectorization
result = np.array(data)**2 + 1

```


**Vectorization** (replacing Python loops):


```python

# SLOW: Python loop
def sum_of_squares_loop(data):
    result = 0
    for x in data:
        result += x**2
    return result

# FAST: NumPy vectorization
def sum_of_squares_vectorized(data):
    return np.sum(data**2)

# Benchmark shows ~100x speedup

```


**Caching/memoization**:


```python
from functools import lru_cache

# SLOW: Recomputes every time
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# FAST: Cached results
@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n < 2:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

# fibonacci(35): ~5 seconds

# fibonacci_cached(35): ~0.0001 seconds

```


### 6. Memory Optimization


**Generators vs lists** (streaming data):


```python

# MEMORY HOG: Builds entire list in memory
def process_large_file(path):
    return [process_line(line) for line in open(path)]

# MEMORY EFFICIENT: Generates values on-demand
def process_large_file_streaming(path):
    return (process_line(line) for line in open(path))

# Usage
for item in process_large_file_streaming("huge.csv"):
    # Process one item at a time
    pass

```


**slots for memory-efficient classes**:


```python

# MEMORY HOG: Each instance has a __dict__
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# MEMORY EFFICIENT: Fixed attributes
class PointSlots:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 1 million Points: ~400 MB

# 1 million PointSlots: ~200 MB

```


**Memory mapping for large datasets**:


```python
import numpy as np

# MEMORY HOG: Loads entire array
data = np.load("large_array.npy")

# MEMORY EFFICIENT: Memory-mapped array
data = np.load("large_array.npy", mmap_mode='r')

# Data loaded on-demand as accessed

```


---


## Optimization Workflow

1. **Profile first**: Measure where time is actually spent
2. **Identify bottleneck**: Focus on functions with highest cumtime
3. **Apply Amdahl's Law**: Is optimization worth it?
4. **Optimize**: Apply appropriate pattern
5. **Benchmark**: Measure speedup
6. **Validate**: Ensure correctness unchanged

---


## Common Pitfalls


| Pitfall                      | Why It's Bad                   | Fix                             |

| ---------------------------- | ------------------------------ | ------------------------------- |

| Optimizing without profiling | Waste time on non-bottlenecks  | Always profile first            |

| Not measuring impact         | Don't know if it helped        | Benchmark before and after      |

| Micro-optimizing hot code    | Algorithmic improvement better | O(n²) → O(n log n) > micro-opts |

| Breaking correctness         | Bugs cost more than slowness   | Test after every optimization   |


---


## Self-Assessment Checklist


Before moving to Day 17, verify you can:

- [ ] Profile code with cProfile and interpret output
- [ ] Use line_profiler to find slow lines
- [ ] Apply Amdahl's Law to predict speedups
- [ ] Write robust microbenchmarks with timeit or pyperf
- [ ] Vectorize Python loops with NumPy

---


## Deliverable


A profiling report (Markdown or PDF) with:

- Annotated profiler output
- Optimization decisions justified by Amdahl's Law
- Before/after benchmarks showing measured speedups

---


## Resources

- [Python Profilers](115)
- [pyperf Documentation](116)
- [Performance Tips](117)
- [Memory Profiling](118)

## Day 17: Python ↔ C++ Boundaries and API Design

## Overview


**Focus**: Wrap C++ functions for Python using pybind11, pass NumPy arrays efficiently, and design hybrid codebases where Python orchestrates and C++ computes.


**Why it matters**: Combine Python's productivity with C++'s performance. Learn when crossing the language boundary is worth the complexity.


---


## Learning Objectives


By the end of Day 17, you will:

- Wrap C++ functions for Python via pybind11
- Pass NumPy arrays to C++ without copying
- Design APIs that minimize boundary-crossing overhead
- Understand when Python ↔ C++ overhead dominates
- Build hybrid codebases effectively

---


## Core Concepts


### 1. pybind11 Basics


**Simple function wrapping**:


```c++
// example.cpp
#include <pybind11/pybind11.h>

double add(double a, double b) {
    return a + b;
}

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    m.doc() = "Example module";
    m.def("add", &add, "Add two numbers",
          py::arg("a"), py::arg("b"));
}

```


**Build with CMake**:


```javascript
cmake_minimum_required(VERSION 3.16)
project(example)

find_package(pybind11 REQUIRED)

pybind11_add_module(example example.cpp)

```


**Or build with setuptools**:


```bash
c++ -O3 -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    example.cpp -o example$(python3-config --extension-suffix)

```


**Use from Python**:


```python
import example
result = example.add(1.0, 2.0)
print(result)  # 3.0

```


### 2. NumPy ↔ C++ Integration


**Accepting NumPy arrays**:


```c++
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double sum_array(py::array_t<double> arr) {
    // Request buffer info
    py::buffer_info buf = arr.request();
    
    // Check dimensions
    if (buf.ndim != 1) {
        throw std::runtime_error("Array must be 1-dimensional");
    }
    
    // Get pointer and size
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    // Compute sum
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += ptr[i];
    }
    
    return sum;
}

PYBIND11_MODULE(nparray, m) {
    m.def("sum_array", &sum_array);
}

```


**Python usage**:


```python
import numpy as np
import nparray

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = nparray.sum_array(arr)
print(result)  # 15.0

# Zero-copy: NumPy array passed by reference

```


**Returning NumPy arrays**:


```c++
py::array_t<double> create_array(size_t n) {
    // Allocate buffer
    auto result = py::array_t<double>(n);
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    // Fill buffer
    for (size_t i = 0; i < n; i++) {
        ptr[i] = static_cast<double>(i);
    }
    
    return result;
}

PYBIND11_MODULE(nparray, m) {
    m.def("create_array", &create_array);
}

```


### 3. Error Handling


**C++ exceptions → Python exceptions**:


```c++
double safe_divide(double a, double b) {
    if (b == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}

// pybind11 automatically converts to Python exceptions

```


**Python side**:


```python
try:
    result = example.safe_divide(10.0, 0.0)
except ValueError as e:  # std::invalid_argument → ValueError
    print(f"Error: {e}")

```


### 4. API Design for Hybrid Code


**Principle: Minimize boundary crossings**


**BAD: Chatty interface** (many small calls):


```python

# Python calls C++ for each element
result = []
for i in range(1000000):
    result.append(cpp_module.process_element(data[i]))  # SLOW!

```


**GOOD: Batched interface** (few large calls):


```python

# Python calls C++ once with entire array
result = cpp_module.process_batch(data)  # FAST!

```


**Design pattern**:

- **Python layer**: Configuration, I/O, orchestration
- **C++ layer**: Tight numerical loops
- **Interface**: Batch operations on arrays

**Example: Bootstrap implementation**


```c++
// C++ side: Process many bootstrap samples at once
py::array_t<double> bootstrap_batch(
    py::array_t<double> data,
    py::array_t<int> indices,  // Flattened: [n_bootstrap * n_samples]
    size_t n_bootstrap
) {
    auto data_buf = data.request();
    auto indices_buf = indices.request();
    
    double* data_ptr = static_cast<double*>(data_buf.ptr);
    int* indices_ptr = static_cast<int*>(indices_buf.ptr);
    
    size_t n_samples = data_buf.shape[0];
    
    // Allocate result
    auto result = py::array_t<double>(n_bootstrap);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Compute means for all bootstrap samples
    for (size_t b = 0; b < n_bootstrap; b++) {
        double sum = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            int idx = indices_ptr[b * n_samples + i];
            sum += data_ptr[idx];
        }
        result_ptr[b] = sum / n_samples;
    }
    
    return result;
}

```


```python

# Python side: Generate seeds, call C++ once
def bootstrap_ci(data, n_bootstrap=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate all indices in Python (cheap)
    n = len(data)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    
    # Process all samples in C++ (expensive computation)
    bootstrap_means = cpp_module.bootstrap_batch(
        data, indices.ravel(), n_bootstrap
    )
    
    # Aggregate in Python (cheap)
    return np.percentile(bootstrap_means, [2.5, 97.5])

```


### 5. Performance Considerations


**Call overhead**: ~1–10 µs per Python → C++ call


**When overhead matters**:

- Tight loops with many calls
- Small arrays (<1000 elements)
- Simple operations (addition, comparison)

**When it doesn't**:

- Batch processing (one call for many operations)
- Large arrays (>10000 elements)
- Complex operations (matrix decompositions)

**Benchmark: Crossing threshold**


```python

# Python loop calling C++ each iteration: SLOW
for i in range(1000000):
    result = cpp.process(data[i])  # 1-10 µs overhead × 1M = 1-10s!

# C++ processes entire array: FAST
result = cpp.process_batch(data)  # 1-10 µs overhead once

```


### 6. Build System Integration


[**setup.py**](http://setup.py/) **with pybind11**:


```python
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'mymodule',
        ['src/mymodule.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name='mypackage',
    ext_modules=ext_modules,
    zip_safe=False,
)

```


**Install in development mode**:


```bash
pip install -e .

```


---


## Design Guidelines

1. **Keep Python for**: Configuration, I/O, visualization, orchestration
2. **Use C++ for**: Tight numerical loops, performance-critical sections
3. **Batch operations**: Minimize boundary crossings
4. **Zero-copy**: Pass NumPy arrays by reference when possible
5. **Error handling**: Let pybind11 convert exceptions automatically

---


## Common Pitfalls


| Pitfall                | Why It's Bad            | Fix                           |

| ---------------------- | ----------------------- | ----------------------------- |

| Chatty interface       | Call overhead dominates | Batch operations              |

| Copying arrays         | Memory + time cost      | Use zero-copy views           |

| No error checking      | Silent crashes          | Validate dimensions and types |

| Building in source dir | Pollutes source tree    | Use build/ directory          |


---


## Self-Assessment Checklist


Before moving to Day 18, verify you can:

- [ ] Wrap a simple C++ function with pybind11
- [ ] Pass NumPy arrays to C++ without copying
- [ ] Return NumPy arrays from C++
- [ ] Design a batched API to minimize overhead
- [ ] Build and install a Python extension module

---


## Deliverable


A working Python package with C++ extension:

- Compiled `.so`/`.pyd` module
- Python wrapper with type hints
- Tests comparing Python and C++ implementations
- Benchmark report showing speedup

---


## Resources

- [pybind11 Documentation](119)
- [NumPy C API](120)
- [Python/C++ Integration Guide](121)
- [Performance Best Practices](122)

## Day 18: Numerical Robustness, Validation, and Stress Testing

## Overview


**Focus**: Validate numerical implementations for correctness, stability, and edge cases using systematic testing strategies and stress tests.


**Why it matters**: Statistical software failures are subtle. Learn to design test suites that catch floating-point errors, boundary conditions, and numerical instability.


---


## Learning Objectives


By the end of Day 18, you will:

- Design unit tests for numerical functions
- Validate against reference implementations (R, SciPy)
- Test edge cases (zeros, infinities, NaNs)
- Stress-test with extreme inputs
- Use property-based testing for invariants

---


## Core Concepts


### 1. Unit Testing for Numerical Code


**Basic structure** (pytest):


```python
import pytest
import numpy as np
from mypackage import compute_mean

def test_mean_simple():
    """Test mean of simple array."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_mean(data)
    expected = 3.0
    assert result == expected

def test_mean_negative():
    """Test mean with negative values."""
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = compute_mean(data)
    expected = 0.0
    assert result == expected

```


**Floating-point comparisons** (use tolerance):


```python
def test_mean_floating_point():
    """Test mean with floating-point arithmetic."""
    data = np.array([0.1, 0.2, 0.3])
    result = compute_mean(data)
    expected = 0.2
    
    # BAD: Exact equality (may fail due to rounding)
    # assert result == expected
    
    # GOOD: Relative tolerance
    np.testing.assert_allclose(result, expected, rtol=1e-10)

```


**Parametrized tests** (test many cases efficiently):


```python
@pytest.mark.parametrize("data,expected", [
    ([1.0, 2.0, 3.0], 2.0),
    ([0.0, 0.0, 0.0], 0.0),
    ([-1.0, 1.0], 0.0),
    ([1e10, 1e10, 1e10], 1e10),
])
def test_mean_parametrized(data, expected):
    """Test mean with various inputs."""
    result = compute_mean(np.array(data))
    np.testing.assert_allclose(result, expected, rtol=1e-10)

```


### 2. Cross-Validation with Reference Implementations


**Comparing to SciPy**:


```python
from scipy import stats
import numpy as np

def test_t_test_against_scipy():
    """Validate t-test implementation against SciPy."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0.5, 1, 100)
    
    # Our implementation
    our_statistic, our_pvalue = my_t_test(x, y)
    
    # SciPy reference
    scipy_result = stats.ttest_ind(x, y)
    
    # Should match within numerical precision
    np.testing.assert_allclose(our_statistic, scipy_result.statistic, rtol=1e-10)
    np.testing.assert_allclose(our_pvalue, scipy_result.pvalue, rtol=1e-10)

```


**Comparing to R** (via rpy2):


```python
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

def test_linear_regression_against_r():
    """Validate linear regression against R's lm()."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.1, 100)
    
    # Our implementation
    our_coef = my_linear_regression(X, y)
    
    # R reference
    ro.globalenv['X'] = X
    ro.globalenv['y'] = y
    r_result = ro.r('lm(y ~ X - 1)')  # No intercept
    r_coef = np.array(ro.r('coef(r_result)'))
    
    np.testing.assert_allclose(our_coef, r_coef, rtol=1e-8)

```


### 3. Edge Case Testing


**Testing special values**:


```python
def test_mean_edge_cases():
    """Test mean with edge cases."""
    
    # Empty array
    with pytest.raises(ValueError):
        compute_mean(np.array([]))
    
    # Single element
    result = compute_mean(np.array([42.0]))
    assert result == 42.0
    
    # All zeros
    result = compute_mean(np.array([0.0, 0.0, 0.0]))
    assert result == 0.0
    
    # Infinity
    result = compute_mean(np.array([np.inf, np.inf]))
    assert result == np.inf
    
    # NaN propagation
    result = compute_mean(np.array([1.0, np.nan, 3.0]))
    assert np.isnan(result)
    
    # Mixed signs
    result = compute_mean(np.array([1e10, -1e10]))
    np.testing.assert_allclose(result, 0.0, atol=1e-5)

```


**Boundary conditions**:


```python
def test_standard_deviation_edge_cases():
    """Test standard deviation edge cases."""
    
    # Zero variance
    result = compute_std(np.array([5.0, 5.0, 5.0]))
    assert result == 0.0
    
    # Very small variance (numerical stability)
    data = np.array([1e10, 1e10 + 1e-5, 1e10 + 2e-5])
    result = compute_std(data)
    assert result > 0  # Should not be exactly zero due to rounding
    
    # Large variance
    data = np.array([1e-10, 1e10])
    result = compute_std(data)
    assert result > 0 and np.isfinite(result)

```


### 4. Stress Testing


**Testing with extreme inputs**:


```python
def test_covariance_matrix_large_scale():
    """Stress test covariance with large matrices."""
    rng = np.random.default_rng(42)
    
    # Large dimensions
    n_samples = 10000
    n_features = 100
    X = rng.normal(0, 1, (n_samples, n_features))
    
    result = compute_covariance(X)
    
    # Validate properties
    assert result.shape == (n_features, n_features)
    assert np.allclose(result, result.T)  # Symmetric
    eigenvalues = np.linalg.eigvals(result)
    assert np.all(eigenvalues >= -1e-10)  # Positive semi-definite

def test_numerical_stability_compensated_sum():
    """Test compensated summation (Kahan algorithm) stability."""
    # Catastrophic cancellation example
    data = np.array([1e10, 1.0, -1e10])
    
    # Naive sum may give 0.0 due to rounding
    naive_result = sum(data)
    
    # Compensated sum should give 1.0
    compensated_result = kahan_sum(data)
    
    assert abs(compensated_result - 1.0) < 1e-10

```


**Testing with ill-conditioned inputs**:


```python
def test_linear_regression_ill_conditioned():
    """Test linear regression with nearly collinear features."""
    rng = np.random.default_rng(42)
    
    # Create nearly collinear features
    X1 = rng.normal(0, 1, 100)
    X2 = X1 + rng.normal(0, 1e-6, 100)  # Almost identical to X1
    X = np.column_stack([X1, X2])
    y = rng.normal(0, 1, 100)
    
    # Should either:
    # 1. Raise a warning about collinearity
    # 2. Use regularization
    # 3. Return finite coefficients (not NaN or inf)
    
    with pytest.warns(UserWarning, match="collinear"):
        coef = my_linear_regression(X, y)
    
    assert np.all(np.isfinite(coef))

```


### 5. Property-Based Testing


**Using Hypothesis**:


```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(npst.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=1000),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
))
def test_mean_properties(data):
    """Test properties that should hold for all inputs."""
    result = compute_mean(data)
    
    # Mean should be between min and max
    assert data.min() <= result <= data.max()
    
    # Mean of constant array is that constant
    if np.all(data == data[0]):
        assert result == data[0]
    
    # Finite input → finite output
    assert np.isfinite(result)

@given(npst.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=100),
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
))
def test_variance_properties(data):
    """Test properties of variance."""
    result = compute_variance(data)
    
    # Variance is non-negative
    assert result >= 0
    
    # Variance is zero iff all elements equal
    if np.allclose(data, data.mean()):
        assert result < 1e-10

```


### 6. Test Coverage


**Measuring coverage** (pytest-cov):


```bash

# Install
pip install pytest-cov

# Run tests with coverage
pytest --cov=mypackage --cov-report=html

# View report
open htmlcov/index.html

```


**Coverage targets**:

- **≥80%**: Minimum for production code
- **≥90%**: Target for critical statistical functions
- **100%**: Unrealistic (diminishing returns)

**What to prioritize**:

- Core numerical algorithms
- Edge case handling
- Error paths

---


## Testing Checklist for Numerical Functions


Before considering a function "tested", verify:

- [ ] Basic correctness with simple inputs
- [ ] Edge cases (empty, single element, zeros)
- [ ] Special values (NaN, inf, -inf)
- [ ] Floating-point precision (use tolerances)
- [ ] Cross-validation with reference implementation
- [ ] Stress test with large/extreme inputs
- [ ] Property-based tests for invariants
- [ ] Test coverage ≥80%

---


## Common Pitfalls


| Pitfall                     | Why It's Bad                  | Fix                                          |

| --------------------------- | ----------------------------- | -------------------------------------------- |

| Exact equality for floats   | Fails due to rounding         | Use `np.testing.assert_allclose`             |

| Not testing edge cases      | Bugs surface in production    | Test empty, zeros, NaN, inf                  |

| No reference validation     | Might be consistently wrong   | Compare to SciPy or R                        |

| Insufficient stress testing | Fails on large/extreme inputs | Test with large arrays, ill-conditioned data |


---


## Self-Assessment Checklist


Before moving to Day 19, verify you can:

- [ ] Write unit tests with pytest
- [ ] Use appropriate floating-point tolerances
- [ ] Validate against SciPy or R implementations
- [ ] Test edge cases systematically
- [ ] Use property-based testing with Hypothesis
- [ ] Achieve ≥80% test coverage

---


## Deliverable


A test suite for your statistical package:

- ≥80% test coverage
- Edge case tests for all public functions
- Cross-validation against reference implementation
- Passing property-based tests

---


## Resources

- [pytest Documentation](123)
- [NumPy Testing Guidelines](124)
- [Hypothesis Documentation](125)
- [Numerical Testing Best Practices](126)

## Day 19: C++ Numerical Patterns and RAII

## Overview


**Focus**: Master C++ resource management with RAII, implement numerical algorithms with proper memory safety, and apply modern C++ patterns.


**Why it matters**: C++ gives control but demands discipline. RAII (Resource Acquisition Is Initialization) ensures leak-free, exception-safe numerical code.


---


## Learning Objectives


By the end of Day 19, you will:

- Implement RAII for automatic resource management
- Use smart pointers (`unique_ptr`, `shared_ptr`) correctly
- Write exception-safe numerical algorithms
- Understand move semantics for efficiency
- Apply modern C++ patterns to statistical code

---


## Core Concepts


### 1. RAII Fundamentals


**Core principle**: Resource lifetime tied to object lifetime


```c++
// BAD: Manual resource management (leak-prone)
double* allocate_array(size_t n) {
    return new double[n];
}

void process() {
    double* data = allocate_array(1000);
    // ... computation ...
    delete[] data;  // Might forget, or exception thrown before this
}

// GOOD: RAII wrapper (automatic cleanup)
class Array {
private:
    double* data_;
    size_t size_;
public:
    Array(size_t n) : data_(new double[n]), size_(n) {}
    ~Array() { delete[] data_; }  // Automatic cleanup
    
    // Prevent copying (or implement correctly)
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    
    // Accessors
    double& operator[](size_t i) { return data_[i]; }
    size_t size() const { return size_; }
};

void process() {
    Array data(1000);
    // ... computation ...
    // Automatic cleanup when data goes out of scope
}

```


### 2. Smart Pointers


**`unique_ptr`** (exclusive ownership):


```c++
#include <memory>
#include <vector>

// Single owner, automatic deletion
std::unique_ptr<double[]> allocate_array(size_t n) {
    return std::make_unique<double[]>(n);
}

void process() {
    auto data = allocate_array(1000);
    // Use data[i] normally
    // Automatic cleanup
}

// Move semantics (transfer ownership)
std::unique_ptr<double[]> source = allocate_array(1000);
std::unique_ptr<double[]> dest = std::move(source);  // source is now nullptr

```


**`shared_ptr`** (shared ownership):


```c++
// Multiple owners, deleted when last owner destroyed
std::shared_ptr<std::vector<double>> create_data() {
    return std::make_shared<std::vector<double>>(1000);
}

void share_data() {
    auto data1 = create_data();
    auto data2 = data1;  // Reference count = 2
    // data deleted when both data1 and data2 go out of scope
}

```


**When to use which**:

- **`unique_ptr`**: Default choice (clear ownership, zero overhead)
- **`shared_ptr`**: Only when genuinely shared ownership needed
- **Raw pointers**: Only for non-owning references

### 3. Exception Safety


**Three guarantees**:

1. **Basic**: Invariants preserved, no leaks
2. **Strong**: Operation succeeds or has no effect (rollback)
3. **No-throw**: Never throws (use `noexcept`)

**Example: Strong exception safety**


```c++
class CovarianceMatrix {
private:
    std::vector<double> data_;
    size_t dim_;
    
public:
    CovarianceMatrix(size_t dim) : data_(dim * dim, 0.0), dim_(dim) {}
    
    // Strong guarantee: either succeeds or unchanged
    void add_observation(const std::vector<double>& obs) {
        if (obs.size() != dim_) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        // Copy state (if exception thrown, original unchanged)
        std::vector<double> new_data = data_;
        
        // Update (might throw)
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                new_data[i * dim_ + j] += obs[i] * obs[j];
            }
        }
        
        // Commit (no-throw operation)
        data_ = std::move(new_data);
    }
};

```


**No-throw guarantee** (for critical operations):


```c++
class Array {
private:
    double* data_;
    size_t size_;
    
public:
    // No-throw swap (enables strong guarantee)
    void swap(Array& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    
    // Strong guarantee via swap
    Array& operator=(Array other) {  // Pass by value (copy)
        swap(other);  // No-throw swap
        return *this;
        // Old data destroyed when 'other' goes out of scope
    }
};

```


### 4. Move Semantics


**Motivation**: Avoid expensive copies


```c++
// BEFORE C++11: Always copies
std::vector<double> create_large_array() {
    std::vector<double> data(1000000);
    // ... fill data ...
    return data;  // Copies entire array
}

// AFTER C++11: Moves (cheap)
std::vector<double> create_large_array() {
    std::vector<double> data(1000000);
    // ... fill data ...
    return data;  // Moves (just pointer swap)
}

```


**Implementing move operations**:


```c++
class Matrix {
private:
    double* data_;
    size_t rows_, cols_;
    
public:
    // Constructor
    Matrix(size_t rows, size_t cols)
        : data_(new double[rows * cols]), rows_(rows), cols_(cols) {}
    
    // Destructor
    ~Matrix() { delete[] data_; }
    
    // Copy constructor (deep copy)
    Matrix(const Matrix& other)
        : data_(new double[other.rows_ * other.cols_]),
          rows_(other.rows_), cols_(other.cols_) {
        std::copy(other.data_, other.data_ + rows_ * cols_, data_);
    }
    
    // Move constructor (transfer ownership)
    Matrix(Matrix&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        other.data_ = nullptr;  // Leave other in valid state
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    // Copy assignment
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Matrix temp(other);  // Copy
            swap(temp);          // Swap (no-throw)
        }
        return *this;
    }
    
    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }
    
    void swap(Matrix& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
    }
};

```


**Rule of Five**: If you define one of these, define all:

1. Destructor
2. Copy constructor
3. Copy assignment
4. Move constructor
5. Move assignment

### 5. Modern C++ Containers for Numerical Code


**Use STL containers** (RAII built-in):


```c++
#include <vector>
#include <array>

// Dynamic size (heap-allocated)
std::vector<double> data(1000);
data.push_back(3.14);

// Fixed size (stack-allocated, no overhead)
std::array<double, 3> point = {1.0, 2.0, 3.0};

// Multi-dimensional (flatten or use Eigen)
std::vector<double> matrix(rows * cols);
auto at = [&](size_t i, size_t j) -> double& {
    return matrix[i * cols + j];
};

```


**Or use Eigen** (expression templates, RAII):


```c++
#include <Eigen/Dense>

Eigen::VectorXd v(1000);
v.setRandom();

Eigen::MatrixXd m(100, 100);
m.setIdentity();

// Automatic memory management, efficient operations
Eigen::VectorXd result = m * v;

```


### 6. Numerical Algorithm Example with RAII


**Bootstrap with proper resource management**:


```c++
#include <vector>
#include <random>
#include <algorithm>

class Bootstrap {
private:
    std::vector<double> data_;
    std::mt19937 rng_;
    
public:
    Bootstrap(std::vector<double> data, unsigned seed = std::random_device{}())
        : data_(std::move(data)), rng_(seed) {}  // Move data (no copy)
    
    std::vector<double> resample(size_t n_bootstrap) {
        std::vector<double> results;
        results.reserve(n_bootstrap);
        
        std::uniform_int_distribution<size_t> dist(0, data_.size() - 1);
        std::vector<double> sample(data_.size());
        
        for (size_t b = 0; b < n_bootstrap; ++b) {
            // Resample
            for (size_t i = 0; i < data_.size(); ++i) {
                sample[i] = data_[dist(rng_)];
            }
            
            // Compute statistic
            double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / sample.size();
            results.push_back(mean);
        }
        
        return results;  // Move (no copy)
    }
    
    std::pair<double, double> confidence_interval(size_t n_bootstrap, double alpha = 0.05) {
        auto bootstrap_means = resample(n_bootstrap);
        std::sort(bootstrap_means.begin(), bootstrap_means.end());
        
        size_t lower_idx = static_cast<size_t>(alpha / 2 * n_bootstrap);
        size_t upper_idx = static_cast<size_t>((1 - alpha / 2) * n_bootstrap);
        
        return {bootstrap_means[lower_idx], bootstrap_means[upper_idx]};
    }
};

// Usage: No manual memory management needed
int main() {
    std::vector<double> data = {1.2, 3.4, 2.1, 5.6, 4.3};
    Bootstrap boot(std::move(data), 42);  // Move data (efficient)
    auto [lower, upper] = boot.confidence_interval(1000);
    // Automatic cleanup
}

```


---


## RAII Patterns Summary


| Pattern      | Use Case          | Example                                      |

| ------------ | ----------------- | -------------------------------------------- |

| `unique_ptr` | Single ownership  | `auto data = std::make_unique<double[]>(n);` |

| `shared_ptr` | Shared ownership  | `auto cache = std::make_shared<Matrix>();`   |

| `vector`     | Dynamic array     | `std::vector<double> data(n);`               |

| `array`      | Fixed-size array  | `std::array<double, 3> point;`               |

| Custom RAII  | Special resources | File handles, GPU memory                     |


---


## Common Pitfalls


| Pitfall                 | Why It's Bad            | Fix                              |

| ----------------------- | ----------------------- | -------------------------------- |

| Raw `new`/`delete`      | Leaks, exception-unsafe | Use smart pointers or containers |

| Returning raw pointers  | Unclear ownership       | Return `unique_ptr` or value     |

| Copying large objects   | Performance cost        | Use move semantics               |

| Forgetting Rule of Five | Broken copy/move        | Define all or delete all         |


---


## Self-Assessment Checklist


Before moving to Day 20, verify you can:

- [ ] Explain RAII principle and benefits
- [ ] Use `unique_ptr` and `shared_ptr` correctly
- [ ] Implement move constructor and move assignment
- [ ] Write exception-safe code (at least basic guarantee)
- [ ] Use STL containers instead of raw arrays

---


## Deliverable


A C++ numerical class that demonstrates:

- RAII (automatic resource management)
- Move semantics (efficient transfers)
- Exception safety (at least basic guarantee)
- No raw `new`/`delete` in user code

---


## Resources

- [C++ Core Guidelines](127)
- [Effective Modern C++](128)
- [RAII and Smart Pointers](129)
- [Exception Safety](130)

## Day 20: Compilation, Optimization Flags, and Microbenchmarking

## Overview


**Focus**: Master compiler optimization flags, write effective microbenchmarks, and validate that optimizations deliver real speedups.


**Why it matters**: Compilers can make code 10x faster with the right flags. Learn to harness optimization without breaking correctness.


---


## Learning Objectives


By the end of Day 20, you will:

- Understand key compiler optimization flags and their effects
- Write robust microbenchmarks that prevent optimizer interference
- Compare debug vs. release build performance
- Use compiler-specific intrinsics when necessary
- Profile optimized code to validate improvements

---


## Core Concepts


### 1. Compiler Optimization Levels


**GCC/Clang flags**:


```bash

# No optimization (debug builds)
g++ -O0 -g code.cpp -o code_debug

# Basic optimizations (default for most projects)
g++ -O2 code.cpp -o code_release

# Aggressive optimizations (may increase compile time)
g++ -O3 code.cpp -o code_fast

# Size optimization (for embedded systems)
g++ -Os code.cpp -o code_small

```


**What each level does**:


| Flag     | Optimizations    | Use Case                           |

| -------- | ---------------- | ---------------------------------- |

| `-O0`    | None             | Debugging (preserves all info)     |

| `-O1`    | Basic            | Fast compile, some speedup         |

| `-O2`    | Standard         | Production default                 |

| `-O3`    | Aggressive       | Performance-critical code          |

| `-Ofast` | Breaks standards | Only if you know what you're doing |


**Key** **`-O2`** **optimizations**:

- Inlining small functions
- Loop unrolling
- Dead code elimination
- Constant folding
- Common subexpression elimination

**Additional** **`-O3`** **optimizations**:

- Vectorization (SIMD)
- More aggressive inlining
- Loop transformations

### 2. Architecture-Specific Flags


**Target CPU architecture**:


```bash

# Use instructions available on this machine
g++ -O3 -march=native code.cpp

# Target specific architecture
g++ -O3 -march=skylake code.cpp
g++ -O3 -march=armv8-a code.cpp

# Enable specific instruction sets
g++ -O3 -mavx2 -mfma code.cpp

```


**Why it matters**:

- `native`: Uses SIMD, FMA, and other extensions on your CPU
- Can give 2-4x speedup for numerical code
- **Caution**: Binary won't work on older CPUs

### 3. Link-Time Optimization (LTO)


**What it does**: Optimizes across compilation units


```bash

# Compile with LTO
g++ -O3 -flto code1.cpp code2.cpp -o program

# Or in two steps
g++ -O3 -flto -c code1.cpp -o code1.o
g++ -O3 -flto -c code2.cpp -o code2.o
g++ -O3 -flto code1.o code2.o -o program

```


**Benefits**:

- Inlines across files
- Removes unused code globally
- 5-15% speedup in many cases

### 4. Fast Math Optimizations


**`-ffast-math`** (breaks IEEE 754 compliance):


```bash
g++ -O3 -ffast-math code.cpp

```


**What it enables**:

- Assumes no NaN or Inf
- Reorders floating-point operations (breaks associativity)
- Approximates math functions

**When to use**:

- ✅ Tight numerical loops with known-finite values
- ❌ General-purpose statistical libraries (NaN handling important)

**Safer alternative** (pick specific optimizations):


```bash
g++ -O3 -fno-math-errno -ffinite-math-only code.cpp

```


### 5. Microbenchmarking Techniques


**Using Google Benchmark**:


```c++
#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

static void BM_Sum_Naive(benchmark::State& state) {
    std::vector<double> data(state.range(0));
    std::iota(data.begin(), data.end(), 0.0);
    
    for (auto _ : state) {
        double sum = 0.0;
        for (auto x : data) {
            sum += x;
        }
        benchmark::DoNotOptimize(sum);  // Prevent optimization away
    }
}
BENCHMARK(BM_Sum_Naive)->Range(8, 8<<10);

static void BM_Sum_Accumulate(benchmark::State& state) {
    std::vector<double> data(state.range(0));
    std::iota(data.begin(), data.end(), 0.0);
    
    for (auto _ : state) {
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_Sum_Accumulate)->Range(8, 8<<10);

BENCHMARK_MAIN();

```


**Compile and run**:


```bash

# Build with optimizations
g++ -O3 -march=native benchmark.cpp -lbenchmark -lpthread -o benchmark

# Run
./benchmark

```


**Output**:


```javascript
-------------------------------------------------------------
Benchmark                   Time             CPU   Iterations
-------------------------------------------------------------
BM_Sum_Naive/8           3.21 ns         3.21 ns    218475392
BM_Sum_Naive/64          24.5 ns         24.5 ns     28594176
BM_Sum_Naive/512          195 ns          195 ns      3589120
BM_Sum_Accumulate/8      2.98 ns         2.98 ns    235233280
BM_Sum_Accumulate/64     22.1 ns         22.1 ns     31664128
BM_Sum_Accumulate/512     177 ns          177 ns      3952640

```


### 6. Preventing Optimizer Interference


**Problem**: Compiler may optimize away benchmarked code


```c++
// BAD: Compiler may optimize this away entirely
for (auto _ : state) {
    double result = expensive_function();
    // result never used → might be eliminated
}

// GOOD: Force use of result
for (auto _ : state) {
    double result = expensive_function();
    benchmark::DoNotOptimize(result);
}

// ALSO GOOD: Prevent input from being constant-folded
std::vector<double> data = create_test_data();
benchmark::DoNotOptimize(data.data());
for (auto _ : state) {
    double result = process(data);
    benchmark::DoNotOptimize(result);
}

```


**`volatile`** **(last resort)**:


```c++
volatile double sink;
for (auto _ : state) {
    sink = expensive_function();
}

```


### 7. Optimization Case Study


**Matrix-vector multiplication**:


```c++
// Naive implementation
void matvec_naive(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Optimized (better cache locality)
void matvec_blocked(const double* A, const double* x, double* y, int n) {
    constexpr int block_size = 64;
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
    }
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; ++j) {
            for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
                y[ii] += A[ii * n + j] * x[j];
            }
        }
    }
}

```


**Benchmark results** (n=1000):


| Version | Flags               | Time (ms) |

| ------- | ------------------- | --------- |

| Naive   | `-O0`               | 8.2       |

| Naive   | `-O2`               | 2.1       |

| Naive   | `-O3 -march=native` | 1.4       |

| Blocked | `-O3 -march=native` | 0.7       |


**Key lessons**:

- `-O2` vs `-O0`: 4x speedup (free lunch)
- `-O3 -march=native`: 1.5x additional speedup
- Algorithmic improvement: 2x on top

### 8. Inspecting Compiler Output


**View assembly**:


```bash

# Generate assembly listing
g++ -O3 -S code.cpp -o code.s

# View with Intel syntax (more readable)
g++ -O3 -S -masm=intel code.cpp -o code.s

```


**Check vectorization**:


```bash

# GCC: Report vectorization
g++ -O3 -march=native -fopt-info-vec-optimized code.cpp

# Clang: Report vectorization
clang++ -O3 -march=native -Rpass=loop-vectorize code.cpp

```


**Example output**:


```javascript
code.cpp:10:5: remark: vectorized loop (vectorization width: 4, interleaved count: 2)

```


---


## Optimization Workflow

1. **Profile first**: Find bottlenecks (don't guess)
2. **Measure baseline**: Benchmark with `-O0`
3. **Try** **`-O2`**: Usually 3-5x speedup
4. **Try** **`-O3 -march=native`**: Additional 1.5-2x
5. **Algorithmic improvements**: Often bigger than compiler opts
6. **Validate correctness**: Optimizations can introduce bugs

---


## Build Configuration Example


**CMakeLists.txt**:


```javascript
cmake_minimum_required(VERSION 3.16)
project(numerical_library)

set(CMAKE_CXX_STANDARD 17)

# Debug build
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Profile-guided optimization (advanced)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")

add_executable(my_program main.cpp)

```


**Build**:


```bash

# Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make

```


---


## Common Pitfalls


| Pitfall                          | Why It's Bad            | Fix                                         |

| -------------------------------- | ----------------------- | ------------------------------------------- |

| Using `-O0` in production        | 5-10x slower            | Always use `-O2` or `-O3`                   |

| Using `-Ofast` blindly           | Breaks correctness      | Use `-O3`, test carefully if using `-Ofast` |

| Not benchmarking changes         | Don't know if it helped | Always measure before/after                 |

| Optimizer removes benchmark code | Invalid results         | Use `DoNotOptimize()`                       |


---


## Self-Assessment Checklist


Before completing Week 3, verify you can:

- [ ] Explain differences between `-O0`, `-O2`, `-O3`
- [ ] Use `-march=native` and understand tradeoffs
- [ ] Write microbenchmarks with Google Benchmark
- [ ] Prevent optimizer from eliminating benchmark code
- [ ] Compare debug vs. release build performance
- [ ] View and interpret vectorization reports

---


## Deliverable


A benchmark suite showing:

- Performance comparison: `-O0` vs. `-O2` vs. `-O3 -march=native`
- Speedup measurements for key functions
- Validation that optimizations preserve correctness
- Markdown report with conclusions

---


## Resources

- [GCC Optimization Options](131)
- [Clang Optimization Options](132)
- [Google Benchmark](133)
- [Optimization Guide](134)

---


## Quick Links

- Proficiency Standards (see below)
- How to Use This Curriculum (see below)
- Interview Preparation Guide (see below)

---


## Proficiency Standards


**Validation Status**: These thresholds are based on instructor experience with similar graduate-level coursework but have not been empirically validated with pilot cohorts. They represent our best estimate of a meaningful proficiency bar. We welcome feedback from learners and institutions using this curriculum.


To claim proficiency from this curriculum, you must meet ALL of the following:


**1. Exercise Performance**

- Score ≥1.5/2.0 average across all Foundational exercises
- Complete ≥75% of Proficiency exercises with ≥1.5/2.0
- Mastery exercises are optional enrichment (not required for proficiency)

**2. Capstone Performance**

- Score ≥1.5/2.0 on EACH dimension of both capstone rubrics
- Complete ALL checklist items for both capstones (see capstone sections for checklists)

**3. Oral Defense Readiness**

- Answer ≥80% of oral defense questions at "strong answer" level
- Demonstrate reasoning and trade-off awareness, not just factual recall
- Practice articulation with a partner or in writing before claiming proficiency

**Self-Scoring Rubric for Oral Defense**:


For each question, score yourself 0-2:

- **0 (Weak)**: Cannot answer, or answer is factually wrong, or relies on vague statements ("it's faster", "it's better")
- **1 (Partial)**: Answer is directionally correct but missing key details, tradeoffs, or concrete examples. Would need prompting in an interview.
- **2 (Strong)**: Answer includes (a) precise technical explanation, (b) concrete example or use case, (c) at least one tradeoff or limitation, (d) delivered in 60-90 seconds without notes

To claim proficiency, you must score ≥1.5 average across all oral defense questions in the curriculum (Algorithmic Thinking section + both capstones). This typically means: mostly 2s with a few 1s, or you can articulate most answers fully with only a few gaps.


**4. Time Investment**

- Expect 50-60 hours total (6-8 hours/day × 2 weeks)
- **Time is NOT a proficiency criterion**: Proficiency is measured by rubric scores (≥1.5/2.0), not speed
- Some learners complete in 40 hours, others in 100+ hours—both can achieve proficiency
- Do NOT rush—proficiency requires deliberate practice, not completion speed

**Interpreting Your Scores**: If you score 1.5/2.0 on an exercise rubric, you should be able to: (1) explain your design choices in 2 minutes without notes, (2) identify 2-3 alternative approaches and their tradeoffs, (3) implement a similar problem in 1.5× the original time. These observable behaviors help calibrate self-assessment.


**What if I don't meet thresholds?**

- Review solution sketches and oral defense "strong answers"
- Re-attempt exercises after reviewing concepts
- Seek help from study group or mentor
- Consider extending timeline (learning pace varies)—better to achieve proficiency in 3 weeks than false confidence in 2

**What proficiency means (and doesn't mean):**


_You CAN:_

- Implement common statistical algorithms from mathematical descriptions
- Debug numerical issues using profiling, assertions, and mathematical reasoning
- Read production NumPy/Eigen/pandas code and understand intent
- Contribute to statistical computing projects with mentorship
- Recognize when Python is sufficient vs when C++ is needed for performance

_You CANNOT yet:_

- Design large-scale statistical software architectures (requires 6-12 months practice)
- Claim expert-level C++ (template metaprogramming, advanced concurrency)
- Independently lead performance optimization projects (need more profiling experience)
- Claim general software engineering proficiency (this is domain-specific training)

**Proficiency is the BEGINNING of mastery, not the end.**


**Partial Proficiency Claims**: If you complete only one language track, you may claim: "Python Proficiency for Statistical Computing" (Week 1 + Python Capstone with ≥1.5/2.0) or "C++ Proficiency for Numerical Computing" (Week 2 + C++ Capstone with ≥1.5/2.0). The full "Statistical Computing Proficiency" requires both language tracks plus demonstrated ability to choose the appropriate language for each computational task.


---


## How to Use This Curriculum


**Time Commitment:**

- Plan for 50-60 hours total across 2 weeks (6-8 hours/day)
- Days 0-7 (Python): ~25-30 hours
- Days 8-14 (C++): ~25-30 hours
- If you consistently exceed 2× expected exercise times, consider extending timeline or seeking additional C++/Python prep

**About Time Estimates**: The "Expected Time (Proficient)" labels are **planning tools**, not proficiency criteria. They represent target times for learners who solidly meet all prerequisites (fluent in probability, statistics, linear algebra, optimization). **These estimates have not been empirically validated.** Actual times vary by 2-5× based on prior programming experience, comfort with mathematical abstraction, and typing speed.


**Critical: Time ≠ Proficiency**. If you:

- Produce correct, well-reasoned solutions that meet rubric criteria (≥1.5/2.0)
- Can articulate design choices and tradeoffs
- Pass oral defense questions (≥80% at "strong answer" level)

...then you have achieved proficiency, **regardless of how long it took**. Taking 2× estimated time does not indicate lack of proficiency—it may simply reflect different learning pace, more thorough exploration, or less prior exposure to these specific tools.


**When time DOES matter**: If you consistently cannot complete exercises even after 3-4× estimated time, this may indicate prerequisite gaps (statistics, linear algebra, or basic programming concepts). In this case, consider supplementary preparation before continuing.


**Note**: Not all exercises have explicit time estimates. Daily exercises (Days 1-14) are embedded in tutorial content without separate time labels. The Algorithmic Thinking section and capstones include explicit "Expected Time (Proficient)" labels where appropriate.


**Exercise Priority:**

- **Foundational:** MUST complete all; these are prerequisites for Proficiency exercises
- **Proficiency:** Attempt all; these test working proficiency; skip only if stuck after 2× expected time
- **Mastery:** Optional enrichment for advanced learners or those continuing to Week 3

**When to Use Solution Sketches:**

- ONLY after spending ≥2× expected time on exercise
- Don't just read solution—understand the reasoning and common mistakes
- Re-attempt exercise from scratch after reviewing solution to verify understanding
- Solution sketches show ONE correct approach; alternatives may exist

**Getting Unstuck:**

1. Re-read the "Concepts" section for that day
2. Check "Common Mistakes" in solution sketch (without reading full solution)
3. Review corresponding sections in "Algorithmic Thinking for Statistical Code"
4. Check oral defense questions for related concepts (practice articulating your confusion)
5. Use study group, mentor, or online community if available
6. If still stuck after 2× expected time: review solution sketch, understand reasoning, retry from scratch

**Daily Workflow Recommendation:**

1. **Morning (2-3 hours):** Read Concepts section carefully; take notes on mental models
2. **Midday (2-3 hours):** Attempt Foundational exercises; verify with solution sketches if stuck
3. **Afternoon (2-3 hours):** Attempt Proficiency exercises; focus on reasoning, not just correct output
4. **End of day (30-60 min):**
5. Review solution sketches for any exercises where stuck
6. Answer 2-3 oral defense questions (practice articulation aloud or in writing)
7. Reflect: What mental models did I apply today? What tradeoffs did I navigate?
5. **Optional evening:** Attempt Mastery exercise if energy permits

**Proficiency Verification:**

- After Day 7: Complete Python Capstone with ≥1.5/2.0 on all rubric dimensions
- After Day 14: Complete C++ Capstone with ≥1.5/2.0 on all rubric dimensions
- Final verification: Answer ≥80% of all oral defense questions at "strong answer" level
- See "Proficiency Standards" section above for full criteria

**Note on Optional Week 3:**
Days 15-20 provide advanced extensions (advanced profiling, stress testing, API design, microbenchmarking). These are NOT required to claim proficiency. Week 3 is for learners who want to deepen skills beyond working proficiency. The core 2-week curriculum (Days 0-14) is self-contained.


---


# Interview Preparation Guide


Completing exercises is necessary but not sufficient for interview success. You must be able to explain your code verbally, defend design decisions, and answer "why" questions without hesitation.


## How to Use the Oral Defense Questions


Every Mastery exercise and both Capstones include Oral Defense Questions. Practice answering these aloud, as if in an interview:

1. **Time yourself**: You should be able to give a clear, structured answer in 60–90 seconds
2. **Explain to a peer**: Have someone who hasn't seen the code ask follow-up questions
3. **Record yourself**: Listen back for filler words, hesitation, and unclear explanations
4. **Practice without code**: Can you explain the concept using only a whiteboard?

## Common Interview Question Patterns


### Python Questions

- "Walk me through how NumPy broadcasting works with a concrete example."
- "Why would I use a Generator instead of np.random.seed()?"
- "What's the difference between a view and a copy? How do you check?"
- "This pandas code raises SettingWithCopyWarning. What's wrong and how do you fix it?"
- "How would you profile this function to find the bottleneck?"
- "Make this code reproducible. Show me exactly what you'd change."

### C++ Questions

- "When should I use stack allocation vs. heap allocation?"
- "Explain RAII. Give me an example from your own code."
- "What happens if you forget to delete heap-allocated memory?"
- "What's the difference between unique_ptr and shared_ptr? When would you use each?"
- "This code has undefined behavior. Find it and fix it."
- "Why is this loop order faster than that one?"

### Cross-Language Questions

- "When would you rewrite Python code in C++? When wouldn't you?"
- "How do you verify that C++ and Python implementations give identical results?"
- "Design a system that uses both languages. What goes where?"

## Mock Interview Format


Practice with a partner using this structure (45 minutes total):

1. **Warm-up** (5 min): Explain a concept from the curriculum without looking at notes
2. **Code Review** (15 min): Partner picks one of your exercises. Explain your design decisions. Partner asks "why" questions.
3. **Live Problem** (15 min): Partner gives you a small problem (e.g., "implement X without using library Y"). Think aloud as you solve it.
4. **Follow-up** (10 min): Partner probes edge cases, asks about performance, suggests modifications

## Red Flags to Avoid

- Saying "I think" or "I believe" when you should know definitively
- Unable to explain what your code does line by line
- Cannot identify the time/space complexity of your solution
- Defensive when asked about tradeoffs or limitations
- Cannot think of test cases for your own code
- Blanking on fundamental concepts (stack vs. heap, view vs. copy, RAII)

## What Differentiates Strong Candidates

- Can explain the "why" behind every design decision
- Proactively mentions edge cases and how they're handled
- Discusses tradeoffs without being asked
- Connects implementation details to performance implications
- Admits uncertainty clearly: "I'm not sure, but I would verify by..."
- Asks clarifying questions before diving into solutions

---

