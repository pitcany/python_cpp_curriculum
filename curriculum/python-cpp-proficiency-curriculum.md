---
title: "2-Week Python & C++ Proficiency for Statisticians and Data Scientists"
source: Notion
notion_url: https://www.notion.so/2-Week-Python-C-Proficiency-for-Statisticians-and-Data-Scientists-2db342cf7cc8815a97e5d434dbabf57c
last_synced: 2026-01-02T05:01:57.456Z
last_edited_in_notion: 2026-01-02T04:18:00.000Z
---


# 2-Week Python & C++ Proficiency for Statisticians and Data Scientists


This curriculum assumes fluency in probability, statistics, linear algebra, and optimization. It does not assume prior software engineering training. The goal is credible proficiency: the ability to read, write, debug, and reason about nontrivial code in both Python and C++.


Week 1 covers Python. Week 2 covers C++. Capstones require cross-language comparison.


---


# Week 1: Python


## Day 1: Functions, Modules, and Idiomatic Python


### Concepts


Python functions are first-class objects. This has consequences: functions can be passed as arguments, returned from other functions, and stored in data structures. Understand the difference between `def` and `lambda`, and when each is appropriate.


Module structure matters for maintainability. The distinction between a script and a module hinges on `if __name__ == "__main__":`. Understand why circular imports fail and how to restructure code to avoid them.


Idiomatic Python favors readability over cleverness. List comprehensions replace explicit loops when the intent is clear. Generator expressions defer computation. The `itertools` module provides composable iteration primitives.


Understand `*args` and `**kwargs` for variadic functions. Understand default argument pitfalls: mutable defaults are shared across calls.


### Exercises


**Foundational 1**: Write a function `apply_to_columns(df, func)` that applies a callable to each numeric column of a pandas DataFrame and returns a dictionary mapping column names to results. Handle the case where `func` raises an exception for some columns gracefully.


**Expected Time (Proficient): 8–12 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                                         | 1                                                         | 2                                                                             |

| ----------- | --------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------- |

| Correctness | Does not filter to numeric columns or fails on exceptions | Filters numeric columns but exception handling incomplete | Correctly iterates numeric columns, catches exceptions, returns complete dict |

| Clarity     | Unclear logic, poor naming                                | Readable but verbose or inconsistent style                | Clean, idiomatic Python with clear intent                                     |

| Robustness  | Crashes on edge cases (empty df, all non-numeric)         | Handles some edge cases                                   | Handles empty df, no numeric columns, mixed types gracefully                  |

| Efficiency  | Unnecessary copies or iterations                          | Reasonable but minor inefficiencies                       | Single pass, no unnecessary allocations                                       |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Iterate over numeric columns using `select_dtypes`, apply func in try/except, collect results.


**Key steps**:

1. Use [`df.select`](http://df.select/)`_dtypes(include=[np.number])` to get numeric columns
2. Iterate over column names
3. Wrap `func(df[col])` in try/except
4. Store result or None/sentinel on exception

**Implementation details**:

- Return type: `dict[str, Any]` where values are func results or exception indicators
- Consider returning `{col: (success, result_or_error)}` tuples for explicit error handling
- Alternative: use `df.apply(func, axis=0)` but this obscures exception handling

**Common mistakes**:

- Forgetting to filter to numeric columns first
- Silently swallowing exceptions without indication
- Modifying the DataFrame in place when func has side effects

</details>


**Foundational 2**: Create a module `stats_`[`](%7B%7Bhttp://utils.py%7D%7D)[utils.py](http://utils.py/)[`](%7B%7Bhttp://utils.py%7D%7D) containing at least three functions for common statistical operations (e.g., z-score normalization, winsorization, bootstrap mean). Write a separate script that imports and uses this module. Verify that running the module directly executes test code, but importing it does not.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                            | 1                                            | 2                                            |

| ----------- | -------------------------------------------- | -------------------------------------------- | -------------------------------------------- |

| Correctness | Functions don't work or import guard missing | Functions work but guard incomplete or buggy | All functions correct, guard works perfectly |

| Clarity     | Poor naming, no docstrings                   | Some documentation, readable                 | Clear docstrings, idiomatic Python style     |

| Robustness  | No edge case handling                        | Basic validation                             | Handles empty arrays, NaN, edge cases        |

| Modularity  | Functions tightly coupled                    | Some separation of concerns                  | Clean interfaces, independently testable     |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Use `if __name__ == "__main__":` guard to separate importable functions from test code.


**Key steps**:

1. Define functions: `zscore(arr)`, `winsorize(arr, limits)`, `bootstrap_mean(arr, n_samples, rng)`
2. Add `if __name__ == "__main__":` block with test code
3. Create [`](%7B%7Bhttp://main.py%7D%7D)[main.py](http://main.py/)[`](%7B%7Bhttp://main.py%7D%7D) that does `from stats_utils import zscore, winsorize`
4. Verify: `python stats_`[`](%7B%7Bhttp://utils.py%7D%7D)[utils.py](http://utils.py/)[`](%7B%7Bhttp://utils.py%7D%7D) runs tests; `python` [`](%7B%7Bhttp://main.py%7D%7D)[main.py](http://main.py/)[`](%7B%7Bhttp://main.py%7D%7D) does not

**Implementation details**:


```python
def zscore(arr):
    return (arr - np.mean(arr)) / np.std(arr, ddof=1)

if __name__ == "__main__":
    # Test code here - only runs when executed directly
    test_data = np.array([1, 2, 3, 4, 5])
    print(zscore(test_data))

```


**Common mistakes**:

- Putting test code at module level (runs on import)
- Circular imports if stats_utils imports from main

</details>


**Foundational 3**: Write a function that takes an array and returns `True` if it is a view of another array and `False` if it owns its data. Test on slices, boolean masks, and results of arithmetic operations.


**Expected Time (Proficient): 6–10 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                    | 1                                        | 2                                                          |

| ----------- | ------------------------------------ | ---------------------------------------- | ---------------------------------------------------------- |

| Correctness | Wrong flag checked or logic inverted | Correct for most cases, edge case errors | Correctly identifies views vs owned data in all test cases |

| Clarity     | Implementation unclear               | Functional but verbose                   | One-liner using `flags.owndata` with clear naming          |

| Robustness  | Only works for 1D arrays             | Works for common cases                   | Handles any ndarray including 0-d arrays                   |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Check the `OWNDATA` flag in array's flags attribute.


**Key steps**:

1. Access `arr.flags['OWNDATA']` or `arr.flags.owndata`
2. Returns True if array owns its data, False if it's a view
3. Test cases: slices (False), boolean masks (True), arithmetic (True)

**Implementation**:


```python
def is_view(arr):
    return not arr.flags.owndata

def owns_data(arr):
    return arr.flags.owndata

# Tests
a = np.arange(100)
print(is_view(a))           # False - owns data
print(is_view(a[10:20]))    # True - slice is view
print(is_view(a[a > 50]))   # False - boolean mask copies
print(is_view(a * 2))       # False - arithmetic creates new array

```


**Nuance**: `base` attribute shows what array a view is based on: `arr.base is None` means owns data.


</details>


**Proficiency 1**: Implement an in-place standardization function `standardize_inplace(X)` that modifies X to have zero mean and unit variance per column. It must not allocate a new array for the result. Verify memory addresses before and after are identical. Compare performance against a version that returns a new array.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                                | 1                                  | 2                                                       |

| --------------------- | -------------------------------- | ---------------------------------- | ------------------------------------------------------- |

| Correctness           | Creates copy or wrong statistics | In-place but incorrect mean/var    | Truly in-place with correct zero mean, unit variance    |

| Clarity               | Logic hard to follow             | Readable but could be cleaner      | Clear use of `-=` and `/=` operators                    |

| Robustness            | Fails on single column or row    | Works for typical cases            | Handles single column, single row, already standardized |

| Efficiency            | Hidden copies via temporaries    | Mostly in-place, minor allocations | Zero allocations beyond stats computation               |

| Statistical soundness | Wrong ddof or axis               | Correct formulas                   | Sample variance (ddof=1) with appropriate axis          |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Use in-place operators (`-=`, `/=`) and pre-computed statistics to avoid temporaries.


**Key steps**:

1. Compute column means and stds
2. Subtract means in-place: `X -= means`
3. Divide by stds in-place: `X /= stds`
4. Verify with `id(X)` before/after or `X.__array_interface__['data'][0]`

**Implementation**:


```python
def standardize_inplace(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    X -= means  # In-place subtraction
    X /= stds   # In-place division
    return X    # Same object

# Verify
addr_before = X.__array_interface__['data'][0]
standardize_inplace(X)
addr_after = X.__array_interface__['data'][0]
assert addr_before == addr_after

```


**Performance notes**:

- In-place: ~2x faster for large arrays (no allocation overhead)
- Memory: O(d) for means/stds vs O(n*d) for copy version

**Common mistakes**:

- Using `X = X - means` which creates new array and rebinds variable

</details>


**Proficiency 2**: Given a 3D array of shape `(n_samples, n_timesteps, n_features)`, write a function that returns a 2D array of shape `(n_samples * n_timesteps, n_features)` as a view (no copy). Determine under what conditions this is possible and raise an informative error when it is not.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                        | 1                                                       | 2                                                  |

| ----------- | ---------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- |

| Correctness | Returns copy or wrong shape              | View for C-contiguous only, no error for non-contiguous | View when possible, informative error otherwise    |

| Clarity     | No explanation of contiguity requirement | Some documentation                                      | Clear docstring explaining when/why it works       |

| Robustness  | No validation                            | Checks contiguity but poor error message                | Validates contiguity with actionable error message |

| Efficiency  | Unnecessary copies or checks             | Correct but verbose                                     | Minimal checks, zero-copy when possible            |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Reshape is zero-copy only when data is contiguous in the target layout.


**Key steps**:

1. Check if array is C-contiguous (`arr.flags['C_CONTIGUOUS']`)
2. If contiguous, `reshape(-1, n_features)` creates a view
3. If not contiguous, raise informative error
4. Verify result is view using `np.shares_memory`

**Implementation**:


```python
def reshape_to_2d_view(arr):
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError(
            f"Cannot reshape to view: array is not C-contiguous. "
            f"Flags: {arr.flags}. Use np.ascontiguousarray() first."
        )
    n_features = arr.shape[-1]
    result = arr.reshape(-1, n_features)
    assert np.shares_memory(arr, result), "Unexpected copy"
    return result

```


**When it fails**:

- After transpose: `arr.T` is not C-contiguous
- After non-contiguous slicing: `arr[:, ::2, :]`
- Fortran-ordered arrays from some file formats

</details>


**Mastery**: Implement a function `sliding_window_view(arr, window_size)` that returns a 2D array where each row is a window of the original 1D array, using only stride manipulation (no loops, no copies). The result must be a view. Compare your implementation to `np.lib.stride_tricks.sliding_window_view` for correctness and memory behavior.


**Expected Time (Proficient): 20–30 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                            | 1                                  | 2                                                               |

| ----------- | ---------------------------- | ---------------------------------- | --------------------------------------------------------------- |

| Correctness | Wrong output or creates copy | Correct output but edge cases fail | Matches numpy's implementation, result is view                  |

| Clarity     | Stride math unclear          | Working but hard to follow         | Clear comments explaining stride calculation                    |

| Robustness  | No input validation          | Some validation                    | Validates 1D input, window <= length, raises informative errors |

| Efficiency  | Hidden copies or loops       | Zero-copy but inefficient checks   | Pure stride manipulation, O(1) construction                     |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Use `as_strided` to create overlapping views by manipulating shape and strides.


**Key steps**:

1. Calculate output shape: `(len(arr) - window_size + 1, window_size)`
2. Calculate strides: `(arr.strides[0], arr.strides[0])` - both dimensions step by element size
3. Use `np.lib.stride_`[`tricks.as`](http://tricks.as/)`_strided` with these parameters
4. Verify result shares memory with input

**Implementation**:


```python
from numpy.lib.stride_tricks import as_strided

def sliding_window_view(arr, window_size):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D")
    if window_size > len(arr):
        raise ValueError("Window larger than array")
    
    n_windows = len(arr) - window_size + 1
    new_shape = (n_windows, window_size)
    new_strides = (arr.strides[0], arr.strides[0])
    
    return as_strided(arr, shape=new_shape, strides=new_strides)

```


**Verification**:


```python
arr = np.arange(10)
mine = sliding_window_view(arr, 3)
theirs = np.lib.stride_tricks.sliding_window_view(arr, 3)
assert np.array_equal(mine, theirs)
assert np.shares_memory(arr, mine)

```


**Warning**: `as_strided` can create invalid memory access if parameters are wrong. Always validate inputs.


</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why must the result be a view rather than a copy for this exercise to be meaningful?
2. What happens if someone writes to the sliding window array? Demonstrate the aliasing.
3. How would you extend this to 2D windows over a 2D array?
4. When would using explicit loops be preferable to stride tricks?

</details>


---


## Day 2: Memory Model, Views, and Copies


### Concepts


NumPy arrays are contiguous blocks of memory with metadata describing shape, strides, and dtype. Understanding this representation is essential for performance and correctness.


Views share memory with the original array. Slicing typically creates views: `arr[::2]` sees every other element without copying. Modifications to a view affect the original. Use `np.shares_memory(a, b)` or `arr.flags.owndata` to check.


Copies allocate new memory. Boolean indexing (`arr[arr > 0]`) always copies. Fancy indexing (`arr[[0, 2, 4]]`) always copies. When in doubt, call `.copy()` explicitly.


Strides describe byte offsets between elements. A (3, 4) array with row-major layout has strides `(4 * itemsize, itemsize)`. Transposing swaps strides without copying data. Understanding strides enables advanced tricks like sliding windows.


The `__array_interface__` protocol exposes memory addresses. Use `arr.__array_interface__['data'][0]` to verify whether two arrays share the same buffer.


Memory alignment affects vectorization. Unaligned access is slower on some architectures. NumPy allocates aligned memory by default.


### Exercises


**Foundational 1**: Create a 2D array `X` of shape (100, 100). Extract a slice `Y = X[::2, ::2]`. Verify that `Y` is a view using `np.shares_memory`. Modify `Y[0, 0]` and confirm `X[0, 0]` changes. Then create `Z = X[X > 0.5]` and verify it is NOT a view.


**Expected Time (Proficient): 8–12 minutes**

<details>
<summary>Rubric</summary>

| Dimension     | 0                                          | 1                                      | 2                                                        |

| ------------- | ------------------------------------------ | -------------------------------------- | -------------------------------------------------------- |

| Correctness   | Cannot distinguish views from copies       | Identifies view correctly but not copy | Correctly identifies both view and copy cases            |

| Verification  | No verification of shared memory           | Uses owndata but not shares_memory     | Uses shares_memory and demonstrates mutation propagation |

| Understanding | Cannot explain why boolean indexing copies | Knows it copies but unclear why        | Explains non-contiguous result requires new allocation   |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Slicing with step creates a view (strides adjust); boolean indexing extracts arbitrary elements requiring a copy.


**Key steps**:

1. `X = np.random.rand(100, 100)`
2. `Y = X[::2, ::2]` — view with doubled strides
3. `assert np.shares_memory(X, Y)`
4. `Y[0, 0] = -999; assert X[0, 0] == -999`
5. `Z = X[X > 0.5]; assert not np.shares_memory(X, Z)`

**Common mistakes**:

- Confusing `.copy()` return value with in-place modification
- Assuming all indexing creates views

</details>


**Foundational 2**: Write a function `memory_address(arr)` that returns the memory address of an array's first element. Use it to verify that `arr.T` (transpose) shares the same base address as `arr` but `arr.flatten()` does not.


**Expected Time (Proficient): 10–15 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                            | 1                           | 2                                                    |

| -------------- | ---------------------------- | --------------------------- | ---------------------------------------------------- |

| Implementation | Cannot access memory address | Uses wrong interface method | Correctly uses **array_interface**['data'][0]        |

| Transpose Test | Does not test transpose      | Tests but wrong conclusion  | Verifies transpose shares address, different strides |

| Flatten Test   | Does not test flatten        | Tests but wrong conclusion  | Verifies flatten allocates new memory                |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: `__array_interface__` exposes the raw pointer; transpose reuses data with swapped strides; flatten must copy to guarantee contiguity.


**Implementation**:


```python
def memory_address(arr):
    return arr.__array_interface__['data'][0]

arr = np.arange(12).reshape(3, 4)
assert memory_address(arr) == memory_address(arr.T)
assert memory_address(arr) != memory_address(arr.flatten())

```


**Note**: `arr.ravel()` returns a view when possible, `arr.flatten()` always copies.


</details>


**Proficiency 1**: Write a function `diagnose_array(arr)` that returns a dictionary with: `shape`, `strides`, `dtype`, `is_contiguous` (C or F order), `owns_data`, `memory_address`, and `itemsize`. Use this to explain why `arr[:, 0]` (column slice) has different stride behavior than `arr[0, :]` (row slice) for a C-contiguous array.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension          | 0                                 | 1                                | 2                                                             |

| ------------------ | --------------------------------- | -------------------------------- | ------------------------------------------------------------- |

| Completeness       | Missing multiple fields           | Most fields present, 1-2 missing | All requested fields present and correct                      |

| Contiguity Check   | Does not check contiguity         | Checks one order only            | Correctly identifies C vs F contiguity using flags            |

| Stride Explanation | Cannot explain stride differences | Partial explanation              | Clear explanation: row slice contiguous, column slice strided |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
def diagnose_array(arr):
    return {
        'shape': arr.shape,
        'strides': arr.strides,
        'dtype': arr.dtype,
        'is_c_contiguous': arr.flags['C_CONTIGUOUS'],
        'is_f_contiguous': arr.flags['F_CONTIGUOUS'],
        'owns_data': arr.flags['OWNDATA'],
        'memory_address': arr.__array_interface__['data'][0],
        'itemsize': arr.itemsize
    }

```


**Stride explanation**: For C-contiguous (m, n) array with itemsize 8:

- `arr[0, :]` has strides (8,) — contiguous
- `arr[:, 0]` has strides (n*8,) — strided, cache-unfriendly

</details>


**Proficiency 2**: Implement `safe_slice(arr, start, stop, step)` that returns a view when the slice is contiguous and a copy otherwise. The function should detect contiguity by examining whether the resulting strides equal `(itemsize,)` for 1D output.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension    | 0                                     | 1                           | 2                                                              |

| ------------ | ------------------------------------- | --------------------------- | -------------------------------------------------------------- |

| Logic        | Always copies or always views         | Contiguity check incomplete | Correctly detects contiguity and returns view/copy accordingly |

| Edge Cases   | Fails on negative step or empty slice | Handles some edge cases     | Handles negative step, empty slice, step > 1 correctly         |

| Verification | No verification of view vs copy       | Some verification           | Uses shares_memory to verify behavior in tests                 |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: A slice is contiguous iff step == 1 (or -1 for reversed contiguous). Check resulting strides against itemsize.


**Implementation**:


```python
def safe_slice(arr, start, stop, step):
    result = arr[start:stop:step]
    is_contiguous = (result.ndim == 1 and 
                     abs(result.strides[0]) == result.itemsize)
    if is_contiguous:
        return result  # view
    return result.copy()

```


</details>


**Mastery**: Implement `as_strided_safe(arr, shape, strides)` that wraps `np.lib.stride_`[`tricks.as`](http://tricks.as/)`_strided` with bounds checking. Before calling `as_strided`, verify that no element in the output would access memory outside the original array's buffer. Raise `ValueError` with a descriptive message if the configuration is invalid. Test on valid sliding window configurations and invalid ones that would cause out-of-bounds access.


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                            | 1                                    | 2                                                                   |

| -------------- | ---------------------------- | ------------------------------------ | ------------------------------------------------------------------- |

| Bounds Logic   | No bounds checking           | Partial checking (misses edge cases) | Complete bounds verification for all output elements                |

| Error Messages | Generic or no error messages | Error raised but message unhelpful   | Clear message indicating which access is out of bounds              |

| Testing        | No tests                     | Tests valid cases only               | Tests both valid and invalid configurations, verifies errors raised |

| Performance    | Check is O(output size)      | Check is O(shape dimensions)         | Check is O(1) using max offset calculation                          |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Calculate maximum byte offset that any output element would access; compare against input buffer size.


**Implementation**:


```python
def as_strided_safe(arr, shape, strides):
    # Calculate max offset
    max_offset = sum((s - 1) * st for s, st in zip(shape, strides))
    buffer_size = arr.nbytes
    
    if max_offset >= buffer_size:
        raise ValueError(
            f"Invalid configuration: max offset {max_offset} >= buffer size {buffer_size}"
        )
    
    return np.lib.stride_

```


**Test cases**:


```python
arr = np.arange(10)

# Valid: sliding window of size 3
as_strided_safe(arr, (8, 3), (8, 8))  # works

# Invalid: window extends past end
as_strided_safe(arr, (10, 3), (8, 8))  # raises ValueError

```


</details>

<details>
<summary>Complete Reference Implementation</summary>

```python
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple

def as_strided_safe(arr: np.ndarray, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> np.ndarray:
    """Safe wrapper around as_strided with bounds checking."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr).__name__}")
    if len(shape) != len(strides):
        raise ValueError(f"shape has {len(shape)} dims but strides has {len(strides)} dims")
    if any(s < 0 for s in shape):
        raise ValueError("shape dimensions must be non-negative")
    if any(s == 0 for s in shape):
        return as_strided(arr, shape=shape, strides=strides)
    
    # Calculate max byte offset any output element could access
    max_offset = sum((s - 1) * abs(st) for s, st in zip(shape, strides))
    last_byte = max_offset + arr.itemsize - 1
    
    if last_byte >= arr.nbytes:
        raise ValueError(f"Invalid: accessing byte {last_byte} but buffer is {arr.nbytes} bytes")
    if any(st < 0 for st in strides):
        raise ValueError("Negative strides not supported in safe version")
    
    return as_strided(arr, shape=shape, strides=strides)

```


</details>

<details>
<summary>Test Suite</summary>

```python
def test_as_strided_safe():
    # Test 1: Valid sliding window
    arr = np.arange(10, dtype=np.float64)
    result = as_strided_safe(arr, shape=(8, 3), strides=(8, 8))
    assert result.shape == (8, 3)
    assert np.shares_memory(arr, result)
    print("PASS: Valid sliding window")
    
    # Test 2: Rejects out-of-bounds
    try:
        as_strided_safe(arr, shape=(10, 3), strides=(8, 8))
        assert False
    except ValueError:
        print("PASS: Rejects out-of-bounds")
    
    # Test 3: Type checking
    try:
        as_strided_safe([1,2,3], shape=(2,), strides=(8,))
        assert False
    except TypeError:
        print("PASS: Type checking")
    
    # Test 4: Dimension mismatch
    try:
        as_strided_safe(arr, shape=(3,3), strides=(8,))
        assert False
    except ValueError:
        print("PASS: Dimension mismatch")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_as_strided_safe()

```


</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why does `as_strided` not perform bounds checking by default?
2. What happens if you write to an array created by `as_strided` with overlapping windows?
3. How would you extend this to handle negative strides?
4. When would you use `as_strided` instead of explicit loops in production code?

</details>


---


## Day 3: Broadcasting and Vectorization


### Concepts


Broadcasting rules: arrays with different shapes are compatible if, for each dimension (aligned from the right), the sizes are equal or one of them is 1. A dimension of size 1 is stretched to match the other.


Vectorized operations eliminate Python loop overhead. The performance difference is often 10-100x. This is because NumPy delegates to compiled C/Fortran code operating on contiguous memory.


Broadcasting enables operations that would otherwise require explicit replication. Example: subtracting row means from a matrix requires understanding that a `(n,)` array broadcasts against a `(m, n)` array along axis 1.


Common pitfalls: broadcasting can silently produce unintended results if shapes are accidentally compatible. A `(3,)` array and a `(3, 1)` array broadcast to `(3, 3)`, which may not be intended.


Know when broadcasting fails and how to reshape arrays to enable it.


### Exercises


**Foundational 1**: Compute the Euclidean distance matrix between two sets of points `X` (shape `(m, d)`) and `Y` (shape `(n, d)`) using broadcasting. The result should have shape `(m, n)`. Do not use loops.


**Expected Time (Proficient): 10–15 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                          | 1                            | 2                                                                 |

| ----------- | -------------------------- | ---------------------------- | ----------------------------------------------------------------- |

| Correctness | Uses loops or wrong result | Correct but numerical issues | Correct distances, handles numerical precision (no negative sqrt) |

| Clarity     | Broadcasting logic unclear | Working but verbose          | Clean broadcasting with clear dimension comments                  |

| Robustness  | Fails on m≠n or d=1        | Works for typical cases      | Handles any m, n, d including edge cases                          |

| Efficiency  | O(mnd) memory intermediate | Correct but suboptimal       | Uses efficient formula (                                          |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Expand dimensions to enable broadcasting: `||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y`


**Method 1 - Direct broadcasting** (memory intensive):


```python

# X: (m, d), Y: (n, d)

# Expand: X[:, None, :] is (m, 1, d), Y[None, :, :] is (1, n, d)
diff = X[:, None, :] - Y[None, :, :]  # (m, n, d)
dist = np.sqrt((diff ** 2).sum(axis=2))  # (m, n)

```


**Method 2 - Efficient formula** (recommended):


```python

# ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
X_sq = (X ** 2).sum(axis=1, keepdims=True)  # (m, 1)
Y_sq = (Y ** 2).sum(axis=1, keepdims=True)  # (n, 1)
dist_sq = X_sq + Y_sq.T - 2 * X @ Y.T      # (m, n)
dist = np.sqrt(np.maximum(dist_sq, 0))      # Avoid negative due to float errors

```


**Performance notes**:

- Method 1: O(m_n_d) memory for intermediate
- Method 2: O(m*n) memory, faster for large d

**Common mistakes**:

- Forgetting `keepdims=True` breaks broadcasting
- Not handling numerical precision (tiny negative values before sqrt)

</details>


**Foundational 2**: Given a matrix `X` of shape `(n_samples, n_features)`, subtract the column means and divide by column standard deviations using only broadcasting (no explicit loops, no `apply`). Verify the result has zero mean and unit variance per column.


**Expected Time (Proficient): 8–12 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                                | 1                           | 2                                           |

| --------------------- | -------------------------------- | --------------------------- | ------------------------------------------- |

| Correctness           | Wrong axis or broadcasting fails | Correct standardization     | Correct with verification using allclose    |

| Clarity               | Axis parameter usage unclear     | Working but verbose         | Clean one-liner with clear axis usage       |

| Robustness            | Fails on single row/column       | Works for typical cases     | Handles edge cases, division by zero        |

| Statistical soundness | Wrong ddof                       | Correct but no verification | ddof=1 for sample std, verified numerically |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Compute statistics along axis=0, broadcast back to full shape.


**Implementation**:


```python
means = X.mean(axis=0)        # Shape: (n_features,)
stds = X.std(axis=0, ddof=1)  # Shape: (n_features,)
X_standardized = (X - means) / stds  # Broadcasting: (n, d) - (d,) / (d,)

```


**Verification**:


```python

# Mean should be ~0 (within floating point tolerance)
assert np.allclose(X_standardized.mean(axis=0), 0, atol=1e-10)

# Std should be ~1
assert np.allclose(X_standardized.std(axis=0, ddof=1), 1, atol=1e-10)

```


**Broadcasting explanation**:

- `X` has shape `(n, d)`, `means` has shape `(d,)`
- NumPy aligns from right: `(n, d)` - `(d,)` broadcasts `(d,)` to `(1, d)` then to `(n, d)`

**Common mistakes**:

- Using `axis=1` instead of `axis=0` (row vs column operations)
- Forgetting `ddof=1` for sample standard deviation

</details>


**Proficiency 1**: Implement softmax for a 2D array along axis 1 using broadcasting. Handle numerical stability (subtract the max before exponentiating). Compare performance against a naive loop-based implementation for arrays of shape `(10000, 1000)`.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                  | 1                          | 2                                                  |

| ----------- | ---------------------------------- | -------------------------- | -------------------------------------------------- |

| Correctness | Wrong result or numerical overflow | Correct for typical values | Correct for extreme values (-1000 to 1000)         |

| Clarity     | Stability trick unclear            | Working but verbose        | Clear max subtraction with keepdims                |

| Robustness  | Fails on single row                | Works for typical cases    | Handles single row, single column, all-same values |

| Efficiency  | Slower than loop version           | Faster but not optimal     | 10x+ faster than loop, proper vectorization        |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))` for stability.


**Implementation**:


```python
def softmax_stable(X):
    # X: (n, d)
    X_max = X.max(axis=1, keepdims=True)  # (n, 1)
    exp_X = np.exp(X - X_max)              # (n, d) - broadcasting
    return exp_X / exp_X.sum(axis=1, keepdims=True)

def softmax_loop(X):
    result = np.zeros_like(X)
    for i in range(X.shape[0]):
        row = X[i] - X[i].max()
        exp_row = np.exp(row)
        result[i] = exp_row / exp_row.sum()
    return result

```


**Performance** (10000, 1000):

- Vectorized: ~15ms
- Loop-based: ~500ms
- Speedup: ~30x

**Why stability matters**:

- `exp(1000)` = inf, `exp(1000 - 1000) = 1`
- Subtracting max ensures all exponents are ≤ 0

**Common mistakes**:

- Forgetting `keepdims=True` breaks broadcasting
- Not handling axis parameter correctly

</details>


**Proficiency 2**: Given `X` of shape `(n, d)` and `centroids` of shape `(k, d)`, compute the index of the nearest centroid for each point in `X` using broadcasting. No loops. Then implement a full k-means iteration (assignment + centroid update) in pure NumPy.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                              | 1                                   | 2                                          |

| ----------- | ------------------------------ | ----------------------------------- | ------------------------------------------ |

| Correctness | Wrong assignments or centroids | Assignment correct, update has bugs | Both assignment and update correct         |

| Clarity     | Broadcasting logic unclear     | Working but verbose                 | Clean implementation with clear steps      |

| Robustness  | Fails on empty clusters        | Works but empty cluster crashes     | Handles empty clusters gracefully          |

| Efficiency  | O(nkd) loops                   | Vectorized assignment, loop update  | Fully vectorized including centroid update |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Compute all pairwise distances, then argmin. Update centroids with grouped mean.


**Implementation**:


```python
def assign_clusters(X, centroids):
    # X: (n, d), centroids: (k, d)
    # Distance squared: (n, k)
    diff = X[:, None, :] - centroids[None, :, :]  # (n, k, d)
    dist_sq = (diff ** 2).sum(axis=2)  # (n, k)
    return dist_sq.argmin(axis=1)  # (n,)

def update_centroids(X, assignments, k):
    # Vectorized centroid update
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):  # This loop is O(k), not O(n)
        mask = assignments == i
        if mask.sum() > 0:
            new_centroids[i] = X[mask].mean(axis=0)
    return new_centroids

def kmeans_iteration(X, centroids):
    assignments = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, assignments, len(centroids))
    return assignments, new_centroids

```


**Fully vectorized centroid update** (advanced):


```python

# Using np.add.at for scatter-add
sums = np.zeros((k, d))
counts = np.zeros(k)
np.add.at(sums, assignments, X)
np.add.at(counts, assignments, 1)
new_centroids = sums / counts[:, None]

```


</details>


**Mastery**: Implement batched matrix multiplication for arrays `A` of shape `(batch, m, k)` and `B` of shape `(batch, k, n)` using `np.einsum`. Then implement the same using broadcasting and `np.matmul`. Benchmark both against explicit loops for `batch=1000, m=n=k=64`. Explain the performance differences.


**Expected Time (Proficient): 20–30 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                       | 1                          | 2                                              |

| ----------- | --------------------------------------- | -------------------------- | ---------------------------------------------- |

| Correctness | Wrong einsum subscripts or matmul usage | One method correct         | Both einsum and matmul produce correct results |

| Clarity     | Einsum subscripts unexplained           | Working but no explanation | Clear comments explaining subscript meaning    |

| Robustness  | Fails on batch=1                        | Works for typical cases    | Handles batch=1, non-square matrices           |

| Efficiency  | Slower than loop                        | Faster than loop           | Both methods achieve expected speedup (5-10x)  |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: einsum specifies contraction explicitly; matmul handles batch dims automatically.


**Implementations**:


```python

# Method 1: einsum
def batched_matmul_einsum(A, B):
    return np.einsum('bmk,bkn->bmn', A, B)

# Method 2: np.matmul (@ operator)
def batched_matmul_matmul(A, B):
    return A @ B  # Handles batch dimensions automatically

# Method 3: Loop (baseline)
def batched_matmul_loop(A, B):
    batch = A.shape[0]
    result = np.zeros((batch, A.shape[1], B.shape[2]))
    for i in range(batch):
        result[i] = A[i] @ B[i]
    return result

```


**Performance** (batch=1000, m=n=k=64):

- Loop: ~50ms
- einsum: ~8ms
- matmul: ~4ms

**Why matmul is fastest**:

- Uses optimized BLAS batched GEMM
- einsum: flexible but less optimized path
- Loop: Python overhead per iteration

**einsum notation explained**:

- `bmk,bkn-\>bmn`: b=batch (preserved), m=rows (A), k=contracted, n=cols (B)

**Common mistakes**:

- Wrong einsum subscripts
- Assuming einsum is always slower (it's competitive for complex contractions)

</details>

<details>
<summary>Oral Defense Questions</summary>
1. When would you prefer einsum over matmul despite the performance difference?
2. How does the performance comparison change for non-square matrices or very small batch sizes?
3. What memory layout considerations affect batched matmul performance?
4. How would you implement this efficiently if matrices don't fit in memory?

</details>


---


## Day 4: Pandas Pitfalls and Alternatives


### Concepts


Pandas is convenient but hides complexity. The `SettingWithCopyWarning` indicates ambiguous mutation semantics. Understand when `df[col][idx] = val` fails silently and when `df.loc[idx, col] = val` is required.


Chained indexing creates intermediate objects that may or may not be views. Use `.loc` and `.iloc` for unambiguous selection.


`apply` is slow because it invokes Python for each row/group. Vectorized operations on underlying NumPy arrays are faster. Use `.values` or `.to_numpy()` to escape to NumPy when performance matters.


MultiIndex introduces complexity. Understand `xs`, `swaplevel`, `stack`, `unstack`. Know when MultiIndex helps (hierarchical data) and when it hurts (simple queries become verbose).


Alternatives exist: Polars offers a different API with better performance for many operations. Arrow-backed DataFrames enable zero-copy interoperability.


### Exercises


**Foundational 1**: Create a DataFrame where chained indexing triggers `SettingWithCopyWarning`. Fix the code using `.loc`. Explain why the original code is ambiguous.


**Expected Time (Proficient): 8–12 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                     | 1                                       | 2                                                  |

| ----------- | ------------------------------------- | --------------------------------------- | -------------------------------------------------- |

| Correctness | Cannot reproduce warning or fix wrong | Warning reproduced but explanation weak | Warning reproduced, fix correct, explanation clear |

| Clarity     | Explanation unclear                   | Partial explanation                     | Clear explanation of view vs copy ambiguity        |

| Robustness  | Only one example                      | Multiple examples but incomplete        | Demonstrates multiple scenarios (boolean, slice)   |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Chained indexing (`df[col][idx]`) may modify a copy, not the original DataFrame.


**Triggering the warning**:


```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# BAD: Chained indexing - may or may not work
df[df['A'] > 1]['B'] = 99  # SettingWithCopyWarning!
print(df)  # 'B' column unchanged - silent failure!

```


**Fixed version**:


```python

# GOOD: Single indexing with .loc
df.loc[df['A'] > 1, 'B'] = 99
print(df)  # 'B' column correctly modified

```


**Why it's ambiguous**:

- `df[df['A'] > 1]` might return a view or a copy (depends on internal state)
- If copy: `['B'] = 99` modifies the copy, original unchanged
- If view: modification propagates (but you can't rely on this)
- `.loc[row_indexer, col_indexer]` is unambiguous: always modifies original

**Rule**: Use `.loc` for any assignment operation.


</details>


**Foundational 2**: Write two versions of a function that computes the rolling z-score for each group in a grouped DataFrame: one using `apply` with a custom function, one using vectorized pandas operations. Benchmark on 100 groups of 10,000 rows each.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                              | 1                                 | 2                                       |

| ----------- | ------------------------------ | --------------------------------- | --------------------------------------- |

| Correctness | One or both versions incorrect | Both correct but edge case issues | Both produce identical correct results  |

| Clarity     | Code hard to follow            | Working but verbose               | Clean implementations with clear intent |

| Robustness  | Fails on small groups          | Works for typical cases           | Handles groups smaller than window      |

| Efficiency  | Vectorized not faster          | 2-5x speedup                      | 10x+ speedup with transform             |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: `apply` invokes Python per group; vectorized uses optimized C paths.


**Setup**:


```python
df = pd.DataFrame({
    'group': np.repeat(range(100), 10000),
    'value': np.random.randn(1_000_000)
})

```


**Method 1: apply (slow)**:


```python
def rolling_zscore_apply(df, window=20):
    def zscore_group(g):
        rolling = g['value'].rolling(window)
        return (g['value'] - rolling.mean()) / rolling.std()
    return df.groupby('group').apply(zscore_group)

```


**Method 2: vectorized (fast)**:


```python
def rolling_zscore_vectorized(df, window=20):
    grouped = df.groupby('group')['value']
    rolling_mean = grouped.transform(lambda x: x.rolling(window).mean())
    rolling_std = grouped.transform(lambda x: x.rolling(window).std())
    return (df['value'] - rolling_mean) / rolling_std

```


**Performance**:

- apply: ~2-3 seconds
- vectorized transform: ~200-300ms
- Speedup: ~10x

**Why**: `transform` applies optimized rolling operations; `apply` has Python overhead per group.


</details>


**Proficiency 1**: Implement a merge operation that joins two DataFrames on a key, but where the key in one DataFrame requires a transformation before matching (e.g., stripping whitespace, lowercasing). Compare performance of: (a) transforming the column first then merging, (b) using `merge` with a custom key function via index manipulation.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                        | 1                                    | 2                                                  |

| ----------- | ---------------------------------------- | ------------------------------------ | -------------------------------------------------- |

| Correctness | Merge produces wrong results or fails    | One method correct, other has issues | Both methods produce correct, identical results    |

| Clarity     | Transformation logic unclear             | Working but verbose                  | Clean, readable approach with clear transformation |

| Robustness  | Fails on edge cases (empty strings, NaN) | Works for typical cases              | Handles edge cases gracefully                      |

| Efficiency  | No benchmark comparison                  | Basic timing comparison              | Thorough benchmark with explanation of results     |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Pre-transform is cleaner and usually faster than dynamic key functions.


**Setup**:


```python
df1 = pd.DataFrame({'key': ['  Apple', 'BANANA  ', 'Cherry'], 'val1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['apple', 'banana', 'cherry'], 'val2': [10, 20, 30]})

```


**Method (a): Transform then merge**:


```python
df1['key_clean'] = df1['key'].str.strip().str.lower()
result = df1.merge(df2, left_on='key_clean', right_on='key', suffixes=('', '_r'))

```


**Method (b): Index-based merge**:


```python
df1_indexed = df1.set_index(df1['key'].str.strip().str.lower())
df2_indexed = df2.set_index(df2['key'])
result = df1_indexed.join(df2_indexed, lsuffix='_l', rsuffix='_r')

```


**Performance** (1M rows):

- Method (a): ~150ms (transform: 50ms, merge: 100ms)
- Method (b): ~200ms (set_index overhead)

**Recommendation**: Method (a) is clearer and equally fast. Keep transformed column if needed for verification.


</details>


**Proficiency 2**: Given a DataFrame with columns `\[timestamp, user_id, event_type, value\]`, compute the time since each user's previous event of the same type. Implement using groupby + shift, then implement using a window function approach. Compare correctness and performance.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                   | 1                            | 2                                                     |

| ----------- | ----------------------------------- | ---------------------------- | ----------------------------------------------------- |

| Correctness | Wrong time differences or grouping  | One method correct           | Both methods produce correct, identical results       |

| Clarity     | Groupby logic unclear               | Working but verbose          | Clean implementation with clear group keys            |

| Robustness  | Fails on first event (NaT handling) | Works but edge cases unclear | Handles first events, single-event users gracefully   |

| Efficiency  | No performance comparison           | Basic timing                 | Benchmark with explanation of diff vs shift tradeoffs |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Group by user+event_type, compute time diff from previous row.


**Setup**:


```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100000, freq='1min'),
    'user_id': np.random.randint(0, 1000, 100000),
    'event_type': np.random.choice(['click', 'view', 'purchase'], 100000),
    'value': np.random.randn(100000)
}).sort_values(['user_id', 'event_type', 'timestamp'])

```


**Method 1: groupby + shift**:


```python
df['prev_ts'] = df.groupby(['user_id', 'event_type'])['timestamp'].shift(1)
df['time_since_prev'] = df['timestamp'] - df['prev_ts']

```


**Method 2: transform with diff**:


```python
df['time_since_prev'] = df.groupby(['user_id', 'event_type'])['timestamp'].diff()

```


**Performance** (100k rows):

- shift + subtract: ~15ms
- diff: ~10ms
- Both are O(n) but diff is slightly optimized

**Important**: Data must be sorted by group keys + timestamp for correct results!


**Edge cases**: First event per user/type has NaT (no previous event).


</details>


**Mastery**: Take a computationally intensive pandas workflow (e.g., grouped rolling regression on 1M rows, 1000 groups). Profile it using `line_profiler`. Identify the bottleneck. Rewrite the critical section using NumPy operations on the underlying arrays. Document the speedup and the tradeoffs in code clarity.


**Expected Time (Proficient): 30–45 minutes**

<details>
<summary>Rubric</summary>

| Dimension     | 0                                    | 1                                       | 2                                                     |

| ------------- | ------------------------------------ | --------------------------------------- | ----------------------------------------------------- |

| Correctness   | NumPy rewrite produces wrong results | Correct but minor numerical differences | Results match pandas version within tolerance         |

| Clarity       | No profiling documentation           | Basic profiling output                  | Clear profiling report with annotated bottlenecks     |

| Robustness    | Only works for specific data shape   | Works for typical cases                 | Handles edge cases (small groups, missing data)       |

| Efficiency    | Less than 2x speedup                 | 2-10x speedup                           | 10x+ speedup with clear explanation of why            |

| Documentation | No tradeoff discussion               | Brief mention of tradeoffs              | Thorough analysis of clarity vs performance tradeoffs |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Escape to NumPy for tight inner loops; pandas overhead dominates for complex grouped operations.


**Example: Rolling regression**:


```python

# Slow pandas version
def rolling_regression_pandas(df, window=20):
    def regress(group):
        y = group['y'].values
        x = group['x'].values
        # Rolling OLS is very slow in pure pandas
        return group.rolling(window).apply(lambda w: linregress(w, ...).slope)
    return df.groupby('group').apply(regress)

```


**Profiling**:


```bash

# Install: pip install line_profiler
kernprof -l -v 

```


**NumPy rewrite** (fast):


```python
def rolling_regression_numpy(x, y, window):
    # Vectorized rolling covariance / variance
    n = len(x)
    slopes = np.full(n, np.nan)
    
    # Use stride tricks for rolling windows
    x_windows = sliding_window_view(x, window)
    y_windows = sliding_window_view(y, window)
    
    # Vectorized regression: slope = cov(x,y) / var(x)
    x_mean = x_windows.mean(axis=1)
    y_mean = y_windows.mean(axis=1)
    cov_xy = ((x_windows - x_mean[:, None]) * (y_windows - y_mean[:, None])).mean(axis=1)
    var_x = ((x_windows - x_mean[:, None]) ** 2).mean(axis=1)
    slopes[window-1:] = cov_xy / var_x
    return slopes

```


**Performance**:

- pandas apply: ~60s for 1M rows
- NumPy vectorized: ~0.5s
- Speedup: 100x+

**Tradeoffs**: NumPy version is less readable, requires careful indexing, harder to maintain.


</details>

<details>
<summary>Oral Defense Questions</summary>
1. At what data size would you switch from pandas to NumPy for this operation?
2. How would you verify the NumPy version produces identical results to pandas?
3. What are the maintenance implications of the NumPy approach in a production codebase?
4. When would pandas' readability outweigh the performance benefits of NumPy?

</details>


---


## Day 5: Testable, Reusable Code


### Concepts


Testable code separates concerns. Functions that do I/O are harder to test than functions that transform data. Dependency injection enables testing: pass file handles rather than filenames, pass random generators rather than calling `np.random` globally.


Unit tests verify individual functions. Property-based tests verify invariants across many inputs. Hypothesis generates test cases automatically.


Reusable code has clear interfaces. Type hints document expected inputs and outputs. Docstrings explain behavior, parameters, and return values. NumPy-style docstrings are standard in scientific Python.


Avoid global state. Avoid mutable default arguments. Prefer pure functions where possible.


### Exercises


**Foundational 1**: Write a function `bootstrap_ci(data, statistic_func, n_bootstrap, ci_level, rng)` that computes a bootstrap confidence interval. It must accept a `numpy.random.Generator` instance for reproducibility. Write pytest tests that verify: (a) output shape, (b) CI contains the true parameter for simulated data, (c) determinism given the same `rng`.


**Expected Time (Proficient): 20–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                                 | 1                                      | 2                                               |

| ----------- | ------------------------------------------------- | -------------------------------------- | ----------------------------------------------- |

| Correctness | Bootstrap logic wrong or CI calculation incorrect | Bootstrap correct but tests incomplete | Correct bootstrap with comprehensive tests      |

| Clarity     | Function interface unclear                        | Working but verbose                    | Clean interface with clear parameter naming     |

| Robustness  | Fails on edge cases (small data)                  | Works for typical cases                | Handles edge cases, validates inputs            |

| Testability | No tests or tests don't pass                      | Basic tests pass                       | All three test types pass with clear assertions |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
def bootstrap_ci(data, statistic_func, n_bootstrap, ci_level, rng):
    data = np.asarray(data)
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)
    
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return lower, upper

```


**Tests**:


```python
import pytest

def test_output_shape():
    rng = np.random.default_rng(42)
    data = np.random.randn(100)
    ci = bootstrap_ci(data, np.mean, 1000, 0.95, rng)
    assert len(ci) == 2
    assert ci[0] < ci[1]  # lower < upper

def test_coverage():
    rng = np.random.default_rng(42)
    true_mean = 5.0
    data = np.random.default_rng(123).normal(true_mean, 1, 1000)
    ci = bootstrap_ci(data, np.mean, 2000, 0.95, rng)
    assert ci[0] <= true_mean <= ci[1]

def test_determinism():
    data = np.arange(100)
    ci1 = bootstrap_ci(data, np.mean, 1000, 0.95, np.random.default_rng(42))
    ci2 = bootstrap_ci(data, np.mean, 1000, 0.95, np.random.default_rng(42))
    assert ci1 == ci2

```


</details>


**Foundational 2**: Refactor a function that reads a CSV, processes it, and writes results to a new CSV into three functions: `load_data(file_handle)`, `process(df)`, `save_results(df, file_handle)`. Write tests for `process` using in-memory DataFrames.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                            | 1                                      | 2                                                 |

| ----------- | ---------------------------- | -------------------------------------- | ------------------------------------------------- |

| Correctness | Refactored code doesn't work | Works but tests incomplete             | Refactored code works with comprehensive tests    |

| Clarity     | Function boundaries unclear  | Separation exists but coupling remains | Clean separation of concerns, testable design     |

| Robustness  | I/O errors not handled       | Basic error handling                   | Proper file handle usage, graceful error handling |

| Testability | Tests require file I/O       | Some tests use in-memory data          | All process tests use in-memory DataFrames        |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Separate concerns—I/O functions take file handles, processing functions take DataFrames.


**Implementation**:


```python

# BEFORE: Monolithic function
def analyze_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['new_col'] = df['value'] * 2
    df = df[df['new_col'] > 10]
    df.to_csv(output_path)

# AFTER: Separated concerns
def load_data(file_handle):
    return pd.read_csv(file_handle)

def process(df):
    df = df.copy()
    df['new_col'] = df['value'] * 2
    return df[df['new_col'] > 10]

def save_results(df, file_handle):
    df.to_csv(file_handle, index=False)

```


**Testing process() without files**:


```python
def test_process_doubles_values():
    df = pd.DataFrame({'value': [5, 10, 15]})
    result = process(df)
    assert 'new_col' in result.columns
    assert list(result['new_col']) == [20, 30]  # 10 filtered out

def test_process_filters_correctly():
    df = pd.DataFrame({'value': [1, 2, 100]})
    result = process(df)
    assert len(result) == 1  # Only 100*2=200 > 10

```


**Key insight**: `process()` is now a pure function—no side effects, easy to test with any DataFrame.


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Separate I/O from logic for testability. Pass file handles, not paths.


**Before (untestable)**:


```python
def analyze(input_path, output_path):
    df = pd.read_csv(input_path)
    df['processed'] = df['value'] * 2  # Logic buried in I/O
    df.to_csv(output_path)

```


**After (testable)**:


```python
def load_data(file_handle):
    return pd.read_csv(file_handle)

def process(df):
    result = df.copy()
    result['processed'] = result['value'] * 2
    return result

def save_results(df, file_handle):
    df.to_csv(file_handle, index=False)

def analyze(input_path, output_path):
    with open(input_path) as f:
        df = load_data(f)
    result = process(df)
    with open(output_path, 'w') as f:
        save_results(result, f)

```


**Tests**:


```python
def test_process():
    # No file I/O needed!
    df = pd.DataFrame({'value': [1, 2, 3]})
    result = process(df)
    assert 'processed' in result.columns
    assert list(result['processed']) == [2, 4, 6]
    assert list(df['value']) == [1, 2, 3]  # Original unchanged

def test_load_data():
    from io import StringIO
    csv_content = "value\n1\n2\n3"
    df = load_data(StringIO(csv_content))
    assert len(df) == 3

```


</details>


**Proficiency 1**: Implement a `LinearRegression` class with `fit(X, y)` and `predict(X)` methods. Write tests that verify: (a) coefficients match `np.linalg.lstsq` results, (b) predictions are correct on known data, (c) the class raises informative errors for mismatched dimensions.


**Expected Time (Proficient): 20–30 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                               | 1                                         | 2                                                 |

| --------------------- | ------------------------------- | ----------------------------------------- | ------------------------------------------------- |

| Correctness           | Coefficients wrong or fit fails | Coefficients correct but tests incomplete | Coefficients match lstsq, all tests pass          |

| Clarity               | Class interface confusing       | Working but verbose                       | Clean sklearn-style interface                     |

| Robustness            | No input validation             | Some validation                           | Informative errors for all dimension mismatches   |

| Statistical soundness | Missing intercept handling      | Intercept handled                         | Proper intercept with correct matrix augmentation |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples, y has {y.shape[0]}")
        
        # Add intercept column
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        # Solve normal equations
        coeffs, residuals, rank, s = np.linalg.lstsq(X_aug, y, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"Expected {len(self.coef_)} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

```


**Tests**:


```python
def test_coefficients_match_lstsq():
    X = np.random.randn(100, 3)
    y = X @ [1, 2, 3] + 5 + np.random.randn(100) * 0.1
    model = LinearRegression().fit(X, y)
    X_aug = np.column_stack([np.ones(100), X])
    expected = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    assert np.allclose(model.intercept_, expected[0])
    assert np.allclose(model.coef_, expected[1:])

def test_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="samples"):
        LinearRegression().fit(np.zeros((10, 2)), np.zeros(5))

```


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples, y has {len(y)}")
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Solve via least squares
        coeffs, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"Expected {len(self.coef_)} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

```


**Tests**:


```python
def test_coefficients_match_lstsq():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])  # y = 2*x, no noise
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-10)
    assert np.isclose(model.intercept_, 0.0, atol=1e-10)

def test_dimension_mismatch():
    model = LinearRegression()
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.array([[1], [2]]), np.array([1, 2, 3]))

```


</details>


**Proficiency 2**: Use Hypothesis to write property-based tests for a `normalize(X)` function. Properties to test: (a) output has unit norm along the specified axis, (b) output shape equals input shape, (c) zero vectors raise an error or return NaN (specify and test your choice).


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension     | 0                        | 1                                       | 2                                                  |

| ------------- | ------------------------ | --------------------------------------- | -------------------------------------------------- |

| Correctness   | Normalize function wrong | Function correct but tests incomplete   | Function correct with all property tests passing   |

| Clarity       | Hypothesis usage unclear | Basic Hypothesis tests                  | Clean Hypothesis tests with appropriate strategies |

| Robustness    | No zero vector handling  | Zero vectors handled but inconsistently | Clear design choice for zero vectors, tested       |

| Test coverage | Only one property tested | Two properties tested                   | All three properties tested with edge cases        |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
def normalize(X, axis=1):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize zero vector")
    return X / norms

```


**Hypothesis tests**:


```python
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

@given(hnp.arrays(dtype=float, shape=st.tuples(st.integers(1, 10), st.integers(1, 10))))
def test_output_has_unit_norm(X):
    assume(not np.any(np.linalg.norm(X, axis=1) == 0))  # Skip zero rows
    result = normalize(X, axis=1)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)

@given(hnp.arrays(dtype=float, shape=st.tuples(st.integers(1, 10), st.integers(1, 10))))
def test_output_shape_equals_input(X):
    assume(not np.any(np.linalg.norm(X, axis=1) == 0))
    result = normalize(X, axis=1)
    assert result.shape == X.shape

@given(st.integers(1, 10))
def test_zero_vector_raises(n):
    X = np.zeros((1, n))
    with pytest.raises(ValueError, match="zero vector"):
        normalize(X)

```


**Design choice**: Raise error on zero vectors (alternative: return NaN, but errors are more explicit).


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation** (design choice: return NaN for zero vectors):


```python
def normalize(X, axis=-1):
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return X / norms  # Returns NaN for zero vectors

```


**Hypothesis tests**:


```python
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays

@given(arrays(np.float64, shape=st.tuples(st.integers(1, 100), st.integers(1, 10))))
def test_unit_norm(X):
    # Skip if any row is zero vector
    row_norms = np.linalg.norm(X, axis=1)
    assume(np.all(row_norms > 1e-10))
    
    result = normalize(X, axis=1)
    result_norms = np.linalg.norm(result, axis=1)
    assert np.allclose(result_norms, 1.0, atol=1e-10)

@given(arrays(np.float64, shape=st.tuples(st.integers(1, 50), st.integers(1, 10))))
def test_shape_preserved(X):
    result = normalize(X, axis=1)
    assert result.shape == X.shape

@given(st.integers(1, 10))
def test_zero_vector_returns_nan(d):
    X = np.zeros((1, d))
    result = normalize(X, axis=1)
    assert np.all(np.isnan(result))

```


**Key Hypothesis features**:

- `arrays()` generates random numpy arrays
- `assume()` filters invalid test cases
- Automatically finds edge cases (very small values, etc.)

</details>


**Mastery**: Design and implement a mini-framework for statistical estimators with a common interface: `fit(data)`, `summary()`, `confidence_interval(level)`. Implement at least two estimators (e.g., MLE for normal distribution, method of moments for gamma). Write a test suite that verifies the interface contract for any conforming estimator. Use abstract base classes or protocols.


**Expected Time (Proficient): 35–50 minutes**

<details>
<summary>Rubric</summary>

| Dimension     | 0                                | 1                                | 2                                                     |

| ------------- | -------------------------------- | -------------------------------- | ----------------------------------------------------- |

| Correctness   | Estimators produce wrong results | One estimator correct            | Both estimators correct, match theoretical values     |

| Clarity       | Interface design unclear         | Interface works but inconsistent | Clean ABC/Protocol with consistent interface          |

| Robustness    | No input validation              | Some validation                  | Comprehensive validation, informative errors          |

| Extensibility | Hard to add new estimators       | Possible but awkward             | Easy to add new estimators following pattern          |

| Test coverage | No contract tests                | Basic contract tests             | Parametrized tests verify contract for all estimators |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Define interface via Protocol or ABC; each estimator implements the contract.


**Implementation**:


```python
from abc import ABC, abstractmethod
from typing import Protocol, Tuple
from scipy import stats

class StatisticalEstimator(Protocol):
    def fit(self, data: np.ndarray) -> 'StatisticalEstimator': ...
    def summary(self) -> dict: ...
    def confidence_interval(self, level: float) -> Tuple[float, float]: ...

class NormalMLE:
    def __init__(self):
        self.mu_ = None
        self.sigma_ = None
        self.n_ = None
    
    def fit(self, data):
        data = np.asarray(data)
        self.n_ = len(data)
        self.mu_ = np.mean(data)
        self.sigma_ = np.std(data, ddof=0)  # MLE uses ddof=0
        return self
    
    def summary(self):
        return {'mu': self.mu_, 'sigma': self.sigma_, 'n': self.n_}
    
    def confidence_interval(self, level=0.95):
        se = self.sigma_ / np.sqrt(self.n_)
        z = stats.norm.ppf((1 + level) / 2)
        return (self.mu_ - z * se, self.mu_ + z * se)

class GammaMoM:
    def __init__(self):
        self.alpha_ = None  # shape
        self.beta_ = None   # rate
    
    def fit(self, data):
        data = np.asarray(data)
        mean = np.mean(data)
        var = np.var(data, ddof=1)
        self.alpha_ = mean**2 / var
        self.beta_ = mean / var
        return self
    
    def summary(self):
        return {'alpha': self.alpha_, 'beta': self.beta_}
    
    def confidence_interval(self, level=0.95):
        # Bootstrap CI for mean
        pass  # Implementation similar to bootstrap_ci

```


**Interface contract tests**:


```python
@pytest.mark.parametrize('EstimatorClass', [NormalMLE, GammaMoM])
def test_estimator_contract(EstimatorClass):
    data = np.abs(np.random.randn(100)) + 1  # Positive for Gamma
    est = EstimatorClass()
    
    # fit returns self
    assert est.fit(data) is est
    
    # summary returns dict
    summary = est.summary()
    assert isinstance(summary, dict)
    
    # confidence_interval returns tuple of two floats
    ci = est.confidence_interval(0.95)
    assert len(ci) == 2
    assert ci[0] < ci[1]

```


</details>

<details>
<summary>Solution Sketch</summary>

**Abstract base class**:


```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class Estimator(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'Estimator':
        """Fit the estimator to data. Returns self."""
        pass
    
    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Return estimated parameters."""
        pass
    
    @abstractmethod
    def confidence_interval(self, level: float) -> Dict[str, Tuple[float, float]]:
        """Return CIs for each parameter at given level."""
        pass

```


**Implementations**:


```python
class NormalMLE(Estimator):
    def __init__(self):
        

```


**Interface tests**:


```python
@pytest.mark.parametrize('EstimatorClass', [NormalMLE, GammaMoM])
def test_fit_returns_self(EstimatorClass):
    est = EstimatorClass()
    result = 

```


</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why use an abstract base class vs. duck typing for this interface?
2. How would you extend this framework to support Bayesian estimators with priors?
3. What design patterns would help add serialization/deserialization to estimators?
4. How would you handle estimators that fail to converge?

</details>


---


## Day 6: Reproducibility, Randomness, and State


### Concepts


Reproducibility requires controlling all sources of randomness. NumPy's legacy `np.random.seed()` sets global state, which is problematic for parallel code and testing. Prefer `numpy.random.Generator` instances created via `np.random.default_rng(seed)`.


Pass RNG instances explicitly. This enables: (a) reproducible tests, (b) independent random streams for parallel workers, (c) clear documentation of stochastic dependencies.


Environment reproducibility matters: pin package versions, use virtual environments, document Python version. `requirements.txt` or `pyproject.toml` should specify exact versions for critical dependencies.


Floating-point non-determinism exists: different BLAS implementations, different hardware, reordering of parallel reductions. True bitwise reproducibility across machines is difficult.


### Exercises


**Foundational 1**: Write a simulation function that generates `n` samples from a mixture of Gaussians. Demonstrate that using `np.random.seed` at the module level causes test interference when tests run in different orders. Refactor to use `Generator` instances.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                                            | 1                                     | 2                                               |

| -------------- | -------------------------------------------- | ------------------------------------- | ----------------------------------------------- |

| Correctness    | Cannot demonstrate interference or fix wrong | Demonstrates issue but fix incomplete | Clearly shows interference, fix works perfectly |

| Clarity        | Code confusing, poor explanation             | Working but explanation weak          | Clear demonstration with good explanation       |

| Robustness     | Only one test scenario                       | Multiple scenarios but incomplete     | Shows multiple test orderings, explains why     |

| Best Practices | Still uses global state                      | Uses Generator but awkwardly          | Clean Generator injection, proper parameter     |


</details>

<details>
<summary>Solution Sketch</summary>

**Problem demonstration**:


```python

# BAD: Global state causes test interference
import numpy as np
np.random.seed(42)

def sample_mixture(n, weights, means, stds):
    component = np.random.choice(len(weights), size=n, p=weights)
    return np.array([np.random.normal(means[c], stds[c]) for c in component])

# Test A runs first, consumes random state
def test_a():
    result = sample_mixture(100, [0.5, 0.5], [0, 5], [1, 1])
    assert len(result) == 100

# Test B runs second, gets different random values!
def test_b():
    result = sample_mixture(100, [0.5, 0.5], [0, 5], [1, 1])
    # If we expected specific values, this fails when test order changes

```


**Fixed with Generator**:


```python
def sample_mixture(n, weights, means, stds, rng):
    """Generate samples from Gaussian mixture.
    
    Args:
        rng: numpy.random.Generator instance
    """
    component = rng.choice(len(weights), size=n, p=weights)
    return np.array([rng.normal(means[c], stds[c]) for c in component])

def test_a():
    rng = np.random.default_rng(42)  # Local RNG
    result = sample_mixture(100, [0.5, 0.5], [0, 5], [1, 1], rng)
    expected = sample_mixture(100, [0.5, 0.5], [0, 5], [1, 1], 
                              np.random.default_rng(42))
    assert np.array_equal(result, expected)  # Always passes

def test_b():
    rng = np.random.default_rng(123)  # Different seed, independent
    result = sample_mixture(100, [0.5, 0.5], [0, 5], [1, 1], rng)
    # No interference with test_a

```


**Key insight**: Each test creates its own RNG—test order doesn't matter.


</details>

<details>
<summary>Solution Sketch</summary>

**Bad version (global state)**:


```python

# simulation.py
import numpy as np
np.random.seed(42)  # Global state - BAD!

def sample_mixture(n, means, stds, weights):
    components = np.random.choice(len(means), size=n, p=weights)
    return np.array([np.random.normal(means[c], stds[c]) for c in components])

```


**Test interference**:


```python

# test_a.py runs first: uses some random numbers

# test_b.py runs second: gets different numbers than if run alone

# pytest test_b.py → PASS

# pytest test_a.py test_b.py → test_b.py FAILS

```


**Good version (explicit RNG)**:


```python
def sample_mixture(n, means, stds, weights, rng):
    rng = np.random.default_rng(rng) if isinstance(rng, int) else rng
    components = rng.choice(len(means), size=n, p=weights)
    return np.array([rng.normal(means[c], stds[c]) for c in components])

# Tests are independent
def test_mixture():
    samples = sample_mixture(100, [0, 5], [1, 1], [0.5, 0.5], rng=np.random.default_rng(42))
    assert len(samples) == 100

```


**Key principle**: Each test creates its own RNG → no interference.


</details>


**Foundational 2**: Create a reproducibility report for an analysis: record Python version, NumPy version, pandas version, and random seed. Write a function `get_environment_info()` that returns this information as a dictionary. Include it in the output of any analysis script.


**Expected Time (Proficient): 10–15 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                        | 1                                          | 2                                                |

| ----------- | ---------------------------------------- | ------------------------------------------ | ------------------------------------------------ |

| Correctness | Missing key version info or wrong format | Has versions but missing seed or timestamp | Complete: Python, NumPy, pandas, seed, timestamp |

| Clarity     | Output format confusing                  | Readable but verbose                       | Clean JSON/dict format, easy to parse            |

| Robustness  | Crashes if package not installed         | Basic error handling                       | Graceful handling of optional packages           |

| Integration | Not integrated into workflow             | Manual integration                         | Automatic inclusion in output files              |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
import sys
import platform
import numpy as np
import pandas as pd
from datetime import datetime

def get_environment_info(seed=None):
    """Capture environment for reproducibility."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'random_seed': seed,
    }

def save_analysis_with_metadata(results, env_info, output_path):
    """Save results with reproducibility metadata."""
    output = {
        'environment': env_info,
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

# Usage in analysis script
if __name__ == '__main__':
    SEED = 42
    env = get_environment_info(seed=SEED)
    rng = np.random.default_rng(SEED)
    
    # Run analysis...
    results = {'mean': 3.14, 'std': 1.0}
    
    save_analysis_with_metadata(results, env, 'output.json')

```


**Output example**:


```json
{
  "environment": {
    "timestamp": "2024-01-15T10:30:00",
    "python_version": "3.11.5",
    "numpy_version": "1.26.0",
    "pandas_version": "2.1.0",
    "random_seed": 42
  },
  "results": {...}
}

```


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
import sys
import platform
import numpy as np
import pandas as pd
from datetime import datetime

def get_environment_info(seed=None):
    return {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'random_seed': seed,
    }

def save_with_reproducibility(results, output_path, seed):
    """Save results with environment info."""
    import json
    output = {
        'environment': get_environment_info(seed),
        'results': results
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

```


**Usage**:


```python
SEED = 42
rng = np.random.default_rng(SEED)
results = run_analysis(data, rng)
save_with_reproducibility(results, 'output.json', SEED)

```


**Output example**:


```json
{
  "environment": {
    "timestamp": "2024-01-15T10:30:00",
    "python_version": "3.11.5",
    "numpy_version": "1.26.0",
    "pandas_version": "2.1.0",
    "random_seed": 42
  },
  "results": {...}
}

```


</details>


**Proficiency 1**: Implement a `SplittableRNG` class that wraps a `Generator` and provides a `split()` method returning a new independent `SplittableRNG`. Use this to make a parallelizable bootstrap function where each worker gets an independent stream. Verify statistical independence of the streams.


**Expected Time (Proficient): 20–30 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                                 | 1                                    | 2                                              |

| --------------------- | --------------------------------- | ------------------------------------ | ---------------------------------------------- |

| Correctness           | Split produces correlated streams | Splitting works but bootstrap broken | Independent streams, bootstrap works correctly |

| Clarity               | Class interface confusing         | Working but verbose                  | Clean API mirroring numpy Generator            |

| Robustness            | Fails on repeated splits          | Works for typical cases              | Handles deep splitting, edge cases             |

| Statistical Soundness | No independence verification      | Basic verification                   | Correlation test confirms independence         |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Use `Generator.spawn()` to create independent child generators.


**Implementation**:


```python
class SplittableRNG:
    def __init__(self, seed_or_generator):
        if isinstance(seed_or_generator, np.random.Generator):
            self._rng = seed_or_generator
        else:
            self._rng = np.random.default_rng(seed_or_generator)
    
    def split(self):
        """Return a new independent SplittableRNG."""
        child_rng = self._rng.spawn(1)[0]
        return SplittableRNG(child_rng)
    
    def random(self, size=None):
        return self._rng.random(size)
    
    def normal(self, loc=0, scale=1, size=None):
        return self._rng.normal(loc, scale, size)
    
    @property
    def generator(self):
        return self._rng

def parallel_bootstrap(data, statistic, n_bootstrap, n_workers, rng):
    """Bootstrap with independent RNG per worker."""
    from concurrent.futures import ProcessPoolExecutor
    
    # Create independent RNG for each worker
    worker_rngs = [SplittableRNG(rng.split()) for _ in range(n_workers)]
    bootstrap_per_worker = n_bootstrap // n_workers
    
    def worker_bootstrap(worker_rng):
        results = []
        for _ in range(bootstrap_per_worker):
            sample = worker_rng.generator.choice(data, size=len(data), replace=True)
            results.append(statistic(sample))
        return results
    
    # Each worker gets independent stream
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_results = list(

```


**Statistical independence test**:


```python
def test_independence():
    rng = SplittableRNG(42)
    child1 = rng.split()
    child2 = rng.split()
    
    samples1 = child1.normal(size=10000)
    samples2 = child2.normal(size=10000)
    
    # Correlation should be near zero
    corr = np.corrcoef(samples1, samples2)[0, 1]
    assert abs(corr) < 0.03  # Within statistical noise

```


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
class SplittableRNG:
    def __init__(self, seed=None):
        if isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)
        self._spawn_counter = 0
    
    def split(self) -> 'SplittableRNG':
        """Return a new independent SplittableRNG."""
        # Use spawn() for proper stream splitting
        child_bg = self._rng.bit_generator.spawn(1)[0]
        child_rng = np.random.Generator(child_bg)
        return SplittableRNG(child_rng)
    
    def random(self, size=None):
        return self._rng.random(size)
    
    def choice(self, a, size=None, replace=True, p=None):
        return self._rng.choice(a, size, replace, p)

```


**Parallel bootstrap**:


```python
from concurrent.futures import ProcessPoolExecutor

def bootstrap_worker(args):
    data, n_samples, rng_state = args
    rng = SplittableRNG(rng_state)  # Each worker gets independent RNG
    results = []
    for _ in range(n_samples):
        sample = rng.choice(data, len(data), replace=True)
        results.append(np.mean(sample))
    return results

def parallel_bootstrap(data, n_total, n_workers, rng):
    # Split RNG for each worker
    worker_rngs = [rng.split() for _ in range(n_workers)]
    samples_per_worker = n_total // n_workers
    
    with ProcessPoolExecutor(n_workers) as ex:
        args = [(data, samples_per_worker, w._rng) for w in worker_rngs]
        results = list(ex.map(bootstrap_worker, args))
    return np.concatenate(results)

```


**Independence test**: Correlation between streams should be ~0.


</details>


**Proficiency 2**: Write a Monte Carlo simulation that exhibits non-reproducibility due to parallel execution with shared RNG state. Fix it using per-worker RNG instances spawned from a parent. Benchmark the overhead of proper RNG management.


**Expected Time (Proficient): 20–28 minutes**

<details>
<summary>Rubric</summary>

| Dimension    | 0                                      | 1                              | 2                                            |

| ------------ | -------------------------------------- | ------------------------------ | -------------------------------------------- |

| Correctness  | Cannot demonstrate non-reproducibility | Shows issue but fix incomplete | Clear before/after demonstration, fix works  |

| Clarity      | Code confusing                         | Working but hard to follow     | Clear parallel pattern, well-documented      |

| Robustness   | Race conditions remain                 | Fixed but fragile              | Robust parallel implementation               |

| Benchmarking | No overhead measurement                | Basic timing                   | Proper benchmark showing negligible overhead |


</details>

<details>
<summary>Solution Sketch</summary>

**Problem: Shared RNG causes non-reproducibility**:


```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# BAD: Shared global RNG
results_bad = []
def worker_bad(n):
    return np.random.random(n).sum()  # Global RNG, race condition!

with ThreadPoolExecutor(max_workers=4) as executor:
    results_bad = list(

```


**Fixed: Per-worker RNG**:


```python
def monte_carlo_parallel(n_simulations, n_workers, seed):
    parent_rng = np.random.default_rng(seed)
    child_rngs = parent_rng.spawn(n_workers)
    
    def worker(args):
        worker_id, n_sims = args
        rng = child_rngs[worker_id]
        return [rng.random(1000).sum() for _ in range(n_sims)]
    
    sims_per_worker = n_simulations // n_workers
    work_items = [(i, sims_per_worker) for i in range(n_workers)]
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(

```


**Overhead benchmark**:


```python

# Overhead of spawn() is negligible: ~1µs per child RNG
import timeit
rng = np.random.default_rng(42)
print(timeit.timeit(lambda: rng.spawn(100), number=1000) / 1000)  # ~0.001s

```


</details>

<details>
<summary>Solution Sketch</summary>

**Broken version (shared state)**:


```python
np.random.seed(42)  # Shared global state

def mc_worker_broken(n_samples):
    # All workers draw from same global state - race condition!
    return [np.random.random() for _ in range(n_samples)]

# Results are non-deterministic due to thread scheduling
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(mc_worker_broken, [1000]*4))

```


**Fixed version (per-worker RNG)**:


```python
def mc_worker_fixed(args):
    n_samples, seed = args
    rng = np.random.default_rng(seed)
    return [rng.random() for _ in range(n_samples)]

def parallel_mc_fixed(n_samples, n_workers, base_seed):
    parent_rng = np.random.default_rng(base_seed)
    # Spawn independent child RNGs
    child_seeds = parent_rng.spawn(n_workers)
    
    with ProcessPoolExecutor(n_workers) as ex:
        args = [(n_samples // n_workers, seed) for seed in child_seeds]
        results = list(ex.map(mc_worker_fixed, args))
    return np.concatenate(results)

# Deterministic: same results every time
r1 = parallel_mc_fixed(10000, 4, 42)
r2 = parallel_mc_fixed(10000, 4, 42)
assert np.array_equal(r1, r2)

```


**Overhead benchmark**:

- RNG creation: ~1µs per `default_rng()` call
- Negligible vs typical simulation workload
- Proper RNG adds <0.1% overhead for most Monte Carlo simulations

</details>


**Mastery**: Implement a decorator `@reproducible(seed_arg='rng')` that intercepts a function, logs the RNG state before execution, and enables replay of the exact same random sequence for debugging. The decorator must work with both `Generator` instances and integer seeds.


**Expected Time (Proficient): 30–45 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                         | 1                                     | 2                                              |

| ----------- | ----------------------------------------- | ------------------------------------- | ---------------------------------------------- |

| Correctness | Decorator breaks function or replay fails | Works for Generator but not int seeds | Works for both Generator and int, replay exact |

| Clarity     | Decorator logic unclear                   | Working but complex                   | Clean decorator pattern, well-documented       |

| Robustness  | Fails on kwargs or nested calls           | Handles typical cases                 | Handles all arg patterns, preserves metadata   |

| Logging     | No state logging                          | Basic logging                         | Complete state capture with replay capability  |

| API Design  | Hard to use                               | Usable but awkward                    | Intuitive replay() method, clear documentation |


</details>

<details>
<summary>Solution Sketch</summary>

**Core idea**: Capture RNG bit_generator state before execution; allow replay.


**Implementation**:


```python
import functools
import numpy as np
import json
from pathlib import Path

def reproducible(seed_arg='rng'):
    """Decorator that logs RNG state for replay."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find RNG in args/kwargs
            rng = kwargs.get(seed_arg)
            if rng is None:
                # Check positional args via signature
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if seed_arg in params:
                    idx = params.index(seed_arg)
                    if idx < len(args):
                        rng = args[idx]
            
            # Handle integer seed
            if isinstance(rng, int):
                rng = np.random.default_rng(rng)
                # Update kwargs for the function
                kwargs[seed_arg] = rng
            
            # Capture state before execution
            if rng is not None:
                state = rng.bit_generator.state
                wrapper._last_state = state
                wrapper._last_call = {'args': str(args), 'kwargs': str(kwargs)}
            
            return func(*args, **kwargs)
        
        wrapper._last_state = None
        wrapper._last_call = None
        
        def replay():
            """Replay last call with captured RNG state."""
            if wrapper._last_state is None:
                raise ValueError("No state captured")
            rng = np.random.default_rng()
            rng.bit_generator.state = wrapper._last_state
            print(f"Replaying with state from: {wrapper._last_call}")
            return rng
        
        wrapper.replay = replay
        return wrapper
    return decorator

# Usage
@reproducible(seed_arg='rng')
def stochastic_analysis(data, n_samples, rng):
    samples = rng.choice(data, size=(n_samples, len(data)), replace=True)
    return samples.mean(axis=1)

# First run
rng = np.random.default_rng(42)
result1 = stochastic_analysis([1, 2, 3, 4, 5], 100, rng)

# Replay exact same sequence
rng_replay = stochastic_analysis.replay()
result2 = stochastic_analysis([1, 2, 3, 4, 5], 100, rng_replay)
assert np.array_equal(result1, result2)

```


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```python
import functools
import json
import logging

logger = logging.getLogger(__name__)

def reproducible(seed_arg='rng', log_file=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the RNG from args/kwargs
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            rng_or_seed = bound.arguments.get(seed_arg)
            
            # Convert to Generator and capture state
            if isinstance(rng_or_seed, int):
                rng = np.random.default_rng(rng_or_seed)
                seed_info = {'type': 'int', 'value': rng_or_seed}
            elif isinstance(rng_or_seed, np.random.Generator):
                rng = rng_or_seed
                # Capture bit_generator state for replay
                state = rng.bit_generator.state
                seed_info = {'type': 'state', 'state': state}
            else:
                raise TypeError(f"Expected int or Generator, got {type(rng_or_seed)}")
            
            # Log for reproducibility
            log_entry = {
                'function': func.__name__,
                'seed_info': seed_info,
                'timestamp': 

```


**Usage**:


```python
@reproducible(seed_arg='rng', log_file='rng_log.jsonl')
def simulate(n, rng):
    return rng.normal(0, 1, n)

# Original run
result = simulate(100, rng=42)

# Later: replay from log file for debugging

# Read state from log, use replay() to recreate exact RNG state

```


</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why capture Generator state rather than just the seed for reproducibility?
2. How does this decorator interact with global random state vs. explicit RNG instances?
3. What are the thread-safety implications of this approach?
4. How would you extend this to handle functions that use multiple random sources?

</details>


---


## Day 7: Debugging, Profiling, and Python Capstone


### Concepts


Debugging strategies: `print` statements are crude but effective. `pdb` and `ipdb` enable interactive debugging. `breakpoint()` drops into the debugger at any point. Post-mortem debugging with [`pdb.pm`](http://pdb.pm/)`()` after an exception.


Profiling identifies bottlenecks. `cProfile` provides function-level timing. `line_profiler` provides line-level timing. `memory_profiler` tracks memory allocation. Profile before optimizing; intuition about bottlenecks is often wrong.


`timeit` measures execution time for small code snippets. For larger code, use profiling tools.


Common performance issues: unnecessary copies, Python-level loops over arrays, repeated DataFrame operations that could be batched.


### Concepts: Capstone Preparation


The Python capstone integrates all Week 1 material. You will implement a complete statistical analysis pipeline with proper structure, testing, reproducibility, and performance.


### Exercises


**Foundational 1**: Given a slow function (provided), use `cProfile` to identify the bottleneck. Write a report documenting: (a) the profiling output, (b) which function consumed the most time, (c) your hypothesis for why.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension   | 0                                        | 1                                       | 2                                                    |

| ----------- | ---------------------------------------- | --------------------------------------- | ---------------------------------------------------- |

| Correctness | Profiling output wrong or misinterpreted | Identifies slowest function incorrectly | Correctly identifies bottleneck with evidence        |

| Clarity     | Report unclear or missing sections       | Has sections but explanation weak       | Clear report with all three parts documented         |

| Analysis    | No hypothesis given                      | Hypothesis superficial                  | Hypothesis explains root cause (loops, copies, etc.) |

| Tool Usage  | Cannot run cProfile                      | Basic cProfile usage                    | Uses pstats to sort and filter results               |


</details>

<details>
<summary>Solution Sketch</summary>

**Slow function example**:


```python
def slow_analysis(data):
    results = []
    for i in range(len(data)):
        for j in range(len(data)):
            dist = np.sqrt(np.sum((data[i] - data[j])**2))
            results.append(dist)
    return np.array(results)

```


**Profiling with cProfile**:


```python
import cProfile
import pstats

data = np.random.randn(500, 10)

```


**Example output**:


```javascript
ncalls  tottime  cumtime  filename:lineno(function)
250000    2.5    2.5      {built-in method numpy.sum}
250000    1.8    4.3      {built-in method numpy.sqrt}
     1    0.8    5.1      slow_analysis

```


**Report**:

- **Bottleneck**: `numpy.sum` called 250,000 times (n² calls)
- **Cumulative time**: 2.5s in sum alone
- **Hypothesis**: Calling NumPy functions inside Python loops has overhead. Each call has fixed cost (~1µs), so 250k calls = 250ms just in call overhead. The loop structure prevents vectorization.

**Fix**: Vectorize: `np.linalg.norm(data\[:, None, :\] - data\[None, :, :\], axis=2)`


</details>

<details>
<summary>Solution Sketch</summary>

**Example slow function**:


```python
def slow_analysis(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            result.append(data[i] * data[j])  # O(n²) list appends
    return np.array(result).reshape(len(data), len(data))

```


**Profiling**:


```python
import cProfile
import pstats

data = np.random.randn(1000)

# Method 1: Simple profiling
cProfile.run('slow_analysis(data)', sort='cumulative')

# Method 2: Detailed analysis
profiler = cProfile.Profile()
profiler.enable()
result = slow_analysis(data)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

```


**Sample output analysis**:


```javascript
 ncalls  tottime  cumtime  filename:lineno(function)
      1    0.100   15.200  script.py:1(slow_analysis)
1000000    5.100    5.100  {method 'append' of 'list'}
      1    8.000    8.000  {built-in method numpy.array}

```


**Report**:

- Bottleneck: `list.append` called 1M times (5.1s) + `numpy.array` conversion (8s)
- Hypothesis: Growing list requires repeated reallocation; final conversion copies all data
- Fix: Use `np.outer(data, data)` → vectorized, single allocation, <10ms

</details>


**Foundational 2**: Use `pdb` to debug a function with an off-by-one error in array indexing. Document the debugging session: what breakpoints you set, what variables you inspected, how you identified the bug.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                       | 1                                 | 2                                                      |

| ----------------- | ----------------------- | --------------------------------- | ------------------------------------------------------ |

| Correctness       | Cannot identify the bug | Finds bug but explanation unclear | Correctly identifies off-by-one with slice explanation |

| Tool Usage        | Cannot use pdb commands | Basic p and n commands only       | Uses p, n, c, s, breakpoint() effectively              |

| Debugging Process | Random inspection       | Some method but inefficient       | Systematic: hypothesis → inspect → verify              |

| Documentation     | No session documented   | Partial documentation             | Complete session with commands and observations        |


</details>

<details>
<summary>Solution Sketch</summary>

**Buggy function**:


```python
def moving_average(data, window):
    result = np.zeros(len(data) - window + 1)
    for i in range(len(result)):
        result[i] = np.mean(data[i:i+window-1])  # BUG: off-by-one
    return result

```


**Debugging session**:


```python

# In terminal or script:
import pdb

def moving_average_debug(data, window):
    result = np.zeros(len(data) - window + 1)
    for i in range(len(result)):
        pdb.set_trace()  # Breakpoint
        result[i] = np.mean(data[i:i+window-1])
    return result

data = np.array([1, 2, 3, 4, 5])
moving_average_debug(data, 3)

```


**pdb commands used**:


```javascript
(Pdb) p i                    # Print i: 0
(Pdb) p window               # Print window: 3
(Pdb) p data[i:i+window-1]   # [1, 2] - only 2 elements!
(Pdb) p data[i:i+window]     # [1, 2, 3] - correct: 3 elements
(Pdb) n                      # Next line
(Pdb) p result[i]            # 1.5 (wrong, should be 2.0)
(Pdb) c                      # Continue

```


**Bug identification**: Slice `data\[i:i+window-1\]` excludes the last element. Python slices are half-open: `\[start, end)`. With window=3, we need indices `i, i+1, i+2`, so slice should be `data\[i:i+window\]`.


**Fix**: Change `data\[i:i+window-1\]` to `data\[i:i+window\]`


</details>

<details>
<summary>Solution Sketch</summary>

**Buggy function**:


```python
def compute_differences(arr):
    """Compute difference between consecutive elements."""
    n = len(arr)
    diffs = np.zeros(n)  # BUG: should be n-1
    for i in range(n):   # BUG: should be range(n-1)
        diffs[i] = arr[i+1] - arr[i]  # IndexError when i = n-1
    return diffs

```


**Debugging session**:


```python
import pdb

arr = np.array([1, 2, 4, 7])

# Method 1: Insert breakpoint in code
def compute_differences_debug(arr):
    n = len(arr)
    diffs = np.zeros(n)
    for i in range(n):
        pdb.set_trace()  # or just: breakpoint()
        diffs[i] = arr[i+1] - arr[i]
    return diffs

# Method 2: Post-mortem debugging
try:
    compute_differences(arr)
except IndexError:
    pdb.post_mortem()  # Drops into debugger at exception

```


**PDB commands used**:


```javascript
(Pdb) p n          # Print n → 4
(Pdb) p i          # Print i → 3 (last iteration)
(Pdb) p arr[i+1]   # IndexError! i+1 = 4, out of bounds
(Pdb) p len(arr)   # 4 - confirms array length
(Pdb) p range(n)   # range(0, 4) - loops 0,1,2,3
(Pdb) q            # Quit debugger

```


**Fix**:


```python
diffs = np.zeros(n - 1)  # Correct size
for i in range(n - 1):   # Correct range

```


**Or vectorized**: `diffs = np.diff(arr)`


</details>


**Proficiency 1**: Profile a data processing pipeline end-to-end. Identify the three slowest operations. For each, either optimize it or document why optimization is not worthwhile (Amdahl's law reasoning).


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                            | 1                                  | 2                                                 |

| -------------- | ---------------------------- | ---------------------------------- | ------------------------------------------------- |

| Profiling      | Cannot profile pipeline      | Profiles but misses key functions  | Complete profile with cumulative times            |

| Identification | Wrong bottlenecks identified | Finds some but not top 3           | Correctly identifies 3 slowest with percentages   |

| Amdahl's Law   | No Amdahl's law reasoning    | Mentions but incorrect math        | Correct speedup bounds calculated                 |

| Optimization   | No optimizations attempted   | Optimizes without measuring impact | Optimizes and verifies speedup matches prediction |


</details>

<details>
<summary>Solution Sketch</summary>

**Pipeline to profile**:


```python
def data_pipeline(filepath):
    df = 

```


**Profiling**:


```python
import cProfile

```


**Results** (hypothetical):


| Step                  | Time | % of Total |

| --------------------- | ---- | ---------- |

| Load (read_csv)       | 2.0s | 20%        |

| Clean (dropna)        | 0.3s | 3%         |

| Normalize (transform) | 1.5s | 15%        |

| Compute (apply)       | 6.2s | 62%        |


**Analysis**:

1. **Compute (62%)**: Worth optimizing. Rewrite `expensive_computation` in vectorized NumPy or Numba. Potential 10x speedup → 6.2s → 0.6s.
2. **Load (20%)**: Could use `pyarrow` or `polars` for 2x speedup, but only saves 1s. Moderate priority.
3. **Normalize (15%)**: Replace `transform(lambda)` with direct pandas operations: `df.groupby()\['value'\].transform('mean')`. 3x speedup.

**Amdahl's law**: If compute is 62% and we make it infinitely fast, max speedup = 1/(1-0.62) = 2.6x. But realistic 10x on compute: new compute = 0.62/10 = 6.2%, total speedup = 10s → 4.4s = 2.3x.


</details>

<details>
<summary>Solution Sketch</summary>

**Example pipeline**:


```python
def pipeline(df):
    df = load_data(df)           # 0.5s (10%)
    df = clean_data(df)          # 0.3s (6%)
    df = compute_features(df)    # 3.0s (60%) <-- BOTTLENECK
    df = train_model(df)         # 1.0s (20%)
    save_results(df)             # 0.2s (4%)
    return df  # Total: 5.0s

```


**Profiling**:


```python
with cProfile.Profile() as pr:
    result = pipeline(df)

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(20)

```


**Three slowest operations**:

1. `compute_features`: 3.0s (60%)
    - **Optimize**: Vectorize, use NumPy instead of loops
    - After: 0.3s. Pipeline: 2.3s. **Speedup: 2.2x**
2. `train_model`: 1.0s (20%)
    - **Optimize**: Already uses sklearn, limited room
    - Could parallelize CV, but adds complexity
    - **Decision**: Skip, Amdahl's law: max 1.25x speedup
3. `load_data`: 0.5s (10%)
    - **Optimize**: Use Parquet instead of CSV
    - After: 0.1s. But only 8% of new total.
    - **Decision**: Worth it if running many times

**Amdahl's Law applied**:

- If `compute_features` is 60% and we make it instant: max speedup = 1/(1-0.6) = 2.5x
- If `train_model` is 20% (now 43% of new total): max additional speedup = 1.75x
- Diminishing returns after the first optimization

</details>


**Proficiency 2**: Use `memory_profiler` to identify a memory leak in a function that processes data in chunks but accidentally retains references to old chunks. Fix the leak and verify memory usage is now constant.


**Expected Time (Proficient): 20–28 minutes**

<details>
<summary>Rubric</summary>

| Dimension           | 0                          | 1                                     | 2                                                        |

| ------------------- | -------------------------- | ------------------------------------- | -------------------------------------------------------- |

| Tool Usage          | Cannot run memory_profiler | Runs but output not interpreted       | Uses @profile decorator, interprets output correctly     |

| Leak Identification | Cannot find leak source    | Finds leak but wrong root cause       | Correctly identifies retained reference pattern          |

| Fix Quality         | Fix doesn't work           | Fix works but introduces other issues | Clean fix, removes accumulation, preserves functionality |

| Verification        | No verification            | Claims fixed without evidence         | Shows before/after memory profile proving constant usage |


</details>

<details>
<summary>Solution Sketch</summary>

**Leaky function**:


```python
def process_chunks_leaky(filepath, chunk_size=10000):
    all_chunks = []  # BUG: Retains all chunks!
    results = []
    
    for chunk in 

```


**Using memory_profiler**:


```bash
pip install memory_profiler
python -m memory_profiler 

```


**Or with decorator**:


```python
from memory_profiler import profile

@profile
def process_chunks_leaky(filepath, chunk_size=10000):
    ...

```


**Output showing leak**:


```javascript
Line #    Mem usage    Increment
     5     50.0 MiB     0.0 MiB   all_chunks = []
     8     60.0 MiB    10.0 MiB   all_chunks.append(chunk)  # Chunk 1
     8     70.0 MiB    10.0 MiB   all_chunks.append(chunk)  # Chunk 2
     8     80.0 MiB    10.0 MiB   all_chunks.append(chunk)  # Chunk 3...

```


**Fixed function**:


```python
def process_chunks_fixed(filepath, chunk_size=10000):
    results = []
    
    for chunk in 

```


**Verification**: Memory stays constant at ~50-60 MiB regardless of file size.


</details>

<details>
<summary>Solution Sketch</summary>

**Buggy function with memory leak**:


```python
class ChunkProcessor:
    def __init__(self):
        self.all_chunks = []  # BUG: accumulates all chunks!
    
    def process_file(self, filepath, chunk_size=10000):
        results = []
        for chunk in 

```


**Profiling with memory_profiler**:


```python

# Install: pip install memory_profiler

# Add decorator to function:
from memory_profiler import profile

@profile
def process_file_leaky(filepath):
    processor = ChunkProcessor()
    return processor.process_file(filepath)

# Run: python -m memory_profiler 

```


**Output showing leak**:


```javascript
Line #    Mem usage    Increment   Line Contents
   10     50.0 MiB     0.0 MiB    for chunk in 

```


**Fixed version**:


```python
class ChunkProcessor:
    def process_file(self, filepath, chunk_size=10000):
        results = []
        for chunk in 

```


**Verification**: Memory stays ~constant (small growth for results only).


</details>


**Proficiency 3 (Read & Critique)**: Review the following code written by a colleague. Identify at least 5 issues (bugs, performance problems, style issues, or missing edge case handling). For each issue, explain what's wrong and provide a fix.


```python
import numpy as np
import pandas as pd

def analyze_experiment(data_file, n_bootstrap=1000):
    np.random.seed(42)
    
    df = pd.read_csv(data_file)
    treatment = df[df.group == 'treatment'].value
    control = df[df.group == 'control'].value
    
    diffs = []
    for i in range(n_bootstrap):
        t_sample = np.random.choice(treatment, len(treatment))
        c_sample = np.random.choice(control, len(control))
        diffs.append(t_sample.mean() - c_sample.mean())
    
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    
    df['normalized'] = (df.value - df.value.mean()) / df.value.std()
    
    return ci_low, ci_high

```


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                                     | 1                            | 2                                                             |

| -------------- | ------------------------------------- | ---------------------------- | ------------------------------------------------------------- |

| Issues Found   | Finds fewer than 3 issues             | Finds 3-4 issues             | Finds 5+ issues with clear explanations                       |

| Fix Quality    | Fixes incorrect or incomplete         | Fixes work but not idiomatic | Fixes are correct, idiomatic, and explained                   |

| Prioritization | Cannot distinguish severity of issues | Some prioritization          | Clearly ranks issues by severity (bugs > performance > style) |


</details>

<details>
<summary>Solution Sketch</summary>

**Issues to identify**:

1. **Global random state** (`np.random.seed(42)`): Not reproducible for parallel code, affects other code. Fix: Use `rng = np.random.default_rng(42)` and pass explicitly.
2. **SettingWithCopyWarning** (`df['normalized'] = ...`): `df[`[`df.group`](http://df.group/) `== ...]` may return a view. Fix: Use `.loc` or work on `.copy()`.
3. **Unused result**: `normalized` column is computed but never returned or used. Fix: Remove dead code or return it.
4. **Python loop for bootstrap**: Slow for large `n_bootstrap`. Fix: Vectorize with NumPy: `t_samples = rng.choice(treatment, (n_bootstrap, len(treatment)))`.
5. **No input validation**: Doesn't check if groups exist, file format, or handle NaN. Fix: Add validation and error messages.
6. **Hardcoded CI level** (2.5, 97.5): Should be parameterized. Fix: Add `ci_level` parameter.
7. **List append in loop**: Minor, but `diffs` could pre-allocate. Fix: `np.empty(n_bootstrap)`.

</details>


---


### Python Capstone


**Task**: Implement a complete Bayesian A/B testing analysis pipeline.


**Expected Time (Proficient): 90–120 minutes**

<details>
<summary>Rubric</summary>

| Dimension               | 0                                     | 1                                     | 2                                                    |

| ----------------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------------------- |

| Statistical Correctness | Wrong statistical methods             | Methods correct but edge cases fail   | All methods correct, handles edge cases              |

| Architecture            | Monolithic, untestable                | Some separation but tight coupling    | Clean modules, dependency injection, pure functions  |

| Reproducibility         | Results vary between runs             | Reproducible with manual seed setting | Bitwise identical results, explicit RNG throughout   |

| Testing                 | No tests or tests don't pass          | Basic unit tests only                 | Unit + property + integration tests, all passing     |

| Performance             | Profile shows major bottleneck (>50%) | No function >30% but memory grows     | Balanced profile, constant memory                    |

| Code Quality            | flake8/mypy errors                    | Passes linting with warnings          | Passes flake8 + mypy --strict, clean code            |

| Documentation           | No design document                    | Document <500 words or superficial    | 500+ words explaining design decisions and tradeoffs |


</details>


**Requirements**:

1. **Data generation module**: Generate synthetic A/B test data with configurable effect sizes, sample sizes, and noise levels. Must use explicit RNG for reproducibility.
2. **Analysis module**: Implement both frequentist (two-sample t-test, permutation test) and Bayesian (conjugate Beta-Binomial for conversion rates) analyses. All functions must be pure (no side effects) and accept RNG instances where randomness is needed.
3. **Visualization module**: Generate publication-quality plots of posterior distributions, credible intervals, and decision boundaries.
4. **CLI interface**: A command-line tool that accepts parameters, runs the analysis, and outputs results to JSON and PNG files.
5. **Test suite**: Pytest tests covering: (a) unit tests for each statistical function with known inputs/outputs, (b) property-based tests for invariants, (c) integration tests for the full pipeline.
6. **Reproducibility**: Running the pipeline twice with the same seed must produce bitwise-identical results.

**Success Criteria**:

- All tests pass
- `cProfile` shows no single function consuming more than 30% of runtime for the default workload
- Memory usage does not grow unboundedly for increasing sample sizes
- Code passes `flake8` and `mypy --strict` with no errors
- A written document (500+ words) explains design decisions: why this module structure, why these abstractions, what tradeoffs were made
<details>
<summary>Oral Defense Questions</summary>
1. Walk me through your decision to separate frequentist and Bayesian analysis into different modules. What coupling exists between them?
2. How did you ensure bitwise reproducibility across the entire pipeline? Show me the code paths that consume randomness.
3. Your profiling shows function X takes 25% of runtime. What would you do if it needed to be faster? What are the tradeoffs?
4. Explain your choice of prior for the Bayesian analysis. How would results change with a different prior?
5. If I wanted to extend this to handle multiple variants (A/B/C/D testing), what would need to change? Would your current abstractions help or hinder?
6. How did you decide which invariants to test with property-based testing? Give me an example of a property test that caught a real bug.

</details>


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


---


# Week 2: C++


## Day 8: Stack vs Heap, Compilation, and Tooling


### Concepts


C++ distinguishes stack and heap allocation. Stack allocation is automatic: variables declared in a scope are destroyed when the scope exits. Heap allocation is manual: `new` allocates, `delete` frees. Failing to `delete` causes memory leaks. Deleting twice causes undefined behavior.


The stack is fast (pointer arithmetic) but limited in size (typically 1-8 MB). The heap is slower (allocator overhead) but large. Choose based on object lifetime and size.


Modern C++ avoids raw `new`/`delete`. Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) or containers that manage memory internally.


Compilation translates source to object files (`.o`). Linking combines object files into an executable. Common flags: `-O2` (optimization), `-g` (debug symbols), `-Wall -Wextra` (warnings), `-std=c++17` (standard version).


Debuggers: `gdb` and `lldb` enable breakpoints, stepping, and variable inspection. Sanitizers (`-fsanitize=address,undefined`) catch memory errors at runtime.


### Build Systems: Makefile vs CMake


**Makefile** is the traditional Unix build tool. Simple for small projects but becomes unwieldy for complex builds. Example:


```makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

main: main.o utils.o
	$(CXX) $(CXXFLAGS) -o main main.o utils.o

main.o: main.cpp utils.h
	$(CXX) $(CXXFLAGS) -c main.cpp

utils.o: utils.cpp utils.h
	$(CXX) $(CXXFLAGS) -c utils.cpp

clean:
	rm -f *.o main

```


**CMake** is the modern standard for C++ projects. It generates platform-specific build files (Makefiles, Ninja, Visual Studio projects). CMake is declarative and handles dependencies, compiler detection, and cross-platform builds automatically.


Basic CMakeLists.txt:


```javascript
cmake_minimum_required(VERSION 3.16)
project(MyProject VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Create executable from sources
add_executable(main main.cpp utils.cpp)

# Optional: Enable sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE -fsanitize=address,undefined)
    target_link_options(main PRIVATE -fsanitize=address,undefined)
endif()

```


**CMake workflow:**


```bash

# Create build directory (out-of-source build)
mkdir build && cd build

# Configure (generates Makefiles)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build .

# Or use make directly
make -j$(nproc)

```


**When to use which:**

- Makefile: Tiny projects, learning compilation, quick scripts
- CMake: Any project with multiple files, dependencies, or collaborators

**Foundational 1**: Write a program that allocates an array of 1,000,000 doubles on the stack. Observe the stack overflow. Then allocate the same array on the heap using `new[]`. Verify it succeeds. Remember to `delete[]`.


**Expected Time (Proficient): 10–15 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                             | 1                                      | 2                                                     |

| ----------------- | ----------------------------- | -------------------------------------- | ----------------------------------------------------- |

| Correctness       | Neither version compiles/runs | Only heap version works, no stack demo | Both versions: stack overflow observed, heap succeeds |

| Memory Management | Missing delete[] (leak)       | delete[] present but wrong form        | Correct delete[] for array, no leaks                  |

| Understanding     | Cannot explain difference     | Vague explanation                      | Explains stack size limit vs heap, allocation speed   |

| Code Quality      | Does not compile              | Compiles with warnings                 | Clean code, no warnings with -Wall                    |


</details>

<details>
<summary>Solution Sketch</summary>

**Stack overflow demonstration**:


```c++
#include <iostream>

int main() {
    // This will likely crash - stack overflow
    // double arr[1000000];  // 8MB on stack, typical limit is 1-8MB
    
    // Heap allocation succeeds
    double* arr = new double[1000000];
    std::cout << "Allocated " << sizeof(double) * 1000000 / 1e6 << " MB on heap" << std::endl;
    
    // Use the array
    for (int i = 0; i < 1000000; i++) {
        arr[i] = i * 0.1;
    }
    
    // MUST free heap memory
    delete[] arr;
    
    return 0;
}

```


**Compile and run**:


```bash
g++ -o stack_heap stack_heap.cpp
./stack_heap  # Works for heap version

# To see stack overflow, uncomment the stack array:

# Segmentation fault (core dumped)

```


**Key differences**:

- Stack: automatic cleanup, limited size (~1-8MB), very fast
- Heap: manual cleanup required, virtually unlimited, slower allocation

</details>

<details>
<summary>Solution Sketch</summary>

**Stack overflow version**:


```c++
#include <iostream>

int main() {
    // Stack allocation - will overflow!
    // 1M doubles = 8MB, typical stack is 1-8MB
    double arr[1000000];  // Segmentation fault on most systems
    arr[0] = 1.0;
    std::cout << arr[0] << std::endl;
    return 0;
}

```


**Compile and run**: `g++ -o stack_test stack.cpp && ./stack_test`


**Result**: Segmentation fault (stack overflow)


**Heap allocation version (correct)**:


```c++
#include <iostream>

int main() {
    // Heap allocation - succeeds
    double* arr = new double[1000000];  // ~8MB on heap
    arr[0] = 1.0;
    arr[999999] = 2.0;
    std::cout << arr[0] << " " << arr[999999] << std::endl;
    
    delete[] arr;  // CRITICAL: free the memory
    arr = nullptr; // Good practice: prevent dangling pointer
    return 0;
}

```


**Key points**:

- Stack: ~1-8MB typical limit, automatic cleanup
- Heap: limited only by system RAM, manual cleanup required
- `delete[]` for arrays, `delete` for single objects

</details>


**Foundational 2**: Compile a simple program with `-g` and no optimization. Use `gdb` to set a breakpoint, step through execution, and print variable values. Document the commands used.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension     | 0                               | 1                              | 2                                             |

| ------------- | ------------------------------- | ------------------------------ | --------------------------------------------- |

| Compilation   | Cannot compile with debug flags | Uses -g but wrong optimization | Correct: -g -O0 for full debug info           |

| GDB Usage     | Cannot start gdb session        | Uses break and run only        | Uses break, run, next, step, print, continue  |

| Debugging     | Cannot inspect variables        | Prints simple variables        | Inspects containers, local vars, expressions  |

| Documentation | No commands documented          | Some commands listed           | Complete session transcript with explanations |


</details>

<details>
<summary>Solution Sketch</summary>

**Simple program** (debug_example.cpp):


```c++
#include <iostream>
#include <vector>

int sum_vector(const std::vector<int>& v) {
    int total = 0;
    for (size_t i = 0; i < v.size(); i++) {
        total += v[i];
    }
    return total;
}

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    int result = sum_vector(nums);
    std::cout << "Sum: " << result << std::endl;
    return 0;
}

```


**Compile with debug symbols**:


```bash
g++ -g -O0 -o debug_example debug_example.cpp

```


**GDB session**:


```bash
gdb ./debug_example

```


**Commands used**:


```javascript
(gdb) break sum_vector           # Set breakpoint at function
(gdb) run                        # Start program
(gdb) print v                    # Print vector (shows size and data)
(gdb) print v.size()             # Print size: 5
(gdb) print total                # Print current total: 0
(gdb) next                       # Step over one line
(gdb) print i                    # Print loop counter
(gdb) print v[i]                 # Print current element
(gdb) continue                   # Run to next breakpoint or end
(gdb) quit                       # Exit gdb

```


**Tip**: Use `layout src` for split view with source code.


</details>

<details>
<summary>Solution Sketch</summary>

**Program** (debug_example.cpp):


```c++
#include <iostream>
#include <vector>

int compute_sum(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) {
        sum += x;
    }
    return sum;
}

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = compute_sum(data);
    std::cout << "Sum: " << result << std::endl;
    return 0;
}

```


**Compile**: `g++ -g -O0 -o debug_example debug_example.cpp`


**GDB session**:


```bash
$ gdb ./debug_example
(gdb) break main              # Set breakpoint at main
(gdb) break compute_sum       # Set breakpoint at function
(gdb) run                     # Start program
(gdb) next                    # Step to next line (n for short)
(gdb) print data              # Print vector (p for short)
(gdb) print data.size()       # Print size
(gdb) continue                # Continue to next breakpoint (c)
(gdb) print sum               # Print local variable
(gdb) print x                 # Print loop variable
(gdb) step                    # Step into function (s)
(gdb) backtrace               # Show call stack (bt)
(gdb) info locals             # Show all local variables
(gdb) quit                    # Exit gdb (q)

```


**Key GDB commands**: break, run, next/step, print, continue, backtrace, quit


</details>


**Proficiency 1**: Write a program with an intentional memory leak (allocate without freeing). Compile with `-fsanitize=address` and observe the error report. Fix the leak. Then introduce a use-after-free bug and observe that report.


**Expected Time (Proficient): 15–22 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                             | 1                                  | 2                                                 |

| -------------- | ----------------------------- | ---------------------------------- | ------------------------------------------------- |

| ASan Setup     | Cannot compile with sanitizer | Compiles but output not understood | Compiles correctly, interprets ASan output        |

| Leak Detection | Cannot create detectable leak | Leak detected but wrong fix        | Leak created, detected, correctly fixed           |

| UAF Detection  | Cannot create UAF bug         | UAF created but not explained      | UAF created, ASan report interpreted correctly    |

| Understanding  | Cannot explain error reports  | Partial explanation                | Explains stack trace, allocation site, error type |


</details>

<details>
<summary>Solution Sketch</summary>

**Memory leak program** (leak.cpp):


```c++
#include <iostream>

int main() {
    int* arr = new int[1000];
    arr[0] = 42;
    std::cout << arr[0] << std::endl;
    // BUG: forgot delete[] arr;
    return 0;
}

```


**Compile with AddressSanitizer**:


```bash
g++ -fsanitize=address -g -o leak leak.cpp
./leak

```


**ASan output**:


```javascript
=================================================================
==12345==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 4000 byte(s) in 1 object(s) allocated from:
    #0 operator new[](unsigned long)
    #1 main leak.cpp:4
SUMMARY: AddressSanitizer: 4000 byte(s) leaked in 1 allocation(s).

```


**Use-after-free program**:


```c++
int main() {
    int* arr = new int[100];
    delete[] arr;
    arr[0] = 42;  // BUG: use after free!
    return 0;
}

```


**ASan output**:


```javascript
=================================================================
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x...
WRITE of size 4 at 0x... thread T0
    #0 main uaf.cpp:4
0x... is located 0 bytes inside of 400-byte region freed by thread T0 here:
    #0 operator delete[](void*)
    #1 main uaf.cpp:3

```


</details>

<details>
<summary>Solution Sketch</summary>

**Memory leak**:


```c++
// leak.cpp
int main() {
    int* p = new int[100];  // Allocated
    p[0] = 42;
    // No delete[] - LEAK!
    return 0;
}

```


**Compile and run**:


```bash
g++ -fsanitize=address -g -o leak leak.cpp
./leak

```


**ASan output**:


```javascript
==12345==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 400 byte(s) in 1 object(s)
    #0 in operator new[](unsigned long)
    #1 in main leak.cpp:3

```


**Use-after-free**:


```c++
// uaf.cpp
int main() {
    int* p = new int[100];
    p[0] = 42;
    delete[] p;
    int x = p[0];  // USE-AFTER-FREE!
    return x;
}

```


**ASan output**:


```javascript
==12346==ERROR: AddressSanitizer: heap-use-after-free
READ of size 4 at 0x... thread T0
    #0 in main uaf.cpp:5
0x... is located 0 bytes inside of 400-byte region freed by
    #0 in operator delete[](void*)
    #1 in main uaf.cpp:4

```


**Fix**: Set pointer to nullptr after delete: `p = nullptr;`


</details>


**Proficiency 2**: Create a multi-file project with a header (`.h`), implementation (`.cpp`), and main file. Write a `Makefile` that compiles incrementally (only recompiles changed files). Verify incremental compilation works by modifying one file and observing which files are recompiled.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                            | 1                                | 2                                               |

| ----------------- | ---------------------------- | -------------------------------- | ----------------------------------------------- |

| Project Structure | Files don't compile together | Compiles but structure unclear   | Clean separation: header, impl, main            |

| Header Guards     | Missing or incorrect         | Present but inconsistent         | Correct #ifndef/#define/#endif pattern          |

| Makefile          | Does not work                | Works but recompiles everything  | Correct dependencies, incremental compilation   |

| Verification      | No verification              | Claims incremental without proof | Demonstrates touch + make shows partial rebuild |


</details>

<details>
<summary>Solution Sketch</summary>

**Project structure**:


```javascript
project/
├── include/
│   └── stats.h
├── src/
│   └── stats.cpp
├── main.cpp
└── Makefile

```


**stats.h**:


```c++
#ifndef STATS_H
#define STATS_H
#include <vector>
double mean(const std::vector<double>& v);
double variance(const std::vector<double>& v);
#endif

```


**stats.cpp**:


```c++
#include "stats.h"
#include <numeric>
double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double variance(const std::vector<double>& v) {
    double m = mean(v);
    double sum = 0;
    for (double x : v) sum += (x - m) * (x - m);
    return sum / (v.size() - 1);
}

```


**Makefile**:


```makefile
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -I include

SRCS = main.cpp src/stats.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = program

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

main.o: include/stats.h
src/stats.o: include/stats.h

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean

```


**Testing incremental compilation**:


```bash
make           # Compiles everything
touch src/stats.cpp
make           # Only recompiles stats.o and relinks

```


</details>

<details>
<summary>Solution Sketch</summary>

**File structure**:


```javascript
project/
  ├── math_utils.h
  ├── math_utils.cpp
  ├── main.cpp
  └── Makefile

```


**math_utils.h**:


```c++
#ifndef MATH_UTILS_H
#define MATH_UTILS_H
double compute_mean(const double* arr, int n);
double compute_std(const double* arr, int n);
#endif

```


**math_utils.cpp**:


```c++
#include "math_utils.h"
#include <cmath>

double compute_mean(const double* arr, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum / n;
}
// ... compute_std implementation

```


**Makefile**:


```makefile
CXX = g++
CXXFLAGS = -Wall -Wextra -g -O2

# Target executable
TARGET = stats_app

# Object files
OBJS = main.o math_utils.o

# Default target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Pattern rule for .cpp -> .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Dependencies
main.o: main.cpp math_utils.h
math_utils.o: math_utils.cpp math_utils.h

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

```


**Incremental compilation test**:


```bash
make           # Compiles everything
touch math_utils.cpp
make           # Only recompiles math_utils.o and relinks

```


</details>


**Mastery**: Write a program that measures allocation time for: (a) stack allocation of a 1000-element array inside a loop (1M iterations), (b) heap allocation with `new[]`/`delete[]` each iteration, (c) heap allocation once, reused across iterations. Report the timing differences and explain them in terms of allocator behavior.


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension               | 0                                  | 1                                  | 2                                                |

| ----------------------- | ---------------------------------- | ---------------------------------- | ------------------------------------------------ |

| Implementation          | Code doesn't compile               | Runs but timing methodology flawed | All three cases implemented, proper timing       |

| Optimization Prevention | Compiler optimizes away allocation | Partial prevention                 | Uses volatile or side effects correctly          |

| Measurement             | No timing reported                 | Times reported without analysis    | Clear comparison table with ratios               |

| Explanation             | No explanation                     | Mentions speed difference only     | Explains allocator overhead, syscalls, free list |

| Compilation             | Wrong optimization level           | Compiles but inconsistent flags    | Compiled with -O2, consistent methodology        |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <iostream>
#include <chrono>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

int main() {
    const int ITERATIONS = 1000000;
    const int SIZE = 1000;
    volatile double sum = 0;  // Prevent optimization
    
    // (a) Stack allocation in loop
    auto start = Clock::now();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double arr[SIZE];  // Stack allocation
        for (int i = 0; i < SIZE; i++) arr[i] = i;
        sum += arr[0];
    }
    auto stack_time = Clock::now() - start;
    
    // (b) Heap allocation each iteration
    start = Clock::now();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double* arr = new double[SIZE];
        for (int i = 0; i < SIZE; i++) arr[i] = i;
        sum += arr[0];
        delete[] arr;
    }
    auto heap_each_time = Clock::now() - start;
    
    // (c) Heap allocation once, reused
    start = Clock::now();
    double* arr = new double[SIZE];
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < SIZE; i++) arr[i] = i;
        sum += arr[0];
    }
    delete[] arr;
    auto heap_once_time = Clock::now() - start;
    
    std::cout << "Stack each iter: " 
              << std::chrono::duration<double>(stack_time).count() << "s\n";
    std::cout << "Heap each iter:  " 
              << std::chrono::duration<double>(heap_each_time).count() << "s\n";
    std::cout << "Heap once:       " 
              << std::chrono::duration<double>(heap_once_time).count() << "s\n";
    
    return 0;
}

```


**Typical results**:


```javascript
Stack each iter: 0.8s
Heap each iter:  2.5s
Heap once:       0.8s

```


**Explanation**:

- Stack: Just pointer adjustment (~1 instruction)
- Heap each: malloc/free overhead (~100-1000 cycles per call)
- Heap once: Amortized to near-zero per iteration

</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <iostream>
#include <chrono>

const int ARRAY_SIZE = 1000;
const int ITERATIONS = 1000000;

volatile double sink;  // Prevent optimization

void benchmark_stack() {
    for (int i = 0; i < ITERATIONS; i++) {
        double arr[ARRAY_SIZE];  // Stack allocation
        arr[0] = i;
        sink = arr[0];
    }
}

void benchmark_heap_repeated() {
    for (int i = 0; i < ITERATIONS; i++) {
        double* arr = new double[ARRAY_SIZE];  // Heap alloc
        arr[0] = i;
        sink = arr[0];
        delete[] arr;  // Heap free
    }
}

void benchmark_heap_reused() {
    double* arr = new double[ARRAY_SIZE];  // Allocate once
    for (int i = 0; i < ITERATIONS; i++) {
        arr[0] = i;
        sink = arr[0];
    }
    delete[] arr;
}

int main() {
    auto time_it = [](auto fn, const char* name) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << name << ": " << ms << " ms" << std::endl;
    };
    
    time_it(benchmark_stack, "Stack");
    time_it(benchmark_heap_repeated, "Heap (repeated)");
    time_it(benchmark_heap_reused, "Heap (reused)");
    return 0;
}

```


**Typical results** (compile with `-O2`):

- Stack: ~5ms
- Heap (repeated): ~500ms
- Heap (reused): ~3ms

**Explanation**:

- Stack: Just pointer arithmetic (move stack pointer)
- Heap repeated: malloc/free overhead (lock, search free list, bookkeeping) × 1M
- Heap reused: One alloc + pure computation ≈ stack speed

</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why does the allocator lock affect performance in single-threaded code?
2. How would these timings change with a custom arena allocator?
3. What compiler optimizations might affect these measurements?
4. When would repeated heap allocation be acceptable in production code?

</details>


---


## Day 9: References, Pointers, and Ownership


### Concepts


Pointers hold memory addresses. Dereferencing (`*ptr`) accesses the value. The address-of operator (`&x`) yields a pointer to `x`. Null pointers (`nullptr`) represent "points to nothing."


References are aliases. Once bound, a reference cannot be rebound. References cannot be null. Use references for function parameters to avoid copying. Use `const` references for input parameters that should not be modified.


Ownership: who is responsible for freeing a resource? Raw pointers do not convey ownership. Smart pointers do: `unique_ptr` has exclusive ownership (cannot be copied, only moved); `shared_ptr` has shared ownership (reference counted); `weak_ptr` observes without owning.


Dangling pointers/references occur when the referent is destroyed. This is undefined behavior. Sanitizers catch some cases; discipline catches others.


### Exercises


**Foundational 1**: Write a function `void swap(int& a, int& b)` that swaps two integers via references. Write a second version `void swap_ptr(int* a, int* b)` using pointers. Demonstrate both work correctly. Explain when you would choose each.


**Expected Time (Proficient): 10–15 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                              | 1                                 | 2                                                      |

| ----------------- | ------------------------------ | --------------------------------- | ------------------------------------------------------ |

| Reference Version | Does not compile or swap fails | Works but uses wrong syntax       | Correct swap using references                          |

| Pointer Version   | Does not compile or crashes    | Works but missing dereference     | Correct swap with proper dereferencing                 |

| Demonstration     | No test output                 | Tests one version only            | Before/after output for both versions                  |

| Explanation       | No comparison given            | Mentions difference superficially | Explains nullability, syntax clarity, style guidelines |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <iostream>

// Reference version
void swap_ref(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// Pointer version
void swap_ptr(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 10, y = 20;
    
    std::cout << "Before: x=" << x << ", y=" << y << std::endl;
    swap_ref(x, y);
    std::cout << "After ref swap: x=" << x << ", y=" << y << std::endl;
    
    swap_ptr(&x, &y);
    std::cout << "After ptr swap: x=" << x << ", y=" << y << std::endl;
    
    return 0;
}

```


**When to choose each**:

- **References**: Cleaner syntax, cannot be null, cannot be reseated. Use when parameter is always valid.
- **Pointers**: Can be null (optional parameters), can be reseated, works with C APIs. Use when nullability is meaningful.

```c++
// Reference: cleaner call syntax
swap_ref(x, y);  // Obvious x and y are modified

// Pointer: explicit that modification happens
swap_ptr(&x, &y);  // & makes modification visible at call site

```


**Style note**: Google style prefers pointers for output parameters to make modification visible; references for const input parameters.


</details>


**Foundational 2**: Write a function that returns a reference to a local variable. Compile and run it. Observe undefined behavior (may appear to work, may crash, may return garbage). Enable sanitizers and observe the diagnostic.


**Expected Time (Proficient): 10–15 minutes**

<details>
<summary>Rubric</summary>

| Dimension       | 0                                | 1                                       | 2                                              |

| --------------- | -------------------------------- | --------------------------------------- | ---------------------------------------------- |

| Bug Creation    | Cannot create dangling reference | Creates bug but returns pointer instead | Correctly returns reference to local           |

| UB Observation  | Does not run the code            | Runs but doesn't note behavior          | Documents observed UB (value, crash, etc.)     |

| Sanitizer Usage | Cannot compile with sanitizers   | Compiles but ignores output             | Uses -fsanitize=address, interprets output     |

| Understanding   | Cannot explain the bug           | Vague explanation                       | Explains stack lifetime, why reference dangles |


</details>

<details>
<summary>Solution Sketch</summary>

**Dangling reference** (dangling.cpp):


```c++
#include <iostream>

int& get_dangling() {
    int local = 42;  // Stack variable
    return local;    // BUG: returning reference to local!
}  // local destroyed here

int main() {
    int& ref = get_dangling();
    std::cout << ref << std::endl;  // Undefined behavior!
    return 0;
}

```


**Without sanitizers**: May print 42, garbage, or crash (undefined behavior).


**With sanitizers**:


```bash
g++ -fsanitize=address -g -o dangling dangling.cpp
./dangling

```


**ASan output**:


```javascript
=================================================================
==12345==ERROR: AddressSanitizer: stack-use-after-return
READ of size 4 at 0x...
    #0 main dangling.cpp:9
Address 0x... is located in stack of thread T0 at offset 32 in frame
    #0 get_dangling() dangling.cpp:3

```


**Compiler warning** (with -Wall):


```javascript
warning: reference to local variable 'local' returned

```


**Fix**: Return by value, or allocate on heap with smart pointer.


</details>


**Proficiency 1**: Implement a `Matrix` class that allocates its data on the heap. Implement the destructor to free memory. Demonstrate that without a proper copy constructor, copying a `Matrix` leads to double-free. Implement the copy constructor and copy assignment operator (Rule of Three).


**Expected Time (Proficient): 22–30 minutes**

<details>
<summary>Rubric</summary>

| Dimension        | 0                               | 1                                 | 2                                              |

| ---------------- | ------------------------------- | --------------------------------- | ---------------------------------------------- |

| Initial Bug      | Cannot demonstrate double-free  | Shows crash but doesn't explain   | Demonstrates shallow copy leads to double-free |

| Copy Constructor | Missing or shallow copy         | Deep copies but misses edge cases | Correct deep copy, allocates new memory        |

| Copy Assignment  | Missing or incorrect            | Works but leaks or unsafe         | Handles self-assignment, frees old memory      |

| Rule of Three    | Missing destructor or operators | Has all three but inconsistent    | Complete: destructor, copy ctor, copy assign   |


</details>

<details>
<summary>Solution Sketch</summary>

**Initial class with double-free bug**:


```c++
class Matrix {
private:
    double* data;
    size_t rows, cols;
public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data = new double[rows * cols]();
    }
    ~Matrix() {
        delete[] data;  // Destructor frees memory
    }
    // No copy constructor/assignment - compiler generates shallow copy!
};

int main() {
    Matrix a(10, 10);
    Matrix b = a;  // Shallow copy: 

```


**Rule of Three implementation**:


```c++
class Matrix {
private:
    double* data;
    size_t rows, cols;
    
public:
    // Constructor
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data = new double[rows * cols]();
    }
    
    // Destructor
    ~Matrix() {
        delete[] data;
    }
    
    // Copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new double[rows * cols];
        std::copy(

```


**Now safe**:


```c++
Matrix a(10, 10);
Matrix b = a;     // Deep copy: separate allocations
b = a;            // Assignment: works correctly

```


</details>


**Proficiency 2**: Refactor the `Matrix` class to use `std::unique_ptr<double[]>` for storage. Implement move constructor and move assignment. Delete copy constructor and copy assignment. Verify that moving is efficient (no data copying) and copying fails to compile.


**Expected Time (Proficient): 20–28 minutes**

<details>
<summary>Rubric</summary>

| Dimension        | 0                               | 1                                 | 2                                             |

| ---------------- | ------------------------------- | --------------------------------- | --------------------------------------------- |

| unique_ptr Usage | Cannot use unique_ptr for array | Uses unique_ptr but wrong syntax  | Correct unique_ptr<double[]> with make_unique |

| Move Constructor | Missing or copies data          | Moves but doesn't null source     | Efficient move with std::move, noexcept       |

| Move Assignment  | Missing or incorrect            | Works but unsafe                  | Handles self-assign, releases old resource    |

| Copy Deletion    | Copy still works                | Deleted but error message unclear | Copy deleted, demonstrates compile error      |


</details>

<details>
<summary>Solution Sketch</summary>

**unique_ptr-based Matrix**:


```c++
#include <memory>
#include <algorithm>

class Matrix {
private:
    std::unique_ptr<double[]> data;
    size_t rows, cols;
    
public:
    // Constructor
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data = std::make_unique<double[]>(rows * cols);
    }
    
    // Move constructor
    Matrix(Matrix&& other) noexcept 
        : data(std::move(

```


**Verification**: Move is O(1), no data copying. After move, source is empty.


</details>


**Mastery**: Implement a graph data structure where nodes are allocated on the heap and edges are `shared_ptr<Node>`. Demonstrate that a cycle (A→B→A) causes a memory leak due to reference counting. Fix using `weak_ptr` for back-edges. Verify no leak with sanitizers.


**Expected Time (Proficient): 28–40 minutes**

<details>
<summary>Rubric</summary>

| Dimension       | 0                                 | 1                               | 2                                              |

| --------------- | --------------------------------- | ------------------------------- | ---------------------------------------------- |

| Graph Structure | Cannot implement node/edge        | Works but incorrect ownership   | Clean Node struct with shared_ptr edges        |

| Cycle Leak Demo | Cannot create cycle               | Creates cycle but no leak proof | Demonstrates leak with ASan or destructor logs |

| weak_ptr Fix    | Cannot use weak_ptr               | Uses weak_ptr incorrectly       | Back-edges as weak_ptr, breaks cycle           |

| Verification    | No verification                   | Claims fixed without proof      | ASan shows no leak, destructors called         |

| Understanding   | Cannot explain reference counting | Partial explanation             | Explains why cycle prevents count reaching 0   |


</details>

<details>
<summary>Solution Sketch</summary>

**Graph with cycle causing leak**:


```c++
#include <memory>
#include <vector>
#include <string>

struct Node {
    std::string name;
    std::vector<std::shared_ptr<Node>> edges;
    
    Node(const std::string& n) : name(n) {
        std::cout << "Node " << name << " created\n";
    }
    ~Node() {
        std::cout << "Node " << name << " destroyed\n";
    }
};

void create_cycle_leak() {
    auto a = std::make_shared<Node>("A");
    auto b = std::make_shared<Node>("B");
    
    a->edges.push_back(b);  // A -> B
    b->edges.push_back(a);  // B -> A (cycle!)
    
    // ref count: a=2 (local + b's edge), b=2 (local + a's edge)
}  // Locals destroyed: a=1, b=1. Never reach 0. LEAK!

```


**Fixed with weak_ptr for back-edges**:


```c++
struct Node {
    std::string name;
    std::vector<std::shared_ptr<Node>> children;   // Strong refs
    std::vector<std::weak_ptr<Node>> back_edges;   // Weak refs
    
    Node(const std::string& n) : name(n) {}
    ~Node() { std::cout << "Node " << name << " destroyed\n"; }
};

void create_cycle_fixed() {
    auto a = std::make_shared<Node>("A");
    auto b = std::make_shared<Node>("B");
    
    a->children.push_back(b);    // A -> B (strong)
    b->back_edges.push_back(a);  // B -> A (weak, doesn't prevent destruction)
    
}  // a destroyed (count 1->0), then b destroyed. No leak!

```


**Verify with ASan**:


```bash
g++ -fsanitize=address -g -o graph graph.cpp
./graph  # No leak reported for fixed version

```


</details>

<details>
<summary>Oral Defense Questions</summary>
1. How do you decide which edges should be weak vs. strong in a general graph?
2. What happens if you try to access a weak_ptr whose target has been destroyed?
3. When would unique_ptr be preferable to shared_ptr for graph edges?
4. How does the destructor order matter when breaking cycles?

</details>


---


## Day 10: RAII and Resource Management


### Concepts


RAII (Resource Acquisition Is Initialization) ties resource lifetime to object lifetime. The constructor acquires the resource; the destructor releases it. This guarantees cleanup even when exceptions occur.


Examples: `std::vector` acquires memory in its constructor and frees it in its destructor. `std::fstream` opens a file in its constructor and closes it in its destructor. `std::lock_guard` acquires a mutex in its constructor and releases it in its destructor.


Custom RAII classes wrap non-RAII resources. Pattern: constructor acquires, destructor releases, delete copy operations (or implement them correctly), implement move operations.


Exception safety levels: basic (no leaks, invariants maintained), strong (operation succeeds or has no effect), no-throw. RAII enables basic safety automatically.


### Exercises


**Foundational 1**: Write an RAII wrapper for `FILE*` (C-style file handle). The constructor opens the file, the destructor closes it. Provide `read` and `write` methods. Demonstrate that the file is closed even if an exception is thrown.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension      | 0                            | 1                                      | 2                                     |

| -------------- | ---------------------------- | -------------------------------------- | ------------------------------------- |

| Constructor    | Does not open file correctly | Opens file but no error handling       | Opens file, throws on failure         |

| Destructor     | Does not close file          | Closes but may double-close            | Closes correctly, handles null        |

| Copy Semantics | Allows dangerous copy        | Deletes copy but no move               | Non-copyable, movable                 |

| Exception Demo | No exception test            | Exception thrown but cleanup not shown | Demonstrates file closed on exception |


</details>

<details>
<summary>Solution Sketch</summary>

**RAII wrapper for FILE***:


```c++
#include <cstdio>
#include <stdexcept>
#include <string>

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
    
    size_t write(const void* buffer, size_t size, size_t count) {
        return std::fwrite(buffer, size, count, file);
    }
};

void process_file() {
    FileHandle f("data.txt", "r");
    
    char buffer[1024];
    

```


</details>


**Foundational 2**: Use `std::lock_guard` to protect a shared counter accessed by multiple threads. Demonstrate that without the guard, data races cause incorrect counts. With the guard, counts are correct. Use `-fsanitize=thread` to detect races.


**Expected Time (Proficient): 15–22 minutes**

<details>
<summary>Rubric</summary>

| Dimension          | 0                        | 1                                        | 2                                       |

| ------------------ | ------------------------ | ---------------------------------------- | --------------------------------------- |

| Thread Creation    | Cannot create threads    | Creates but doesn't join properly        | Creates and joins all threads correctly |

| Race Demonstration | No race shown            | Race exists but output correct by chance | Shows incorrect count without lock      |

| lock_guard Usage   | Missing or incorrect     | Works but scope too wide/narrow          | Correct RAII lock scope, count correct  |

| TSan Usage         | Cannot compile with TSan | Compiles but ignores warnings            | TSan detects race, interprets output    |


</details>

<details>
<summary>Solution Sketch</summary>

**Without lock_guard (data race)**:


```c++
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>

int counter = 0;
std::mutex mtx;

void increment_unsafe(int n) {
    for (int i = 0; i < n; i++) {
        counter++;  // DATA RACE!
    }
}

void increment_safe(int n) {
    for (int i = 0; i < n; i++) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII lock
        counter++;
    }  // lock released automatically
}

int main() {
    std::vector<std::thread> threads;
    
    // Unsafe version
    counter = 0;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(increment_unsafe, 10000);
    }
    for (auto& t : threads) t.join();
    std::cout << "Unsafe: " << counter << " (expected 100000)\n";
    
    // Safe version
    threads.clear();
    counter = 0;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(increment_safe, 10000);
    }
    for (auto& t : threads) t.join();
    std::cout << "Safe: " << counter << " (expected 100000)\n";
    
    return 0;
}

```


**Compile with ThreadSanitizer**:


```bash
g++ -fsanitize=thread -g -o race race.cpp -pthread
./race

```


**TSan output** (unsafe version):


```javascript
WARNING: ThreadSanitizer: data race
  Write of size 4 at 0x... by thread T2:
    #0 increment_unsafe(int) race.cpp:9
  Previous write of size 4 at 0x... by thread T1:
    #0 increment_unsafe(int) race.cpp:9

```


</details>


**Proficiency 1**: Implement an RAII wrapper for a database connection (simulated: constructor prints "connected", destructor prints "disconnected"). Ensure the wrapper is move-only (non-copyable). Write code that demonstrates connection cleanup on scope exit, including when exceptions are thrown.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension        | 0                               | 1                                          | 2                                         |

| ---------------- | ------------------------------- | ------------------------------------------ | ----------------------------------------- |

| RAII Pattern     | Resource not tied to lifetime   | Constructor/destructor work but incomplete | Clean acquire in ctor, release in dtor    |

| Move Semantics   | Copy allowed (dangerous)        | Move works but copy not deleted            | Move-only: move ctor/assign, copy deleted |

| Move Correctness | Move leaves source in bad state | Move works but no null check in dtor       | Move transfers ownership, source safe     |

| Exception Demo   | No exception test               | Exception thrown but output unclear        | Shows "disconnected" printed on exception |


</details>

<details>
<summary>Solution Sketch</summary>

**Move-only database connection wrapper**:


```c++
#include <iostream>
#include <string>
#include <stdexcept>

class DatabaseConnection {
private:
    bool connected = false;
    std::string name;
    
public:
    explicit DatabaseConnection(const std::string& db_name) : name(db_name) {
        std::cout << "Connected to " << name << std::endl;
        connected = true;
    }
    
    ~DatabaseConnection() {
        if (connected) {
            std::cout << "Disconnected from " << name << std::endl;
        }
    }
    
    // Non-copyable
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;
    
    // Movable
    DatabaseConnection(DatabaseConnection&& other) noexcept 
        : connected(other.connected), name(std::move(

```


</details>


**Proficiency 2**: Implement a `ScopedTimer` class that records the start time in its constructor and prints the elapsed time in its destructor. Use it to measure function execution time by placing a `ScopedTimer` at the start of a function. Demonstrate it works correctly even if the function returns early or throws.


**Expected Time (Proficient): 15–22 minutes**

<details>
<summary>Rubric</summary>

| Dimension            | 0                         | 1                                  | 2                                            |

| -------------------- | ------------------------- | ---------------------------------- | -------------------------------------------- |

| Timer Implementation | Cannot measure time       | Measures but wrong units/precision | Uses chrono correctly, millisecond precision |

| RAII Pattern         | Requires manual stop call | Auto-prints but leaks or crashes   | Clean auto-print in destructor               |

| Early Return         | Not tested                | Tested but timer doesn't fire      | Timer prints on early return                 |

| Exception Safety     | Not tested                | Tested but timer doesn't fire      | Timer prints when exception thrown           |


</details>

<details>
<summary>Solution Sketch</summary>

**ScopedTimer implementation**:


```c++
#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer {
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    
public:
    explicit ScopedTimer(const std::string& n = "Timer") : name(n) {
        start = std::chrono::high_resolution_clock::now();
    }
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        std::cout << name << ": " << duration.count() << " ms" << std::endl;
    }
    
    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};

// Usage examples
double compute_sum(int n) {
    ScopedTimer timer("compute_sum");
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += i * 0.001;
        if (i == n/2) return sum;  // Early return: timer still fires
    }
    return sum;
}

void risky_function() {
    ScopedTimer timer("risky_function");
    // ... some work ...
    throw std::runtime_error("Error!");  // Exception: timer still fires
}

int main() {
    compute_sum(1000000);     // Prints: compute_sum: X.XX ms
    
    try {
        risky_function();      // Prints: risky_function: X.XX ms
    } catch (...) {}
    
    return 0;
}

```


</details>


**Mastery**: Implement a memory pool allocator as an RAII class. The constructor allocates a large block. The `allocate(size)` method returns pointers into the block (bump allocator). The destructor frees the block. Demonstrate: (a) many small allocations are fast, (b) all memory is freed when the pool is destroyed, (c) no use-after-free is possible if allocations are only used within the pool's lifetime.


**Expected Time (Proficient): 30–45 minutes**

<details>
<summary>Rubric</summary>

| Dimension        | 0                         | 1                                    | 2                                            |

| ---------------- | ------------------------- | ------------------------------------ | -------------------------------------------- |

| Allocator Design | Cannot allocate from pool | Allocates but no bounds checking     | Bump allocator with alignment, throws on OOM |

| RAII Cleanup     | Memory not freed          | Freed but manual call needed         | All memory freed in destructor automatically |

| Performance Demo | No benchmark              | Benchmark but no comparison          | Shows pool faster than individual new calls  |

| Safety Demo      | No safety discussion      | Mentions safety but no demonstration | Shows lifetime guarantees, no UAF possible   |

| Alignment        | No alignment handling     | Fixed alignment only                 | Configurable alignment, handles max_align_t  |


</details>

<details>
<summary>Solution Sketch</summary>

**Memory pool allocator**:


```c++
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

class MemoryPool {
private:
    std::vector<char> buffer;  // RAII: automatically freed
    size_t offset = 0;
    
public:
    explicit MemoryPool(size_t size) : buffer(size) {}
    
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        // Align offset
        size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
        
        if (aligned_offset + size > buffer.size()) {
            throw std::bad_alloc();
        }
        
        void* ptr = 

```


**Key properties**:

- Allocation: O(1) bump pointer, no fragmentation
- Deallocation: O(1) reset (or automatic at scope end)
- Safety: All memory freed when pool destroyed; no per-allocation delete needed

</details>

<details>
<summary>Complete Reference Implementation</summary>

```c++
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cassert>
#include <new>

/**
 * A simple bump allocator / memory pool using RAII.
 * 
 * Key design decisions:
 * - Uses std::vector<char> for automatic cleanup (RAII)
 * - Bump pointer allocation: O(1) allocate, no individual free
 * - Configurable alignment (defaults to max_align_t)
 * - Clear error messages for debugging
 * 
 * Limitations:
 * - Cannot free individual allocations (only reset entire pool)
 * - Fixed capacity set at construction
 * - Not thread-safe
 */
class MemoryPool {
private:
    std::vector<char> buffer_;  // RAII: automatically freed in destructor
    size_t offset_ = 0;
    size_t peak_usage_ = 0;     // Track high water mark
    size_t allocation_count_ = 0;

public:
    /**
     * Construct a memory pool with given capacity.
     * @param size Total bytes available for allocation
     */
    explicit MemoryPool(size_t size) : buffer_(size) {
        if (size == 0) {
            throw std::invalid_argument("MemoryPool size must be > 0");
        }
    }
    
    // Delete copy operations to prevent accidental copies
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Allow move operations
    MemoryPool(MemoryPool&& other) noexcept
        : buffer_(std::move(other.buffer_)),
          offset_(other.offset_),
          peak_usage_(other.peak_usage_),
          allocation_count_(other.allocation_count_) {
        other.offset_ = 0;
        other.peak_usage_ = 0;
        other.allocation_count_ = 0;
    }
    
    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this != &other) {
            buffer_ = std::move(other.buffer_);
            offset_ = other.offset_;
            peak_usage_ = other.peak_usage_;
            allocation_count_ = other.allocation_count_;
            other.offset_ = 0;
            other.peak_usage_ = 0;
            other.allocation_count_ = 0;
        }
        return *this;
    }
    
    /**
     * Allocate memory from the pool.
     * @param size Number of bytes to allocate
     * @param alignment Required alignment (must be power of 2)
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if pool is exhausted
     */
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        if (size == 0) {
            return nullptr;
        }
        
        // Validate alignment is power of 2
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Alignment must be power of 2");
        }
        
        // Calculate aligned offset using bitmask trick
        // (offset + alignment - 1) & ~(alignment - 1) rounds up to alignment
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
        
        // Check if we have enough space
        if (aligned_offset + size > buffer_.size()) {
            throw std::bad_alloc();
        }
        
        void* ptr = buffer_.data() + aligned_offset;
        offset_ = aligned_offset + size;
        allocation_count_++;
        
        if (offset_ > peak_usage_) {
            peak_usage_ = offset_;
        }
        
        return ptr;
    }
    
    /**
     * Allocate and construct an object of type T.
     */
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        return new (mem) T(std::forward<Args>(args)...);
    }
    
    /**
     * Reset the pool, making all memory available again.
     * WARNING: Does not call destructors! Only use for POD types
     * or manually call destructors first.
     */
    void reset() noexcept {
        offset_ = 0;
        allocation_count_ = 0;
        // Note: peak_usage_ intentionally preserved for diagnostics
    }
    
    // Accessors
    size_t capacity() const noexcept { return buffer_.size(); }
    size_t used() const noexcept { return offset_; }
    size_t available() const noexcept { return buffer_.size() - offset_; }
    size_t peak_usage() const noexcept { return peak_usage_; }
    size_t allocation_count() const noexcept { return allocation_count_; }
};


// ============ TEST SUITE ============
void test_memory_pool() {
    std::cout << "Running MemoryPool tests...\n";
    
    // Test 1: Basic allocation
    {
        MemoryPool pool(1024);
        void* p1 = pool.allocate(100);
        void* p2 = pool.allocate(200);
        assert(p1 != nullptr);
        assert(p2 != nullptr);
        assert(p1 != p2);
        assert(pool.used() >= 300);  // >= due to alignment
        std::cout << "  [PASS] Basic allocation\n";
    }
    
    // Test 2: Alignment
    {
        MemoryPool pool(1024);
        pool.allocate(1);  // 1 byte, misaligns next
        void* p = pool.allocate(8, 64);  // Request 64-byte alignment
        assert(reinterpret_cast<uintptr_t>(p) % 64 == 0);
        std::cout << "  [PASS] Alignment works\n";
    }
    
    // Test 3: Exhaustion throws
    {
        MemoryPool pool(100);
        pool.allocate(50);
        bool threw = false;
        try {
            pool.allocate(100);  // Too big
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        assert(threw);
        std::cout << "  [PASS] Exhaustion throws bad_alloc\n";
    }
    
    // Test 4: Reset works
    {
        MemoryPool pool(100);
        pool.allocate(50);
        pool.allocate(40);
        assert(pool.used() >= 90);
        pool.reset();
        assert(pool.used() == 0);
        pool.allocate(90);  // Should work after reset
        std::cout << "  [PASS] Reset frees all memory\n";
    }
    
    // Test 5: Many small allocations are fast
    {
        MemoryPool pool(1000000);  // 1 MB
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100000; i++) {
            pool.allocate(8);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  [PASS] 100K allocations in " << ms << " ms\n";
    }
    
    // Test 6: RAII cleanup
    {
        void* leaked_ptr;
        {
            MemoryPool pool(1024);
            leaked_ptr = pool.allocate(100);
            // Pool goes out of scope here - memory is freed
        }
        // Cannot use leaked_ptr - would be use-after-free!
        // This is the RAII guarantee: memory freed at scope end
        std::cout << "  [PASS] RAII cleanup (no leak)\n";
    }
    
    // Test 7: Create typed objects
    {
        MemoryPool pool(1024);
        struct Point { double x, y, z; };
        Point* p = pool.create<Point>();
        p->x = 1.0; p->y = 2.0; p->z = 3.0;
        assert(p->x == 1.0);
        std::cout << "  [PASS] Typed object creation\n";
    }
    
    std::cout << "\nAll tests passed!\n";
}

int main() {
    test_memory_pool();
    return 0;
}

```


**Compile and run**:


```bash
g++ -std=c++17 -O2 -fsanitize=address -o pool_test pool.cpp
./pool_test

```


</details>

<details>
<summary>Oral Defense Questions</summary>
1. What are the limitations of a bump allocator compared to a general-purpose allocator?
2. How would you handle individual deallocations in this design?
3. When would you use a memory pool vs. std::pmr::monotonic_buffer_resource?
4. How does alignment affect the effective capacity of the pool?

</details>


---


## Day 11: STL Containers and Algorithms


### Concepts


`std::vector` is the default container: contiguous memory, amortized O(1) append, O(1) random access. Know the difference between `size()` and `capacity()`. Know that `push_back` may invalidate iterators and references.


`std::array` is a fixed-size array on the stack. Use when size is known at compile time.


`std::unordered_map` provides O(1) average lookup. `std::map` provides O(log n) lookup but maintains sorted order.


STL algorithms operate on iterator ranges. Examples: `std::sort`, `std::transform`, `std::accumulate`, `std::find_if`. Algorithms are generic: they work on any container with appropriate iterators.


Lambdas provide inline function objects for algorithms. Capture by value `[=]` or by reference `[&]`. Prefer explicit captures `[x, &y]` for clarity.


### Exercises


**Foundational 1**: Implement a function that computes the mean and variance of a `std::vector<double>` using `std::accumulate`. Do not write explicit loops.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension            | 0                       | 1                                     | 2                                                   |

| -------------------- | ----------------------- | ------------------------------------- | --------------------------------------------------- |

| Mean Calculation     | Incorrect or uses loop  | Uses accumulate but wrong formula     | Correct mean with std::accumulate                   |

| Variance Calculation | Incorrect or uses loop  | Uses accumulate but wrong formula     | Correct variance with lambda, sample variance (n-1) |

| No Loops             | Uses explicit for/while | Mixed: some accumulate, some loops    | Pure STL algorithms, no explicit loops              |

| Edge Cases           | Crashes on empty vector | Returns wrong value for small vectors | Handles empty vector, size=1                        |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation using std::accumulate**:


```c++
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>

std::pair<double, double> mean_variance(const std::vector<double>& v) {
    if (v.empty()) {
        throw std::invalid_argument("Empty vector");
    }
    
    // Mean using accumulate
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    
    // Variance using accumulate with custom operation
    double sum_sq = std::accumulate(v.begin(), v.end(), 0.0,
        [mean](double acc, double x) {
            return acc + (x - mean) * (x - mean);
        });
    
    double variance = sum_sq / (v.size() - 1);  // Sample variance
    
    return {mean, variance};
}

int main() {
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto [mean, var] = mean_variance(data);
    
    std::cout << "Mean: " << mean << std::endl;          // 5.5
    std::cout << "Variance: " << var << std::endl;       // 9.166...
    std::cout << "Std Dev: " << std::sqrt(var) << std::endl;
    
    return 0;
}

```


**Note**: No explicit for loops—all iteration via std::accumulate.


</details>


**Foundational 2**: Sort a vector of pairs `(value, index)` by value using `std::sort` with a lambda comparator. Then use `std::stable_sort` and explain when stability matters.


**Expected Time (Proficient): 12–18 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                   | 1                                | 2                                                    |

| --------------------- | ------------------- | -------------------------------- | ---------------------------------------------------- |

| Lambda Comparator     | Cannot write lambda | Lambda incorrect or verbose      | Clean lambda comparing first element                 |

| std::sort Usage       | Does not compile    | Works but wrong element accessed | Correct sort by value                                |

| stable_sort Demo      | Not implemented     | Implemented but no comparison    | Shows difference in output for ties                  |

| Stability Explanation | No explanation      | Vague or incorrect               | Clear examples: secondary sort, rankings, DB records |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<std::pair<double, int>> data = {
        {3.5, 0}, {1.2, 1}, {3.5, 2}, {2.8, 3}, {1.2, 4}
    };
    
    // Sort by value using lambda
    std::sort(data.begin(), data.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    
    std::cout << "After sort: ";
    for (const auto& [val, idx] : data) {
        std::cout << "(" << val << "," << idx << ") ";
    }
    // Output: (1.2,4) (1.2,1) (2.8,3) (3.5,2) (3.5,0)
    // Note: original order of equal elements NOT preserved
    
    // Reset and use stable_sort
    data = {{3.5, 0}, {1.2, 1}, {3.5, 2}, {2.8, 3}, {1.2, 4}};
    
    std::stable_sort(data.begin(), data.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    
    std::cout << "\nAfter stable_sort: ";
    for (const auto& [val, idx] : data) {
        std::cout << "(" << val << "," << idx << ") ";
    }
    // Output: (1.2,1) (1.2,4) (2.8,3) (3.5,0) (3.5,2)
    // Original order of equal elements IS preserved
    
    return 0;
}

```


**When stability matters**:

- Sorting by secondary key after primary sort
- Maintaining original order for ties (e.g., sports rankings)
- Sorting database records where insertion order is meaningful

**Performance**: stable_sort is O(n log n) but uses O(n) extra memory.


</details>


**Proficiency 1**: Given a large vector of integers, use `std::partition` to separate even and odd numbers in-place. Then use `std::nth_element` to find the median without fully sorting. Benchmark against `std::sort` followed by indexing.


**Expected Time (Proficient): 20–28 minutes**

<details>
<summary>Rubric</summary>

| Dimension           | 0                      | 1                               | 2                                         |

| ------------------- | ---------------------- | ------------------------------- | ----------------------------------------- |

| partition Usage     | Cannot use partition   | Works but predicate wrong       | Correct even/odd separation in-place      |

| nth_element Usage   | Cannot use nth_element | Finds element but wrong index   | Correct median found without full sort    |

| Benchmarking        | No timing              | Times but unfair comparison     | Fair benchmark showing nth_element faster |

| Complexity Analysis | No analysis            | Mentions O notation incorrectly | Explains O(n) vs O(n log n) difference    |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

int main() {
    std::vector<int> data(10000000);
    std::iota(data.begin(), data.end(), 0);
    std::shuffle(data.begin(), data.end(), std::mt19937{42});
    
    auto data_copy = data;
    
    // Using partition for even/odd
    auto start = std::chrono::high_resolution_clock::now();
    auto mid = std::partition(data.begin(), data.end(),
        [](int x) { return x % 2 == 0; });
    auto partition_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Evens: first " << (mid - data.begin()) << " elements\n";
    
    // Using nth_element to find median
    data = data_copy;
    start = std::chrono::high_resolution_clock::now();
    size_t median_idx = data.size() / 2;
    std::nth_element(data.begin(), data.begin() + median_idx, data.end());
    auto nth_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Median: " << data[median_idx] << "\n";
    std::cout << "nth_element time: " 
              << std::chrono::duration<double, std::milli>(nth_time).count() << " ms\n";
    
    // Compare to full sort
    data = data_copy;
    start = std::chrono::high_resolution_clock::now();
    std::sort(data.begin(), data.end());
    auto sort_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Sort time: " 
              << std::chrono::duration<double, std::milli>(sort_time).count() << " ms\n";
    
    // nth_element: O(n) average, sort: O(n log n)
    return 0;
}

```


**Typical results** (10M elements):

- nth_element: ~50ms
- full sort: ~800ms
- Speedup: 16x for finding median

</details>


**Proficiency 2**: Implement a sparse vector as `std::unordered_map<size_t, double>`. Implement dot product with another sparse vector and with a dense `std::vector<double>`. Analyze the complexity of each.


**Expected Time (Proficient): 20–28 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                              | 1                                       | 2                                                    |

| --------------------- | ------------------------------ | --------------------------------------- | ---------------------------------------------------- |

| Sparse Representation | Cannot implement sparse vector | Works but inefficient zero handling     | Correct: stores non-zeros only, handles missing keys |

| Sparse-Sparse Dot     | Incorrect result               | Correct but O(nnz1 + nnz2)              | Optimal: O(min(nnz1, nnz2))                          |

| Sparse-Dense Dot      | Incorrect result               | Correct but iterates dense              | Optimal: O(nnz) iterating sparse only                |

| Complexity Analysis   | No analysis                    | States complexity without justification | Correct analysis with nnz notation explained         |


</details>

<details>
<summary>Solution Sketch</summary>

**Sparse vector implementation**:


```c++
#include <unordered_map>
#include <vector>
#include <iostream>

class SparseVector {
private:
    std::unordered_map<size_t, double> data;
    size_t dim;
    
public:
    explicit SparseVector(size_t dimension) : dim(dimension) {}
    
    void set(size_t idx, double val) {
        if (val != 0.0) {
            data[idx] = val;
        } else {
            data.erase(idx);
        }
    }
    
    double get(size_t idx) const {
        auto it = data.find(idx);
        return it != data.end() ? it->second : 0.0;
    }
    
    // Sparse-sparse dot product: O(min(nnz1, nnz2))
    double dot(const SparseVector& other) const {
        double result = 0.0;
        const auto& smaller = data.size() < 

```


**Complexity analysis**:

- Sparse-sparse: O(min(nnz₁, nnz₂)) — iterate smaller, lookup in larger (O(1) average)
- Sparse-dense: O(nnz) — only iterate non-zeros, O(1) dense access
- Dense-dense: O(n) — must touch every element

</details>


**Mastery**: Implement an LRU (Least Recently Used) cache using `std::list` and `std::unordered_map`. `get(key)` and `put(key, value)` must be O(1). Verify correctness with tests that check eviction order. Profile to confirm O(1) behavior empirically.


**Expected Time (Proficient): 30–45 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                       | 1                                        | 2                                            |

| ----------------- | ----------------------- | ---------------------------------------- | -------------------------------------------- |

| Data Structure    | Wrong containers chosen | Correct containers but wrong linkage     | list + unordered_map with iterator storage   |

| get() Correctness | Does not work           | Returns value but doesn't update recency | O(1) lookup, moves to front                  |

| put() Correctness | Does not work           | Inserts but eviction wrong               | O(1) insert, correct LRU eviction            |

| Testing           | No tests                | Basic tests only                         | Tests verify eviction order, update behavior |

| Profiling         | No profiling            | Times but doesn't vary size              | Shows constant time across cache sizes       |


</details>

<details>
<summary>Solution Sketch</summary>

**LRU Cache implementation**:


```c++
#include <list>
#include <unordered_map>
#include <optional>

template<typename K, typename V>
class LRUCache {
private:
    size_t capacity;
    std::list<std::pair<K, V>> items;  // Front = most recent
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> index;
    
public:
    explicit LRUCache(size_t cap) : capacity(cap) {}
    
    std::optional<V> get(const K& key) {
        auto it = index.find(key);
        if (it == index.end()) {
            return std::nullopt;
        }
        // Move to front (most recently used)
        items.splice(items.begin(), items, it->second);
        return it->second->second;
    }
    
    void put(const K& key, const V& value) {
        auto it = index.find(key);
        if (it != index.end()) {
            // Update existing
            it->second->second = value;
            items.splice(items.begin(), items, it->second);
            return;
        }
        // Evict if at capacity
        if (items.size() >= capacity) {
            auto& lru = items.back();
            index.erase(lru.first);
            items.pop_back();
        }
        // Insert new
        items.emplace_front(key, value);
        index[key] = items.begin();
    }
    
    size_t size() const { return items.size(); }
};

// Test eviction order
int main() {
    LRUCache<int, std::string> cache(3);
    
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");
    cache.get(1);          // Access 1, now order: 1, 3, 2
    cache.put(4, "four");  // Evicts 2 (LRU)
    
    assert(!cache.get(2).has_value());  // 2 was evicted
    assert(cache.get(1).value() == "one");  // 1 still there
    
    return 0;
}

```


**O(1) operations**:

- get: O(1) hash lookup + O(1) list splice
- put: O(1) hash lookup + O(1) list operations

</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why is std::list necessary for O(1) LRU operations instead of std::vector?
2. How does iterator invalidation affect the map-list linkage?
3. What cache eviction policy would you use for a database buffer pool?
4. How would you make this LRU cache thread-safe?

</details>


---


## Day 12: Numerical Stability and Floating-Point


### Concepts


Floating-point representation (IEEE 754): finite precision causes rounding errors. Understand machine epsilon, denormalized numbers, infinity, and NaN.


Catastrophic cancellation occurs when subtracting nearly equal numbers. Example: computing variance via `E[X²] - E[X]²` is numerically unstable. Welford's algorithm is stable.


Order of operations matters. Summing many small numbers into a large accumulator loses precision. Kahan summation compensates for this.


Compare floating-point numbers with tolerance, not equality. Use `std::abs(a - b) < epsilon` or relative tolerance.


Overflow and underflow: intermediate results may exceed representable range. Log-space computation (log-sum-exp) avoids overflow in probability calculations.


### Exercises


**Foundational 1**: Compute the variance of `{1e10, 1e10 + 1, 1e10 + 2}` using the naive formula. Then compute using Welford's algorithm. Compare results to the true variance and explain the discrepancy.


**Expected Time (Proficient): 15–22 minutes**

<details>
<summary>Rubric</summary>

| Dimension              | 0                            | 1                                    | 2                                               |

| ---------------------- | ---------------------------- | ------------------------------------ | ----------------------------------------------- |

| Naive Implementation   | Cannot implement E[X²]-E[X]² | Implements but doesn't observe error | Correct implementation showing wrong result     |

| Welford Implementation | Cannot implement             | Implements but incorrect formula     | Correct online algorithm, right result          |

| Comparison             | No comparison to true value  | Compares but doesn't quantify error  | Shows naive=0, Welford=1, true=1                |

| Explanation            | No explanation               | Mentions precision vaguely           | Explains catastrophic cancellation, lost digits |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Naive (unstable) variance: E[X²] - E[X]²
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

// Welford's algorithm (stable)
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

int main() {
    std::vector<double> data = {1e10, 1e10 + 1, 1e10 + 2};
    
    // True variance: Var({0, 1, 2}) = 1.0
    double true_var = 1.0;
    
    std::cout << std::setprecision(15);
    std::cout << "True variance:    " << true_var << std::endl;
    std::cout << "Naive variance:   " << variance_naive(data) << std::endl;
    std::cout << "Welford variance: " << variance_welford(data) << std::endl;
    
    return 0;
}

```


**Output**:


```javascript
True variance:    1
Naive variance:   0                  // WRONG! Lost all precision
Welford variance: 1.00000000000000   // Correct

```


**Explanation**: In naive formula, we compute 1e20 - 1e20 (both terms ≈ 1e20). With ~16 digits of precision, the difference of 1 is lost in rounding. Welford's algorithm only subtracts values close in magnitude.


</details>


**Foundational 2**: Sum one million values of `1e-10` using naive summation and Kahan summation. Compare to the true sum (`1e-4`). Report the relative error of each method.


**Expected Time (Proficient): 15–20 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                 | 1                                   | 2                                          |

| ----------------- | ----------------- | ----------------------------------- | ------------------------------------------ |

| Naive Summation   | Cannot implement  | Implements but wrong true value     | Correct naive sum computed                 |

| Kahan Summation   | Cannot implement  | Implements but missing compensation | Correct: y, t, c variables used properly   |

| Error Calculation | No error reported | Absolute error only                 | Relative error for both methods            |

| Analysis          | No analysis       | Notes Kahan is better               | Explains how compensation tracks lost bits |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    const int N = 1000000;
    const double value = 1e-10;
    const double true_sum = N * value;  // 1e-4 = 0.0001
    
    // Naive summation
    double naive_sum = 0.0;
    for (int i = 0; i < N; i++) {
        naive_sum += value;
    }
    
    // Kahan (compensated) summation
    double kahan_sum = 0.0;
    double c = 0.0;  // Compensation for lost low-order bits
    for (int i = 0; i < N; i++) {
        double y = value - c;      // Compensated value
        double t = kahan_sum + y;  // Tentative sum
        c = (t - kahan_sum) - y;   // Recover lost bits
        kahan_sum = t;
    }
    
    double naive_error = std::abs(naive_sum - true_sum) / true_sum;
    double kahan_error = std::abs(kahan_sum - true_sum) / true_sum;
    
    std::cout << std::setprecision(15);
    std::cout << "True sum:      " << true_sum << std::endl;
    std::cout << "Naive sum:     " << naive_sum << std::endl;
    std::cout << "Kahan sum:     " << kahan_sum << std::endl;
    std::cout << "Naive error:   " << naive_error * 100 << "%" << std::endl;
    std::cout << "Kahan error:   " << kahan_error * 100 << "%" << std::endl;
    
    return 0;
}

```


**Typical output**:


```javascript
True sum:      0.0001
Naive sum:     0.000100000000169
Kahan sum:     0.0001
Naive error:   0.000169%
Kahan error:   0%  (or near machine epsilon)

```


**Note**: Naive accumulates rounding errors; Kahan tracks and compensates for them.


</details>


**Proficiency 1**: Implement log-sum-exp for a vector of log-probabilities. Demonstrate that the naive implementation (`log(sum(exp(x)))`) overflows for large values. Your implementation must not overflow.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                    | 1                                    | 2                                                   |

| --------------------- | -------------------- | ------------------------------------ | --------------------------------------------------- |

| Naive Implementation  | Cannot implement     | Implements but doesn't show overflow | Shows naive returns inf for large values            |

| Stable Implementation | Still overflows      | Works but subtracts wrong value      | Correct: subtracts max, handles edge cases          |

| Math Understanding    | Cannot explain trick | Knows to subtract max but not why    | Explains log(sum(exp)) = max + log(sum(exp(x-max))) |

| Testing               | No test cases        | Tests one range only                 | Tests extreme positive, negative, mixed values      |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// UNSTABLE: Overflows for large values
double logsumexp_naive(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += std::exp(x);  // exp(1000) = inf!
    }
    return std::log(sum);
}

// STABLE: Subtract max before exp
double logsumexp_stable(const std::vector<double>& v) {
    if (v.empty()) return -std::numeric_limits<double>::infinity();
    
    double max_val = *std::max_element(v.begin(), v.end());
    
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return max_val;
    }
    
    double sum = 0.0;
    for (double x : v) {
        sum += std::exp(x - max_val);  // exp(x - max) <= 1, no overflow
    }
    
    return max_val + std::log(sum);
}

int main() {
    std::vector<double> log_probs = {-1000, -999, -998};
    std::vector<double> large_vals = {1000, 1001, 1002};
    
    std::cout << "Small values:\n";
    std::cout << "  Naive:  " << logsumexp_naive(log_probs) << std::endl;   // Works
    std::cout << "  Stable: " << logsumexp_stable(log_probs) << std::endl;  // Works
    
    std::cout << "Large values:\n";
    std::cout << "  Naive:  " << logsumexp_naive(large_vals) << std::endl;  // inf!
    std::cout << "  Stable: " << logsumexp_stable(large_vals) << std::endl; // ~1002.4
    
    return 0;
}

```


**Math**: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))


</details>


**Proficiency 2**: Implement numerically stable softmax. Test on vectors with values ranging from -1000 to 1000. Verify the output sums to 1.0 (within tolerance) and contains no NaN or infinity.


**Expected Time (Proficient): 15–22 minutes**

<details>
<summary>Rubric</summary>

| Dimension       | 0                | 1                             | 2                                         |

| --------------- | ---------------- | ----------------------------- | ----------------------------------------- |

| Implementation  | Produces NaN/inf | Works for small values only   | Stable for extreme values (-1000 to 1000) |

| Max Subtraction | Not used         | Subtracts but not max element | Correctly subtracts max before exp        |

| Sum-to-One Test | No verification  | Checks but wrong tolerance    | Verifies sum within 1e-10 of 1.0          |

| NaN/Inf Check   | No check         | Checks output only            | Checks both intermediate and final values |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

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

int main() {
    // Test with extreme values
    std::vector<double> extreme = {-1000, 0, 1000};
    auto probs = softmax_stable(extreme);
    
    // Verify sum to 1
    double sum = 0;
    for (double p : probs) {
        sum += p;
        assert(!std::isnan(p) && !std::isinf(p));
    }
    assert(std::abs(sum - 1.0) < 1e-10);
    
    // Result: [0, 0, 1] approximately (1000 dominates)
    std::cout << "Probs: ";
    for (double p : probs) std::cout << p << " ";
    std::cout << "\nSum: " << sum << std::endl;
    
    return 0;
}

```


**Why stable**: Without subtracting max, exp(1000) = inf. After subtracting max=1000, we compute exp(0)=1, exp(-1000)≈0, exp(-2000)≈0. No overflow.


</details>


**Mastery**: Implement Cholesky decomposition with pivoting for positive semi-definite matrices. Handle the case where the matrix is numerically singular (eigenvalue below tolerance). Compare your implementation to naive Cholesky on a nearly-singular covariance matrix. Document the stability improvements.


**Expected Time (Proficient): 35–50 minutes**

<details>
<summary>Rubric</summary>

| Dimension            | 0                      | 1                                   | 2                                                      |

| -------------------- | ---------------------- | ----------------------------------- | ------------------------------------------------------ |

| Basic Cholesky       | Cannot implement       | Works for well-conditioned matrices | Correct L*L^T decomposition                            |

| Pivoting             | No pivoting            | Pivots but wrong selection          | Correctly selects largest diagonal, permutes rows/cols |

| Singularity Handling | Crashes on singular    | Detects but wrong handling          | Gracefully stops at rank, reports numerical rank       |

| Comparison           | No comparison to naive | Compares but same test matrix       | Shows naive fails, pivoted succeeds on near-singular   |

| Documentation        | No explanation         | Mentions stability                  | Explains why pivoting improves numerical stability     |


</details>

<details>
<summary>Solution Sketch</summary>

**Cholesky with pivoting**:


```c++
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

class CholeskyPivoted {
public:
    std::vector<std::vector<double>> L;
    std::vector<int> perm;  // Permutation indices
    int rank;               // Numerical rank
    
    void decompose(std::vector<std::vector<double>> A, double tol = 1e-10) {
        int n = A.size();
        L.assign(n, std::vector<double>(n, 0.0));
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), 0);
        
        for (int k = 0; k < n; k++) {
            // Find pivot: largest diagonal element in remaining submatrix
            int pivot = k;
            double max_diag = A[k][k];
            for (int i = k + 1; i < n; i++) {
                if (A[i][i] > max_diag) {
                    max_diag = A[i][i];
                    pivot = i;
                }
            }
            
            // Check for numerical singularity
            if (max_diag < tol) {
                rank = k;
                return;  // Stop: remaining eigenvalues too small
            }
            
            // Swap rows and columns
            if (pivot != k) {
                std::swap(perm[k], perm[pivot]);
                for (int j = 0; j < n; j++) std::swap(A[k][j], A[pivot][j]);
                for (int i = 0; i < n; i++) std::swap(A[i][k], A[i][pivot]);
                for (int j = 0; j < k; j++) std::swap(L[k][j], L[pivot][j]);
            }
            
            // Standard Cholesky step
            double sum = 0;
            for (int j = 0; j < k; j++) sum += L[k][j] * L[k][j];
            L[k][k] = std::sqrt(A[k][k] - sum);
            
            for (int i = k + 1; i < n; i++) {
                sum = 0;
                for (int j = 0; j < k; j++) sum += L[i][j] * L[k][j];
                L[i][k] = (A[i][k] - sum) / L[k][k];
            }
        }
        rank = n;
    }
};

```


**Stability improvement**: Pivoting selects largest diagonal, avoiding division by small numbers and accumulation of errors. Naive Cholesky fails on near-singular matrices; pivoted version gracefully handles rank deficiency.


</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why does pivoting improve numerical stability for Cholesky decomposition?
2. How do you choose the tolerance for detecting numerical singularity?
3. What is the relationship between pivot order and eigenvalue ordering?
4. When would you use Cholesky vs. LU decomposition for solving linear systems?

</details>


---


## Day 13: Performance Reasoning and Optimization


### Concepts


Performance reasoning starts with understanding the memory hierarchy: registers, L1/L2/L3 cache, RAM, disk. Cache misses dominate performance for memory-bound code.


Data layout matters: row-major (C-style) vs column-major (Fortran-style). Accessing contiguous memory is fast; strided access causes cache misses.


Compiler optimizations: `-O2` enables most optimizations, `-O3` enables aggressive optimizations (may increase code size). `-march=native` enables CPU-specific instructions. Profile before and after to verify improvements.


Benchmarking pitfalls: warm-up runs, preventing dead code elimination (use `volatile` or `benchmark::DoNotOptimize`), measuring steady state not transients.


Amdahl's law: optimizing a fraction f of the code yields at most 1/(1-f) speedup. Profile to find bottlenecks before optimizing.


### Exercises


**Foundational 1**: Implement matrix multiplication with row-major access pattern (ijk order) and compare to column-major access (ikj order). Measure the performance difference for 1000x1000 matrices. Explain in terms of cache behavior.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension          | 0                             | 1                          | 2                                                      |

| ------------------ | ----------------------------- | -------------------------- | ------------------------------------------------------ |

| ijk Implementation | Does not compile or incorrect | Works but wrong loop order | Correct ijk nested loops                               |

| ikj Implementation | Does not compile or incorrect | Works but same as ijk      | Correct ikj with row-wise B access                     |

| Benchmarking       | No timing                     | Times but not 1000x1000    | Proper timing, shows 2-5x difference                   |

| Cache Explanation  | No explanation                | Mentions cache vaguely     | Explains stride-1 vs stride-n access, cache line reuse |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <chrono>
#include <iostream>

using Matrix = std::vector<std::vector<double>>;

// ijk order: row-major access pattern (cache-friendly for A and C)
void matmul_ijk(const Matrix& A, const Matrix& B, Matrix& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];  // B access strides by row
            }
            C[i][j] = sum;
        }
    }
}

// ikj order: better cache usage for B
void matmul_ikj(const Matrix& A, const Matrix& B, Matrix& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) C[i][j] = 0;
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += a_ik * B[k][j];  // B[k] row accessed sequentially!
            }
        }
    }
}

int main() {
    const int n = 1000;
    Matrix A(n, std::vector<double>(n, 1.0));
    Matrix B(n, std::vector<double>(n, 1.0));
    Matrix C(n, std::vector<double>(n, 0.0));
    
    auto start = std::chrono::high_resolution_clock::now();
    matmul_ijk(A, B, C, n);
    auto ijk_time = std::chrono::high_resolution_clock::now() - start;
    
    start = std::chrono::high_resolution_clock::now();
    matmul_ikj(A, B, C, n);
    auto ikj_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "ijk: " << std::chrono::duration<double>(ijk_time).count() << "s\n";
    std::cout << "ikj: " << std::chrono::duration<double>(ikj_time).count() << "s\n";
    
    return 0;
}

```


**Typical results**: ikj is 2-5x faster than ijk.


**Cache explanation**: In ijk, `B\[k\]\[j\]` strides down column (cache miss per access). In ikj, `B\[k\]\[j\]` accesses row sequentially (cache line reuse).


</details>


**Foundational 2**: Compare the performance of `std::vector<int>` vs `std::list<int>` for: (a) sequential traversal, (b) random access by index, (c) insertion in the middle. Explain the results in terms of memory layout.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension          | 0               | 1                                | 2                                                 |

| ------------------ | --------------- | -------------------------------- | ------------------------------------------------- |

| Traversal Test     | Not implemented | Tests but unfair comparison      | Fair test, shows vector 10-15x faster             |

| Random Access Test | Not implemented | Tests but uses iterator for both | Shows O(1) vs O(n) difference clearly             |

| Insertion Test     | Not implemented | Tests but wrong position         | Mid-insertion: shows list O(1), vector O(n)       |

| Memory Explanation | No explanation  | Mentions contiguous/scattered    | Explains cache lines, prefetcher, pointer chasing |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <list>
#include <chrono>
#include <numeric>
#include <iostream>

int main() {
    const int N = 1000000;
    
    std::vector<int> vec(N);
    std::list<int> lst;
    std::iota(vec.begin(), vec.end(), 0);
    for (int i = 0; i < N; i++) lst.push_back(i);
    
    // (a) Sequential traversal
    auto start = std::chrono::high_resolution_clock::now();
    long sum = 0;
    for (int x : vec) sum += x;
    auto vec_trav = std::chrono::high_resolution_clock::now() - start;
    
    start = std::chrono::high_resolution_clock::now();
    sum = 0;
    for (int x : lst) sum += x;
    auto lst_trav = std::chrono::high_resolution_clock::now() - start;
    
    // (b) Random access by index
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) sum += vec[rand() % N];
    auto vec_rand = std::chrono::high_resolution_clock::now() - start;
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        auto it = lst.begin();
        std::advance(it, rand() % N);
        sum += *it;
    }
    auto lst_rand = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Traversal - vector: " << vec_trav.count() / 1e6 << "ms, "
              << "list: " << lst_trav.count() / 1e6 << "ms\n";
    std::cout << "Random - vector: " << vec_rand.count() / 1e6 << "ms, "
              << "list: " << lst_rand.count() / 1e6 << "ms\n";
    
    return 0;
}

```


**Typical results**:

- Traversal: vector ~1ms, list ~15ms (15x slower)
- Random access: vector ~0.01ms, list ~5000ms (500,000x slower!)

**Explanation**:

- Vector: contiguous memory → prefetcher works, cache lines reused
- List: nodes scattered → every access is cache miss, no prefetch benefit
- Random access: vector O(1), list O(n) to reach element

</details>


**Proficiency 1**: Implement a naive sum-of-squares function and an unrolled version (process 4 elements per loop iteration). Measure speedup. Then enable `-O3` and measure again. Explain why the difference changes.


**Expected Time (Proficient): 18–25 minutes**

<details>
<summary>Rubric</summary>

| Dimension               | 0                      | 1                                     | 2                                                               |

| ----------------------- | ---------------------- | ------------------------------------- | --------------------------------------------------------------- |

| Naive Implementation    | Incorrect result       | Correct but optimized away            | Correct with volatile/DoNotOptimize                             |

| Unrolled Implementation | Incorrect or not 4-way | Unrolls but wrong accumulator pattern | 4 accumulators, handles remainder                               |

| -O0 Benchmark           | No benchmark           | Benchmarks but no speedup shown       | Shows ~2x speedup from unrolling                                |

| -O3 Comparison          | Not tested             | Tests but doesn't explain             | Shows similar performance, explains compiler auto-vectorization |


</details>

<details>
<summary>Solution Sketch</summary>

**Implementation**:


```c++
#include <vector>
#include <chrono>
#include <iostream>

double sum_squares_naive(const std::vector<double>& v) {
    double sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

double sum_squares_unrolled(const std::vector<double>& v) {
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    size_t n = v.size();
    size_t i = 0;
    
    // Process 4 elements per iteration
    for (; i + 3 < n; i += 4) {
        sum0 += v[i] * v[i];
        sum1 += v[i+1] * v[i+1];
        sum2 += v[i+2] * v[i+2];
        sum3 += v[i+3] * v[i+3];
    }
    // Handle remainder
    for (; i < n; i++) {
        sum0 += v[i] * v[i];
    }
    return sum0 + sum1 + sum2 + sum3;
}

int main() {
    std::vector<double> data(10000000, 1.0);
    
    auto time_func = [&](auto func, const char* name) {
        auto start = std::chrono::high_resolution_clock::now();
        volatile double result = func(data);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << name << ": " 
                  << std::chrono::duration<double, std::milli>(end - start).count() 
                  << " ms\n";
    };
    
    time_func(sum_squares_naive, "Naive -O0");
    time_func(sum_squares_unrolled, "Unrolled -O0");
    
    return 0;
}

```


**Results with -O0**: Unrolled ~2x faster (less loop overhead, more ILP).


**Results with -O3**: Both nearly identical—compiler auto-unrolls and vectorizes.


**Lesson**: Manual optimization often redundant with modern compilers at -O3.


</details>


**Proficiency 2**: Profile a matrix multiplication implementation using `perf stat` (or equivalent). Report cache miss rates. Implement loop tiling (blocking) and measure the improvement in cache behavior and execution time.


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension            | 0                | 1                                | 2                                                |

| -------------------- | ---------------- | -------------------------------- | ------------------------------------------------ |

| perf Usage           | Cannot run perf  | Runs but wrong counters          | Reports cache-misses, cache-references correctly |

| Tiled Implementation | Does not compile | Tiling but wrong block iteration | Correct 6-loop nest with blocking                |

| Cache Improvement    | No cache metrics | Reports but no improvement       | Shows reduced miss rate (e.g., 25%→3%)           |

| Time Improvement     | No timing        | Times but no improvement         | Shows 3-5x speedup from tiling                   |


</details>

<details>
<summary>Solution Sketch</summary>

**Tiled matrix multiplication**:


```c++
#include <vector>
#include <algorithm>

void matmul_tiled(const std::vector<std::vector<double>>& A,
                  const std::vector<std::vector<double>>& B,
                  std::vector<std::vector<double>>& C,
                  int n, int block_size = 64) {
    // Initialize C to zero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0;
    
    // Tiled loop nest
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                // Mini matrix multiply within tile
                int i_end = std::min(ii + block_size, n);
                int j_end = std::min(jj + block_size, n);
                int k_end = std::min(kk + block_size, n);
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double a_ik = A[i][k];
                        for (int j = jj; j < j_end; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

```


**Profiling with perf**:


```bash
g++ -O2 -o matmul matmul.cpp
perf stat -e cache-misses,cache-references ./matmul

```


**Results** (n=1024):


| Version          | Time | Cache miss rate |

| ---------------- | ---- | --------------- |

| Naive ijk        | 8.5s | 25%             |

| Tiled (block=64) | 2.1s | 3%              |


**Explanation**: Tiling ensures block fits in L1/L2 cache. Each element reused block_size times before eviction.


</details>


**Mastery**: Implement dense matrix-vector multiplication. Compare: (a) naive implementation, (b) loop-unrolled, (c) using SIMD intrinsics (e.g., AVX if available). Benchmark on 10000x10000 matrix. Document speedups and when each optimization is worthwhile.


**Expected Time (Proficient): 35–50 minutes**

<details>
<summary>Rubric</summary>

| Dimension               | 0                        | 1                             | 2                                             |

| ----------------------- | ------------------------ | ----------------------------- | --------------------------------------------- |

| Naive Implementation    | Incorrect                | Correct but slow              | Correct, proper memory layout                 |

| Unrolled Implementation | Same as naive            | Unrolls but no speedup        | 4-way unroll with separate accumulators       |

| SIMD Implementation     | Not attempted or crashes | Uses intrinsics but incorrect | Correct AVX with FMA, horizontal sum          |

| Benchmarking            | No benchmark             | Benchmarks but wrong size     | 10000x10000, all three versions compared      |

| Documentation           | No analysis              | Reports speedups only         | Explains when each optimization is worthwhile |


</details>

<details>
<summary>Solution Sketch</summary>

**Naive implementation**:


```c++
void matvec_naive(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

```


**Unrolled version**:


```c++
void matvec_unrolled(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        int j = 0;
        for (; j + 3 < n; j += 4) {
            sum0 += A[i * n + j] * x[j];
            sum1 += A[i * n + j + 1] * x[j + 1];
            sum2 += A[i * n + j + 2] * x[j + 2];
            sum3 += A[i * n + j + 3] * x[j + 3];
        }
        for (; j < n; j++) sum0 += A[i * n + j] * x[j];
        y[i] = sum0 + sum1 + sum2 + sum3;
    }
}

```


**AVX SIMD version**:


```c++
#include <immintrin.h>

void matvec_avx(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        __m256d sum = _mm256_setzero_pd();
        int j = 0;
        for (; j + 3 < n; j += 4) {
            __m256d a = _mm256_loadu_pd(&A[i * n + j]);
            __m256d xv = _mm256_loadu_pd(&x[j]);
            sum = _mm256_fmadd_pd(a, xv, sum);  // FMA: sum += a * x
        }
        // Horizontal sum of 4 doubles
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        double result = temp[0] + temp[1] + temp[2] + temp[3];
        // Handle remainder
        for (; j < n; j++) result += A[i * n + j] * x[j];
        y[i] = result;
    }
}

```


**Benchmark results** (10000x10000, compiled with -O3 -march=native):

- Naive: ~180ms
- Unrolled: ~120ms (1.5x speedup)
- AVX: ~50ms (3.6x speedup)

**When each is worthwhile**:

- Naive: Readable, use when not performance-critical or with -O3 (compiler vectorizes)
- Unrolled: Marginal benefit with -O3; useful without optimization or on older compilers
- SIMD: Maximum performance; use in production numerical kernels where every cycle matters

**Compile flags**: `g++ -O3 -march=native -mavx2 -mfma -o matvec matvec.cpp`


</details>


---

<details>
<summary>Solution Sketch</summary>

**Implementations**:


```c++
#include <vector>
#include <immintrin.h>  // AVX intrinsics

// (a) Naive
void matvec_naive(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// (b) Loop unrolled
void matvec_unrolled(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        int j = 0;
        for (; j + 3 < n; j += 4) {
            sum0 += A[i * n + j] * x[j];
            sum1 += A[i * n + j + 1] * x[j + 1];
            sum2 += A[i * n + j + 2] * x[j + 2];
            sum3 += A[i * n + j + 3] * x[j + 3];
        }
        for (; j < n; j++) sum0 += A[i * n + j] * x[j];
        y[i] = sum0 + sum1 + sum2 + sum3;
    }
}

// (c) AVX SIMD (if available)
void matvec_avx(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        __m256d sum = _mm256_setzero_pd();
        int j = 0;
        for (; j + 3 < n; j += 4) {
            __m256d a = _mm256_loadu_pd(&A[i * n + j]);
            __m256d xv = _mm256_loadu_pd(&x[j]);
            sum = _mm256_fmadd_pd(a, xv, sum);
        }
        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        double result = temp[0] + temp[1] + temp[2] + temp[3];
        // Handle remainder
        for (; j < n; j++) result += A[i * n + j] * x[j];
        y[i] = result;
    }
}

```


**Benchmark** (n=10000):


| Version  | Time  | Speedup |

| -------- | ----- | ------- |

| Naive    | 850ms | 1x      |

| Unrolled | 420ms | 2x      |

| AVX      | 110ms | 8x      |


**When worthwhile**: AVX worth it only if n≥100 and function is hot. Unrolling often redundant with -O3.


</details>

<details>
<summary>Oral Defense Questions</summary>
1. Why does the compiler not auto-vectorize the naive implementation with -O3?
2. What happens to SIMD performance with non-aligned memory access?
3. How do you balance portability vs. performance with intrinsics?
4. When would you prefer library BLAS over hand-written SIMD?

</details>


---


## Day 14: C++ vs Python Performance and C++ Capstone


### Concepts


Python's overhead: interpreter dispatch, dynamic typing, GIL (Global Interpreter Lock) for threading. NumPy reduces overhead by delegating to compiled code, but Python-level loops are still slow.


C++ eliminates interpreter overhead. Proper use of stack allocation, cache-friendly data structures, and compiler optimizations yields performance near hardware limits.


Interoperability: `pybind11` exposes C++ functions to Python. This enables writing performance-critical code in C++ and calling it from Python. The NumPy C API enables zero-copy array passing.


When to use C++: tight loops over data, performance-critical inner kernels, real-time constraints. When to stay in Python: prototyping, I/O-bound code, code that is already fast enough.


### Exercises


**Foundational 1**: Implement the same function (e.g., pairwise distances) in Python (with loops), NumPy (vectorized), and C++. Benchmark all three on 10,000 points in 100 dimensions. Report speedups.


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension           | 0                | 1                         | 2                                                |

| ------------------- | ---------------- | ------------------------- | ------------------------------------------------ |

| Python Loop Version | Incorrect result | Correct but n≤10000       | Correct pairwise distances, runs on 10000 points |

| NumPy Version       | Still uses loops | Vectorized but slow       | Fully vectorized using broadcasting              |

| C++ Version         | Does not compile | Compiles but wrong result | Correct, compiled with -O3                       |

| Benchmark           | No timing        | Times but not all three   | All three timed, speedups reported               |


</details>

<details>
<summary>Solution Sketch</summary>

**Python with loops** (pairwise_[loop.py](http://loop.py/)):


```python
import numpy as np
import time

def pairwise_loop(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
    return D

```


**NumPy vectorized**:


```python
def pairwise_numpy(X):
    X_sq = (X**2).sum(axis=1, keepdims=True)
    D_sq = X_sq + X_sq.T - 2 * X @ X.T
    return np.sqrt(np.maximum(D_sq, 0))

```


**C++ implementation**:


```c++
void pairwise_distances(const double* X, double* D, int n, int d) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < d; k++) {
                double diff = X[i * d + k] - X[j * d + k];
                sum += diff * diff;
            }
            D[i * n + j] = std::sqrt(sum);
        }
    }
}

```


**Benchmark** (n=10000, d=100):


| Version      | Time | Speedup |

| ------------ | ---- | ------- |

| Python loops | 180s | 1x      |

| NumPy        | 1.2s | 150x    |

| C++ (-O3)    | 0.8s | 225x    |


**Lesson**: NumPy gets you 90% of the way. C++ gives another 1.5x for compute-bound code.


</details>


**Foundational 2**: Identify the bottleneck in a Python data processing script using `cProfile`. Rewrite only the bottleneck in C++. Measure the speedup of the overall script.


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension           | 0                       | 1                               | 2                                                    |

| ------------------- | ----------------------- | ------------------------------- | ---------------------------------------------------- |

| Profiling           | Cannot run cProfile     | Profiles but wrong bottleneck   | Correctly identifies hottest function with %         |

| C++ Rewrite         | Does not compile        | Compiles but incorrect          | Correct C++ implementation of bottleneck             |

| Integration         | Cannot call from Python | Calls but data conversion wrong | Clean ctypes or pybind11 integration                 |

| Speedup Measurement | No measurement          | Measures C++ only, not overall  | Reports overall script speedup with Amdahl reasoning |


</details>

<details>
<summary>Solution Sketch</summary>

**Workflow**:

1. Profile Python script:

```python
import cProfile

```

1. Identify bottleneck (e.g., `compute_kernel` at 85% of time)
2. Rewrite only that function in C++:

```c++
// kernel.cpp
extern "C" double compute_kernel(double* data, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += std::exp(-data[i] * data[i]);
    }
    return result;
}

```

1. Compile as shared library:

```bash
g++ -O3 -shared -fPIC -o 

```

1. Call from Python via ctypes:

```python
import ctypes
import numpy as np

lib = ctypes.CDLL('./

```


**Result**: If bottleneck was 85% of 10s runtime, and C++ is 10x faster: 8.5s → 0.85s. Total: 10s → 2.35s (4x overall speedup).


</details>


**Proficiency 1**: Use `pybind11` to expose a C++ function computing the log-determinant of a positive definite matrix. Call it from Python and verify it matches `np.linalg.slogdet`. Benchmark against the NumPy version.


**Expected Time (Proficient): 28–40 minutes**

<details>
<summary>Rubric</summary>

| Dimension         | 0                          | 1                                 | 2                                        |

| ----------------- | -------------------------- | --------------------------------- | ---------------------------------------- |

| pybind11 Setup    | Cannot build extension     | Builds but wrong module name      | Clean build, imports correctly in Python |

| Log-det Algorithm | Wrong algorithm or crashes | Works but not via Cholesky        | Correct Cholesky-based log-det           |

| Verification      | No comparison to NumPy     | Compares but not within tolerance | Matches np.linalg.slogdet within 1e-10   |

| Benchmark         | No benchmark               | Benchmarks but not fair           | Fair comparison on same matrix sizes     |


</details>

<details>
<summary>Solution Sketch</summary>

**C++ implementation** (logdet.cpp):


```c++
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

double log_determinant(py::array_t<double> matrix) {
    auto buf = matrix.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Input must be square matrix");
    
    int n = buf.shape[0];
    double* data = static_cast<double*>(buf.ptr);
    
    // Make a copy for Cholesky decomposition
    std::vector<double> L(n * n);
    std::copy(data, data + n * n, L.begin());
    
    // Cholesky decomposition (in-place on L)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = L[i * n + j];
            for (int k = 0; k < j; k++)
                sum -= L[i * n + k] * L[j * n + k];
            
            if (i == j) {
                if (sum <= 0) throw std::runtime_error("Not positive definite");
                L[i * n + j] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    
    // log det = 2 * sum(log(diag(L)))
    double logdet = 0;
    for (int i = 0; i < n; i++)
        logdet += std::log(L[i * n + i]);
    return 2 * logdet;
}

PYBIND11_MODULE(logdet_cpp, m) {
    m.def("log_determinant", &log_determinant);
}

```


**Build**:


```bash
c++ -O3 -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    logdet.cpp -o logdet_cpp$(python3-config --extension-suffix)

```


**Python usage and verification**:


```python
import numpy as np
import logdet_cpp

A = np.random.randn(1000, 1000)
A = A @ A.T + np.eye(1000)  # Make positive definite

# NumPy reference
sign, logdet_np = np.linalg.slogdet(A)

# C++ version
logdet_cpp_val = logdet_cpp.log_determinant(A)

assert np.isclose(logdet_np, logdet_cpp_val)

```


</details>


**Proficiency 2**: Implement a Monte Carlo simulation in both Python (NumPy) and C++. The simulation should involve generating random numbers and aggregating statistics. Compare performance and code complexity. Identify the crossover point (problem size) where C++ becomes worthwhile.


**Expected Time (Proficient): 25–35 minutes**

<details>
<summary>Rubric</summary>

| Dimension             | 0                       | 1                            | 2                                                      |

| --------------------- | ----------------------- | ---------------------------- | ------------------------------------------------------ |

| NumPy Implementation  | Incorrect or uses loops | Correct but slow             | Fully vectorized, uses Generator                       |

| C++ Implementation    | Does not compile        | Compiles but wrong RNG       | Correct with std::mt19937, reproducible                |

| Crossover Analysis    | No analysis             | Times but no crossover found | Tests multiple sizes, identifies crossover point       |

| Complexity Discussion | No discussion           | Mentions lines of code       | Discusses LOC, development time, when each appropriate |


</details>

<details>
<summary>Solution Sketch</summary>

**Monte Carlo: Estimating Pi**


**Python/NumPy version**:


```python
import numpy as np

def monte_carlo_pi_numpy(n_samples):
    rng = np.random.default_rng(42)
    x = rng.random(n_samples)
    y = rng.random(n_samples)
    inside = np.sum(x**2 + y**2 <= 1)
    return 4 * inside / n_samples

```


**C++ version**:


```c++
#include <random>
#include <cmath>

double monte_carlo_pi_cpp(size_t n_samples, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    size_t inside = 0;
    for (size_t i = 0; i < n_samples; i++) {
        double x = dist(rng);
        double y = dist(rng);
        if (x * x + y * y <= 1.0) inside++;
    }
    return 4.0 * inside / n_samples;
}

```


**Benchmark results**:


| N samples   | NumPy | C++ (-O3) | Winner |

| ----------- | ----- | --------- | ------ |

| 10,000      | 0.2ms | 0.3ms     | NumPy  |

| 100,000     | 1.5ms | 2.8ms     | NumPy  |

| 1,000,000   | 15ms  | 28ms      | NumPy  |

| 100,000,000 | 1.5s  | 2.8s      | NumPy  |


**Crossover analysis**: For this simple simulation, NumPy wins at all sizes because:

1. NumPy's vectorized RNG is highly optimized
2. NumPy's aggregation (sum) uses SIMD internally
3. C++ loop has no SIMD (unless manually added)

**When C++ wins**:

1. Complex per-sample logic that can't vectorize
2. Stateful simulations (Markov chains, particle systems)
3. Memory-constrained scenarios (can't allocate n_samples array)

**Modified C++ with SIMD** (competitive):


```c++
// With AVX2 vectorization, C++ becomes ~2x faster than NumPy
// Requires manual SIMD or OpenMP SIMD pragmas
#pragma omp simd reduction(+:inside)
for (size_t i = 0; i < n_samples; i++) { ... }

```


**Code complexity comparison**:

- NumPy: 5 lines, instant to write
- C++: 15 lines basic, 40+ lines with SIMD
- Recommendation: Use NumPy unless you need custom logic or are already in a C++ codebase

</details>


---

<details>
<summary>Solution Sketch</summary>

**Monte Carlo simulation**: Estimate π via random sampling.


**Python/NumPy version**:


```python
import numpy as np
import time

def estimate_pi_numpy(n_samples, rng):
    x = rng.random(n_samples)
    y = rng.random(n_samples)
    inside = np.sum(x**2 + y**2 <= 1)
    return 4 * inside / n_samples

```


**C++ version**:


```c++
#include <random>
#include <cmath>

double estimate_pi_cpp(int n_samples, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    int inside = 0;
    for (int i = 0; i < n_samples; i++) {
        double x = dist(rng);
        double y = dist(rng);
        if (x*x + y*y <= 1.0) inside++;
    }
    return 4.0 * inside / n_samples;
}

```


**Benchmark**:


```python
import time
import numpy as np

sizes = [1000, 10000, 100000, 1000000, 10000000]

for n in sizes:
    rng = np.random.default_rng(42)
    
    start = time.time()
    pi_numpy = estimate_pi_numpy(n, rng)
    numpy_time = time.time() - start
    
    start = time.time()
    pi_cpp = estimate_pi_cpp(n, 42)  # Via pybind11
    cpp_time = time.time() - start
    
    print(f"n={n:>10}: NumPy={numpy_time:.4f}s, C++={cpp_time:.4f}s, "
          f"speedup={numpy_time/cpp_time:.1f}x")

```


**Results**:


| n          | NumPy   | C++      | Speedup |

| ---------- | ------- | -------- | ------- |

| 1,000      | 0.0001s | 0.00005s | 2x      |

| 10,000     | 0.0003s | 0.0002s  | 1.5x    |

| 100,000    | 0.002s  | 0.002s   | 1x      |

| 1,000,000  | 0.02s   | 0.015s   | 1.3x    |

| 10,000,000 | 0.18s   | 0.14s    | 1.3x    |


**Crossover**: NumPy is competitive at all sizes for this RNG-bound task. C++ wins more when computation dominates (e.g., complex per-sample calculations).


**Code complexity**: Python 4 lines vs C++ 15 lines. For <2x speedup, Python often preferred.


</details>


---


### C++ Capstone


**Task**: Implement a Gaussian Mixture Model (GMM) fitting library in C++.


**Expected Time (Proficient): 120–180 minutes**

<details>
<summary>Rubric</summary>

| Dimension       | 0                            | 1                                        | 2                                              |

| --------------- | ---------------------------- | ---------------------------------------- | ---------------------------------------------- |

| Data Structures | Raw pointers, memory leaks   | RAII but missing move semantics          | Clean Matrix/Vector with copy/move, RAII       |

| EM Algorithm    | E-step or M-step incorrect   | Both steps work but numerically unstable | Correct E/M steps with log-sum-exp stability   |

| Initialization  | Random only                  | k-means but not k-means++                | Correct k-means++ initialization               |

| Testing         | No tests                     | Some tests but not comprehensive         | Unit tests for each component, all passing     |

| Performance     | >10s for 10K points          | <10s but memory errors                   | <1s, no ASan errors                            |

| Python Binding  | Not implemented              | Binds but crashes or wrong types         | Clean pybind11, accepts/returns NumPy arrays   |

| Sklearn Match   | Results differ significantly | Close but not within tolerance           | Matches sklearn within 1e-4 on same init       |

| Documentation   | No document                  | <500 words or superficial                | 500+ words comparing C++ vs Python design/perf |


</details>


**Requirements**:

1. **Data structures**: Implement `Matrix` and `Vector` classes with appropriate constructors, destructors, and copy/move semantics. Use RAII for memory management.
2. **Core algorithms**: Implement E-step (compute responsibilities), M-step (update parameters), and log-likelihood computation. Handle numerical stability (log-sum-exp, positive-definite covariance enforcement).
3. **Initialization**: Implement k-means++ initialization for cluster centers.
4. **Convergence**: Iterate until log-likelihood change is below a threshold or maximum iterations reached.
5. **API**: Provide a clean public interface: `fit(data, n_components, max_iter, tol)`, `predict(data)`, `score(data)`.
6. **Testing**: Unit tests for each component using a testing framework (e.g., Catch2, Google Test).
7. **Python binding**: Expose the GMM class to Python via `pybind11`. Demonstrate usage from Python.

**Success Criteria**:

- All tests pass
- No memory errors detected by AddressSanitizer
- Fits 10,000 points in 20 dimensions with 5 components in under 1 second (on reasonable hardware)
- Results match `sklearn.mixture.GaussianMixture` within tolerance on the same data and initialization
- Python binding works and accepts/returns NumPy arrays
- A written document (500+ words) compares the C++ implementation to an equivalent Python/NumPy implementation: design differences, performance differences, and where each would be appropriate in a production system
<details>
<summary>Oral Defense Questions</summary>
1. Walk me through your Matrix class. Why did you choose this memory layout? How does it affect performance?
2. Explain your log-sum-exp implementation. What happens mathematically without the stability trick?
3. Show me how k-means++ initialization works. Why is it better than random initialization?
4. Your E-step computes responsibilities. What are the edge cases and how do you handle them?
5. How did you verify your C++ implementation matches sklearn? What tolerance is acceptable and why?
6. If I wanted to parallelize this GMM implementation, where would you start? What synchronization would be needed?
7. Compare the memory access patterns in your C++ implementation vs. a NumPy implementation. Where does C++ have an advantage?

</details>


---


# If You Completed This, You Are Proficient


This is not a summary. This is a concrete checklist of abilities. If you cannot do these things, you are not proficient.


## Python Proficiency Checklist

- [ ] Write a Python module with multiple files, proper imports, and no circular dependencies
- [ ] Implement a decorator that modifies function behavior (e.g., memoization, timing)
- [ ] Explain why a given NumPy operation returns a view or a copy
- [ ] Use stride manipulation to create a sliding window view without copying data
- [ ] Compute pairwise distances for 10,000 points without Python loops
- [ ] Identify and fix a `SettingWithCopyWarning` in pandas code
- [ ] Write a function that is 10x faster than an equivalent `df.apply()` call
- [ ] Write pytest tests with fixtures and parametrization
- [ ] Use Hypothesis to generate property-based tests for a numerical function
- [ ] Pass an explicit `numpy.random.Generator` to all stochastic functions
- [ ] Reproduce an analysis exactly given the same seed and environment
- [ ] Use `cProfile` to identify the slowest function in a pipeline
- [ ] Use `pdb` to inspect variable state at a breakpoint
- [ ] Explain why a function has a memory leak and fix it

## C++ Proficiency Checklist

- [ ] Explain when to use stack vs heap allocation for a given problem
- [ ] Write a class with correct destructor, copy constructor, and copy assignment (Rule of Three)
- [ ] Write a move constructor and move assignment operator
- [ ] Use `std::unique_ptr` to manage heap memory without manual `delete`
- [ ] Write an RAII class that acquires and releases a resource correctly
- [ ] Use `std::vector` and `std::unordered_map` appropriately
- [ ] Write a lambda with explicit captures for use with STL algorithms
- [ ] Implement numerically stable variance computation (Welford's algorithm)
- [ ] Implement log-sum-exp without overflow
- [ ] Compile with `-fsanitize=address` and interpret the output
- [ ] Use `gdb` or `lldb` to set breakpoints and inspect variables
- [ ] Explain why one loop order is faster than another for matrix operations
- [ ] Measure and explain cache miss rates for different data access patterns
- [ ] Expose a C++ function to Python via `pybind11`

## Cross-Language Checklist

- [ ] Given a Python function, predict whether rewriting in C++ will yield significant speedup
- [ ] Identify when NumPy is fast enough and C++ is unnecessary
- [ ] Design a system where Python handles I/O and orchestration while C++ handles computation
- [ ] Write equivalent code in both languages and verify they produce identical results
- [ ] Reason about numerical precision differences between implementations

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
