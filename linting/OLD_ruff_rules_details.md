# Ruff/Flake8 Lint Rules Research

## A: flake8-builtins

Checks for Python builtins being shadowed by variables, arguments, or imports.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| A001 | builtin-variable-shadowing | Variable is shadowing a Python builtin | Common issue with `max`, `min`, `sum`, `filter` which are frequently used in ML code |
| A002 | builtin-argument-shadowing | Function argument is shadowing a Python builtin | Avoid using `input`, `type`, `id` as parameter names in ML functions |
| A003 | builtin-attribute-shadowing | Class attribute is shadowing a Python builtin | Be careful with class attributes like `type`, `id` in model classes |
| A004 | builtin-import-shadowing | Import statement is shadowing a Python builtin | Rarely an issue in ML code |
| A005 | builtin-module-shadowing | Module is shadowing a Python standard library module | Avoid naming modules `random`, `math`, `statistics` |
| A006 | builtin-lambda-argument-shadowing | Lambda argument is shadowing a Python builtin | Common in data processing lambdas |

## ANN: flake8-annotations

Enforces type annotations in Python code.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| ANN001 | missing-type-function-argument | Missing type annotation for function argument | Can be verbose for data processing functions; consider selective enforcement |
| ANN002 | missing-type-args | Missing type annotation for *args | Often unnecessary for ML utility functions |
| ANN003 | missing-type-kwargs | Missing type annotation for **kwargs | Often unnecessary for ML configuration functions |
| ANN101 | missing-type-self | Missing type annotation for self in method | Generally not needed, self type is implicit |
| ANN102 | missing-type-cls | Missing type annotation for cls in classmethod | Generally not needed, cls type is implicit |
| ANN201 | missing-return-type-undocumented-public-function | Missing return type annotation for public function | Important for API functions |
| ANN202 | missing-return-type-private-function | Missing return type annotation for private function | Can be overly strict for internal ML utilities |
| ANN204 | missing-return-type-special-method | Missing return type annotation for special method | Usually unnecessary (__init__, __str__, etc.) |
| ANN205 | missing-return-type-static-method | Missing return type annotation for static method | Useful for utility functions |
| ANN206 | missing-return-type-class-method | Missing return type annotation for class method | Useful for factory methods |
| ANN401 | any-type | Dynamically typed expressions (typing.Any) are disallowed | Too strict for ML code that deals with various data types |

## ARG: flake8-unused-arguments

Detects unused function arguments.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| ARG001 | unused-function-argument | Unused function argument | Common in callbacks, event handlers; may need suppression |
| ARG002 | unused-method-argument | Unused method argument | Common in abstract base classes and interfaces |
| ARG003 | unused-class-method-argument | Unused class method argument | Often occurs in factory methods |
| ARG004 | unused-static-method-argument | Unused static method argument | Less common, usually indicates a real issue |
| ARG005 | unused-lambda-argument | Unused lambda argument | Common in data processing pipelines |

## B: flake8-bugbear

Detects likely bugs and design problems.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| B002 | unary-prefix-increment | No support for unary prefix increment (++) | Rarely an issue |
| B003 | assignment-to-os-environ | Assigning to os.environ doesn't clear environment | Important for experiment configuration |
| B004 | unreliable-callable-check | hasattr(x, "__call__") is unreliable for callable check | Use callable() instead |
| B005 | strip-with-multi-character-string | .strip() with multi-char strings is misleading | Can cause data preprocessing bugs |
| B006 | mutable-argument-default | Do not use mutable data structures for defaults | Critical - causes shared state bugs |
| B007 | unused-loop-control-variable | Loop control variable not used within loop body | Common with enumerate() in data iteration |
| B008 | function-call-in-default-argument | Do not perform function calls in argument defaults | Important for model initialization |
| B009 | get-attr-with-constant | Do not call getattr with a constant attribute | Use direct attribute access |
| B010 | set-attr-with-constant | Do not call setattr with a constant attribute | Use direct attribute assignment |
| B011 | assert-false | Do not assert False, raise AssertionError() | Clearer error handling |
| B012 | jump-statement-in-finally | break/continue/return in finally blocks problematic | Can silence exceptions |
| B013 | redundant-tuple-in-exception-handler | Length-one tuple literal is redundant in except | Minor style issue |
| B014 | duplicate-handler-exception | Redundant exception types in except clause | Simplify exception handling |
| B015 | useless-comparison | Pointless comparison | Wastes CPU cycles, important for training loops |
| B016 | cannot-raise-literal | Cannot raise a literal | Syntax error |
| B017 | assert-raises-exception | assertRaises(Exception) is too broad | Use specific exceptions |
| B018 | useless-expression | Found useless expression | Common in Jupyter notebooks (forgotten outputs) |
| B019 | cached-instance-method | functools.lru_cache on methods can cause memory leaks | Critical for model methods |
| B020 | loop-variable-overrides-iterator | Loop control variable overrides iterable | Causes confusing bugs |
| B021 | f-string-docstring | f-string used as docstring | Won't work as intended |
| B022 | useless-contextlib-suppress | No arguments to contextlib.suppress | Context manager does nothing |
| B023 | function-uses-loop-variable | Function in loop uses loop variable | Late-binding closure issue |
| B024 | abstract-base-class-without-abstract-method | ABC has no abstract methods | Might be missing decorators |
| B025 | duplicate-try-block-exception | try-except with duplicate exceptions | First match wins, others ignored |
| B026 | star-arg-unpacking-after-keyword-argument | Star-arg after keyword argument discouraged | Can be confusing |
| B027 | empty-method-without-abstract-decorator | Empty method in ABC without abstract decorator | Should be marked abstract |
| B028 | no-explicit-stacklevel | warnings.warn without explicit stacklevel | Use stacklevel=2 or higher |
| B029 | except-with-empty-tuple | except () catches nothing | Syntax error |
| B030 | except-with-non-exception-type | Except handlers should be exception classes | Type error |

## BLE: flake8-blind-except

Detects overly broad exception handling.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| BLE001 | blind-except | Do not catch blind exception | Avoid `except:` or `except Exception:`, be specific |

## COM: flake8-commas

Enforces trailing comma conventions.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| COM812 | missing-trailing-comma | Trailing comma missing | Helps with git diffs in configuration dicts |
| COM818 | trailing-comma-on-bare-tuple | Trailing comma on bare tuple prohibited | Style consistency |
| COM819 | prohibited-trailing-comma | Trailing comma prohibited | Single-line expressions don't need trailing commas |

## C4: flake8-comprehensions

Simplifies comprehensions and calls.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| C400 | unnecessary-generator-list | Unnecessary generator (rewrite using list()) | Memory efficiency matters for large datasets |
| C401 | unnecessary-generator-set | Unnecessary generator (rewrite using set()) | Set operations common in data deduplication |
| C402 | unnecessary-generator-dict | Unnecessary generator (rewrite as dict comprehension) | Cleaner configuration building |
| C403 | unnecessary-list-comprehension-set | Unnecessary list comprehension (rewrite as set comp) | More efficient for unique values |
| C404 | unnecessary-list-comprehension-dict | Unnecessary list comprehension (rewrite as dict comp) | Cleaner key-value transformations |
| C405 | unnecessary-literal-set | Unnecessary list/tuple literal (use set literal) | {1, 2} instead of set([1, 2]) |
| C406 | unnecessary-literal-dict | Unnecessary list/tuple literal (use dict literal) | {1: 2} instead of dict([(1, 2)]) |
| C408 | unnecessary-collection-call | Unnecessary dict/list/tuple call (use literal) | {} faster than dict() |
| C409 | unnecessary-literal-within-tuple-call | Unnecessary list/tuple passed to tuple() | Simplify tuple construction |
| C410 | unnecessary-literal-within-list-call | Unnecessary list passed to list() | Remove redundant call |
| C411 | unnecessary-list-call | Unnecessary list call around comprehension | [x for x in y] not list([x for x in y]) |
| C413 | unnecessary-call-around-sorted | Unnecessary list/reversed call around sorted() | sorted() returns a list |
| C414 | unnecessary-double-cast-or-process | Unnecessary call within another call | Avoid double processing |
| C415 | unnecessary-subscript-reversal | Unnecessary subscript reversal | Use reversed() properly |
| C416 | unnecessary-comprehension | Unnecessary comprehension (use function directly) | list(x) instead of [i for i in x] |
| C417 | unnecessary-map | Unnecessary map usage (use generator/comprehension) | Comprehensions often clearer than map with lambda |

## C90: mccabe

Measures code complexity.

| Rule Code | Rule Name | Description | ML Consideration |
|-----------|-----------|-------------|------------------|
| C901 | complex-structure | Function is too complex (cyclomatic complexity) | ML code often has high complexity; consider higher thresholds |

## ML-Specific Recommendations

### Rules to Consider Disabling/Configuring:
1. **ANN001-003**: Can be overly verbose for data processing functions
2. **ANN401**: Too strict for ML code dealing with dynamic data
3. **ARG001**: Common in callbacks and event handlers
4. **B007**: Common with enumerate() when only index is needed
5. **C901**: ML algorithms often have high complexity by nature

### Critical Rules for ML:
1. **B006**: Mutable defaults cause shared state bugs
2. **B019**: Memory leaks with cached instance methods
3. **B023**: Late-binding closures in training loops
4. **A001-002**: Shadowing builtins like max, min, sum, filter
5. **B015**: Useless comparisons waste cycles in training loops

### Performance-Focused Rules:
1. **C4xx rules**: Optimize comprehensions and data structures
2. **B015**: Remove pointless comparisons
3. **C408**: Use literals instead of function calls
4. **COM812**: Trailing commas help with configuration management