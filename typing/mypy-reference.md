# Mypy Reference

## Output Format

### Message Structure
```
<file>:<line>:<column>: <severity>: <message> [<error-code>]
```

Example:
```
src/module.py:42:13: error: Argument 1 has incompatible type "str"; expected "int"  [arg-type]
src/module.py:42:13: note: See https://mypy.rtfd.io/en/stable/_refs.html#code-arg-type for more info
```

### Severity Types
- `error:` - Type checking violations (always include error code)
- `warning:` - Issues with enabled warning flags (always include error code)
- `note:` - Multiple purposes (never include error code):
  - Supplementary context for preceding error/warning
  - Standalone information (reveal_type, reveal_locals)
  - Context headers (e.g., "In member 'foo' of class 'Test':")

### Message Characteristics
- **Errors** and **warnings** ALWAYS have error codes in brackets
- **Notes** NEVER have error codes
- Only errors affect exit code (exit 1)
- Warnings require explicit flags to enable (e.g., --warn-return-any)

### Note Types
1. **Supplementary notes** - Follow errors/warnings with additional context:
   ```
   file.py:10: error: Function is missing a return type annotation  [no-untyped-def]
   file.py:10: note: Use "-> None" if function does not return a value
   ```

2. **Context notes** - Provide location context for errors:
   ```
   file.py: note: In member "process" of class "Handler":
   file.py:25: error: Unsupported operand types for + ("int" and "str")  [operator]
   ```

3. **Standalone notes** - Independent information:
   ```
   file.py:30: note: Revealed type is "dict[str, Any]"
   file.py:35: note: Revealed local types are:
   file.py:35: note:     x: int
   file.py:35: note:     result: str
   ```

## Python API

### Import
```python
from mypy import api
from typing import Tuple, List
```

### Functions

#### `api.run(args: List[str]) -> Tuple[str, str, int]`
- **Returns**: `(stdout, stderr, exit_status)`
- **stdout**: Type checking results (errors, warnings, notes)
- **stderr**: System-level errors
- **exit_status**: 0 (success), 1 (type errors), 2 (system error)
- **Thread-safe**: Yes

#### `api.run_dmypy(args: List[str]) -> Tuple[str, str, int]`
- **Returns**: Same as `api.run()`
- **Uses**: dmypy daemon for incremental checking
- **Thread-safe**: No (modifies sys.stdout/stderr)
- **Performance**: Faster for repeated runs

### dmypy Commands
```python
api.run_dmypy(["start"])          # Start daemon
api.run_dmypy(["status"])         # Check daemon status
api.run_dmypy(["check", "path"]) # Type check files
api.run_dmypy(["stop"])          # Stop daemon
api.run_dmypy(["restart"])       # Restart daemon
```

## Command-Line Options

### Output Control
- `--show-error-codes` - Display error codes in brackets (default)
- `--hide-error-codes` - Hide error codes
- `--show-column-numbers` - Include column numbers
- `--show-error-end` - Show end position (line:column)
- `--pretty` - Pretty output with source snippets
- `--show-error-context` - Additional context in messages

### Output Format Components
1. File path (relative or absolute)
2. Line number (always shown)
3. Column number (optional with `--show-column-numbers`)
4. End position (optional with `--show-error-end`)
5. Severity (`error:`, `warning:`, `note:`)
6. Message text
7. Error code in brackets (required for errors/warnings, never for notes)

## Exit Codes
- **0**: No type errors found
- **1**: Type errors detected (warnings don't affect exit code)
- **2**: Fatal/system error

## Warning Flags
Common warning flags that enable additional checks:
- `--warn-return-any` - Warn about returning Any from typed functions
- `--warn-redundant-casts` - Warn about redundant type casts
- `--warn-unreachable` - Warn about unreachable code
- `--warn-unused-ignores` - Warn about unused `# type: ignore` comments
- `--warn-incomplete-stub` - Warn about incomplete stub packages

## Summary Line Format
```
Found <X> errors in <Y> files (checked <Z> source files)
```

## Daemon Behavior
- Caches type information between runs
- Must restart on configuration changes
- Can crash on certain errors (handle with retry logic)
- Status codes: 0 (running), 1 (not running)

## Output Parsing

### Current Limitations
- **No native JSON output**: Mypy doesn't support `--output json` (feature request #10816, #13874)
- **Third-party tools**: `mypy-json-report` provides limited JSON conversion (error counts only)
- **IDE parsing**: Most IDEs use regex patterns to parse text output

### Regex Patterns for Parsing

#### Basic Error/Warning Pattern
```regex
^(?P<file>[^:]+):(?P<line>\d+):\s*(?P<level>error|warning):\s*(?P<msg>.*?)\s*\[(?P<code>[^\]]+)\]$
```

#### With Column Numbers
```regex
^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<level>error|warning):\s*(?P<msg>.*?)\s*\[(?P<code>[^\]]+)\]$
```

#### With Error Spans (--show-error-end)
```regex
^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):(?P<end_line>\d+):(?P<end_col>\d+):\s*(?P<level>error|warning):\s*(?P<msg>.*?)\s*\[(?P<code>[^\]]+)\]$
```

#### Note Patterns
```regex
# Standard note with location
^(?P<file>[^:]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s*note:\s*(?P<msg>.*)$

# Context note (no line number)
^(?P<file>[^:]+):\s*note:\s*(?P<msg>.*)$

# Reveal type notes
^(?P<file>[^:]+):(?P<line>\d+):\s*note:\s*Revealed (?:type is|local types are)
```

#### Summary Line Pattern
```regex
^Found (?P<errors>\d+) errors? in (?P<files>\d+) files? \(checked (?P<total>\d+) source files?\)$
```

### Parsing Considerations

1. **Stateful parsing required**: Notes must be associated with preceding errors/warnings
2. **Multiple patterns needed**: Different output formats require different regex patterns
3. **Order matters**: Errors appear before their associated notes
4. **Graceful fallback**: Not all lines match patterns (empty lines, special messages)

### Parsing Gotchas and Solutions

#### Optional Regex Groups
When using regex patterns, not all groups may be present:
```python
# DANGEROUS: Assumes column group exists
column = int(match.group("column")) if match.group("column") else None

# SAFE: Check group existence first
groupdict = match.groupdict()
column = int(groupdict["column"]) if "column" in groupdict and groupdict["column"] else None
```

#### Line Number Tracking
Track line numbers consistently:
```python
# Use 1-based numbering to match editors
for line_number, line in enumerate(lines, start=1):
    # line_number matches what users see in editors
```

#### Parse Errors as Data
Instead of failing on unparseable lines, collect them:
```python
@dataclass
class ParseError:
    line_number: int
    line_content: str
    reason: str | None = None

# Collect errors instead of raising
parse_errors = []
if not any_pattern_matched:
    parse_errors.append(ParseError(line_num, line, "No pattern matched"))
```

### Parser Configuration Strategies

#### Minimal Output (No Columns)
For mypy run with `--no-column-numbers`:
```python
config = ParserConfig(show_column_numbers=False)
```

#### Full Output (With End Positions)
For mypy run with `--show-column-numbers --show-error-end`:
```python
config = ParserConfig(
    show_column_numbers=True,
    show_error_end=True
)
```

#### Debug Mode
Enable debug output to troubleshoot parsing:
```python
config = ParserConfig(debug=True)
# Outputs: [DEBUG] Line 1: Parsed as diagnostic
```

### Integration Recommendations

1. **Use a dedicated parser library**: See `dr_cli.typecheck.parser` for a reference implementation
2. **Support configuration**: Allow users to match their mypy settings
3. **Handle malformed output**: Real-world mypy output may include unexpected lines
4. **Preserve line numbers**: Essential for correlating parse errors with input
5. **Test with various formats**: Mypy output varies based on configuration flags

### Alternative Output Formats

#### JUnit XML (--junit-xml)
```bash
mypy --junit-xml=mypy-results.xml src/
```
Generates XML test results compatible with CI/CD systems.

#### Coverage Reports (--html-report, --xml-report)
```bash
mypy --html-report mypy-coverage src/
mypy --xml-report mypy-coverage src/
```
Generates type coverage reports, not error listings.

### Integration Approaches

1. **Direct API usage**: Use `mypy.api.run()` and parse stdout
2. **Subprocess with parsing**: Run mypy CLI and parse output with regex
3. **Watch mode parsing**: Parse incremental output from `dmypy check`
4. **Third-party wrappers**: Limited tools like `mypy-json-report`

### Future JSON Support
- **Pyright compatibility**: Proposed to match Pyright's `--outputjson` format
- **Feature requests**: Track #10816 and #13874 on GitHub for updates
- **Workaround**: Parse text output and generate custom JSON/structured formats

## Context-Dependent Error Counting

### The Cross-Directory Error Phenomenon

When checking multiple directories, mypy can produce different error counts than the sum of individual directory checks:

```bash
# Individual checks
mypy src/    # 230 errors
mypy tests/  # 255 errors  
mypy scripts/ # 158 errors
# Total: 643 errors

# Combined check
mypy src/ scripts/ tests/  # 1261 errors (!)
# Difference: 618 additional "cross-directory" errors
```

### Why This Happens

Mypy generates context-dependent errors based on how code is used across module boundaries:

1. **Import Resolution**: When checking directories together, mypy can resolve imports and detect type mismatches:
   ```python
   # src/utils.py
   def process_data(x):  # Missing type annotation
       return x * 2
   
   # tests/test_utils.py
   result: str = process_data(5)  # Error only when checked together!
   ```

2. **Usage Context Errors**: Untyped functions generate additional errors where they're called:
   ```
   # When checking src/ alone:
   src/utils.py:10: error: Function is missing a type annotation
   
   # When checking with tests/:
   src/utils.py:10: error: Function is missing a type annotation
   tests/test.py:5: error: Call to untyped function "process_data" in typed context
   ```

3. **Transitive Dependencies**: Test files often import src files, causing mypy to check those src files multiple times with different contexts.

### Implications for Error Reduction

1. **Cascade Effect**: Fixing type annotations in core modules eliminates both:
   - Direct errors (missing annotations)
   - Indirect errors (calls to untyped functions)

2. **Prioritization**: Focus on heavily-imported modules for maximum impact

3. **True Baseline**: Always check all directories together for accurate error counts

### Detecting Cross-Directory Errors

```bash
# Script to identify cross-directory errors
for dir in src scripts tests; do 
    mypy --output-format jsonl --output-file ".${dir}_errors.jsonl" "$dir"
done

# Check combined
mypy --output-format jsonl --output-file ".all_errors.jsonl" src scripts tests

# Compare counts
individual_total=$(jq -r '.error_count' .src_errors.jsonl .scripts_errors.jsonl .tests_errors.jsonl | awk '{sum+=$1} END {print sum}')
combined_total=$(jq -r '.error_count' .all_errors.jsonl)
echo "Cross-directory errors: $((combined_total - individual_total))"
```

### Best Practices

1. **Always run mypy on all directories** for accurate error counts
2. **Fix src/ types first** to maximize cascade benefits  
3. **Expect non-additive error counts** when dealing with interconnected modules
4. **Use file-specific error analysis** to understand which modules generate the most cross-directory errors