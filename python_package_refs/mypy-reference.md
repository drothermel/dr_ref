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