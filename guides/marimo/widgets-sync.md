# Python ↔ Svelte Reactivity Pattern

This document explains the bidirectional reactivity pattern used in the Config File Manager widget to keep Python (Marimo) and Svelte UI in sync.

## Context

- **Python widget class** (`ConfigFileManager`): exposes traitlets such as `selected_config` that live in the notebook runtime.
- **Svelte widget component** (`ConfigFileManager.svelte`): renders the UI and stores `lastWritten*` variables via `$state`.
- **Shared helpers** (`use-file-bindings.ts`, `createFileBindingHandlers`): own the `writeSelectedConfig` logic that actually touches traitlets.

Throughout this guide, “widget” refers to the Python `ConfigFileManager` class, while “component” refers to the Svelte implementation.

## Table of Contents

1. [Quick start wiring](#quick-start-wiring)
2. [The Problem](#the-problem)
3. [Why We Need the Write-Back Pattern](#why-we-need-the-write-back-pattern)
4. [The Implementation](#the-implementation)
5. [The Complete Flow](#the-complete-flow)
6. [Why Each Part is Necessary](#why-each-part-is-necessary)
7. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
8. [Future Considerations](#future-considerations)

## Quick start wiring

Use this minimal setup as a template before diving into the details below.

```python
# Python notebook cell
from widgets import ConfigFileManager

widget = ConfigFileManager()
widget.selected_config  # traitlet that will stay synced via write-back
```

```svelte
<!-- ConfigFileManager.svelte -->
<script lang="ts">
  import { createFileBindingHandlers } from "./use-file-bindings";

  const bindings = $props.bindings; // traitlets from Python
  let lastWrittenConfig = $state<string | null | undefined>(undefined);

  const bindingHandlers = createFileBindingHandlers({
    bindings,
    writeCallback: (contents) => {
      lastWrittenConfig = contents;
    },
  });

  $effect(() => {
    if (lastWrittenConfig !== bindings.selected_config) {
      bindingHandlers.writeSelectedConfig(bindings.selected_config);
    }
  });
</script>
```

This snippet highlights the only moving parts: track the last value, call the callback before touching the traitlet, and gate the write-back in `$effect`.

## The Problem

When building AnyWidget components for Marimo notebooks, we need **bidirectional reactivity**:

1. **Svelte → Python**: User interacts with widget UI → Python notebook cells should re-run
2. **Python → Svelte**: Notebook code updates widget state → UI should reflect changes

AnyWidget provides traitlet syncing that handles data transfer, but there's a subtle issue with **Marimo's reactivity system**.

## Why We Need the Write-Back Pattern

### The Issue: Python Updates Don't Always Trigger Reactivity

Consider this Marimo notebook:

```python
# Cell 1: Create widget
widget = ConfigFileManager()

# Cell 2: Update config from Python
config = {"harvest_window": 100}
widget.selected_config = json.dumps(config)

# Cell 3: Display config (should re-run when Cell 2 runs)
display(json.loads(widget.selected_config))
```

**What should happen**: When Cell 2 runs, Cell 3 should detect the change and re-run.

**What might happen without write-back**: Marimo might not detect the change in Cell 3 because:
- The traitlet value changed on the Python side
- But Marimo's dependency graph may not register this as a "reactive update"
- Cell 3 doesn't re-run, showing stale data

### The Solution: Svelte Writes Back to Trigger Reactivity

The pattern ensures that **any change to the traitlet, even from Python, gets written back through Svelte**, which guarantees Marimo's reactivity system sees it as a change.

## The Implementation

### 1. Track Last Written Value (Svelte-side only)

```typescript
let lastWrittenConfig = $state<string | undefined | null>(undefined);
```

This variable lives **only in Svelte**, not synced as a traitlet. It tracks the last value Svelte wrote to the binding.

### 2. Write Callback (Update Tracker Before Writing)

```typescript
const writeConfigCallback = (contents?: string | null) => {
    lastWrittenConfig = contents;
};

const bindingHandlers = createFileBindingHandlers({
    bindings,
    maxFiles,
    writeCallback: writeConfigCallback,
});
```

Whenever we write to the traitlet, we **first** record what we're about to write in `lastWrittenConfig`.

### 3. The Write Function (use-file-bindings.ts)

```typescript
const writeSelectedConfig = (contents: string | null | undefined): void => {
    writeCallback?.(contents);           // ← Update lastWrittenConfig FIRST
    bindings.selected_config = contents ?? "";  // ← THEN update traitlet
};
```

**Critical order**: We update `lastWrittenConfig` **before** updating the traitlet binding.

### 4. The Reactive Effect (Detect Changes and Write Back)

```typescript
$effect(() => {
    console.log("CHECK NEW === LAST WRITTEN: selected config updated")
    console.log("NEW:", bindings.selected_config);
    console.log("OLD:", lastWrittenConfig);
    if (lastWrittenConfig !== bindings.selected_config) {
        console.log("WRITING THE SELECTED CONFIG TO FIX MISMATCH")
        bindingHandlers.writeSelectedConfig(bindings.selected_config);
    }
});
```

This effect runs whenever `bindings.selected_config` changes. It asks: "Did I write this value myself, or did it come from Python?"

## The Complete Flow

### Scenario A: Python Updates the Config

```
1. Python (Cell 2): widget.selected_config = json.dumps({"new": "data"})

2. AnyWidget traitlet sync → Svelte receives update
   bindings.selected_config = '{"new": "data"}'

3. $effect fires (bindings changed)
   lastWrittenConfig = undefined (or old value)
   bindings.selected_config = '{"new": "data"}'
   They don't match!

4. Call writeSelectedConfig('{"new": "data"}')
   a. writeCallback sets: lastWrittenConfig = '{"new": "data"}'
   b. Then sets: bindings.selected_config = '{"new": "data"}'

5. Traitlet update (from step 4b) triggers Marimo reactivity
   → Cell 3 detects dependency change → re-runs

6. $effect fires again (bindings changed from step 4b)
   lastWrittenConfig = '{"new": "data"}'
   bindings.selected_config = '{"new": "data"}'
   They match! Do nothing (prevents infinite loop)
```

### Scenario B: Svelte UI Updates the Config

```
1. User clicks "Load Config" → handleLoadConfig() runs

2. Calls bindingHandlers.writeSelectedConfig(previewText)
   a. writeCallback sets: lastWrittenConfig = previewText
   b. Then sets: bindings.selected_config = previewText

3. Traitlet sync → Python receives update
   widget.selected_config = previewText

4. Marimo detects change → dependent cells re-run

5. $effect fires (bindings changed)
   lastWrittenConfig = previewText (we set this in step 2a)
   bindings.selected_config = previewText
   They match! Do nothing (no unnecessary write-back)
```

### Scenario C: Widget Initialization with Data

```
1. Python: widget = ConfigFileManager(config_dict={"initial": "data"})
   Python __init__ sets: self.selected_config = '{"initial": "data"}'

2. Svelte component mounts
   Receives: bindings.selected_config = '{"initial": "data"}'
   Initializes: lastWrittenConfig = undefined

3. $effect fires (on mount)
   lastWrittenConfig = undefined
   bindings.selected_config = '{"initial": "data"}'
   They don't match!

4. Call writeSelectedConfig('{"initial": "data"}')
   a. Sets: lastWrittenConfig = '{"initial": "data"}'
   b. Sets: bindings.selected_config = '{"initial": "data"}'

5. Initial write-back triggers Marimo reactivity
   → Ensures dependent cells know about initial widget state

6. $effect fires again
   lastWrittenConfig = '{"initial": "data"}'
   bindings.selected_config = '{"initial": "data"}'
   They match! Stabilizes (no infinite loop)
```

## Why Each Part is Necessary

### Q: "Why not just use bindings.selected_config directly? Why track lastWrittenConfig?"

**A:** Without `lastWrittenConfig`, we get an infinite loop:

```typescript
// BAD: Without lastWrittenConfig
$effect(() => {
    bindingHandlers.writeSelectedConfig(bindings.selected_config);
    // This updates bindings.selected_config
    // Which triggers this $effect again
    // Which updates bindings.selected_config again
    // → INFINITE LOOP
});
```

The `lastWrittenConfig` variable breaks the loop by letting us detect: "I just wrote this value, so I should ignore this change."

### Q: "Why write back at all? We're just writing the same value we received!"

**A:** The write-back **triggers Marimo's reactivity system**. Without it:

```python
# Cell 1
widget = ConfigFileManager()

# Cell 2
widget.selected_config = json.dumps({"a": 1})

# Cell 3 - expects to see {"a": 1} but might show stale data
display(json.loads(widget.selected_config))
```

When Cell 2 runs, it sets the traitlet value. But Marimo's reactive dependency tracking needs to see a "change event" to know Cell 3 should re-run. The Svelte write-back ensures this event fires.

### Q: "Why update lastWrittenConfig BEFORE updating the binding?"

**A:** Order matters for the comparison in `$effect`:

```typescript
// CORRECT ORDER (what we do)
const writeSelectedConfig = (contents) => {
    writeCallback(contents);           // lastWrittenConfig = contents
    bindings.selected_config = contents;  // Trigger sync
};

$effect(() => {
    if (lastWrittenConfig !== bindings.selected_config) {
        // When we write, lastWrittenConfig is already set
        // So this condition will be false, preventing re-trigger
    }
});

// WRONG ORDER
const writeSelectedConfig = (contents) => {
    bindings.selected_config = contents;  // Trigger sync
    writeCallback(contents);           // lastWrittenConfig = contents (too late!)
};

$effect(() => {
    if (lastWrittenConfig !== bindings.selected_config) {
        // lastWrittenConfig still has old value when this runs
        // Condition is true → writes again → infinite loop
    }
});
```

If we update `lastWrittenConfig` **after** the binding, the `$effect` will see a mismatch and trigger another write.

### Q: "Can we simplify by removing the write callback pattern?"

**A:** No. The callback is what lets us update `lastWrittenConfig` **synchronously before** the binding update. Without it:

```typescript
// WRONG: No callback
const writeSelectedConfig = (contents) => {
    lastWrittenConfig = contents;  // This is in ConfigFileManager.svelte
    bindings.selected_config = contents;  // This triggers $effect
};

// But writeSelectedConfig is in use-file-bindings.ts!
// It doesn't have access to lastWrittenConfig
// We need the callback to communicate between files
```

The callback pattern lets the shared `use-file-bindings` hook communicate with the component-specific reactive state.

## Common Pitfalls to Avoid

### ❌ DON'T: Remove lastWrittenConfig

```typescript
// This causes infinite loops
$effect(() => {
    bindingHandlers.writeSelectedConfig(bindings.selected_config);
});
```

### ❌ DON'T: Update lastWrittenConfig after the binding

```typescript
const writeSelectedConfig = (contents) => {
    bindings.selected_config = contents;
    writeCallback(contents);  // Too late!
};
```

### ❌ DON'T: Skip the write-back

```typescript
// This breaks Python → Svelte → Python reactivity
$effect(() => {
    // Just update local state, don't write back
    previewText = bindings.selected_config;
});
```

### ✅ DO: Follow the pattern exactly

1. Track last written value in component state
2. Provide callback that updates tracker
3. In write function: call callback FIRST, then update binding
4. In effect: compare and write back only if values differ

## Future Considerations

When adding new synchronized state (like `current_state`, `baseline_state`), apply this same pattern:

```typescript
// For each synced traitlet:
let lastWrittenCurrentState = $state<string | undefined>(undefined);
let lastWrittenBaselineState = $state<string | undefined>(undefined);

// Provide callbacks:
const writeCurrentStateCallback = (contents) => {
    lastWrittenCurrentState = contents;
};
const writeBaselineStateCallback = (contents) => {
    lastWrittenBaselineState = contents;
};

// Use in binding handlers:
const writeCurrentState = (contents) => {
    writeCurrentStateCallback(contents);
    bindings.current_state = contents;
};

// Check in effects:
$effect(() => {
    if (lastWrittenCurrentState !== bindings.current_state) {
        writeCurrentState(bindings.current_state);
    }
});
```

This ensures all Python ↔ Svelte state stays reactive in both directions.

### Additional edge cases to plan for

- **Batch traitlet updates:** if you set multiple bindings in a single action, ensure **each traitlet has its own `lastWritten*` tracker and callback** before the shared helper writes to Python. Never reuse one tracker across bindings or only the first change will short-circuit the effect.
- **Large payloads (e.g., >1 MB configs):** serialization still occurs twice (Python → Svelte → Python). Keep the write-back but consider compressing/streaming payloads and throttling user-triggered writes to avoid UI jank.
- **Error handling:** if `writeSelectedConfig` can throw (network interruption, validation failure), surface the error to the component and reset `lastWritten*` so the effect retries once the underlying issue is resolved; otherwise you can get stuck with mismatched values.
- **Initialization defaults:** when the widget creates default data in Python, explicitly document in the component that the first `$effect` will fire and perform a write-back—avoid adding other `bindings.*` mutations inside the same mount frame or you may interleave trackers.
