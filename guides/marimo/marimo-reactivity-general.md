# Marimo Reactivity Reference

**Pointers for More Info**
* Marimo AnyWidget Wrapper Impl (`mo.ui.anywidget`): [marimo/_plugins/ui/_impl/from_anywidget.py](https://raw.githubusercontent.com/marimo-team/marimo/main/marimo/_plugins/ui/_impl/from_anywidget.py)
* Marimo UIElement Impl: [marimo/_plugins/ui/_core/ui_element.py](https://raw.githubusercontent.com/marimo-team/marimo/main/marimo/_plugins/ui/_core/ui_element.py#L90)
* AnyWidget Class Impl: [anywidget/widget.py](https://raw.githubusercontent.com/manzt/anywidget/main/anywidget/widget.py)

**Legend — rule types**

* **[RUNTIME]** enforced by marimo at compile/run time
* **[STRUCTURAL]** enforced by marimo’s static model/linter (DAG, setup cell rules, etc.)
* **[PROJECT]** your declared, project‑level convention (takes precedence in your repos)
* **[GUIDANCE]** preferred design pattern (not enforced)

---

## 1) Core mental model

* **Static analysis → DAG → reactive runs.** marimo parses each cell (no execution) to find **globals it defines** and **globals it references**, builds a **DAG over cells**, then enforces data‑flow order. When a cell runs, marimo **automatically runs every other cell that references any global it defined**. [STRUCTURAL] 
* **Execution is variable‑driven, not page‑order.** Cell position doesn’t determine order; data‑flow does. [STRUCTURAL] 
* **Deleting a cell deletes its globals.** Dependents re‑run or are marked stale (lazy mode). [STRUCTURAL] 

---

## 2) What **triggers** vs **doesn’t** trigger reactivity

**Triggers**

1. Running a cell → descendants run. [STRUCTURAL] 
2. Interacting with a **UI element bound to a global** → cells that read `element.value` run. [STRUCTURAL] 
3. Calling a **`mo.state` setter** → other cells that read the getter run (caller does **not** re‑run by default). [STRUCTURAL] 

**Non‑triggers**

* **Mutations / attribute assignments aren’t tracked** (`list.append`, `obj.attr = ...`) → derive a **new object** or perform mutation where object is defined. [STRUCTURAL] 

---

## 3) Globals, locals, naming, and setup

* **One global name, one defining cell.** Redefinition across cells is an error; cycles are illegal. [STRUCTURAL] 
* **Locals require the `_` prefix.** In marimo, names assigned at top level **without** `_` are globals; names **with** `_` are local to the defining cell and invisible to others. Use `_name` for any temporary that must be redefined safely later. [RUNTIME] 
* **No `global` hacks.** “Never define anything using `global`.” [RUNTIME] 
* **Setup cell semantics matter.** The **first cell acts as a setup cell**; it must not depend on other cells (MB004). **Put *all imports* in the first cell** so import state is correctly initialized in setup. [STRUCTURAL][PROJECT] 

---

## 4) Display model: last expression & output APIs

* **Only the last, top‑level expression is displayed.** If output occurs inside control flow, it won’t display unless routed via `mo.output`. [RUNTIME] 
* **Matplotlib:** **return `plt.gca()` as the last expression** (don’t use `plt.show()`). [RUNTIME] 
* **`mo.output.append/replace`** can display non‑final or conditional results. [RUNTIME] 

---

## 5) UI elements and functional reactivity

* **Define UI in one cell and read `.value` in another.** [STRUCTURAL] 
* **You cannot read an element’s `.value` in the same cell where it’s defined** (this is an error). [RUNTIME] 
* **UI must be bound to a global.** A bare `mo.ui.slider(...)` that isn’t assigned to a name won’t drive reactivity. [STRUCTURAL] 
* **Composite UIs:** for dynamic groups, use `mo.ui.array([...])` or `mo.ui.dictionary({...})` so the group itself is reactive (a plain Python list/dict won’t be). [STRUCTURAL] 

**Button vs run button (use the run button):**

* `mo.ui.button(...)` is a general element whose `.value` you manage.
* **`mo.ui.run_button()`** produces a **pulse**: `.value` becomes `True` when clicked and **auto‑resets** to `False` after dependents run. In practice, **use `run_button` for nearly all “button to run something” use‑cases.** [GUIDANCE][PROJECT] 

**Avoid callbacks for propagation.** Let `.value` drive reactivity; reserve `on_change` for imperative side effects. [GUIDANCE] 

---

## 6) Forms and gating recomputation

* **Forms (`.form()`)** collect values and submit once (avoid recompute on every keystroke). [GUIDANCE] 
* **Project stance:** Prefer **`run_button`** (and/or forms) to gate expensive work. **Don’t use `mo.stop()`;** prefer explicit gating with a run button and controlled output. [PROJECT] 

**Pattern (gated compute without `mo.stop`)**:

```python
# cell: control
run = mo.ui.run_button("Run analysis")
run

# cell: expensive
if not run.value:
    mo.output.replace(mo.md("Click **Run analysis**"))
else:
    result = heavy_compute(...)
    mo.output.replace(result)
```

---

## 7) `mo.state`: advanced, rarely needed

* **Default: avoid it.** 99% of use cases rely on UI `.value` + reactive cells. [GUIDANCE] 
* **API:** `get, set = mo.state(initial, allow_self_loops=False)`; **calling the setter runs *other* cells that read the getter; the caller doesn’t re‑run** (avoid accidental loops). [STRUCTURAL] 
* **Two‑way binding:** put tied widgets in **separate cells**, each calling the setter; render in a third cell that reads the getter. [GUIDANCE] 
* **Do not store UI elements in state.** [GUIDANCE] 

---

## 8) Data, charts, and interop

* **Dataframes:** **Use pandas exclusively** in this project; ignore “prefer Polars” guidance in generic docs. [PROJECT] (generic: Polars was recommended) 
* **Altair:** return a chart object for rendering; use **`mo.ui.altair_chart(chart)`** when you need to **read interactions via `chart.value`**. [GUIDANCE] 
* **Plotly:** return the figure object directly. [RUNTIME] 
* **SQL:** use `mo.sql("""...""", engine=...)` for queries. **Do not put comments in SQL cells**; the SQL interpreter will treat them as SQL and may error. [RUNTIME] 

---

## 9) Performance, caching, and runtime modes

* **Runtime modes:** *autorun* (default) vs *lazy* (mark dependents stale; run on demand). [STRUCTURAL] 
* **Disable cells** when editing to avoid fan‑out; re‑enable to recompute. [GUIDANCE] 
* **Caching:** `@mo.cache`, `@mo.persistent_cache`, `@mo.lru_cache` for expensive pure functions; keys include args and closed‑over variables; persistent cache survives restarts (consider `pin_modules=True`). [GUIDANCE] 
* **Lazy compute/render:** `mo.lazy(...)` defers expensive components until visible. [GUIDANCE] 

---

## 10) Idempotence, linting, and debugging

* **Cells must be effectively idempotent.** Because upstream edits re‑run descendants, non‑idempotent cells will be immediately overwritten or oscillate; in practice **you must write idempotent cells**. [STRUCTURAL] 
* **No cycles (MB003).** Break with refactoring; static cycles are illegal. [STRUCTURAL] 
* **Setup cells (MB004):** must not depend on others. [STRUCTURAL] 
* **Tools:** minimap, dependency graph, variables explorer help diagnose unexpected runs/misses. [GUIDANCE] 

---

## 11) Known constraints & pitfalls

* **Multiple `run_button`s inside the *same reactive container*** (e.g., `mo.ui.dictionary`, `mo.ui.array`, or a `batch(...).form()`) can co‑trigger. **Do not co‑locate run buttons in the same container.** Place them in **separate cells** or separate containers. [STRUCTURAL][PROJECT] 
* **UI keeps “resetting”.** Usually because the defining cell re‑ran; isolate the definition to its own cell or persist needed value with state (only if warranted). [GUIDANCE] 

---

## 12) Patterns (concise, battle‑tested)

### A) Correct use of UI value (separate cells)

```python
# cell 1
slider = mo.ui.slider(1, 10, value=5)
slider  # display control

# cell 2
mo.md(f"Selected: {slider.value}")  # reacts to interactions
```

> Accessing `.value` in the same cell as definition is an error. [RUNTIME] 

### B) Dynamic groups of controls

```python
# cell 1
sliders = mo.ui.array([mo.ui.slider(0, 10) for _ in range(n)])
sliders

# cell 2
values = sliders.value  # -> list[int], reactive
```

> Plain Python lists are non‑reactive. [STRUCTURAL] 

### C) Gating expensive work with a run button (no `mo.stop`)

```python
# cell 1
run = mo.ui.run_button("Run embedding")
run

# cell 2
if not run.value:
    mo.output.replace(mo.md("Click **Run embedding**"))
else:
    @mo.persistent_cache
    def embed(text: str): ...
    mo.output.replace(embed(corpus.value))
```

[PROJECT] prefer `run_button`; avoid `mo.stop`. 

### D) Altair: render vs read interactions

```python
# Render only
chart = alt.Chart(df).mark_point().encode(x="x", y="y")  # last expr renders

# Capture interactions
chart_ui = mo.ui.altair_chart(chart)
chart_ui
# elsewhere:
sel = chart_ui.value  # read selections / params
```

[GUIDANCE] 

### E) Matplotlib last expression

```python
fig, ax = plt.subplots()
ax.plot(y)
plt.gca()  # last, top-level expression -> display
```

[REQUIRED DISPLAY] 

### F) Two‑way binding with state (only if you truly need it)

```python
# cell 1
get_x, set_x = mo.state(0)

# cell 2
s = mo.ui.slider(0, 100, value=get_x(), on_change=set_x)

# cell 3
n = mo.ui.number(0, 100, value=get_x(), on_change=set_x)

# cell 4
mo.md(f"x = {get_x()}")
```

Setter re‑runs **other** cells that read `get_x()`; caller does not re‑run. [STRUCTURAL] 

---

## 13) Rules vs recommendations (final checklist)

**Hard rules (marimo‑enforced)**

* Unique global per name; **no redeclare across cells**. [STRUCTURAL] 
* **No cycles** between cells (MB003). [STRUCTURAL] 
* **Setup cell** must not depend on others (MB004). [STRUCTURAL] 
* **Locals require `_` prefix**; otherwise the name is global. [RUNTIME] 
* **Do not access `UI.value` in its defining cell.** [RUNTIME] 
* **Only last, top‑level expression displays**, or use `mo.output`. [RUNTIME] 
* **Matplotlib:** display with `plt.gca()` as last expression. [RUNTIME] 
* **SQL:** no comments in `mo.sql(...)` cells. [RUNTIME] 

**Project‑level conventions (your repo)**

* **All imports in the first (setup) cell;** always include `import marimo as mo`. [PROJECT] 
* **Use pandas exclusively** (ignore generic “prefer Polars”). [PROJECT] 
* **Prefer `mo.ui.run_button()` over `mo.ui.button()`** for task triggers. [PROJECT] 
* **Avoid `mo.stop()`**; gate with run buttons and explicit output control. [PROJECT] 
* **Do not place multiple `run_button`s in the same reactive container.** [PROJECT] 

**Guidance (recommended, not enforced)**

* Prefer **reading `.value`** over `on_change` callbacks. 
* Use forms, caching, lazy mode, and cell disabling to control recomputation. 
* Keep cells **idempotent** (functionally required for sanity under reactivity). 
* Encapsulate logic in functions to minimize globals. 

---

## 14) Minimal “starter” template (pandas‑first, setup‑cell imports)

```python
# ── Setup cell (imports only) ───────────────────────────────────────────────
import marimo as mo
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
# (project convention: all imports live here)

# ── Controls ────────────────────────────────────────────────────────────────
n = mo.ui.slider(100, 10_000, value=1000, label="rows")
run = mo.ui.run_button("Generate")
mo.hstack([n, run])

# ── Data & compute (gated) ─────────────────────────────────────────────────
if not run.value:
    mo.output.replace(mo.md("Click **Generate**"))
else:
    @mo.persistent_cache  # survives restarts
    def make_df(n: int) -> pd.DataFrame:
        return pd.DataFrame({"x": range(n), "y": range(n)})

    df = make_df(n.value)
    mo.output.replace(df)

# ── Altair: render-only chart ──────────────────────────────────────────────
chart = alt.Chart(df).mark_point().encode(x="x", y="y")
chart  # last, top-level expression

# ── Matplotlib: last expression must be gca() ──────────────────────────────
fig, ax = plt.subplots()
ax.plot(df["y"])
plt.gca()  # required for display
```
