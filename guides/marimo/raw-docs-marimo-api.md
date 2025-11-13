# Docs.marimo.io Documentation

*Generated on 2025-11-05 14:56:07*

- **Base URL**: https://docs.marimo.io
- **Total Pages**: 85
- **Max Depth**: 8

---

## Table of Contents

  - [API Reference - marimoAPI Reference¶](#api-reference-marimoapi-reference¶)
    - [App - marimoApp¶](#app-marimoapp¶)
    - [Cell - marimoCell¶](#cell-marimocell¶)
    - [Command Line Arguments - marimoCommand Line Arguments¶](#command-line-arguments-marimocommand-line-arguments¶)
    - [Diagrams - marimoDiagrams¶](#diagrams-marimodiagrams¶)
    - [Markdown - marimoMarkdown¶](#markdown-marimomarkdown¶)
    - [State - marimoState¶](#state-marimostate¶)
    - [Watch - marimoWatch¶](#watch-marimowatch¶)
      - [HTML - marimoHTML¶](#html-marimohtml¶)
      - [Visualize outputs - marimoVisualizing outputs¶](#visualize-outputs-marimovisualizing-outputs¶)
      - [Run notebooks as scripts - marimoRun as a script¶](#run-notebooks-as-scripts-marimorun-as-a-script¶)
        - [Control Flow - marimoControl flow¶](#control-flow-marimocontrol-flow¶)
        - [Navigation Menu - marimoNavigation Menu¶](#navigation-menu-marimonavigation-menu¶)
        - [Table - marimoTable¶](#table-marimotable¶)
        - [Tabs - marimoTabs¶](#tabs-marimotabs¶)
        - [Layouts - marimoLayouts¶](#layouts-marimolayouts¶)
        - [Accordion - marimoAccordion¶](#accordion-marimoaccordion¶)
        - [Sidebar - marimoSidebar¶](#sidebar-marimosidebar¶)
        - [Stacks - marimoStacks¶](#stacks-marimostacks¶)
        - [Miscellaneous - marimoMiscellaneous¶](#miscellaneous-marimomiscellaneous¶)
        - [Outputs - marimoOutputs¶](#outputs-marimooutputs¶)
        - [Lint Rules - marimoLint Rules¶](#lint-rules-marimolint-rules¶)
          - [Refresh - marimoRefresh¶](#refresh-marimorefresh¶)
          - [Run Button - marimoRun Button¶](#run-button-marimorun-button¶)
          - [Callout - marimoCallout¶](#callout-marimocallout¶)
          - [Carousel - marimoCarousel¶](#carousel-marimocarousel¶)
          - [Json - marimoJson¶](#json-marimojson¶)
          - [Justify - marimoJustify¶](#justify-marimojustify¶)
          - [Lazy - marimoLazy¶](#lazy-marimolazy¶)
          - [Outline - marimoOutline¶](#outline-marimooutline¶)
          - [Plain - marimoPlain¶](#plain-marimoplain¶)
          - [Routes - marimoRoutes¶](#routes-marimoroutes¶)
          - [Stat - marimoStat¶](#stat-marimostat¶)
          - [Tree - marimoTree¶](#tree-marimotree¶)
          - [Commands - marimomarimo¶](#commands-marimomarimo¶)
          - [Programmatic - marimoRunning the marimo backend programmatically¶](#programmatic-marimorunning-the-marimo-backend-programmatically¶)
          - [Expensive notebooks - marimoWorking with expensive notebooks¶](#expensive-notebooks-marimoworking-with-expensive-notebooks¶)
          - [Cycle dependencies - marimoMB003: cycle-dependencies¶](#cycle-dependencies-marimomb003-cycle-dependencies¶)
          - [Empty cells - marimoMF004: empty-cells¶](#empty-cells-marimomf004-empty-cells¶)
          - [General formatting - marimoMF001: general-formatting¶](#general-formatting-marimomf001-general-formatting¶)
          - [Syntax error - marimoMB005: invalid-syntax¶](#syntax-error-marimomb005-invalid-syntax¶)
          - [Markdown Indentation - marimoMF007: markdown-indentation¶](#markdown-indentation-marimomf007-markdown-indentation¶)
          - [Misc parse log - marimoMF006: misc-log-capture¶](#misc-parse-log-marimomf006-misc-log-capture¶)
          - [Multiple definitions - marimoMB002: multiple-definitions¶](#multiple-definitions-marimomb002-multiple-definitions¶)
          - [Parse stderr - marimoMF003: parse-stderr¶](#parse-stderr-marimomf003-parse-stderr¶)
          - [Parse stdout - marimoMF002: parse-stdout¶](#parse-stdout-marimomf002-parse-stdout¶)
          - [Self Import - marimoMR001: self-import¶](#self-import-marimomr001-self-import¶)
          - [Setup cell dependencies - marimoMB004: setup-cell-dependencies¶](#setup-cell-dependencies-marimomb004-setup-cell-dependencies¶)
          - [SQL parse error - marimoMF005: sql-parse-error¶](#sql-parse-error-marimomf005-sql-parse-error¶)
          - [Unparsable cells - marimoMB001: unparsable-cells¶](#unparsable-cells-marimomb001-unparsable-cells¶)
            - [Query Parameters - marimoQuery Parameters¶](#query-parameters-marimoquery-parameters¶)
            - [Best practices - marimoBest practices¶](#best-practices-marimobest-practices¶)
            - [Runtime configuration - marimoRuntime configuration¶](#runtime-configuration-marimoruntime-configuration¶)
            - [Export to other formats - marimoExporting to HTML and other formats¶](#export-to-other-formats-marimoexporting-to-html-and-other-formats¶)
            - [Running cells - marimoRunning cells¶](#running-cells-marimorunning-cells¶)
            - [Understanding errors - marimoUnderstanding errors¶](#understanding-errors-marimounderstanding-errors¶)
              - [Caching - marimoCaching¶](#caching-marimocaching¶)
              - [FAQ - marimoFAQ](#faq-marimofaq)
              - [Interactive elements - marimoInteractive elements¶](#interactive-elements-marimointeractive-elements¶)
              - [marimo islands 🏝️ - marimomarimo islands 🏝️¶](#marimo-islands-🏝️-marimomarimo-islands-🏝️¶)
              - [Notebooks in existing projects - marimoNotebooks in existing projects¶](#notebooks-in-existing-projects-marimonotebooks-in-existing-projects¶)
              - [Cloudflare - marimoPublish to Cloudflare¶](#cloudflare-marimopublish-to-cloudflare¶)
              - [Publishing with Quarto - marimoPublishing with Quarto¶](#publishing-with-quarto-marimopublishing-with-quarto¶)
              - [Cycles - marimoCycles¶](#cycles-marimocycles¶)
              - [Import star - marimoStar imports¶](#import-star-marimostar-imports¶)
              - [Multiple definitions - marimoMultiple definitions¶](#multiple-definitions-marimomultiple-definitions¶)
              - [Setup References - marimoSetup references¶](#setup-references-marimosetup-references¶)
              - [WebAssembly notebooks - marimoWebAssembly Notebooks¶](#webassembly-notebooks-marimowebassembly-notebooks¶)
                - [Inputs - marimoInputs¶](#inputs-marimoinputs¶)
                - [Array - marimoArray¶](#array-marimoarray¶)
                - [Batch - marimoBatch¶](#batch-marimobatch¶)
                - [Dictionary - marimoDictionary¶](#dictionary-marimodictionary¶)
                - [Form - marimoForm¶](#form-marimoform¶)
                - [Plotting - marimoPlotting¶](#plotting-marimoplotting¶)
                - [Getting Started - marimoGetting Started¶](#getting-started-marimogetting-started¶)
                - [Migrate from Jupyter - marimoComing from Jupyter¶](#migrate-from-jupyter-marimocoming-from-jupyter¶)
                - [Custom UI plugins - marimoCustom UI plugins](#custom-ui-plugins-marimocustom-ui-plugins)
                - [Package management - marimoPackage management¶](#package-management-marimopackage-management¶)
                - [Inlining dependencies - marimoInlining dependencies¶](#inlining-dependencies-marimoinlining-dependencies¶)
                - [Using uv - marimoUsing uv¶](#using-uv-marimousing-uv¶)
                - [Community Cloud - marimoCommunity Cloud¶](#community-cloud-marimocommunity-cloud¶)
                - [GitHub Pages - marimoPublish to GitHub Pages¶](#github-pages-marimopublish-to-github-pages¶)
                - [Publishing with MkDocs - marimoPublishing with MkDocs¶](#publishing-with-mkdocs-marimopublishing-with-mkdocs¶)
                - [Reuse functions and classes - marimoImporting functions and classes defined in notebooks¶](#reuse-functions-and-classes-marimoimporting-functions-and-classes-defined-in-notebooks¶)
                - [Recipes - marimoRecipes¶](#recipes-marimorecipes¶)

---

## API Reference - marimoAPI Reference¶

*Source: [https://docs.marimo.io/api/](https://docs.marimo.io/api/)*

<!-- Source: https://docs.marimo.io/api/ -->
<!-- Title: API Reference - marimoAPI Reference¶ -->
<!-- Depth: 1 -->

[Skip to content](#api-reference)

# API Reference [¶](\#api-reference "Permanent link")

Use the marimo library in marimo notebooks ( `import marimo as mo`) to

- connect interactive inputs like sliders, dropdowns, and tables to Python,
- express yourself with dynamically created markdown,
- layout information with tabs or grids,
- output media like images and audio,
- and more!

|  |  |
| --- | --- |
| [markdown](markdown/) | Write markdown with `mo.md` |
| [inputs](inputs/) | Connect sliders, dropdowns, tables, and more to Python |
| [layouts](layouts/) | Customize outputs with accordions, tabs, stacks, and more |
| [plotting](plotting/) | Output interactive plots |
| [media](media/) | Output media like images, audio, PDFs, and plain text |
| [diagrams](diagrams/) | Flow charts, graphs, statistic cards, and more |
| [status](status/) | Display progress indicators |
| [outputs](outputs/) | Modify cell outputs, redirect console output |
| [control\_flow](control_flow/) | Control how cells execute |
| [html](html/) | Manipulate HTML objects |
| [query\_params](query_params/) | Access and set query parameters with `mo.query_params` |
| [cli\_args](cli_args/) | Access command-line arguments with `mo.cli_args` |
| [caching](caching/) | Cache expensive computations in memory or on disk |
| [state](state/) | Synchronize multiple UI elements with `mo.state` |
| [app](app/) | Embed notebooks in other notebooks |
| [cell](cell/) | Run cells defined in another notebook |
| [watch](watch/) | Reactively respond to file changes on disk |
| [miscellaneous](miscellaneous/) | Miscellaneous utilities |

Back to top

# API Reference [¶](\#api-reference "Permanent link")

Use the marimo library in marimo notebooks ( `import marimo as mo`) to

- connect interactive inputs like sliders, dropdowns, and tables to Python,
- express yourself with dynamically created markdown,
- layout information with tabs or grids,
- output media like images and audio,
- and more!

|  |  |
| --- | --- |
| [markdown](markdown/) | Write markdown with `mo.md` |
| [inputs](inputs/) | Connect sliders, dropdowns, tables, and more to Python |
| [layouts](layouts/) | Customize outputs with accordions, tabs, stacks, and more |
| [plotting](plotting/) | Output interactive plots |
| [media](media/) | Output media like images, audio, PDFs, and plain text |
| [diagrams](diagrams/) | Flow charts, graphs, statistic cards, and more |
| [status](status/) | Display progress indicators |
| [outputs](outputs/) | Modify cell outputs, redirect console output |
| [control\_flow](control_flow/) | Control how cells execute |
| [html](html/) | Manipulate HTML objects |
| [query\_params](query_params/) | Access and set query parameters with `mo.query_params` |
| [cli\_args](cli_args/) | Access command-line arguments with `mo.cli_args` |
| [caching](caching/) | Cache expensive computations in memory or on disk |
| [state](state/) | Synchronize multiple UI elements with `mo.state` |
| [app](app/) | Embed notebooks in other notebooks |
| [cell](cell/) | Run cells defined in another notebook |
| [watch](watch/) | Reactively respond to file changes on disk |
| [miscellaneous](miscellaneous/) | Miscellaneous utilities |

 Back to top

# API Reference [¶](\#api-reference "Permanent link")

Use the marimo library in marimo notebooks ( `import marimo as mo`) to

- connect interactive inputs like sliders, dropdowns, and tables to Python,
- express yourself with dynamically created markdown,
- layout information with tabs or grids,
- output media like images and audio,
- and more!

|  |  |
| --- | --- |
| [markdown](markdown/) | Write markdown with `mo.md` |
| [inputs](inputs/) | Connect sliders, dropdowns, tables, and more to Python |
| [layouts](layouts/) | Customize outputs with accordions, tabs, stacks, and more |
| [plotting](plotting/) | Output interactive plots |
| [media](media/) | Output media like images, audio, PDFs, and plain text |
| [diagrams](diagrams/) | Flow charts, graphs, statistic cards, and more |
| [status](status/) | Display progress indicators |
| [outputs](outputs/) | Modify cell outputs, redirect console output |
| [control\_flow](control_flow/) | Control how cells execute |
| [html](html/) | Manipulate HTML objects |
| [query\_params](query_params/) | Access and set query parameters with `mo.query_params` |
| [cli\_args](cli_args/) | Access command-line arguments with `mo.cli_args` |
| [caching](caching/) | Cache expensive computations in memory or on disk |
| [state](state/) | Synchronize multiple UI elements with `mo.state` |
| [app](app/) | Embed notebooks in other notebooks |
| [cell](cell/) | Run cells defined in another notebook |
| [watch](watch/) | Reactively respond to file changes on disk |
| [miscellaneous](miscellaneous/) | Miscellaneous utilities |


---

### App - marimoApp¶

*Source: [https://docs.marimo.io/api/app/](https://docs.marimo.io/api/app/)*

<!-- Source: https://docs.marimo.io/api/app/ -->
<!-- Title: App - marimoApp¶ -->
<!-- Depth: 2 -->

[Skip to content](#app)

# App [¶](\#app "Permanent link")

## `` marimo.App [¶](\#marimo.App "Permanent link")

A marimo notebook.

A marimo notebook is a dataflow graph, with each node computing a Python
function.

### `` embed`async`[¶](\#marimo.App.embed "Permanent link")

Embed a notebook into another notebook.

The `embed` method lets you embed the output of a notebook
into another notebook and access the values of its variables.

Running `await app.embed()` executes the notebook and results an object
encapsulating the notebook visual output and its definitions.

Embedded notebook outputs are interactive: when you interact with
UI elements in an embedded notebook's output, any cell referring
to the `app` object other than the one that imported it is marked for
execution, and its internal state is automatically updated. This lets
you use notebooks as building blocks or components to create
higher-level notebooks.

Multiple levels of nesting are supported: it's possible to embed a
notebook that in turn embeds another notebook, and marimo will do the
right thing.

Example

To embed independent copies of same app object, first clone the
app with `app.clone()`:

Returns:
An object `result` with two attributes: `result.output` (visual
output of the notebook) and `result.defs` (a dictionary mapping
variable names defined by the notebook to their values).

### `` run [¶](\#marimo.App.run "Permanent link")

Run the marimo app and return its outputs and definitions.

Use this method to run marimo apps programmatically and retrieve their
outputs and definitions. This lets you execute notebooks from other
Python scripts. By providing definitions to `app.run()`, you can
override specific cells in the notebook with your own values.

Examples:

Consider a notebook `my_notebook.py`:

To run this app programmatically:

Definition Override Behavior

When you provide definitions to `app.run()`, you are **completely**
**overriding** the definitions of cells that define those variables:

- The cells that originally defined those variables will not execute
- You must provide **all** the definitions that a cell would normally produce
- Cells that depend on the overridden variables will use your provided values

| PARAMETER | DESCRIPTION |
| --- | --- |
| `defs` | You may pass values for any variable definitions as keyword<br>arguments. marimo will use these values instead of executing<br>the cells that would normally define them. Cells that depend<br>on these variables will use your provided values.<br>**TYPE:** `dict[str, Any]`**DEFAULT:** `None` |
| `**kwargs` | For forward-compatibility with future arguments.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Any]` | A tuple containing: |
| `Mapping[str, Any]` | - Sequence of cell outputs (visual outputs from each cell) |
| `tuple[Sequence[Any], Mapping[str, Any]]` | - Mapping of variable names to their values (definitions) |

Environment Variables

## AppMeta [¶](\#appmeta "Permanent link")

## `` marimo.app\_meta [¶](\#marimo.app_meta "Permanent link")

Get the metadata of a marimo app.

The `AppMeta` class provides access to runtime metadata about a marimo app,
such as its display theme and execution mode.

Examples:

Get the current theme and conditionally set a plotting library's theme:

Show content only in edit mode:

Get the current request headers or user info:

| RETURNS | DESCRIPTION |
| --- | --- |
| `AppMeta` | An AppMeta object containing the app's metadata.<br>**TYPE:** `AppMeta` |

## `` marimo.AppMeta [¶](\#marimo.AppMeta "Permanent link")

Metadata about the app.

This class provides access to runtime metadata about a marimo app, such as
its display theme and execution mode.

### `` mode`property`[¶](\#marimo.AppMeta.mode "Permanent link")

The execution mode of the app.

Examples:

Show content only in edit mode:

| RETURNS | DESCRIPTION |
| --- | --- |
| `Optional[RunMode]` | - "edit": The notebook is being edited in the marimo editor |
| `Optional[RunMode]` | - "run": The notebook is being run as an app |
| `Optional[RunMode]` | - "script": The notebook is being run as a script |
| `Optional[RunMode]` | - "test": The cell has been invoked by a test |
| `Optional[RunMode]` | - None: The mode could not be determined |

### `` request`property`[¶](\#marimo.AppMeta.request "Permanent link")

The current HTTP request if any. The shape of the request object depends on the ASGI framework used,
but typically includes:

- `headers`: Request headers
- `cookies`: Request cookies
- `query_params`: Query parameters
- `path_params`: Path parameters
- `user`: User data added by authentication middleware
- `url`: URL information including path, query parameters

Examples:

Get the current request and print the path:

| RETURNS | DESCRIPTION |
| --- | --- |
| `Optional[HTTPRequest]` | Optional\[HTTPRequest\]: The current request object if available, None otherwise. |

### `` theme`property`[¶](\#marimo.AppMeta.theme "Permanent link")

The display theme of the app.

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | Either "light" or "dark". If the user's configuration is set to<br>"system", currently returns "light".<br>**TYPE:** `str` |

Examples:

Get the current theme and conditionally set a plotting library's theme:

Back to top

# App [¶](\#app "Permanent link")

## `` marimo.App [¶](\#marimo.App "Permanent link")

A marimo notebook.

A marimo notebook is a dataflow graph, with each node computing a Python
function.

### `` embed`async`[¶](\#marimo.App.embed "Permanent link")

Embed a notebook into another notebook.

The `embed` method lets you embed the output of a notebook
into another notebook and access the values of its variables.

Running `await app.embed()` executes the notebook and results an object
encapsulating the notebook visual output and its definitions.

Embedded notebook outputs are interactive: when you interact with
UI elements in an embedded notebook's output, any cell referring
to the `app` object other than the one that imported it is marked for
execution, and its internal state is automatically updated. This lets
you use notebooks as building blocks or components to create
higher-level notebooks.

Multiple levels of nesting are supported: it's possible to embed a
notebook that in turn embeds another notebook, and marimo will do the
right thing.

Example

To embed independent copies of same app object, first clone the
app with `app.clone()`:

Returns:
An object `result` with two attributes: `result.output` (visual
output of the notebook) and `result.defs` (a dictionary mapping
variable names defined by the notebook to their values).

### `` run [¶](\#marimo.App.run "Permanent link")

Run the marimo app and return its outputs and definitions.

Use this method to run marimo apps programmatically and retrieve their
outputs and definitions. This lets you execute notebooks from other
Python scripts. By providing definitions to `app.run()`, you can
override specific cells in the notebook with your own values.

Examples:

Consider a notebook `my_notebook.py`:

To run this app programmatically:

Definition Override Behavior

When you provide definitions to `app.run()`, you are **completely**
**overriding** the definitions of cells that define those variables:

- The cells that originally defined those variables will not execute
- You must provide **all** the definitions that a cell would normally produce
- Cells that depend on the overridden variables will use your provided values

| PARAMETER | DESCRIPTION |
| --- | --- |
| `defs` | You may pass values for any variable definitions as keyword<br>arguments. marimo will use these values instead of executing<br>the cells that would normally define them. Cells that depend<br>on these variables will use your provided values.<br>**TYPE:** `dict[str, Any]`**DEFAULT:** `None` |
| `**kwargs` | For forward-compatibility with future arguments.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Any]` | A tuple containing: |
| `Mapping[str, Any]` | - Sequence of cell outputs (visual outputs from each cell) |
| `tuple[Sequence[Any], Mapping[str, Any]]` | - Mapping of variable names to their values (definitions) |

Environment Variables

## AppMeta [¶](\#appmeta "Permanent link")

## `` marimo.app\_meta [¶](\#marimo.app_meta "Permanent link")

Get the metadata of a marimo app.

The `AppMeta` class provides access to runtime metadata about a marimo app,
such as its display theme and execution mode.

Examples:

Get the current theme and conditionally set a plotting library's theme:

Show content only in edit mode:

Get the current request headers or user info:

| RETURNS | DESCRIPTION |
| --- | --- |
| `AppMeta` | An AppMeta object containing the app's metadata.<br>**TYPE:** `AppMeta` |

## `` marimo.AppMeta [¶](\#marimo.AppMeta "Permanent link")

Metadata about the app.

This class provides access to runtime metadata about a marimo app, such as
its display theme and execution mode.

### `` mode`property`[¶](\#marimo.AppMeta.mode "Permanent link")

The execution mode of the app.

Examples:

Show content only in edit mode:

| RETURNS | DESCRIPTION |
| --- | --- |
| `Optional[RunMode]` | - "edit": The notebook is being edited in the marimo editor |
| `Optional[RunMode]` | - "run": The notebook is being run as an app |
| `Optional[RunMode]` | - "script": The notebook is being run as a script |
| `Optional[RunMode]` | - "test": The cell has been invoked by a test |
| `Optional[RunMode]` | - None: The mode could not be determined |

### `` request`property`[¶](\#marimo.AppMeta.request "Permanent link")

The current HTTP request if any. The shape of the request object depends on the ASGI framework used,
but typically includes:

- `headers`: Request headers
- `cookies`: Request cookies
- `query_params`: Query parameters
- `path_params`: Path parameters
- `user`: User data added by authentication middleware
- `url`: URL information including path, query parameters

Examples:

Get the current request and print the path:

| RETURNS | DESCRIPTION |
| --- | --- |
| `Optional[HTTPRequest]` | Optional\[HTTPRequest\]: The current request object if available, None otherwise. |

### `` theme`property`[¶](\#marimo.AppMeta.theme "Permanent link")

The display theme of the app.

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | Either "light" or "dark". If the user's configuration is set to<br>"system", currently returns "light".<br>**TYPE:** `str` |

Examples:

Get the current theme and conditionally set a plotting library's theme:

 Back to top

# App [¶](\#app "Permanent link")

## `` marimo.App [¶](\#marimo.App "Permanent link")

A marimo notebook.

A marimo notebook is a dataflow graph, with each node computing a Python
function.

### `` embed`async`[¶](\#marimo.App.embed "Permanent link")

Embed a notebook into another notebook.

The `embed` method lets you embed the output of a notebook
into another notebook and access the values of its variables.

Running `await app.embed()` executes the notebook and results an object
encapsulating the notebook visual output and its definitions.

Embedded notebook outputs are interactive: when you interact with
UI elements in an embedded notebook's output, any cell referring
to the `app` object other than the one that imported it is marked for
execution, and its internal state is automatically updated. This lets
you use notebooks as building blocks or components to create
higher-level notebooks.

Multiple levels of nesting are supported: it's possible to embed a
notebook that in turn embeds another notebook, and marimo will do the
right thing.

Example

To embed independent copies of same app object, first clone the
app with `app.clone()`:

Returns:
An object `result` with two attributes: `result.output` (visual
output of the notebook) and `result.defs` (a dictionary mapping
variable names defined by the notebook to their values).

### `` run [¶](\#marimo.App.run "Permanent link")

Run the marimo app and return its outputs and definitions.

Use this method to run marimo apps programmatically and retrieve their
outputs and definitions. This lets you execute notebooks from other
Python scripts. By providing definitions to `app.run()`, you can
override specific cells in the notebook with your own values.

Examples:

Consider a notebook `my_notebook.py`:

To run this app programmatically:

Definition Override Behavior

When you provide definitions to `app.run()`, you are **completely**
**overriding** the definitions of cells that define those variables:

- The cells that originally defined those variables will not execute
- You must provide **all** the definitions that a cell would normally produce
- Cells that depend on the overridden variables will use your provided values

| PARAMETER | DESCRIPTION |
| --- | --- |
| `defs` | You may pass values for any variable definitions as keyword<br>arguments. marimo will use these values instead of executing<br>the cells that would normally define them. Cells that depend<br>on these variables will use your provided values.<br>**TYPE:** `dict[str, Any]`**DEFAULT:** `None` |
| `**kwargs` | For forward-compatibility with future arguments.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Any]` | A tuple containing: |
| `Mapping[str, Any]` | - Sequence of cell outputs (visual outputs from each cell) |
| `tuple[Sequence[Any], Mapping[str, Any]]` | - Mapping of variable names to their values (definitions) |

Environment Variables

## AppMeta [¶](\#appmeta "Permanent link")

## `` marimo.app\_meta [¶](\#marimo.app_meta "Permanent link")

Get the metadata of a marimo app.

The `AppMeta` class provides access to runtime metadata about a marimo app,
such as its display theme and execution mode.

Examples:

Get the current theme and conditionally set a plotting library's theme:

Show content only in edit mode:

Get the current request headers or user info:

| RETURNS | DESCRIPTION |
| --- | --- |
| `AppMeta` | An AppMeta object containing the app's metadata.<br>**TYPE:** `AppMeta` |

## `` marimo.AppMeta [¶](\#marimo.AppMeta "Permanent link")

Metadata about the app.

This class provides access to runtime metadata about a marimo app, such as
its display theme and execution mode.

### `` mode`property`[¶](\#marimo.AppMeta.mode "Permanent link")

The execution mode of the app.

Examples:

Show content only in edit mode:

| RETURNS | DESCRIPTION |
| --- | --- |
| `Optional[RunMode]` | - "edit": The notebook is being edited in the marimo editor |
| `Optional[RunMode]` | - "run": The notebook is being run as an app |
| `Optional[RunMode]` | - "script": The notebook is being run as a script |
| `Optional[RunMode]` | - "test": The cell has been invoked by a test |
| `Optional[RunMode]` | - None: The mode could not be determined |

### `` request`property`[¶](\#marimo.AppMeta.request "Permanent link")

The current HTTP request if any. The shape of the request object depends on the ASGI framework used,
but typically includes:

- `headers`: Request headers
- `cookies`: Request cookies
- `query_params`: Query parameters
- `path_params`: Path parameters
- `user`: User data added by authentication middleware
- `url`: URL information including path, query parameters

Examples:

Get the current request and print the path:

| RETURNS | DESCRIPTION |
| --- | --- |
| `Optional[HTTPRequest]` | Optional\[HTTPRequest\]: The current request object if available, None otherwise. |

### `` theme`property`[¶](\#marimo.AppMeta.theme "Permanent link")

The display theme of the app.

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | Either "light" or "dark". If the user's configuration is set to<br>"system", currently returns "light".<br>**TYPE:** `str` |

Examples:

Get the current theme and conditionally set a plotting library's theme:


---

### Cell - marimoCell¶

*Source: [https://docs.marimo.io/api/cell/](https://docs.marimo.io/api/cell/)*

<!-- Source: https://docs.marimo.io/api/cell/ -->
<!-- Title: Cell - marimoCell¶ -->
<!-- Depth: 2 -->

[Skip to content](#cell)

# Cell [¶](\#cell "Permanent link")

## `` marimo.Cell`dataclass`[¶](\#marimo.Cell "Permanent link")

An executable notebook cell

Cells are the fundamental unit of execution in marimo. They represent
a single unit of execution, which can be run independently and reused
across notebooks.

Cells are defined using the `@app.cell` decorator, which registers the
function as a cell in marimo.

For example:

Cells can be invoked as functions, and picked up by external frameworks
(like `pytest` if their name starts with `test_`). However, consider
implementing reusable functions (@app.function) in your notebook for
granular control of the output.

A `Cell` object can also be executed without arguments via its `run()`
method, which returns the cell's last expression (output) and a mapping from
its defined names to its values.

For example:

Cells can be named via the marimo editor in the browser, or by
changing the cell's function name in the notebook file.

See the documentation of `run` for info and examples.

### `` defs`property`[¶](\#marimo.Cell.defs "Permanent link")

The definitions made by this cell

### `` name`property`[¶](\#marimo.Cell.name "Permanent link")

### `` refs`property`[¶](\#marimo.Cell.refs "Permanent link")

The references that this cell takes as input

### `` run [¶](\#marimo.Cell.run "Permanent link")

Run this cell and return its visual output and definitions.

Use this method to run **named cells** and retrieve their output and
definitions. This lets you reuse cells defined in one notebook in another
notebook or Python file. It also makes it possible to write and execute
unit tests for notebook cells using a test framework like `pytest`.

Examples:

marimo cells can be given names either through the editor cell menu
or by manually changing the function name in the notebook file. For
example, consider a notebook `notebook.py`:

To reuse the `add` cell in another notebook, you'd simply write:

When `run` is called without arguments, it automatically computes
the values that the cell depends on (in this case, `mo`, `x`, and
`y`). You can override these values by providing any subset of them
as keyword arguments. For example,

Defined UI Elements

If the cell's `output` has UI elements that are in `defs`, interacting
with the output in the frontend will trigger reactive execution of
cells that reference the `defs` object. For example, if `output` has
a slider defined by the cell, then scrubbing the slider will cause
cells that reference `defs` to run.

Async cells

If this cell is a coroutine function (starting with `async`), or if
any of its ancestors are coroutine functions, then you'll need to
`await` the result: `output, defs = await cell.run()`. You can check
whether the result is an awaitable using:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**refs` | You may pass values for any of this cell's references as keyword<br>arguments. marimo will automatically compute values for any refs<br>that are not provided by executing the parent cells that compute<br>them.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[Any, Mapping[str, Any]] | Awaitable[tuple[Any, Mapping[str, Any]]]` | tuple `(output, defs)`, or an awaitable of the same:<br>`output` is the cell's last expression and `defs` is a `Mapping`<br>from the cell's defined names to their values. |

Back to top

# Cell [¶](\#cell "Permanent link")

## `` marimo.Cell`dataclass`[¶](\#marimo.Cell "Permanent link")

An executable notebook cell

Cells are the fundamental unit of execution in marimo. They represent
a single unit of execution, which can be run independently and reused
across notebooks.

Cells are defined using the `@app.cell` decorator, which registers the
function as a cell in marimo.

For example:

Cells can be invoked as functions, and picked up by external frameworks
(like `pytest` if their name starts with `test_`). However, consider
implementing reusable functions (@app.function) in your notebook for
granular control of the output.

A `Cell` object can also be executed without arguments via its `run()`
method, which returns the cell's last expression (output) and a mapping from
its defined names to its values.

For example:

Cells can be named via the marimo editor in the browser, or by
changing the cell's function name in the notebook file.

See the documentation of `run` for info and examples.

### `` defs`property`[¶](\#marimo.Cell.defs "Permanent link")

The definitions made by this cell

### `` name`property`[¶](\#marimo.Cell.name "Permanent link")

### `` refs`property`[¶](\#marimo.Cell.refs "Permanent link")

The references that this cell takes as input

### `` run [¶](\#marimo.Cell.run "Permanent link")

Run this cell and return its visual output and definitions.

Use this method to run **named cells** and retrieve their output and
definitions. This lets you reuse cells defined in one notebook in another
notebook or Python file. It also makes it possible to write and execute
unit tests for notebook cells using a test framework like `pytest`.

Examples:

marimo cells can be given names either through the editor cell menu
or by manually changing the function name in the notebook file. For
example, consider a notebook `notebook.py`:

To reuse the `add` cell in another notebook, you'd simply write:

When `run` is called without arguments, it automatically computes
the values that the cell depends on (in this case, `mo`, `x`, and
`y`). You can override these values by providing any subset of them
as keyword arguments. For example,

Defined UI Elements

If the cell's `output` has UI elements that are in `defs`, interacting
with the output in the frontend will trigger reactive execution of
cells that reference the `defs` object. For example, if `output` has
a slider defined by the cell, then scrubbing the slider will cause
cells that reference `defs` to run.

Async cells

If this cell is a coroutine function (starting with `async`), or if
any of its ancestors are coroutine functions, then you'll need to
`await` the result: `output, defs = await cell.run()`. You can check
whether the result is an awaitable using:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**refs` | You may pass values for any of this cell's references as keyword<br>arguments. marimo will automatically compute values for any refs<br>that are not provided by executing the parent cells that compute<br>them.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[Any, Mapping[str, Any]] | Awaitable[tuple[Any, Mapping[str, Any]]]` | tuple `(output, defs)`, or an awaitable of the same:<br>`output` is the cell's last expression and `defs` is a `Mapping`<br>from the cell's defined names to their values. |

 Back to top

# Cell [¶](\#cell "Permanent link")

## `` marimo.Cell`dataclass`[¶](\#marimo.Cell "Permanent link")

An executable notebook cell

Cells are the fundamental unit of execution in marimo. They represent
a single unit of execution, which can be run independently and reused
across notebooks.

Cells are defined using the `@app.cell` decorator, which registers the
function as a cell in marimo.

For example:

Cells can be invoked as functions, and picked up by external frameworks
(like `pytest` if their name starts with `test_`). However, consider
implementing reusable functions (@app.function) in your notebook for
granular control of the output.

A `Cell` object can also be executed without arguments via its `run()`
method, which returns the cell's last expression (output) and a mapping from
its defined names to its values.

For example:

Cells can be named via the marimo editor in the browser, or by
changing the cell's function name in the notebook file.

See the documentation of `run` for info and examples.

### `` defs`property`[¶](\#marimo.Cell.defs "Permanent link")

The definitions made by this cell

### `` name`property`[¶](\#marimo.Cell.name "Permanent link")

### `` refs`property`[¶](\#marimo.Cell.refs "Permanent link")

The references that this cell takes as input

### `` run [¶](\#marimo.Cell.run "Permanent link")

Run this cell and return its visual output and definitions.

Use this method to run **named cells** and retrieve their output and
definitions. This lets you reuse cells defined in one notebook in another
notebook or Python file. It also makes it possible to write and execute
unit tests for notebook cells using a test framework like `pytest`.

Examples:

marimo cells can be given names either through the editor cell menu
or by manually changing the function name in the notebook file. For
example, consider a notebook `notebook.py`:

To reuse the `add` cell in another notebook, you'd simply write:

When `run` is called without arguments, it automatically computes
the values that the cell depends on (in this case, `mo`, `x`, and
`y`). You can override these values by providing any subset of them
as keyword arguments. For example,

Defined UI Elements

If the cell's `output` has UI elements that are in `defs`, interacting
with the output in the frontend will trigger reactive execution of
cells that reference the `defs` object. For example, if `output` has
a slider defined by the cell, then scrubbing the slider will cause
cells that reference `defs` to run.

Async cells

If this cell is a coroutine function (starting with `async`), or if
any of its ancestors are coroutine functions, then you'll need to
`await` the result: `output, defs = await cell.run()`. You can check
whether the result is an awaitable using:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**refs` | You may pass values for any of this cell's references as keyword<br>arguments. marimo will automatically compute values for any refs<br>that are not provided by executing the parent cells that compute<br>them.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[Any, Mapping[str, Any]] | Awaitable[tuple[Any, Mapping[str, Any]]]` | tuple `(output, defs)`, or an awaitable of the same:<br>`output` is the cell's last expression and `defs` is a `Mapping`<br>from the cell's defined names to their values. |


---

### Command Line Arguments - marimoCommand Line Arguments¶

*Source: [https://docs.marimo.io/api/cli_args/](https://docs.marimo.io/api/cli_args/)*

<!-- Source: https://docs.marimo.io/api/cli_args/ -->
<!-- Title: Command Line Arguments - marimoCommand Line Arguments¶ -->
<!-- Depth: 2 -->

[Skip to content](#command-line-arguments)

# Command Line Arguments [¶](\#command-line-arguments "Permanent link")

When running as a script with `python notebook.py`, command-line arguments
are available to your program in `sys.argv`, just like any other Python
program. This means you can use
[`argparse`](https://docs.python.org/3/library/argparse.html),
[`simple-parsing`](https://github.com/lebrice/SimpleParsing), and other tools
for specifying and parsing arguments.

You can also use tools like `argparse` when running as a notebook ( `marimo
edit` or `marimo run`) or exporting to HTML, IPYNB or another format ( `marimo
export`). In these cases, `sys.argv` is set to the notebook filename followed by any args
following the `--` separator.

For example, for

sets `sys.argv` to `["notebook.py", "--lr", "1e-4", "--epochs", "15"]`.

For usage examples, see the guide on [running as a script](../../guides/scripts/).

## Utility function for parsing arguments [¶](\#utility-function-for-parsing-arguments "Permanent link")

marimo provides a utility function called `mo.cli_args()` for parsing arguments
from strings into primitive data types ( `int`, `bool`, `float`, `str`).
However, unlike `argparse` and `simple-parsing`, this function does not let you
declare your program's arguments, nor does it generate help text. **For these**
**reasons, we recommend using `argparse` or `simple-parsing` instead.**

## `` marimo.cli\_args [¶](\#marimo.cli_args "Permanent link")

Get the command line arguments of a marimo notebook.

Examples:

`marimo edit notebook.py -- -size 10`

| RETURNS | DESCRIPTION |
| --- | --- |
| `CLIArgs` | A dictionary containing the command line arguments.<br>This dictionary is read-only and cannot be mutated.<br>**TYPE:** `CLIArgs` |

Query Parameters

You can also access query parameters passed to the notebook using
`mo.query_params`. This allows you to pass arguments to the notebook that can be controlled by the user.

Back to top

# Command Line Arguments [¶](\#command-line-arguments "Permanent link")

When running as a script with `python notebook.py`, command-line arguments
are available to your program in `sys.argv`, just like any other Python
program. This means you can use
[`argparse`](https://docs.python.org/3/library/argparse.html),
[`simple-parsing`](https://github.com/lebrice/SimpleParsing), and other tools
for specifying and parsing arguments.

You can also use tools like `argparse` when running as a notebook ( `marimo
edit` or `marimo run`) or exporting to HTML, IPYNB or another format ( `marimo
export`). In these cases, `sys.argv` is set to the notebook filename followed by any args
following the `--` separator.

For example, for

sets `sys.argv` to `["notebook.py", "--lr", "1e-4", "--epochs", "15"]`.

For usage examples, see the guide on [running as a script](../../guides/scripts/).

## Utility function for parsing arguments [¶](\#utility-function-for-parsing-arguments "Permanent link")

marimo provides a utility function called `mo.cli_args()` for parsing arguments
from strings into primitive data types ( `int`, `bool`, `float`, `str`).
However, unlike `argparse` and `simple-parsing`, this function does not let you
declare your program's arguments, nor does it generate help text. **For these**
**reasons, we recommend using `argparse` or `simple-parsing` instead.**

## `` marimo.cli\_args [¶](\#marimo.cli_args "Permanent link")

Get the command line arguments of a marimo notebook.

Examples:

`marimo edit notebook.py -- -size 10`

| RETURNS | DESCRIPTION |
| --- | --- |
| `CLIArgs` | A dictionary containing the command line arguments.<br>This dictionary is read-only and cannot be mutated.<br>**TYPE:** `CLIArgs` |

Query Parameters

You can also access query parameters passed to the notebook using
`mo.query_params`. This allows you to pass arguments to the notebook that can be controlled by the user.

 Back to top

# Command Line Arguments [¶](\#command-line-arguments "Permanent link")

When running as a script with `python notebook.py`, command-line arguments
are available to your program in `sys.argv`, just like any other Python
program. This means you can use
[`argparse`](https://docs.python.org/3/library/argparse.html),
[`simple-parsing`](https://github.com/lebrice/SimpleParsing), and other tools
for specifying and parsing arguments.

You can also use tools like `argparse` when running as a notebook ( `marimo
edit` or `marimo run`) or exporting to HTML, IPYNB or another format ( `marimo
export`). In these cases, `sys.argv` is set to the notebook filename followed by any args
following the `--` separator.

For example, for

sets `sys.argv` to `["notebook.py", "--lr", "1e-4", "--epochs", "15"]`.

For usage examples, see the guide on [running as a script](../../guides/scripts/).

## Utility function for parsing arguments [¶](\#utility-function-for-parsing-arguments "Permanent link")

marimo provides a utility function called `mo.cli_args()` for parsing arguments
from strings into primitive data types ( `int`, `bool`, `float`, `str`).
However, unlike `argparse` and `simple-parsing`, this function does not let you
declare your program's arguments, nor does it generate help text. **For these**
**reasons, we recommend using `argparse` or `simple-parsing` instead.**

## `` marimo.cli\_args [¶](\#marimo.cli_args "Permanent link")

Get the command line arguments of a marimo notebook.

Examples:

`marimo edit notebook.py -- -size 10`

| RETURNS | DESCRIPTION |
| --- | --- |
| `CLIArgs` | A dictionary containing the command line arguments.<br>This dictionary is read-only and cannot be mutated.<br>**TYPE:** `CLIArgs` |

Query Parameters

You can also access query parameters passed to the notebook using
`mo.query_params`. This allows you to pass arguments to the notebook that can be controlled by the user.


---

### Diagrams - marimoDiagrams¶

*Source: [https://docs.marimo.io/api/diagrams/](https://docs.marimo.io/api/diagrams/)*

<!-- Source: https://docs.marimo.io/api/diagrams/ -->
<!-- Title: Diagrams - marimoDiagrams¶ -->
<!-- Depth: 2 -->

[Skip to content](#diagrams)

# Diagrams [¶](\#diagrams "Permanent link")

## Mermaid diagrams [¶](\#mermaid-diagrams "Permanent link")

## `` marimo.mermaid [¶](\#marimo.mermaid "Permanent link")

Render a diagram with Mermaid.

Mermaid is a tool for making diagrams such as flow charts and graphs. See
the [Mermaid documentation](https://github.com/mermaid-js/mermaid#readme)
for details.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `diagram` | a string containing a Mermaid diagram<br>**TYPE:** `str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

Back to top

# Diagrams [¶](\#diagrams "Permanent link")

## Mermaid diagrams [¶](\#mermaid-diagrams "Permanent link")

## `` marimo.mermaid [¶](\#marimo.mermaid "Permanent link")

Render a diagram with Mermaid.

Mermaid is a tool for making diagrams such as flow charts and graphs. See
the [Mermaid documentation](https://github.com/mermaid-js/mermaid#readme)
for details.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `diagram` | a string containing a Mermaid diagram<br>**TYPE:** `str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

 Back to top

# Diagrams [¶](\#diagrams "Permanent link")

## Mermaid diagrams [¶](\#mermaid-diagrams "Permanent link")

## `` marimo.mermaid [¶](\#marimo.mermaid "Permanent link")

Render a diagram with Mermaid.

Mermaid is a tool for making diagrams such as flow charts and graphs. See
the [Mermaid documentation](https://github.com/mermaid-js/mermaid#readme)
for details.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `diagram` | a string containing a Mermaid diagram<br>**TYPE:** `str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example


---

### Markdown - marimoMarkdown¶

*Source: [https://docs.marimo.io/api/markdown/](https://docs.marimo.io/api/markdown/)*

<!-- Source: https://docs.marimo.io/api/markdown/ -->
<!-- Title: Markdown - marimoMarkdown¶ -->
<!-- Depth: 2 -->

[Skip to content](#markdown)

# Markdown [¶](\#markdown "Permanent link")

Write markdown with `mo.md`; make your markdown **interactive**, **dynamic**,
and **visually rich** by interpolating arbitrary Python values and marimo
elements.

## `` marimo.md [¶](\#marimo.md "Permanent link")

Write markdown

This function takes a string of markdown as input and returns an Html
object. Output the object as the last expression of a cell to render
the markdown in your app.

**Interpolation.**

You can interpolate Python values into your markdown strings, for example
using f-strings. Html objects and UI elements can be directly interpolated.
For example:

For other objects, like plots, use marimo's `as_html` method to embed
them in markdown:

**LaTeX.**

Enclose LaTeX in single '$' signs for inline math, and double '$$' for
display math or square brackets for display math. (Use raw strings,
prefixed with an "r", to use single backslashes.) For example:

renders:

The exponential function \\(f(x) = e^x\\) can be represented as

\\\[
f(x) = 1 + x + \\frac{x^2}{2!} + \\frac{x^3}{3!} + \\ldots.
\\\]

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | a string of markdown<br>**TYPE:** `str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

## Loading LaTeX macros [¶](\#loading-latex-macros "Permanent link")

You can load LaTeX macros using `mo.latex(filename=...)`.

## `` marimo.latex [¶](\#marimo.latex "Permanent link")

Load LaTeX from a file or URL.

or

| PARAMETER | DESCRIPTION |
| --- | --- |
| `filename` | Path to a LaTeX file<br>**TYPE:** `Union[str, Path]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `None` | An `Html` object |

Side effects

The `mo.latex()` function has side effects (registering the LaTeX macros) and should be used in the same cell as `import marimo`. Otherwise, the LaTeX macros may not be loaded before the cells that use them.

## Icons [¶](\#icons "Permanent link")

We support rendering icons from [Iconify](https://icon-sets.iconify.design/).

When is inside markdown, you can render an icon with the syntax `::iconset:icon-name::` for example `::lucide:rocket::` or `::mdi:home::`. This is useful for quickly adding an icon, however, it does not support advanced configuration such as size, color, and rotation.

For other advanced features, use `mo.icon()` such as `mo.icon("lucide:rocket", size=20)` or `mo.icon("mdi:home", color="blue")`.

## `` marimo.icon [¶](\#marimo.icon "Permanent link")

Displays an icon. These icons are referenced by name from the
[Iconify](https://iconify.design/) library.

They are named in the format `icon-set:icon-name`, e.g.
`lucide:leaf`.

Icons are lazily loaded from a CDN, so they will not be loaded when
not connected to the internet.

These can be used in buttons, tabs, and other UI elements.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `icon_name` | the name of the icon to display<br>**TYPE:** `str` |
| `size` | the size of the icon in pixels<br>**TYPE:** `Optional[int]`**DEFAULT:** `None` |
| `color` | the color of the icon<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `inline` | whether to display the icon inline or as a block element<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `flip` | whether to flip the icon horizontally, vertically, or both<br>**TYPE:** `Optional[Literal['horizontal', 'vertical', 'horizontal,vertical']]`**DEFAULT:** `None` |
| `rotate` | whether to rotate the icon 90, 180, or 270 degrees<br>**TYPE:** `Optional[Literal['90deg', '180deg', '270deg']]`**DEFAULT:** `None` |
| `style` | a dictionary of CSS styles to apply to the icon<br>**TYPE:** `Optional[dict[str, Union[str, int, float, None]]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object.<br>**TYPE:** `Html` |

## Tooltips [¶](\#tooltips "Permanent link")

You can render a tooltip by adding the `data-tooltip` attribute to an element.

## Rendering images [¶](\#rendering-images "Permanent link")

You can render images from a local `public/` folder:

See [Static files](../../guides/outputs/#static-files) for information about serving images and other static assets.

Back to top

# Markdown [¶](\#markdown "Permanent link")

Write markdown with `mo.md`; make your markdown **interactive**, **dynamic**,
and **visually rich** by interpolating arbitrary Python values and marimo
elements.

## `` marimo.md [¶](\#marimo.md "Permanent link")

Write markdown

This function takes a string of markdown as input and returns an Html
object. Output the object as the last expression of a cell to render
the markdown in your app.

**Interpolation.**

You can interpolate Python values into your markdown strings, for example
using f-strings. Html objects and UI elements can be directly interpolated.
For example:

For other objects, like plots, use marimo's `as_html` method to embed
them in markdown:

**LaTeX.**

Enclose LaTeX in single '$' signs for inline math, and double '$$' for
display math or square brackets for display math. (Use raw strings,
prefixed with an "r", to use single backslashes.) For example:

renders:

The exponential function \\(f(x) = e^x\\) can be represented as

\\\[
f(x) = 1 + x + \\frac{x^2}{2!} + \\frac{x^3}{3!} + \\ldots.
\\\]

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | a string of markdown<br>**TYPE:** `str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

## Loading LaTeX macros [¶](\#loading-latex-macros "Permanent link")

You can load LaTeX macros using `mo.latex(filename=...)`.

## `` marimo.latex [¶](\#marimo.latex "Permanent link")

Load LaTeX from a file or URL.

or

| PARAMETER | DESCRIPTION |
| --- | --- |
| `filename` | Path to a LaTeX file<br>**TYPE:** `Union[str, Path]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `None` | An `Html` object |

Side effects

The `mo.latex()` function has side effects (registering the LaTeX macros) and should be used in the same cell as `import marimo`. Otherwise, the LaTeX macros may not be loaded before the cells that use them.

## Icons [¶](\#icons "Permanent link")

We support rendering icons from [Iconify](https://icon-sets.iconify.design/).

When is inside markdown, you can render an icon with the syntax `::iconset:icon-name::` for example `::lucide:rocket::` or `::mdi:home::`. This is useful for quickly adding an icon, however, it does not support advanced configuration such as size, color, and rotation.

For other advanced features, use `mo.icon()` such as `mo.icon("lucide:rocket", size=20)` or `mo.icon("mdi:home", color="blue")`.

## `` marimo.icon [¶](\#marimo.icon "Permanent link")

Displays an icon. These icons are referenced by name from the
[Iconify](https://iconify.design/) library.

They are named in the format `icon-set:icon-name`, e.g.
`lucide:leaf`.

Icons are lazily loaded from a CDN, so they will not be loaded when
not connected to the internet.

These can be used in buttons, tabs, and other UI elements.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `icon_name` | the name of the icon to display<br>**TYPE:** `str` |
| `size` | the size of the icon in pixels<br>**TYPE:** `Optional[int]`**DEFAULT:** `None` |
| `color` | the color of the icon<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `inline` | whether to display the icon inline or as a block element<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `flip` | whether to flip the icon horizontally, vertically, or both<br>**TYPE:** `Optional[Literal['horizontal', 'vertical', 'horizontal,vertical']]`**DEFAULT:** `None` |
| `rotate` | whether to rotate the icon 90, 180, or 270 degrees<br>**TYPE:** `Optional[Literal['90deg', '180deg', '270deg']]`**DEFAULT:** `None` |
| `style` | a dictionary of CSS styles to apply to the icon<br>**TYPE:** `Optional[dict[str, Union[str, int, float, None]]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object.<br>**TYPE:** `Html` |

## Tooltips [¶](\#tooltips "Permanent link")

You can render a tooltip by adding the `data-tooltip` attribute to an element.

## Rendering images [¶](\#rendering-images "Permanent link")

You can render images from a local `public/` folder:

See [Static files](../../guides/outputs/#static-files) for information about serving images and other static assets.

 Back to top

# Markdown [¶](\#markdown "Permanent link")

Write markdown with `mo.md`; make your markdown **interactive**, **dynamic**,
and **visually rich** by interpolating arbitrary Python values and marimo
elements.

## `` marimo.md [¶](\#marimo.md "Permanent link")

Write markdown

This function takes a string of markdown as input and returns an Html
object. Output the object as the last expression of a cell to render
the markdown in your app.

**Interpolation.**

You can interpolate Python values into your markdown strings, for example
using f-strings. Html objects and UI elements can be directly interpolated.
For example:

For other objects, like plots, use marimo's `as_html` method to embed
them in markdown:

**LaTeX.**

Enclose LaTeX in single '$' signs for inline math, and double '$$' for
display math or square brackets for display math. (Use raw strings,
prefixed with an "r", to use single backslashes.) For example:

renders:

The exponential function \\(f(x) = e^x\\) can be represented as

\\\[
f(x) = 1 + x + \\frac{x^2}{2!} + \\frac{x^3}{3!} + \\ldots.
\\\]

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | a string of markdown<br>**TYPE:** `str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

## Loading LaTeX macros [¶](\#loading-latex-macros "Permanent link")

You can load LaTeX macros using `mo.latex(filename=...)`.

## `` marimo.latex [¶](\#marimo.latex "Permanent link")

Load LaTeX from a file or URL.

or

| PARAMETER | DESCRIPTION |
| --- | --- |
| `filename` | Path to a LaTeX file<br>**TYPE:** `Union[str, Path]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `None` | An `Html` object |

Side effects

The `mo.latex()` function has side effects (registering the LaTeX macros) and should be used in the same cell as `import marimo`. Otherwise, the LaTeX macros may not be loaded before the cells that use them.

## Icons [¶](\#icons "Permanent link")

We support rendering icons from [Iconify](https://icon-sets.iconify.design/).

When is inside markdown, you can render an icon with the syntax `::iconset:icon-name::` for example `::lucide:rocket::` or `::mdi:home::`. This is useful for quickly adding an icon, however, it does not support advanced configuration such as size, color, and rotation.

For other advanced features, use `mo.icon()` such as `mo.icon("lucide:rocket", size=20)` or `mo.icon("mdi:home", color="blue")`.

## `` marimo.icon [¶](\#marimo.icon "Permanent link")

Displays an icon. These icons are referenced by name from the
[Iconify](https://iconify.design/) library.

They are named in the format `icon-set:icon-name`, e.g.
`lucide:leaf`.

Icons are lazily loaded from a CDN, so they will not be loaded when
not connected to the internet.

These can be used in buttons, tabs, and other UI elements.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `icon_name` | the name of the icon to display<br>**TYPE:** `str` |
| `size` | the size of the icon in pixels<br>**TYPE:** `Optional[int]`**DEFAULT:** `None` |
| `color` | the color of the icon<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `inline` | whether to display the icon inline or as a block element<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `flip` | whether to flip the icon horizontally, vertically, or both<br>**TYPE:** `Optional[Literal['horizontal', 'vertical', 'horizontal,vertical']]`**DEFAULT:** `None` |
| `rotate` | whether to rotate the icon 90, 180, or 270 degrees<br>**TYPE:** `Optional[Literal['90deg', '180deg', '270deg']]`**DEFAULT:** `None` |
| `style` | a dictionary of CSS styles to apply to the icon<br>**TYPE:** `Optional[dict[str, Union[str, int, float, None]]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object.<br>**TYPE:** `Html` |

## Tooltips [¶](\#tooltips "Permanent link")

You can render a tooltip by adding the `data-tooltip` attribute to an element.

## Rendering images [¶](\#rendering-images "Permanent link")

You can render images from a local `public/` folder:

See [Static files](../../guides/outputs/#static-files) for information about serving images and other static assets.


---

### State - marimoState¶

*Source: [https://docs.marimo.io/api/state/](https://docs.marimo.io/api/state/)*

<!-- Source: https://docs.marimo.io/api/state/ -->
<!-- Title: State - marimoState¶ -->
<!-- Depth: 2 -->

[Skip to content](#state)

# State [¶](\#state "Permanent link")

Stop! Read the interactivity guide first!

**Read the guide on [creating interactive\**
**elements](../../guides/interactivity/)** before reading this one!

Advanced topic!

This guide covers reactive state ( `mo.state`), an advanced topic.

**You likely don't need `mo.state`**. UI elements already have built-in
state, their associated value, which you can access with their `value` attribute.
For example, `mo.ui.slider()` has a value that is its current position on an
interval, while `mo.ui.button()` has a value that can be configured to
count the number of times it has been clicked, or to toggle between `True` and
`False`. Additionally, interacting with UI elements bound to global variables
[automatically executes cells](../../guides/interactivity/) that reference those
variables, letting you react to changes by just reading their
`value` attributes. **This functional paradigm is the preferred way of**
**reacting to UI interactions in marimo.** **Chances are, the reactive**
**execution built into UI elements will suffice.** (For example, [you don't need\
reactive state to handle a button click](../../recipes/#working-with-buttons).)

That said, here are some signs you might need `mo.state`:

- you need to maintain historical state related to a UI element that can't
be computed from its built-in `value` ( _e.g._, all values the user has
ever input into a form)
- you need to synchronize two different UI elements ( _e.g._, so that
interacting with either one controls the other)
- you need to introduce cycles across cells

**In over 99% of cases, you don't need and shouldn't use `mo.state`.** This
feature can introduce hard-to-find bugs.

## `` marimo.state [¶](\#marimo.state "Permanent link")

Mutable reactive state.

Warning: reactive state is an advanced feature that you likely don't need;
it makes it possible to introduce cycles and hard-to-debug code execution
paths. **In almost all cases, you should prefer using marimo's built-in**
**[reactive execution](https://docs.marimo.io/guides/reactivity) and**
**[interactivity](https://docs.marimo.io/guides/interactivity).**

This function takes an initial value and returns:

- a getter function that reads the state value
- a setter function to set the state's value

When you call the setter function and update the state value in one cell,
all _other_ cells that read any global variables assigned to the getter
will automatically run. By default, the cell that called the setter
function won't be re-run, even if it references the getter. To allow a
state setter to possibly run the caller cell, set the keyword argument
`allow_self_loops=True`.

You can use this function with `UIElement` `on_change` handlers to trigger
side-effects when an element's value is updated; however, you should
prefer using marimo's built-in [reactive execution for interactive\
elements](https://docs.marimo.io/guides/interactivity).

For example, you can tie multiple UI elements to derive their values from
shared state.

Examples:

Create state:

Read the value:

Update the state:

Update based on current value:

Never mutate the state directly. You should only change its value through
its setter.

**Synchronizing multiple UI elements:**

Warning

Do not store `marimo.ui` elements in state; doing so can cause
hard-to-diagnose bugs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | Initial value of the state.<br>**TYPE:** `T` |
| `allow_self_loops` | If True, a cell that calls a state setter and also<br>references its getter will be re-run. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `State[T]` | A tuple of (getter function, setter function). The getter function |
| `Callable[[T], None]` | retrieves the state value; the setter function takes a new value or a |
| `tuple[State[T], Callable[[T], None]]` | function that updates the current value. |

Back to top

# State [¶](\#state "Permanent link")

Stop! Read the interactivity guide first!

**Read the guide on [creating interactive\**
**elements](../../guides/interactivity/)** before reading this one!

Advanced topic!

This guide covers reactive state ( `mo.state`), an advanced topic.

**You likely don't need `mo.state`**. UI elements already have built-in
state, their associated value, which you can access with their `value` attribute.
For example, `mo.ui.slider()` has a value that is its current position on an
interval, while `mo.ui.button()` has a value that can be configured to
count the number of times it has been clicked, or to toggle between `True` and
`False`. Additionally, interacting with UI elements bound to global variables
[automatically executes cells](../../guides/interactivity/) that reference those
variables, letting you react to changes by just reading their
`value` attributes. **This functional paradigm is the preferred way of**
**reacting to UI interactions in marimo.** **Chances are, the reactive**
**execution built into UI elements will suffice.** (For example, [you don't need\
reactive state to handle a button click](../../recipes/#working-with-buttons).)

That said, here are some signs you might need `mo.state`:

- you need to maintain historical state related to a UI element that can't
be computed from its built-in `value` ( _e.g._, all values the user has
ever input into a form)
- you need to synchronize two different UI elements ( _e.g._, so that
interacting with either one controls the other)
- you need to introduce cycles across cells

**In over 99% of cases, you don't need and shouldn't use `mo.state`.** This
feature can introduce hard-to-find bugs.

## `` marimo.state [¶](\#marimo.state "Permanent link")

Mutable reactive state.

Warning: reactive state is an advanced feature that you likely don't need;
it makes it possible to introduce cycles and hard-to-debug code execution
paths. **In almost all cases, you should prefer using marimo's built-in**
**[reactive execution](https://docs.marimo.io/guides/reactivity) and**
**[interactivity](https://docs.marimo.io/guides/interactivity).**

This function takes an initial value and returns:

- a getter function that reads the state value
- a setter function to set the state's value

When you call the setter function and update the state value in one cell,
all _other_ cells that read any global variables assigned to the getter
will automatically run. By default, the cell that called the setter
function won't be re-run, even if it references the getter. To allow a
state setter to possibly run the caller cell, set the keyword argument
`allow_self_loops=True`.

You can use this function with `UIElement` `on_change` handlers to trigger
side-effects when an element's value is updated; however, you should
prefer using marimo's built-in [reactive execution for interactive\
elements](https://docs.marimo.io/guides/interactivity).

For example, you can tie multiple UI elements to derive their values from
shared state.

Examples:

Create state:

Read the value:

Update the state:

Update based on current value:

Never mutate the state directly. You should only change its value through
its setter.

**Synchronizing multiple UI elements:**

Warning

Do not store `marimo.ui` elements in state; doing so can cause
hard-to-diagnose bugs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | Initial value of the state.<br>**TYPE:** `T` |
| `allow_self_loops` | If True, a cell that calls a state setter and also<br>references its getter will be re-run. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `State[T]` | A tuple of (getter function, setter function). The getter function |
| `Callable[[T], None]` | retrieves the state value; the setter function takes a new value or a |
| `tuple[State[T], Callable[[T], None]]` | function that updates the current value. |

 Back to top

# State [¶](\#state "Permanent link")

Stop! Read the interactivity guide first!

**Read the guide on [creating interactive\**
**elements](../../guides/interactivity/)** before reading this one!

Advanced topic!

This guide covers reactive state ( `mo.state`), an advanced topic.

**You likely don't need `mo.state`**. UI elements already have built-in
state, their associated value, which you can access with their `value` attribute.
For example, `mo.ui.slider()` has a value that is its current position on an
interval, while `mo.ui.button()` has a value that can be configured to
count the number of times it has been clicked, or to toggle between `True` and
`False`. Additionally, interacting with UI elements bound to global variables
[automatically executes cells](../../guides/interactivity/) that reference those
variables, letting you react to changes by just reading their
`value` attributes. **This functional paradigm is the preferred way of**
**reacting to UI interactions in marimo.** **Chances are, the reactive**
**execution built into UI elements will suffice.** (For example, [you don't need\
reactive state to handle a button click](../../recipes/#working-with-buttons).)

That said, here are some signs you might need `mo.state`:

- you need to maintain historical state related to a UI element that can't
be computed from its built-in `value` ( _e.g._, all values the user has
ever input into a form)
- you need to synchronize two different UI elements ( _e.g._, so that
interacting with either one controls the other)
- you need to introduce cycles across cells

**In over 99% of cases, you don't need and shouldn't use `mo.state`.** This
feature can introduce hard-to-find bugs.

## `` marimo.state [¶](\#marimo.state "Permanent link")

Mutable reactive state.

Warning: reactive state is an advanced feature that you likely don't need;
it makes it possible to introduce cycles and hard-to-debug code execution
paths. **In almost all cases, you should prefer using marimo's built-in**
**[reactive execution](https://docs.marimo.io/guides/reactivity) and**
**[interactivity](https://docs.marimo.io/guides/interactivity).**

This function takes an initial value and returns:

- a getter function that reads the state value
- a setter function to set the state's value

When you call the setter function and update the state value in one cell,
all _other_ cells that read any global variables assigned to the getter
will automatically run. By default, the cell that called the setter
function won't be re-run, even if it references the getter. To allow a
state setter to possibly run the caller cell, set the keyword argument
`allow_self_loops=True`.

You can use this function with `UIElement` `on_change` handlers to trigger
side-effects when an element's value is updated; however, you should
prefer using marimo's built-in [reactive execution for interactive\
elements](https://docs.marimo.io/guides/interactivity).

For example, you can tie multiple UI elements to derive their values from
shared state.

Examples:

Create state:

Read the value:

Update the state:

Update based on current value:

Never mutate the state directly. You should only change its value through
its setter.

**Synchronizing multiple UI elements:**

Warning

Do not store `marimo.ui` elements in state; doing so can cause
hard-to-diagnose bugs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | Initial value of the state.<br>**TYPE:** `T` |
| `allow_self_loops` | If True, a cell that calls a state setter and also<br>references its getter will be re-run. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `State[T]` | A tuple of (getter function, setter function). The getter function |
| `Callable[[T], None]` | retrieves the state value; the setter function takes a new value or a |
| `tuple[State[T], Callable[[T], None]]` | function that updates the current value. |


---

### Watch - marimoWatch¶

*Source: [https://docs.marimo.io/api/watch/](https://docs.marimo.io/api/watch/)*

<!-- Source: https://docs.marimo.io/api/watch/ -->
<!-- Title: Watch - marimoWatch¶ -->
<!-- Depth: 2 -->

[Skip to content](#watch)

# Watch [¶](\#watch "Permanent link")

## `` marimo.watch.file [¶](\#marimo.watch.file "Permanent link")

A reactive wrapper for file paths.

This function takes a file path to watch and returns a wrapper to reactively
read and write from the file.

The "wrapped" file Path object exposes most of the same methods as the
[pathlib.Path object](https://docs.python.org/3/library/pathlib.html#pathlib.Path),
with a few exceptions. The following methods are not available:

- `open()`
- `rename()`
- `replace()`

This object will trigger dependent cells to re-evaluate when the file is
changed.

Warning

It is possible to misuse this API in similar ways to `state()`. Consider
reading the warning and caveats in the
[`state()` documentation](../state/), and using this function only when
reading file paths, and not when writing them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path to watch.<br>**TYPE:** `Path | str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `FileState` | A reactive wrapper for watching the file path. |

## `` marimo.watch.directory [¶](\#marimo.watch.directory "Permanent link")

A reactive wrapper for directory paths.

This function takes a directory path to watch and returns a wrapper to
reactively list the contents of the directory.

This object will trigger dependent cells to re-evaluate when the directory
structure is changed (i.e., files are added or removed).

Note

This function does NOT react to file content changes, only to changes in
the directory structure. Utilize `mo.watch.file()` to watch for changes
in specific files. Additional note: this will not follow symlinks.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path to watch.<br>**TYPE:** `Path | str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `DirectoryState` | A reactive wrapper for watching the directory. |

Back to top

# Watch [¶](\#watch "Permanent link")

## `` marimo.watch.file [¶](\#marimo.watch.file "Permanent link")

A reactive wrapper for file paths.

This function takes a file path to watch and returns a wrapper to reactively
read and write from the file.

The "wrapped" file Path object exposes most of the same methods as the
[pathlib.Path object](https://docs.python.org/3/library/pathlib.html#pathlib.Path),
with a few exceptions. The following methods are not available:

- `open()`
- `rename()`
- `replace()`

This object will trigger dependent cells to re-evaluate when the file is
changed.

Warning

It is possible to misuse this API in similar ways to `state()`. Consider
reading the warning and caveats in the
[`state()` documentation](../state/), and using this function only when
reading file paths, and not when writing them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path to watch.<br>**TYPE:** `Path | str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `FileState` | A reactive wrapper for watching the file path. |

## `` marimo.watch.directory [¶](\#marimo.watch.directory "Permanent link")

A reactive wrapper for directory paths.

This function takes a directory path to watch and returns a wrapper to
reactively list the contents of the directory.

This object will trigger dependent cells to re-evaluate when the directory
structure is changed (i.e., files are added or removed).

Note

This function does NOT react to file content changes, only to changes in
the directory structure. Utilize `mo.watch.file()` to watch for changes
in specific files. Additional note: this will not follow symlinks.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path to watch.<br>**TYPE:** `Path | str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `DirectoryState` | A reactive wrapper for watching the directory. |

 Back to top

# Watch [¶](\#watch "Permanent link")

## `` marimo.watch.file [¶](\#marimo.watch.file "Permanent link")

A reactive wrapper for file paths.

This function takes a file path to watch and returns a wrapper to reactively
read and write from the file.

The "wrapped" file Path object exposes most of the same methods as the
[pathlib.Path object](https://docs.python.org/3/library/pathlib.html#pathlib.Path),
with a few exceptions. The following methods are not available:

- `open()`
- `rename()`
- `replace()`

This object will trigger dependent cells to re-evaluate when the file is
changed.

Warning

It is possible to misuse this API in similar ways to `state()`. Consider
reading the warning and caveats in the
[`state()` documentation](../state/), and using this function only when
reading file paths, and not when writing them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path to watch.<br>**TYPE:** `Path | str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `FileState` | A reactive wrapper for watching the file path. |

## `` marimo.watch.directory [¶](\#marimo.watch.directory "Permanent link")

A reactive wrapper for directory paths.

This function takes a directory path to watch and returns a wrapper to
reactively list the contents of the directory.

This object will trigger dependent cells to re-evaluate when the directory
structure is changed (i.e., files are added or removed).

Note

This function does NOT react to file content changes, only to changes in
the directory structure. Utilize `mo.watch.file()` to watch for changes
in specific files. Additional note: this will not follow symlinks.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path to watch.<br>**TYPE:** `Path | str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `DirectoryState` | A reactive wrapper for watching the directory. |


---

#### HTML - marimoHTML¶

*Source: [https://docs.marimo.io/api/html/](https://docs.marimo.io/api/html/)*

<!-- Source: https://docs.marimo.io/api/html/ -->
<!-- Title: HTML - marimoHTML¶ -->
<!-- Depth: 3 -->

[Skip to content](#html)

# HTML [¶](\#html "Permanent link")

All marimo elements extend the HTML element class.

## `` marimo.as\_html [¶](\#marimo.as_html "Permanent link")

Convert a value to HTML that can be embedded into markdown.

This function returns an `Html` object representing `value`. Use it to
embed values into Markdown or other HTML strings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | An object<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object |

Example

## `` marimo.Html`dataclass`[¶](\#marimo.Html "Permanent link")

Bases: `MIME`

A wrapper around HTML text that can be used as an output.

Output an `Html` object as the last expression of a cell to render it in
your app.

Use f-strings to embed Html objects as text into other HTML or markdown
strings. For example:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `text` | a string of HTML<br>**TYPE:** `str` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | a string of HTML<br>**TYPE:** `str` |

| METHOD | DESCRIPTION |
| --- | --- |
| `batch` | convert this HTML element into a batched UI element |
| `callout` | wrap this element in a callout |
| `center` | center this element in the output area |
| `right` | right-justify this element in the output area |

Initialize the HTML element.

Subclasses of HTML MUST call this method.

### `` text`property`[¶](\#marimo.Html.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.Html.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.Html.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.Html.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` left [¶](\#marimo.Html.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.Html.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.Html.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

## `` marimo.iframe [¶](\#marimo.iframe "Permanent link")

Embed an HTML string in an iframe.

Scripts by default are not executed using `mo.as_html` or `mo.Html`,
so if you have a script tag (written as `<script></script>`),
you can use `mo.iframe` for scripts to be executed.

You may also want to use this function to display HTML content
that may contain styles that could interfere with the rest of the
page.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html` | An HTML string<br>**TYPE:** `str` |
| `width` | The width of the iframe<br>**TYPE:** `str`**DEFAULT:** `'100%'` |
| `height` | The height of the iframe<br>**TYPE:** `str`**DEFAULT:** `'400px'` |

Back to top

# HTML [¶](\#html "Permanent link")

All marimo elements extend the HTML element class.

## `` marimo.as\_html [¶](\#marimo.as_html "Permanent link")

Convert a value to HTML that can be embedded into markdown.

This function returns an `Html` object representing `value`. Use it to
embed values into Markdown or other HTML strings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | An object<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object |

Example

## `` marimo.Html`dataclass`[¶](\#marimo.Html "Permanent link")

Bases: `MIME`

A wrapper around HTML text that can be used as an output.

Output an `Html` object as the last expression of a cell to render it in
your app.

Use f-strings to embed Html objects as text into other HTML or markdown
strings. For example:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `text` | a string of HTML<br>**TYPE:** `str` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | a string of HTML<br>**TYPE:** `str` |

| METHOD | DESCRIPTION |
| --- | --- |
| `batch` | convert this HTML element into a batched UI element |
| `callout` | wrap this element in a callout |
| `center` | center this element in the output area |
| `right` | right-justify this element in the output area |

Initialize the HTML element.

Subclasses of HTML MUST call this method.

### `` text`property`[¶](\#marimo.Html.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.Html.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.Html.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.Html.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` left [¶](\#marimo.Html.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.Html.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.Html.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

## `` marimo.iframe [¶](\#marimo.iframe "Permanent link")

Embed an HTML string in an iframe.

Scripts by default are not executed using `mo.as_html` or `mo.Html`,
so if you have a script tag (written as `<script></script>`),
you can use `mo.iframe` for scripts to be executed.

You may also want to use this function to display HTML content
that may contain styles that could interfere with the rest of the
page.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html` | An HTML string<br>**TYPE:** `str` |
| `width` | The width of the iframe<br>**TYPE:** `str`**DEFAULT:** `'100%'` |
| `height` | The height of the iframe<br>**TYPE:** `str`**DEFAULT:** `'400px'` |

 Back to top

# HTML [¶](\#html "Permanent link")

All marimo elements extend the HTML element class.

## `` marimo.as\_html [¶](\#marimo.as_html "Permanent link")

Convert a value to HTML that can be embedded into markdown.

This function returns an `Html` object representing `value`. Use it to
embed values into Markdown or other HTML strings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | An object<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object |

Example

## `` marimo.Html`dataclass`[¶](\#marimo.Html "Permanent link")

Bases: `MIME`

A wrapper around HTML text that can be used as an output.

Output an `Html` object as the last expression of a cell to render it in
your app.

Use f-strings to embed Html objects as text into other HTML or markdown
strings. For example:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `text` | a string of HTML<br>**TYPE:** `str` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | a string of HTML<br>**TYPE:** `str` |

| METHOD | DESCRIPTION |
| --- | --- |
| `batch` | convert this HTML element into a batched UI element |
| `callout` | wrap this element in a callout |
| `center` | center this element in the output area |
| `right` | right-justify this element in the output area |

Initialize the HTML element.

Subclasses of HTML MUST call this method.

### `` text`property`[¶](\#marimo.Html.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.Html.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.Html.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.Html.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` left [¶](\#marimo.Html.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.Html.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.Html.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

## `` marimo.iframe [¶](\#marimo.iframe "Permanent link")

Embed an HTML string in an iframe.

Scripts by default are not executed using `mo.as_html` or `mo.Html`,
so if you have a script tag (written as `<script></script>`),
you can use `mo.iframe` for scripts to be executed.

You may also want to use this function to display HTML content
that may contain styles that could interfere with the rest of the
page.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html` | An HTML string<br>**TYPE:** `str` |
| `width` | The width of the iframe<br>**TYPE:** `str`**DEFAULT:** `'100%'` |
| `height` | The height of the iframe<br>**TYPE:** `str`**DEFAULT:** `'400px'` |


---

#### Visualize outputs - marimoVisualizing outputs¶

*Source: [https://docs.marimo.io/guides/outputs/](https://docs.marimo.io/guides/outputs/)*

<!-- Source: https://docs.marimo.io/guides/outputs/ -->
<!-- Title: Visualize outputs - marimoVisualizing outputs¶ -->
<!-- Depth: 3 -->

[Skip to content](#visualizing-outputs)

# Visualizing outputs [¶](\#visualizing-outputs "Permanent link")

The last expression of a cell is its visual output, rendered above the cell.
Outputs are included in the "app" or read-only view of the notebook. marimo
comes out of the box a number of elements to help you make rich outputs,
documented in the [API reference](../../api/).

## Markdown [¶](\#markdown "Permanent link")

Markdown is written with the marimo library function [`mo.md`](../../api/markdown/#marimo.md "            marimo.md").
Writing markdown programmatically lets you make dynamic markdown: interpolate
Python values into markdown strings, conditionally render your markdown, and
embed markdown in other objects.

Here's a simple hello world example:

Notice that marimo knows how to render marimo objects in markdown: you can just
embed them in [`mo.md()`](../../api/markdown/#marimo.md "            marimo.md") using an f-string, and marimo will
figure out how to display them!

For other objects, like matplotlib plots, wrap
them in [`mo.as_html()`](../../api/html/#marimo.as_html "            marimo.as_html") to tap into marimo's
media viewer:

### Markdown editor [¶](\#markdown-editor "Permanent link")

marimo automatically renders cells that only use `mo.md` in a markdown editor
that supports common hotkeys. You can switch between the Markdown and Python
editors by clicking the button in the top right:

marimo is pure Python, even when you're using markdown.

**Writing LaTeX.** The markdown editor supports writing LaTeX. You should typically
use a raw string for markdown with LaTeX, which you can activate by checking the `r`
box in the bottom-right corner of the markdown editor.

**Interpolating Python values.** Interpolating Python values requires using an
`f`-string, which you can activate by checking the `f` box in the bottom-right
corner of the markdown editor.

### Markdown extensions [¶](\#markdown-extensions "Permanent link")

#### Details [¶](\#details "Permanent link")

Create expandable details with additional context:

Source code for `examples/markdown/details.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

#### Admonitions [¶](\#admonitions "Permanent link")

Highlight text using admonitions:

Source code for `examples/markdown/admonitions.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

#### Emoji [¶](\#emoji "Permanent link")

Use `:emoji:` syntax to add emojis; for example, `:rocket:` creates 🚀.

### Static files [¶](\#static-files "Permanent link")

marimo supports serving static files from a `public/` folder located next to your notebook. This is useful for including images or other static assets in your notebook.

To use files from the public folder, create a `public` directory next to your notebook and reference files using the `public/` path prefix:

For security reasons:

- Only files within the `public` directory can be accessed
- Symlinks are not followed
- Path traversal attempts (e.g., `../`) are blocked

## Layout [¶](\#layout "Permanent link")

The marimo library also comes with elements for laying out outputs, including
[`mo.hstack`](../../api/layouts/stacks/#marimo.hstack "            marimo.hstack"), [`mo.vstack`](../../api/layouts/stacks/#marimo.vstack "            marimo.vstack"),
[`mo.accordion`](../../api/layouts/accordion/#marimo.accordion "            marimo.accordion"), [`mo.ui.tabs`](../../api/inputs/tabs/#marimo.ui.tabs "            marimo.ui.tabs"), [`mo.sidebar`](../../api/layouts/sidebar/#marimo.sidebar "            marimo.sidebar"),
[`mo.nav_menu`](../../api/inputs/nav_menu/#marimo.nav_menu "            marimo.nav_menu"), [`mo.ui.table`](../../api/inputs/table/#marimo.ui.table "            marimo.ui.table"),
and [many more](../../api/layouts/).

## Progress bars [¶](\#progress-bars "Permanent link")

Use [`mo.status.progress_bar`](../../api/status/#marimo.status.progress_bar "            marimo.status.progress_bar") and
[`mo.status.spinner`](../../api/status/#marimo.status.spinner "            marimo.status.spinner") to create progress indicators:

## Media [¶](\#media "Permanent link")

marimo comes with functions to display media, including images, audio,
video, pdfs, and more. See the [API docs](../../api/media/) for more info.

## Inspecting objects [¶](\#inspecting-objects "Permanent link")

marimo has built-in formatters for many objects, but sometimes the default
representation isn't useful (e.g., `<MyClass at 0x...>`). In these cases, use
[`mo.inspect()`](../../api/miscellaneous/#marimo.inspect "            marimo.inspect") to explore an object's attributes, methods,
and documentation as an output. See the [API\
docs](../../api/outputs/#object-inspection) for more details.

## Imperatively adding outputs [¶](\#imperatively-adding-outputs "Permanent link")

While a cell's output is its last expression, it can at times be helpful
to imperatively add to the output area while a cell is running. marimo
provides utility functions like
[`mo.output.append`](../../api/outputs/#marimo.output.append "            marimo.output.append") for accomplishing this; see the
[API docs](../../api/outputs/) for more information.

## Console Outputs [¶](\#console-outputs "Permanent link")

Console outputs, such as print statements, show up below a cell in the console
output area; they are not included in the output area or app view by default.

To include console outputs in the cell output area, use
[`mo.redirect_stdout`](../../api/outputs/#marimo.redirect_stdout "            marimo.redirect_stdout") or
[`mo.redirect_stderr`](../../api/outputs/#marimo.redirect_stderr "            marimo.redirect_stderr"):

marimo also includes utility functions for [capturing standard out](../../api/outputs/#marimo.capture_stdout "            marimo.capture_stdout") and [standard error](../../api/outputs/#marimo.capture_stderr "            marimo.capture_stderr") without redirecting them. See the [API docs](../../api/outputs/#console-outputs) for more.

## Threading [¶](\#threading "Permanent link")

To create a thread that can reliably communicate outputs to the frontend,
use [`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread"), which has exactly the same API as
as `threading.Thread`.

### Cleaning up your thread [¶](\#cleaning-up-your-thread "Permanent link")

When the cell that spawned a [`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread") is invalidated
(re-run, deleted, interrupted, or otherwise errored), the thread's
`should_exit` property will evaluate to `True`, at which point it is your
responsibility to clean up your thread. You can retrieve the current
[`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread") with [`mo.current_thread`](../../api/control_flow/#marimo.current_thread "            marimo.current_thread").

**Example.**

### Patching threads created by third-party code [¶](\#patching-threads-created-by-third-party-code "Permanent link")

If you need to forward outputs from threads spawned by third-party code, try
patching `threading.Thread`:

This however may leak threads, since the patched threads won't know to check the `mo.Thread`'s
`should_exit` property.

Back to top

# Visualizing outputs [¶](\#visualizing-outputs "Permanent link")

The last expression of a cell is its visual output, rendered above the cell.
Outputs are included in the "app" or read-only view of the notebook. marimo
comes out of the box a number of elements to help you make rich outputs,
documented in the [API reference](../../api/).

## Markdown [¶](\#markdown "Permanent link")

Markdown is written with the marimo library function [`mo.md`](../../api/markdown/#marimo.md "            marimo.md").
Writing markdown programmatically lets you make dynamic markdown: interpolate
Python values into markdown strings, conditionally render your markdown, and
embed markdown in other objects.

Here's a simple hello world example:

Notice that marimo knows how to render marimo objects in markdown: you can just
embed them in [`mo.md()`](../../api/markdown/#marimo.md "            marimo.md") using an f-string, and marimo will
figure out how to display them!

For other objects, like matplotlib plots, wrap
them in [`mo.as_html()`](../../api/html/#marimo.as_html "            marimo.as_html") to tap into marimo's
media viewer:

### Markdown editor [¶](\#markdown-editor "Permanent link")

marimo automatically renders cells that only use `mo.md` in a markdown editor
that supports common hotkeys. You can switch between the Markdown and Python
editors by clicking the button in the top right:

marimo is pure Python, even when you're using markdown.

**Writing LaTeX.** The markdown editor supports writing LaTeX. You should typically
use a raw string for markdown with LaTeX, which you can activate by checking the `r`
box in the bottom-right corner of the markdown editor.

**Interpolating Python values.** Interpolating Python values requires using an
`f`-string, which you can activate by checking the `f` box in the bottom-right
corner of the markdown editor.

### Markdown extensions [¶](\#markdown-extensions "Permanent link")

#### Details [¶](\#details "Permanent link")

Create expandable details with additional context:

Source code for `examples/markdown/details.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

#### Admonitions [¶](\#admonitions "Permanent link")

Highlight text using admonitions:

Source code for `examples/markdown/admonitions.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

#### Emoji [¶](\#emoji "Permanent link")

Use `:emoji:` syntax to add emojis; for example, `:rocket:` creates 🚀.

### Static files [¶](\#static-files "Permanent link")

marimo supports serving static files from a `public/` folder located next to your notebook. This is useful for including images or other static assets in your notebook.

To use files from the public folder, create a `public` directory next to your notebook and reference files using the `public/` path prefix:

For security reasons:

- Only files within the `public` directory can be accessed
- Symlinks are not followed
- Path traversal attempts (e.g., `../`) are blocked

## Layout [¶](\#layout "Permanent link")

The marimo library also comes with elements for laying out outputs, including
[`mo.hstack`](../../api/layouts/stacks/#marimo.hstack "            marimo.hstack"), [`mo.vstack`](../../api/layouts/stacks/#marimo.vstack "            marimo.vstack"),
[`mo.accordion`](../../api/layouts/accordion/#marimo.accordion "            marimo.accordion"), [`mo.ui.tabs`](../../api/inputs/tabs/#marimo.ui.tabs "            marimo.ui.tabs"), [`mo.sidebar`](../../api/layouts/sidebar/#marimo.sidebar "            marimo.sidebar"),
[`mo.nav_menu`](../../api/inputs/nav_menu/#marimo.nav_menu "            marimo.nav_menu"), [`mo.ui.table`](../../api/inputs/table/#marimo.ui.table "            marimo.ui.table"),
and [many more](../../api/layouts/).

## Progress bars [¶](\#progress-bars "Permanent link")

Use [`mo.status.progress_bar`](../../api/status/#marimo.status.progress_bar "            marimo.status.progress_bar") and
[`mo.status.spinner`](../../api/status/#marimo.status.spinner "            marimo.status.spinner") to create progress indicators:

## Media [¶](\#media "Permanent link")

marimo comes with functions to display media, including images, audio,
video, pdfs, and more. See the [API docs](../../api/media/) for more info.

## Inspecting objects [¶](\#inspecting-objects "Permanent link")

marimo has built-in formatters for many objects, but sometimes the default
representation isn't useful (e.g., `<MyClass at 0x...>`). In these cases, use
[`mo.inspect()`](../../api/miscellaneous/#marimo.inspect "            marimo.inspect") to explore an object's attributes, methods,
and documentation as an output. See the [API\
docs](../../api/outputs/#object-inspection) for more details.

## Imperatively adding outputs [¶](\#imperatively-adding-outputs "Permanent link")

While a cell's output is its last expression, it can at times be helpful
to imperatively add to the output area while a cell is running. marimo
provides utility functions like
[`mo.output.append`](../../api/outputs/#marimo.output.append "            marimo.output.append") for accomplishing this; see the
[API docs](../../api/outputs/) for more information.

## Console Outputs [¶](\#console-outputs "Permanent link")

Console outputs, such as print statements, show up below a cell in the console
output area; they are not included in the output area or app view by default.

To include console outputs in the cell output area, use
[`mo.redirect_stdout`](../../api/outputs/#marimo.redirect_stdout "            marimo.redirect_stdout") or
[`mo.redirect_stderr`](../../api/outputs/#marimo.redirect_stderr "            marimo.redirect_stderr"):

marimo also includes utility functions for [capturing standard out](../../api/outputs/#marimo.capture_stdout "            marimo.capture_stdout") and [standard error](../../api/outputs/#marimo.capture_stderr "            marimo.capture_stderr") without redirecting them. See the [API docs](../../api/outputs/#console-outputs) for more.

## Threading [¶](\#threading "Permanent link")

To create a thread that can reliably communicate outputs to the frontend,
use [`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread"), which has exactly the same API as
as `threading.Thread`.

### Cleaning up your thread [¶](\#cleaning-up-your-thread "Permanent link")

When the cell that spawned a [`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread") is invalidated
(re-run, deleted, interrupted, or otherwise errored), the thread's
`should_exit` property will evaluate to `True`, at which point it is your
responsibility to clean up your thread. You can retrieve the current
[`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread") with [`mo.current_thread`](../../api/control_flow/#marimo.current_thread "            marimo.current_thread").

**Example.**

### Patching threads created by third-party code [¶](\#patching-threads-created-by-third-party-code "Permanent link")

If you need to forward outputs from threads spawned by third-party code, try
patching `threading.Thread`:

This however may leak threads, since the patched threads won't know to check the `mo.Thread`'s
`should_exit` property.

 Back to top

# Visualizing outputs [¶](\#visualizing-outputs "Permanent link")

The last expression of a cell is its visual output, rendered above the cell.
Outputs are included in the "app" or read-only view of the notebook. marimo
comes out of the box a number of elements to help you make rich outputs,
documented in the [API reference](../../api/).

## Markdown [¶](\#markdown "Permanent link")

Markdown is written with the marimo library function [`mo.md`](../../api/markdown/#marimo.md "            marimo.md").
Writing markdown programmatically lets you make dynamic markdown: interpolate
Python values into markdown strings, conditionally render your markdown, and
embed markdown in other objects.

Here's a simple hello world example:

Notice that marimo knows how to render marimo objects in markdown: you can just
embed them in [`mo.md()`](../../api/markdown/#marimo.md "            marimo.md") using an f-string, and marimo will
figure out how to display them!

For other objects, like matplotlib plots, wrap
them in [`mo.as_html()`](../../api/html/#marimo.as_html "            marimo.as_html") to tap into marimo's
media viewer:

### Markdown editor [¶](\#markdown-editor "Permanent link")

marimo automatically renders cells that only use `mo.md` in a markdown editor
that supports common hotkeys. You can switch between the Markdown and Python
editors by clicking the button in the top right:

marimo is pure Python, even when you're using markdown.

**Writing LaTeX.** The markdown editor supports writing LaTeX. You should typically
use a raw string for markdown with LaTeX, which you can activate by checking the `r`
box in the bottom-right corner of the markdown editor.

**Interpolating Python values.** Interpolating Python values requires using an
`f`-string, which you can activate by checking the `f` box in the bottom-right
corner of the markdown editor.

### Markdown extensions [¶](\#markdown-extensions "Permanent link")

#### Details [¶](\#details "Permanent link")

Create expandable details with additional context:

Source code for `examples/markdown/details.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

#### Admonitions [¶](\#admonitions "Permanent link")

Highlight text using admonitions:

Source code for `examples/markdown/admonitions.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

#### Emoji [¶](\#emoji "Permanent link")

Use `:emoji:` syntax to add emojis; for example, `:rocket:` creates 🚀.

### Static files [¶](\#static-files "Permanent link")

marimo supports serving static files from a `public/` folder located next to your notebook. This is useful for including images or other static assets in your notebook.

To use files from the public folder, create a `public` directory next to your notebook and reference files using the `public/` path prefix:

For security reasons:

- Only files within the `public` directory can be accessed
- Symlinks are not followed
- Path traversal attempts (e.g., `../`) are blocked

## Layout [¶](\#layout "Permanent link")

The marimo library also comes with elements for laying out outputs, including
[`mo.hstack`](../../api/layouts/stacks/#marimo.hstack "            marimo.hstack"), [`mo.vstack`](../../api/layouts/stacks/#marimo.vstack "            marimo.vstack"),
[`mo.accordion`](../../api/layouts/accordion/#marimo.accordion "            marimo.accordion"), [`mo.ui.tabs`](../../api/inputs/tabs/#marimo.ui.tabs "            marimo.ui.tabs"), [`mo.sidebar`](../../api/layouts/sidebar/#marimo.sidebar "            marimo.sidebar"),
[`mo.nav_menu`](../../api/inputs/nav_menu/#marimo.nav_menu "            marimo.nav_menu"), [`mo.ui.table`](../../api/inputs/table/#marimo.ui.table "            marimo.ui.table"),
and [many more](../../api/layouts/).

## Progress bars [¶](\#progress-bars "Permanent link")

Use [`mo.status.progress_bar`](../../api/status/#marimo.status.progress_bar "            marimo.status.progress_bar") and
[`mo.status.spinner`](../../api/status/#marimo.status.spinner "            marimo.status.spinner") to create progress indicators:

## Media [¶](\#media "Permanent link")

marimo comes with functions to display media, including images, audio,
video, pdfs, and more. See the [API docs](../../api/media/) for more info.

## Inspecting objects [¶](\#inspecting-objects "Permanent link")

marimo has built-in formatters for many objects, but sometimes the default
representation isn't useful (e.g., `<MyClass at 0x...>`). In these cases, use
[`mo.inspect()`](../../api/miscellaneous/#marimo.inspect "            marimo.inspect") to explore an object's attributes, methods,
and documentation as an output. See the [API\
docs](../../api/outputs/#object-inspection) for more details.

## Imperatively adding outputs [¶](\#imperatively-adding-outputs "Permanent link")

While a cell's output is its last expression, it can at times be helpful
to imperatively add to the output area while a cell is running. marimo
provides utility functions like
[`mo.output.append`](../../api/outputs/#marimo.output.append "            marimo.output.append") for accomplishing this; see the
[API docs](../../api/outputs/) for more information.

## Console Outputs [¶](\#console-outputs "Permanent link")

Console outputs, such as print statements, show up below a cell in the console
output area; they are not included in the output area or app view by default.

To include console outputs in the cell output area, use
[`mo.redirect_stdout`](../../api/outputs/#marimo.redirect_stdout "            marimo.redirect_stdout") or
[`mo.redirect_stderr`](../../api/outputs/#marimo.redirect_stderr "            marimo.redirect_stderr"):

marimo also includes utility functions for [capturing standard out](../../api/outputs/#marimo.capture_stdout "            marimo.capture_stdout") and [standard error](../../api/outputs/#marimo.capture_stderr "            marimo.capture_stderr") without redirecting them. See the [API docs](../../api/outputs/#console-outputs) for more.

## Threading [¶](\#threading "Permanent link")

To create a thread that can reliably communicate outputs to the frontend,
use [`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread"), which has exactly the same API as
as `threading.Thread`.

### Cleaning up your thread [¶](\#cleaning-up-your-thread "Permanent link")

When the cell that spawned a [`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread") is invalidated
(re-run, deleted, interrupted, or otherwise errored), the thread's
`should_exit` property will evaluate to `True`, at which point it is your
responsibility to clean up your thread. You can retrieve the current
[`mo.Thread`](../../api/control_flow/#marimo.Thread "            marimo.Thread") with [`mo.current_thread`](../../api/control_flow/#marimo.current_thread "            marimo.current_thread").

**Example.**

### Patching threads created by third-party code [¶](\#patching-threads-created-by-third-party-code "Permanent link")

If you need to forward outputs from threads spawned by third-party code, try
patching `threading.Thread`:

This however may leak threads, since the patched threads won't know to check the `mo.Thread`'s
`should_exit` property.


---

#### Run notebooks as scripts - marimoRun as a script¶

*Source: [https://docs.marimo.io/guides/scripts/](https://docs.marimo.io/guides/scripts/)*

<!-- Source: https://docs.marimo.io/guides/scripts/ -->
<!-- Title: Run notebooks as scripts - marimoRun as a script¶ -->
<!-- Depth: 3 -->

[Skip to content](#run-as-a-script)

# Run as a script [¶](\#run-as-a-script "Permanent link")

You can run marimo notebooks as scripts at the command line, just like
any other Python script. For example,

Running a notebook as a script is useful when your notebook has side-effects,
like writing to disk. Print statements and other console outputs will show
up in your terminal.

Check before running

Before running a notebook as a script, you can use marimo's linter to check for issues that might prevent execution:

See the [Lint Rules](../lint_rules/) guide for more information about marimo's linting system.

Saving notebook outputs

To run as a script while also saving HTML of the notebook outputs, use

You can also pass command-line arguments to your notebook during export.
Separate these args from the command with two dashes:

Exporting to other formats, such as ipynb, is also possible:

## Command-line arguments [¶](\#command-line-arguments "Permanent link")

When run as a script, you can access your notebook's command-line arguments
through `sys.argv`, just like any other Python program. This also
means you can declare your notebook's command-line arguments using Python
libraries like [`argparse`](https://docs.python.org/3/library/argparse.html)
and [`simple-parsing`](https://github.com/lebrice/SimpleParsing).

These examples shows how to conditionally assign values to variables based on
command-line arguments when running as a script, and use default values when
running as a notebook.

### argparse [¶](\#argparse "Permanent link")

Source code for `examples/running_as_a_script/sharing_arguments.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

### simple-parsing [¶](\#simple-parsing "Permanent link")

Source code for `examples/running_as_a_script/with_simple_parsing.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

Back to top

# Run as a script [¶](\#run-as-a-script "Permanent link")

You can run marimo notebooks as scripts at the command line, just like
any other Python script. For example,

Running a notebook as a script is useful when your notebook has side-effects,
like writing to disk. Print statements and other console outputs will show
up in your terminal.

Check before running

Before running a notebook as a script, you can use marimo's linter to check for issues that might prevent execution:

See the [Lint Rules](../lint_rules/) guide for more information about marimo's linting system.

Saving notebook outputs

To run as a script while also saving HTML of the notebook outputs, use

You can also pass command-line arguments to your notebook during export.
Separate these args from the command with two dashes:

Exporting to other formats, such as ipynb, is also possible:

## Command-line arguments [¶](\#command-line-arguments "Permanent link")

When run as a script, you can access your notebook's command-line arguments
through `sys.argv`, just like any other Python program. This also
means you can declare your notebook's command-line arguments using Python
libraries like [`argparse`](https://docs.python.org/3/library/argparse.html)
and [`simple-parsing`](https://github.com/lebrice/SimpleParsing).

These examples shows how to conditionally assign values to variables based on
command-line arguments when running as a script, and use default values when
running as a notebook.

### argparse [¶](\#argparse "Permanent link")

Source code for `examples/running_as_a_script/sharing_arguments.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

### simple-parsing [¶](\#simple-parsing "Permanent link")

Source code for `examples/running_as_a_script/with_simple_parsing.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

 Back to top

# Run as a script [¶](\#run-as-a-script "Permanent link")

You can run marimo notebooks as scripts at the command line, just like
any other Python script. For example,

Running a notebook as a script is useful when your notebook has side-effects,
like writing to disk. Print statements and other console outputs will show
up in your terminal.

Check before running

Before running a notebook as a script, you can use marimo's linter to check for issues that might prevent execution:

See the [Lint Rules](../lint_rules/) guide for more information about marimo's linting system.

Saving notebook outputs

To run as a script while also saving HTML of the notebook outputs, use

You can also pass command-line arguments to your notebook during export.
Separate these args from the command with two dashes:

Exporting to other formats, such as ipynb, is also possible:

## Command-line arguments [¶](\#command-line-arguments "Permanent link")

When run as a script, you can access your notebook's command-line arguments
through `sys.argv`, just like any other Python program. This also
means you can declare your notebook's command-line arguments using Python
libraries like [`argparse`](https://docs.python.org/3/library/argparse.html)
and [`simple-parsing`](https://github.com/lebrice/SimpleParsing).

These examples shows how to conditionally assign values to variables based on
command-line arguments when running as a script, and use default values when
running as a notebook.

### argparse [¶](\#argparse "Permanent link")

Source code for `examples/running_as_a_script/sharing_arguments.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

### simple-parsing [¶](\#simple-parsing "Permanent link")

Source code for `examples/running_as_a_script/with_simple_parsing.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you


---

##### Control Flow - marimoControl flow¶

*Source: [https://docs.marimo.io/api/control_flow/](https://docs.marimo.io/api/control_flow/)*

<!-- Source: https://docs.marimo.io/api/control_flow/ -->
<!-- Title: Control Flow - marimoControl flow¶ -->
<!-- Depth: 4 -->

[Skip to content](#control-flow)

# Control flow [¶](\#control-flow "Permanent link")

## Controlling when cells run [¶](\#controlling-when-cells-run "Permanent link")

- Use [`mo.stop`](#marimo.stop "            marimo.stop") to halt execution of a cell when a condition is met.
- Combine [`mo.stop`](#marimo.stop "            marimo.stop") with [`mo.ui.run_button`](../inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to gate execution on button click.
- Use [`mo.ui.refresh`](../inputs/refresh/#marimo.ui.refresh "            marimo.ui.refresh") to make cells run periodically.

Lazy execution

In addition to these utilities, you can [configure the runtime to be lazy](../../guides/expensive_notebooks/#configure-how-marimo-runs-cells), marking cells as stale instead of automatically running them.

## `` marimo.stop [¶](\#marimo.stop "Permanent link")

Stops execution of a cell when `predicate` is `True`

When `predicate` is `True`, this function raises a `MarimoStopError`. If
uncaught, this exception stops execution of the current cell and makes
`output` its output. Any descendants of this cell that were previously
scheduled to run will not be run, and their defs will be removed from
program memory.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `predicate` | The predicate indicating whether to stop.<br>**TYPE:** `bool` |
| `output` | The output to be assigned to the current cell.<br>**TYPE:** `bool`**DEFAULT:** `None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `MarimoStopError` | When `predicate` is `True` |

## Threading [¶](\#threading "Permanent link")

## `` marimo.Thread [¶](\#marimo.Thread "Permanent link")

Bases: `Thread`

A Thread subclass that can communicate with the frontend.

`mo.Thread` has the same API as threading.Thread,
but `mo.Thread` s are able to communicate with the marimo
frontend, whereas `threading.Thread` can't.

Threads can append to a cell's output using `mo.output.append`, or to the
console output area using `print`. The corresponding outputs will be
forwarded to the frontend.

Writing directly to sys.stdout or sys.stderr, or to file descriptors 1 and
2, is not yet supported.

**Thread lifecycle.** When the cell that spawned this thread is invalidated
(re-run, deleted, interrupted, or otherwise errored), this thread's
`should_exit` property will evaluate to `True`, at which point it
is the developer's responsibility to clean up their thread.

**Example.**

### `` should\_exit`property`[¶](\#marimo.Thread.should_exit "Permanent link")

Whether this thread should exit.

Returns `True` when the cell that spawned this thread has been invalidated;
for example, if the cell:

- was re-run
- was deleted
- was interrupted

then this property evaluates to True.

It is the developer's responsibility to clean up and finish their
thread when this flag is set. Retrieve the current `mo.Thread` with

### `` run [¶](\#marimo.Thread.run "Permanent link")

## `` marimo.current\_thread [¶](\#marimo.current_thread "Permanent link")

Return the `marimo.Thread` object for the caller's thread of control.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Thread` | The current `marimo.Thread` object. |

| RAISES | DESCRIPTION |
| --- | --- |
| `RuntimeError` | If the current thread of control is not a `marimo.Thread`. |

Back to top

# Control flow [¶](\#control-flow "Permanent link")

## Controlling when cells run [¶](\#controlling-when-cells-run "Permanent link")

- Use [`mo.stop`](#marimo.stop "            marimo.stop") to halt execution of a cell when a condition is met.
- Combine [`mo.stop`](#marimo.stop "            marimo.stop") with [`mo.ui.run_button`](../inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to gate execution on button click.
- Use [`mo.ui.refresh`](../inputs/refresh/#marimo.ui.refresh "            marimo.ui.refresh") to make cells run periodically.

Lazy execution

In addition to these utilities, you can [configure the runtime to be lazy](../../guides/expensive_notebooks/#configure-how-marimo-runs-cells), marking cells as stale instead of automatically running them.

## `` marimo.stop [¶](\#marimo.stop "Permanent link")

Stops execution of a cell when `predicate` is `True`

When `predicate` is `True`, this function raises a `MarimoStopError`. If
uncaught, this exception stops execution of the current cell and makes
`output` its output. Any descendants of this cell that were previously
scheduled to run will not be run, and their defs will be removed from
program memory.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `predicate` | The predicate indicating whether to stop.<br>**TYPE:** `bool` |
| `output` | The output to be assigned to the current cell.<br>**TYPE:** `bool`**DEFAULT:** `None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `MarimoStopError` | When `predicate` is `True` |

## Threading [¶](\#threading "Permanent link")

## `` marimo.Thread [¶](\#marimo.Thread "Permanent link")

Bases: `Thread`

A Thread subclass that can communicate with the frontend.

`mo.Thread` has the same API as threading.Thread,
but `mo.Thread` s are able to communicate with the marimo
frontend, whereas `threading.Thread` can't.

Threads can append to a cell's output using `mo.output.append`, or to the
console output area using `print`. The corresponding outputs will be
forwarded to the frontend.

Writing directly to sys.stdout or sys.stderr, or to file descriptors 1 and
2, is not yet supported.

**Thread lifecycle.** When the cell that spawned this thread is invalidated
(re-run, deleted, interrupted, or otherwise errored), this thread's
`should_exit` property will evaluate to `True`, at which point it
is the developer's responsibility to clean up their thread.

**Example.**

### `` should\_exit`property`[¶](\#marimo.Thread.should_exit "Permanent link")

Whether this thread should exit.

Returns `True` when the cell that spawned this thread has been invalidated;
for example, if the cell:

- was re-run
- was deleted
- was interrupted

then this property evaluates to True.

It is the developer's responsibility to clean up and finish their
thread when this flag is set. Retrieve the current `mo.Thread` with

### `` run [¶](\#marimo.Thread.run "Permanent link")

## `` marimo.current\_thread [¶](\#marimo.current_thread "Permanent link")

Return the `marimo.Thread` object for the caller's thread of control.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Thread` | The current `marimo.Thread` object. |

| RAISES | DESCRIPTION |
| --- | --- |
| `RuntimeError` | If the current thread of control is not a `marimo.Thread`. |

 Back to top

# Control flow [¶](\#control-flow "Permanent link")

## Controlling when cells run [¶](\#controlling-when-cells-run "Permanent link")

- Use [`mo.stop`](#marimo.stop "            marimo.stop") to halt execution of a cell when a condition is met.
- Combine [`mo.stop`](#marimo.stop "            marimo.stop") with [`mo.ui.run_button`](../inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to gate execution on button click.
- Use [`mo.ui.refresh`](../inputs/refresh/#marimo.ui.refresh "            marimo.ui.refresh") to make cells run periodically.

Lazy execution

In addition to these utilities, you can [configure the runtime to be lazy](../../guides/expensive_notebooks/#configure-how-marimo-runs-cells), marking cells as stale instead of automatically running them.

## `` marimo.stop [¶](\#marimo.stop "Permanent link")

Stops execution of a cell when `predicate` is `True`

When `predicate` is `True`, this function raises a `MarimoStopError`. If
uncaught, this exception stops execution of the current cell and makes
`output` its output. Any descendants of this cell that were previously
scheduled to run will not be run, and their defs will be removed from
program memory.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `predicate` | The predicate indicating whether to stop.<br>**TYPE:** `bool` |
| `output` | The output to be assigned to the current cell.<br>**TYPE:** `bool`**DEFAULT:** `None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `MarimoStopError` | When `predicate` is `True` |

## Threading [¶](\#threading "Permanent link")

## `` marimo.Thread [¶](\#marimo.Thread "Permanent link")

Bases: `Thread`

A Thread subclass that can communicate with the frontend.

`mo.Thread` has the same API as threading.Thread,
but `mo.Thread` s are able to communicate with the marimo
frontend, whereas `threading.Thread` can't.

Threads can append to a cell's output using `mo.output.append`, or to the
console output area using `print`. The corresponding outputs will be
forwarded to the frontend.

Writing directly to sys.stdout or sys.stderr, or to file descriptors 1 and
2, is not yet supported.

**Thread lifecycle.** When the cell that spawned this thread is invalidated
(re-run, deleted, interrupted, or otherwise errored), this thread's
`should_exit` property will evaluate to `True`, at which point it
is the developer's responsibility to clean up their thread.

**Example.**

### `` should\_exit`property`[¶](\#marimo.Thread.should_exit "Permanent link")

Whether this thread should exit.

Returns `True` when the cell that spawned this thread has been invalidated;
for example, if the cell:

- was re-run
- was deleted
- was interrupted

then this property evaluates to True.

It is the developer's responsibility to clean up and finish their
thread when this flag is set. Retrieve the current `mo.Thread` with

### `` run [¶](\#marimo.Thread.run "Permanent link")

## `` marimo.current\_thread [¶](\#marimo.current_thread "Permanent link")

Return the `marimo.Thread` object for the caller's thread of control.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Thread` | The current `marimo.Thread` object. |

| RAISES | DESCRIPTION |
| --- | --- |
| `RuntimeError` | If the current thread of control is not a `marimo.Thread`. |


---

##### Navigation Menu - marimoNavigation Menu¶

*Source: [https://docs.marimo.io/api/inputs/nav_menu/](https://docs.marimo.io/api/inputs/nav_menu/)*

<!-- Source: https://docs.marimo.io/api/inputs/nav_menu/ -->
<!-- Title: Navigation Menu - marimoNavigation Menu¶ -->
<!-- Depth: 4 -->

[Skip to content](#navigation-menu)

# Navigation Menu [¶](\#navigation-menu "Permanent link")

## `` marimo.nav\_menu [¶](\#marimo.nav_menu "Permanent link")

Navigation menu component.

This is useful for creating a navigation menu with hyperlinks,
most used when creating multi-page applications, with
`marimo.create_asgi_app` ( [docs](https://docs.marimo.io/guides/deploying/programmatically.html)).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `menu` | a dictionary of tab names to tab content;<br>the content can also be nested dictionaries (one level deep)<br>strings are interpreted as markdown<br>**TYPE:** `dict[str, JSONType]` |
| `orientation` | The orientation of the menu.<br>**TYPE:** `Literal['horizontal', 'vertical']`**DEFAULT:** `'horizontal'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

### You can also nest dictionaries to create submenus [¶](\#marimo.nav_menu--you-can-also-nest-dictionaries-to-create-submenus "Permanent link")

Back to top

# Navigation Menu [¶](\#navigation-menu "Permanent link")

## `` marimo.nav\_menu [¶](\#marimo.nav_menu "Permanent link")

Navigation menu component.

This is useful for creating a navigation menu with hyperlinks,
most used when creating multi-page applications, with
`marimo.create_asgi_app` ( [docs](https://docs.marimo.io/guides/deploying/programmatically.html)).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `menu` | a dictionary of tab names to tab content;<br>the content can also be nested dictionaries (one level deep)<br>strings are interpreted as markdown<br>**TYPE:** `dict[str, JSONType]` |
| `orientation` | The orientation of the menu.<br>**TYPE:** `Literal['horizontal', 'vertical']`**DEFAULT:** `'horizontal'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

### You can also nest dictionaries to create submenus [¶](\#marimo.nav_menu--you-can-also-nest-dictionaries-to-create-submenus "Permanent link")

 Back to top

# Navigation Menu [¶](\#navigation-menu "Permanent link")

## `` marimo.nav\_menu [¶](\#marimo.nav_menu "Permanent link")

Navigation menu component.

This is useful for creating a navigation menu with hyperlinks,
most used when creating multi-page applications, with
`marimo.create_asgi_app` ( [docs](https://docs.marimo.io/guides/deploying/programmatically.html)).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `menu` | a dictionary of tab names to tab content;<br>the content can also be nested dictionaries (one level deep)<br>strings are interpreted as markdown<br>**TYPE:** `dict[str, JSONType]` |
| `orientation` | The orientation of the menu.<br>**TYPE:** `Literal['horizontal', 'vertical']`**DEFAULT:** `'horizontal'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

### You can also nest dictionaries to create submenus [¶](\#marimo.nav_menu--you-can-also-nest-dictionaries-to-create-submenus "Permanent link")


---

##### Table - marimoTable¶

*Source: [https://docs.marimo.io/api/inputs/table/](https://docs.marimo.io/api/inputs/table/)*

<!-- Source: https://docs.marimo.io/api/inputs/table/ -->
<!-- Title: Table - marimoTable¶ -->
<!-- Depth: 4 -->

[Skip to content](#table)

# Table [¶](\#table "Permanent link")

## `` marimo.ui.table [¶](\#marimo.ui.table "Permanent link")

Bases: `UIElement[Union[list[str], list[int], list[dict[str, Any]]], Union[list[JSONType], IntoDataFrame, list[TableCell]]]`

A table component with selectable rows.

Get the selected rows with `table.value`. The table data can be supplied as:

1. a list of dicts, with one dict for each row, keyed by column names;
2. a list of values, representing a table with a single column;
3. a Pandas dataframe; or
4. a Polars dataframe; or
5. an Ibis dataframe; or
6. a PyArrow table.

Examples:

Create a table from a list of dicts, one for each row:

Create a table from a single column of data:

Create a table from a dataframe:

Create a table with format mapping:

Create a table with conditional cell formatting:

Create a table with per-cell hover text (plain text only):

Note: when sorting is applied, per-cell hover and styling may not be
correctly aligned with rows, similar to cell styling behavior.

In each case, access the table data with `table.value`.

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The selected rows, in the same format<br>as the original data, or None if no selection.<br>**TYPE:** `Union[List[JSONType], IntoDataFrame]` |
| `data` | The original table data.<br>**TYPE:** `Union[List[JSONType], IntoDataFrame]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | Values can be primitives ( `str`, `int`, `float`, `bool`, or `None`) or marimo elements:<br>e.g. `mo.ui.button(...)`, `mo.md(...)`, `mo.as_html(...)`, etc. Data can be passed in many ways:<br>\- as dataframes: a pandas dataframe, a polars dataframe<br>\- as rows: a list of dicts, where each dict represents a row in the table<br>\- as columns: a dict keyed by column names, where the value of each entry is a list representing a column<br>\- as a single column: a list of values<br>**TYPE:** `Union[List[Union[str, int, float, bool, MIME, None]], List[Dict[str, JSONType]], Dict[str, List[JSONType]], IntoDataFrame]` |
| `pagination` | Whether to paginate; if False, all rows will be shown.<br>Defaults to True when above 10 rows, False otherwise.<br>**TYPE:** `bool`**DEFAULT:** `None` |
| `selection` | 'single' or 'multi' to enable row selection,<br>'single-cell' or 'multi-cell' to enable cell selection<br>or None to disable. Defaults to "multi".<br>**TYPE:** `Literal['single', 'multi', 'single-cell', 'multi-cell']`**DEFAULT:** `'multi'` |
| `initial_selection` | Indices of the rows you want selected by default.<br>**TYPE:** `Union[List[int], List[tuple[str, str]]`**DEFAULT:** `None` |
| `page_size` | The number of rows to show per page. Defaults to 10.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `show_column_summaries` | Whether to show column summaries.<br>Defaults to True when the table has less than 40 columns and at least 10 rows, False otherwise.<br>If "stats", only show stats. If "chart", only show charts.<br>**TYPE:** `Union[bool, Literal['stats', 'chart']]`**DEFAULT:** `None` |
| `show_data_types` | Whether to show data types of columns in the table header.<br>Defaults to True.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `show_download` | Whether to show the download button.<br>Defaults to True for dataframes, False otherwise.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `format_mapping` | A mapping from<br>column names to formatting strings or functions.<br>**TYPE:** `Dict[str, Union[str, Callable[..., Any]]]`**DEFAULT:** `None` |
| `freeze_columns_left` | List of column names to freeze on the left.<br>**TYPE:** `Sequence[str]`**DEFAULT:** `None` |
| `freeze_columns_right` | List of column names to freeze on the right.<br>**TYPE:** `Sequence[str]`**DEFAULT:** `None` |
| `text_justify_columns` | Dictionary of column names to text justification options: left, center, right.<br>**TYPE:** `Dict[str, Literal['left', 'center', 'right']]`**DEFAULT:** `None` |
| `wrapped_columns` | List of column names to wrap.<br>**TYPE:** `List[str]`**DEFAULT:** `None` |
| `header_tooltip` | Mapping from column names to tooltip text on the column header.<br>**TYPE:** `Dict[str, str]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to run when this element's value changes.<br>**TYPE:** `Callable[[Union[List[JSONType], Dict[str, List[JSONType]], IntoDataFrame, List[TableCell]]], None]`**DEFAULT:** `None` |
| `style_cell` | A function that takes the row id, column name and value and returns a dictionary of CSS styles.<br>**TYPE:** `Callable[[str, str, Any], Dict[str, Any]]`**DEFAULT:** `None` |
| `hover_template` | Either a string template applied at the row level, or a callable<br>that computes plain-text hover titles for individual visible cells.<br>When a callable is provided, values are computed per page in Python<br>and passed to the frontend; native HTML `title` is used for display.<br>Plain text only is supported.<br>**TYPE:** `Union[str, Callable[[str, str, Any], str]]`**DEFAULT:** `None` |
| `max_columns` | Maximum number of columns to display. Defaults to the<br>configured default\_table\_max\_columns (50 by default). Set to None to show all columns.<br>**TYPE:** `int`**DEFAULT:** `MAX_COLUMNS_NOT_PROVIDED` |
| `max_height` | Maximum height of the table body in pixels. When set,<br>the table becomes vertically scrollable and the header will be made sticky<br>in the UI to remain visible while scrolling. Defaults to None.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `label` | A descriptive name for the table. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |

### `` data`property`[¶](\#marimo.ui.table.data "Permanent link")

Get the original table data.

| RETURNS | DESCRIPTION |
| --- | --- |
| `TableData` | The original data passed to the table constructor, in its<br>original format (list, dict, dataframe, etc.).<br>**TYPE:** `TableData` |

### `` default\_page\_size`cached``property`[¶](\#marimo.ui.table.default_page_size "Permanent link")

### `` text`property`[¶](\#marimo.ui.table.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.table.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.table.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.table.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.table.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.table.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.table.from_args "Permanent link")

### `` lazy`staticmethod`[¶](\#marimo.ui.table.lazy "Permanent link")

Create a table from a Polars LazyFrame.

This won't load the data into memory until requested by the user.
Once requested, only the first 10 rows will be loaded.

Pagination and selection are not supported for lazy tables.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | The data to display.<br>**TYPE:** `IntoLazyFrame` |
| `page_size` | The number of rows to show per page.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `preload` | Whether to load the first page of data<br>without user confirmation. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

### `` left [¶](\#marimo.ui.table.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.table.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.table.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Table [¶](\#table "Permanent link")

## `` marimo.ui.table [¶](\#marimo.ui.table "Permanent link")

Bases: `UIElement[Union[list[str], list[int], list[dict[str, Any]]], Union[list[JSONType], IntoDataFrame, list[TableCell]]]`

A table component with selectable rows.

Get the selected rows with `table.value`. The table data can be supplied as:

1. a list of dicts, with one dict for each row, keyed by column names;
2. a list of values, representing a table with a single column;
3. a Pandas dataframe; or
4. a Polars dataframe; or
5. an Ibis dataframe; or
6. a PyArrow table.

Examples:

Create a table from a list of dicts, one for each row:

Create a table from a single column of data:

Create a table from a dataframe:

Create a table with format mapping:

Create a table with conditional cell formatting:

Create a table with per-cell hover text (plain text only):

Note: when sorting is applied, per-cell hover and styling may not be
correctly aligned with rows, similar to cell styling behavior.

In each case, access the table data with `table.value`.

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The selected rows, in the same format<br>as the original data, or None if no selection.<br>**TYPE:** `Union[List[JSONType], IntoDataFrame]` |
| `data` | The original table data.<br>**TYPE:** `Union[List[JSONType], IntoDataFrame]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | Values can be primitives ( `str`, `int`, `float`, `bool`, or `None`) or marimo elements:<br>e.g. `mo.ui.button(...)`, `mo.md(...)`, `mo.as_html(...)`, etc. Data can be passed in many ways:<br>\- as dataframes: a pandas dataframe, a polars dataframe<br>\- as rows: a list of dicts, where each dict represents a row in the table<br>\- as columns: a dict keyed by column names, where the value of each entry is a list representing a column<br>\- as a single column: a list of values<br>**TYPE:** `Union[List[Union[str, int, float, bool, MIME, None]], List[Dict[str, JSONType]], Dict[str, List[JSONType]], IntoDataFrame]` |
| `pagination` | Whether to paginate; if False, all rows will be shown.<br>Defaults to True when above 10 rows, False otherwise.<br>**TYPE:** `bool`**DEFAULT:** `None` |
| `selection` | 'single' or 'multi' to enable row selection,<br>'single-cell' or 'multi-cell' to enable cell selection<br>or None to disable. Defaults to "multi".<br>**TYPE:** `Literal['single', 'multi', 'single-cell', 'multi-cell']`**DEFAULT:** `'multi'` |
| `initial_selection` | Indices of the rows you want selected by default.<br>**TYPE:** `Union[List[int], List[tuple[str, str]]`**DEFAULT:** `None` |
| `page_size` | The number of rows to show per page. Defaults to 10.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `show_column_summaries` | Whether to show column summaries.<br>Defaults to True when the table has less than 40 columns and at least 10 rows, False otherwise.<br>If "stats", only show stats. If "chart", only show charts.<br>**TYPE:** `Union[bool, Literal['stats', 'chart']]`**DEFAULT:** `None` |
| `show_data_types` | Whether to show data types of columns in the table header.<br>Defaults to True.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `show_download` | Whether to show the download button.<br>Defaults to True for dataframes, False otherwise.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `format_mapping` | A mapping from<br>column names to formatting strings or functions.<br>**TYPE:** `Dict[str, Union[str, Callable[..., Any]]]`**DEFAULT:** `None` |
| `freeze_columns_left` | List of column names to freeze on the left.<br>**TYPE:** `Sequence[str]`**DEFAULT:** `None` |
| `freeze_columns_right` | List of column names to freeze on the right.<br>**TYPE:** `Sequence[str]`**DEFAULT:** `None` |
| `text_justify_columns` | Dictionary of column names to text justification options: left, center, right.<br>**TYPE:** `Dict[str, Literal['left', 'center', 'right']]`**DEFAULT:** `None` |
| `wrapped_columns` | List of column names to wrap.<br>**TYPE:** `List[str]`**DEFAULT:** `None` |
| `header_tooltip` | Mapping from column names to tooltip text on the column header.<br>**TYPE:** `Dict[str, str]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to run when this element's value changes.<br>**TYPE:** `Callable[[Union[List[JSONType], Dict[str, List[JSONType]], IntoDataFrame, List[TableCell]]], None]`**DEFAULT:** `None` |
| `style_cell` | A function that takes the row id, column name and value and returns a dictionary of CSS styles.<br>**TYPE:** `Callable[[str, str, Any], Dict[str, Any]]`**DEFAULT:** `None` |
| `hover_template` | Either a string template applied at the row level, or a callable<br>that computes plain-text hover titles for individual visible cells.<br>When a callable is provided, values are computed per page in Python<br>and passed to the frontend; native HTML `title` is used for display.<br>Plain text only is supported.<br>**TYPE:** `Union[str, Callable[[str, str, Any], str]]`**DEFAULT:** `None` |
| `max_columns` | Maximum number of columns to display. Defaults to the<br>configured default\_table\_max\_columns (50 by default). Set to None to show all columns.<br>**TYPE:** `int`**DEFAULT:** `MAX_COLUMNS_NOT_PROVIDED` |
| `max_height` | Maximum height of the table body in pixels. When set,<br>the table becomes vertically scrollable and the header will be made sticky<br>in the UI to remain visible while scrolling. Defaults to None.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `label` | A descriptive name for the table. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |

### `` data`property`[¶](\#marimo.ui.table.data "Permanent link")

Get the original table data.

| RETURNS | DESCRIPTION |
| --- | --- |
| `TableData` | The original data passed to the table constructor, in its<br>original format (list, dict, dataframe, etc.).<br>**TYPE:** `TableData` |

### `` default\_page\_size`cached``property`[¶](\#marimo.ui.table.default_page_size "Permanent link")

### `` text`property`[¶](\#marimo.ui.table.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.table.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.table.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.table.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.table.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.table.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.table.from_args "Permanent link")

### `` lazy`staticmethod`[¶](\#marimo.ui.table.lazy "Permanent link")

Create a table from a Polars LazyFrame.

This won't load the data into memory until requested by the user.
Once requested, only the first 10 rows will be loaded.

Pagination and selection are not supported for lazy tables.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | The data to display.<br>**TYPE:** `IntoLazyFrame` |
| `page_size` | The number of rows to show per page.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `preload` | Whether to load the first page of data<br>without user confirmation. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

### `` left [¶](\#marimo.ui.table.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.table.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.table.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Table [¶](\#table "Permanent link")

## `` marimo.ui.table [¶](\#marimo.ui.table "Permanent link")

Bases: `UIElement[Union[list[str], list[int], list[dict[str, Any]]], Union[list[JSONType], IntoDataFrame, list[TableCell]]]`

A table component with selectable rows.

Get the selected rows with `table.value`. The table data can be supplied as:

1. a list of dicts, with one dict for each row, keyed by column names;
2. a list of values, representing a table with a single column;
3. a Pandas dataframe; or
4. a Polars dataframe; or
5. an Ibis dataframe; or
6. a PyArrow table.

Examples:

Create a table from a list of dicts, one for each row:

Create a table from a single column of data:

Create a table from a dataframe:

Create a table with format mapping:

Create a table with conditional cell formatting:

Create a table with per-cell hover text (plain text only):

Note: when sorting is applied, per-cell hover and styling may not be
correctly aligned with rows, similar to cell styling behavior.

In each case, access the table data with `table.value`.

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The selected rows, in the same format<br>as the original data, or None if no selection.<br>**TYPE:** `Union[List[JSONType], IntoDataFrame]` |
| `data` | The original table data.<br>**TYPE:** `Union[List[JSONType], IntoDataFrame]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | Values can be primitives ( `str`, `int`, `float`, `bool`, or `None`) or marimo elements:<br>e.g. `mo.ui.button(...)`, `mo.md(...)`, `mo.as_html(...)`, etc. Data can be passed in many ways:<br>\- as dataframes: a pandas dataframe, a polars dataframe<br>\- as rows: a list of dicts, where each dict represents a row in the table<br>\- as columns: a dict keyed by column names, where the value of each entry is a list representing a column<br>\- as a single column: a list of values<br>**TYPE:** `Union[List[Union[str, int, float, bool, MIME, None]], List[Dict[str, JSONType]], Dict[str, List[JSONType]], IntoDataFrame]` |
| `pagination` | Whether to paginate; if False, all rows will be shown.<br>Defaults to True when above 10 rows, False otherwise.<br>**TYPE:** `bool`**DEFAULT:** `None` |
| `selection` | 'single' or 'multi' to enable row selection,<br>'single-cell' or 'multi-cell' to enable cell selection<br>or None to disable. Defaults to "multi".<br>**TYPE:** `Literal['single', 'multi', 'single-cell', 'multi-cell']`**DEFAULT:** `'multi'` |
| `initial_selection` | Indices of the rows you want selected by default.<br>**TYPE:** `Union[List[int], List[tuple[str, str]]`**DEFAULT:** `None` |
| `page_size` | The number of rows to show per page. Defaults to 10.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `show_column_summaries` | Whether to show column summaries.<br>Defaults to True when the table has less than 40 columns and at least 10 rows, False otherwise.<br>If "stats", only show stats. If "chart", only show charts.<br>**TYPE:** `Union[bool, Literal['stats', 'chart']]`**DEFAULT:** `None` |
| `show_data_types` | Whether to show data types of columns in the table header.<br>Defaults to True.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `show_download` | Whether to show the download button.<br>Defaults to True for dataframes, False otherwise.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `format_mapping` | A mapping from<br>column names to formatting strings or functions.<br>**TYPE:** `Dict[str, Union[str, Callable[..., Any]]]`**DEFAULT:** `None` |
| `freeze_columns_left` | List of column names to freeze on the left.<br>**TYPE:** `Sequence[str]`**DEFAULT:** `None` |
| `freeze_columns_right` | List of column names to freeze on the right.<br>**TYPE:** `Sequence[str]`**DEFAULT:** `None` |
| `text_justify_columns` | Dictionary of column names to text justification options: left, center, right.<br>**TYPE:** `Dict[str, Literal['left', 'center', 'right']]`**DEFAULT:** `None` |
| `wrapped_columns` | List of column names to wrap.<br>**TYPE:** `List[str]`**DEFAULT:** `None` |
| `header_tooltip` | Mapping from column names to tooltip text on the column header.<br>**TYPE:** `Dict[str, str]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to run when this element's value changes.<br>**TYPE:** `Callable[[Union[List[JSONType], Dict[str, List[JSONType]], IntoDataFrame, List[TableCell]]], None]`**DEFAULT:** `None` |
| `style_cell` | A function that takes the row id, column name and value and returns a dictionary of CSS styles.<br>**TYPE:** `Callable[[str, str, Any], Dict[str, Any]]`**DEFAULT:** `None` |
| `hover_template` | Either a string template applied at the row level, or a callable<br>that computes plain-text hover titles for individual visible cells.<br>When a callable is provided, values are computed per page in Python<br>and passed to the frontend; native HTML `title` is used for display.<br>Plain text only is supported.<br>**TYPE:** `Union[str, Callable[[str, str, Any], str]]`**DEFAULT:** `None` |
| `max_columns` | Maximum number of columns to display. Defaults to the<br>configured default\_table\_max\_columns (50 by default). Set to None to show all columns.<br>**TYPE:** `int`**DEFAULT:** `MAX_COLUMNS_NOT_PROVIDED` |
| `max_height` | Maximum height of the table body in pixels. When set,<br>the table becomes vertically scrollable and the header will be made sticky<br>in the UI to remain visible while scrolling. Defaults to None.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `label` | A descriptive name for the table. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |

### `` data`property`[¶](\#marimo.ui.table.data "Permanent link")

Get the original table data.

| RETURNS | DESCRIPTION |
| --- | --- |
| `TableData` | The original data passed to the table constructor, in its<br>original format (list, dict, dataframe, etc.).<br>**TYPE:** `TableData` |

### `` default\_page\_size`cached``property`[¶](\#marimo.ui.table.default_page_size "Permanent link")

### `` text`property`[¶](\#marimo.ui.table.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.table.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.table.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.table.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.table.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.table.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.table.from_args "Permanent link")

### `` lazy`staticmethod`[¶](\#marimo.ui.table.lazy "Permanent link")

Create a table from a Polars LazyFrame.

This won't load the data into memory until requested by the user.
Once requested, only the first 10 rows will be loaded.

Pagination and selection are not supported for lazy tables.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | The data to display.<br>**TYPE:** `IntoLazyFrame` |
| `page_size` | The number of rows to show per page.<br>**TYPE:** `int`**DEFAULT:** `None` |
| `preload` | Whether to load the first page of data<br>without user confirmation. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

### `` left [¶](\#marimo.ui.table.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.table.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.table.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

##### Tabs - marimoTabs¶

*Source: [https://docs.marimo.io/api/inputs/tabs/](https://docs.marimo.io/api/inputs/tabs/)*

<!-- Source: https://docs.marimo.io/api/inputs/tabs/ -->
<!-- Title: Tabs - marimoTabs¶ -->
<!-- Depth: 4 -->

[Skip to content](#tabs)

# Tabs [¶](\#tabs "Permanent link")

Source code for `examples/ui/tabs.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.ui.tabs [¶](\#marimo.ui.tabs "Permanent link")

Bases: `UIElement[str, str]`

Display objects in a tabbed view.

Examples:

Show content in tabs:

Control which tab is selected:

Tab content can be lazily loaded:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The name of the selected tab.<br>**TYPE:** `str` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tabs` | A dictionary of tab names to tab content; strings<br>are interpreted as markdown.<br>**TYPE:** `dict[str, object]` |
| `value` | The name of the tab to open. Defaults to the first tab.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `lazy` | Whether to lazily load the tab content.<br>This is a convenience that wraps each tab in a `mo.lazy`<br>component. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `label` | A descriptive name for the tab. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Callable[[dict[str, object]], None]`**DEFAULT:** `None` |

### `` text`property`[¶](\#marimo.ui.tabs.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.tabs.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.tabs.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.tabs.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.tabs.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.tabs.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.tabs.from_args "Permanent link")

### `` left [¶](\#marimo.ui.tabs.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.tabs.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.tabs.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Tabs [¶](\#tabs "Permanent link")

Source code for `examples/ui/tabs.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.ui.tabs [¶](\#marimo.ui.tabs "Permanent link")

Bases: `UIElement[str, str]`

Display objects in a tabbed view.

Examples:

Show content in tabs:

Control which tab is selected:

Tab content can be lazily loaded:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The name of the selected tab.<br>**TYPE:** `str` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tabs` | A dictionary of tab names to tab content; strings<br>are interpreted as markdown.<br>**TYPE:** `dict[str, object]` |
| `value` | The name of the tab to open. Defaults to the first tab.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `lazy` | Whether to lazily load the tab content.<br>This is a convenience that wraps each tab in a `mo.lazy`<br>component. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `label` | A descriptive name for the tab. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Callable[[dict[str, object]], None]`**DEFAULT:** `None` |

### `` text`property`[¶](\#marimo.ui.tabs.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.tabs.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.tabs.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.tabs.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.tabs.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.tabs.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.tabs.from_args "Permanent link")

### `` left [¶](\#marimo.ui.tabs.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.tabs.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.tabs.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Tabs [¶](\#tabs "Permanent link")

Source code for `examples/ui/tabs.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.ui.tabs [¶](\#marimo.ui.tabs "Permanent link")

Bases: `UIElement[str, str]`

Display objects in a tabbed view.

Examples:

Show content in tabs:

Control which tab is selected:

Tab content can be lazily loaded:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The name of the selected tab.<br>**TYPE:** `str` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tabs` | A dictionary of tab names to tab content; strings<br>are interpreted as markdown.<br>**TYPE:** `dict[str, object]` |
| `value` | The name of the tab to open. Defaults to the first tab.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `lazy` | Whether to lazily load the tab content.<br>This is a convenience that wraps each tab in a `mo.lazy`<br>component. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `label` | A descriptive name for the tab. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Callable[[dict[str, object]], None]`**DEFAULT:** `None` |

### `` text`property`[¶](\#marimo.ui.tabs.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.tabs.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.tabs.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.tabs.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.tabs.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.tabs.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.tabs.from_args "Permanent link")

### `` left [¶](\#marimo.ui.tabs.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.tabs.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.tabs.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

##### Layouts - marimoLayouts¶

*Source: [https://docs.marimo.io/api/layouts/](https://docs.marimo.io/api/layouts/)*

<!-- Source: https://docs.marimo.io/api/layouts/ -->
<!-- Title: Layouts - marimoLayouts¶ -->
<!-- Depth: 4 -->

[Skip to content](#layouts)

# Layouts [¶](\#layouts "Permanent link")

marimo has higher-order layout functions that you can use to arrange outputs
in rows, columns, tables, tabs, and more.

## Stateless [¶](\#stateless "Permanent link")

Unlike elements in `marimo.ui`, these don't have any values associated with
them but just render their children in a certain way.

| Function | Description |
| --- | --- |
| [`marimo.accordion`](accordion/#marimo.accordion "            marimo.accordion") | Create collapsible sections |
| [`marimo.carousel`](carousel/#marimo.carousel "            marimo.carousel") | Create a slideshow |
| [`marimo.callout`](callout/#marimo.callout "            marimo.callout") | Create highlighted sections |
| [`marimo.center`](justify/#marimo.center "            marimo.center") | Center content |
| [`marimo.hstack`](stacks/#marimo.hstack "            marimo.hstack") | Stack elements horizontally |
| [`marimo.lazy`](lazy/#marimo.lazy "            marimo.lazy") | Lazy load content |
| [`marimo.left`](justify/#marimo.left "            marimo.left") | Left-align content |
| [`marimo.nav_menu`](../inputs/nav_menu/#marimo.nav_menu "            marimo.nav_menu") | Create navigation menus |
| [`marimo.outline`](outline/#marimo.outline "            marimo.outline") | Display table of contents outline |
| [`marimo.plain`](plain/#marimo.plain "            marimo.plain") | Display content without styling |
| [`marimo.right`](justify/#marimo.right "            marimo.right") | Right-align content |
| [`marimo.routes`](routes/#marimo.routes "            marimo.routes") | Create page routing |
| [`marimo.stat`](stat/#marimo.stat "            marimo.stat") | Display statistics |
| [`marimo.sidebar`](sidebar/#marimo.sidebar "            marimo.sidebar") | Create sidebars |
| [`marimo.tree`](tree/#marimo.tree "            marimo.tree") | Create tree structures |
| [`marimo.json`](json/#marimo.json "            marimo.json") | Create JSON structures |
| [`marimo.vstack`](stacks/#marimo.vstack "            marimo.vstack") | Stack elements vertically |

## Stateful [¶](\#stateful "Permanent link")

Some elements in `marimo.ui` are also helpful for layout. These elements
do have values associated with them: for example, `tabs` tracks the
selected tab name, and `table` tracks the selected rows.

| Function | Description |
| --- | --- |
| [`marimo.ui.tabs`](../inputs/tabs/#marimo.ui.tabs "            marimo.ui.tabs") | Create tabbed interfaces |
| [`marimo.ui.table`](../inputs/table/#marimo.ui.table "            marimo.ui.table") | Create interactive tables |

Back to top

# Layouts [¶](\#layouts "Permanent link")

marimo has higher-order layout functions that you can use to arrange outputs
in rows, columns, tables, tabs, and more.

## Stateless [¶](\#stateless "Permanent link")

Unlike elements in `marimo.ui`, these don't have any values associated with
them but just render their children in a certain way.

| Function | Description |
| --- | --- |
| [`marimo.accordion`](accordion/#marimo.accordion "            marimo.accordion") | Create collapsible sections |
| [`marimo.carousel`](carousel/#marimo.carousel "            marimo.carousel") | Create a slideshow |
| [`marimo.callout`](callout/#marimo.callout "            marimo.callout") | Create highlighted sections |
| [`marimo.center`](justify/#marimo.center "            marimo.center") | Center content |
| [`marimo.hstack`](stacks/#marimo.hstack "            marimo.hstack") | Stack elements horizontally |
| [`marimo.lazy`](lazy/#marimo.lazy "            marimo.lazy") | Lazy load content |
| [`marimo.left`](justify/#marimo.left "            marimo.left") | Left-align content |
| [`marimo.nav_menu`](../inputs/nav_menu/#marimo.nav_menu "            marimo.nav_menu") | Create navigation menus |
| [`marimo.outline`](outline/#marimo.outline "            marimo.outline") | Display table of contents outline |
| [`marimo.plain`](plain/#marimo.plain "            marimo.plain") | Display content without styling |
| [`marimo.right`](justify/#marimo.right "            marimo.right") | Right-align content |
| [`marimo.routes`](routes/#marimo.routes "            marimo.routes") | Create page routing |
| [`marimo.stat`](stat/#marimo.stat "            marimo.stat") | Display statistics |
| [`marimo.sidebar`](sidebar/#marimo.sidebar "            marimo.sidebar") | Create sidebars |
| [`marimo.tree`](tree/#marimo.tree "            marimo.tree") | Create tree structures |
| [`marimo.json`](json/#marimo.json "            marimo.json") | Create JSON structures |
| [`marimo.vstack`](stacks/#marimo.vstack "            marimo.vstack") | Stack elements vertically |

## Stateful [¶](\#stateful "Permanent link")

Some elements in `marimo.ui` are also helpful for layout. These elements
do have values associated with them: for example, `tabs` tracks the
selected tab name, and `table` tracks the selected rows.

| Function | Description |
| --- | --- |
| [`marimo.ui.tabs`](../inputs/tabs/#marimo.ui.tabs "            marimo.ui.tabs") | Create tabbed interfaces |
| [`marimo.ui.table`](../inputs/table/#marimo.ui.table "            marimo.ui.table") | Create interactive tables |

 Back to top

# Layouts [¶](\#layouts "Permanent link")

marimo has higher-order layout functions that you can use to arrange outputs
in rows, columns, tables, tabs, and more.

## Stateless [¶](\#stateless "Permanent link")

Unlike elements in `marimo.ui`, these don't have any values associated with
them but just render their children in a certain way.

| Function | Description |
| --- | --- |
| [`marimo.accordion`](accordion/#marimo.accordion "            marimo.accordion") | Create collapsible sections |
| [`marimo.carousel`](carousel/#marimo.carousel "            marimo.carousel") | Create a slideshow |
| [`marimo.callout`](callout/#marimo.callout "            marimo.callout") | Create highlighted sections |
| [`marimo.center`](justify/#marimo.center "            marimo.center") | Center content |
| [`marimo.hstack`](stacks/#marimo.hstack "            marimo.hstack") | Stack elements horizontally |
| [`marimo.lazy`](lazy/#marimo.lazy "            marimo.lazy") | Lazy load content |
| [`marimo.left`](justify/#marimo.left "            marimo.left") | Left-align content |
| [`marimo.nav_menu`](../inputs/nav_menu/#marimo.nav_menu "            marimo.nav_menu") | Create navigation menus |
| [`marimo.outline`](outline/#marimo.outline "            marimo.outline") | Display table of contents outline |
| [`marimo.plain`](plain/#marimo.plain "            marimo.plain") | Display content without styling |
| [`marimo.right`](justify/#marimo.right "            marimo.right") | Right-align content |
| [`marimo.routes`](routes/#marimo.routes "            marimo.routes") | Create page routing |
| [`marimo.stat`](stat/#marimo.stat "            marimo.stat") | Display statistics |
| [`marimo.sidebar`](sidebar/#marimo.sidebar "            marimo.sidebar") | Create sidebars |
| [`marimo.tree`](tree/#marimo.tree "            marimo.tree") | Create tree structures |
| [`marimo.json`](json/#marimo.json "            marimo.json") | Create JSON structures |
| [`marimo.vstack`](stacks/#marimo.vstack "            marimo.vstack") | Stack elements vertically |

## Stateful [¶](\#stateful "Permanent link")

Some elements in `marimo.ui` are also helpful for layout. These elements
do have values associated with them: for example, `tabs` tracks the
selected tab name, and `table` tracks the selected rows.

| Function | Description |
| --- | --- |
| [`marimo.ui.tabs`](../inputs/tabs/#marimo.ui.tabs "            marimo.ui.tabs") | Create tabbed interfaces |
| [`marimo.ui.table`](../inputs/table/#marimo.ui.table "            marimo.ui.table") | Create interactive tables |


---

##### Accordion - marimoAccordion¶

*Source: [https://docs.marimo.io/api/layouts/accordion/](https://docs.marimo.io/api/layouts/accordion/)*

<!-- Source: https://docs.marimo.io/api/layouts/accordion/ -->
<!-- Title: Accordion - marimoAccordion¶ -->
<!-- Depth: 4 -->

[Skip to content](#accordion)

# Accordion [¶](\#accordion "Permanent link")

Source code for `examples/outputs/accordion.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.accordion [¶](\#marimo.accordion "Permanent link")

Accordion of one or more items.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | a dictionary of item names to item content; strings are<br>interpreted as markdown<br>**TYPE:** `dict[str, object]` |
| `multiple` | whether to allow multiple items to be open simultaneously<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `lazy` | a boolean, whether to lazily load the accordion content.<br>This is a convenience that wraps each accordion in a `mo.lazy`<br>component.<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

Accordion content can be lazily loaded:

where `expensive_item` is the item to render, or a callable that
returns the item to render.

Back to top

# Accordion [¶](\#accordion "Permanent link")

Source code for `examples/outputs/accordion.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.accordion [¶](\#marimo.accordion "Permanent link")

Accordion of one or more items.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | a dictionary of item names to item content; strings are<br>interpreted as markdown<br>**TYPE:** `dict[str, object]` |
| `multiple` | whether to allow multiple items to be open simultaneously<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `lazy` | a boolean, whether to lazily load the accordion content.<br>This is a convenience that wraps each accordion in a `mo.lazy`<br>component.<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

Accordion content can be lazily loaded:

where `expensive_item` is the item to render, or a callable that
returns the item to render.

 Back to top

# Accordion [¶](\#accordion "Permanent link")

Source code for `examples/outputs/accordion.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.accordion [¶](\#marimo.accordion "Permanent link")

Accordion of one or more items.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | a dictionary of item names to item content; strings are<br>interpreted as markdown<br>**TYPE:** `dict[str, object]` |
| `multiple` | whether to allow multiple items to be open simultaneously<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `lazy` | a boolean, whether to lazily load the accordion content.<br>This is a convenience that wraps each accordion in a `mo.lazy`<br>component.<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

Accordion content can be lazily loaded:

where `expensive_item` is the item to render, or a callable that
returns the item to render.


---

##### Sidebar - marimoSidebar¶

*Source: [https://docs.marimo.io/api/layouts/sidebar/](https://docs.marimo.io/api/layouts/sidebar/)*

<!-- Source: https://docs.marimo.io/api/layouts/sidebar/ -->
<!-- Title: Sidebar - marimoSidebar¶ -->
<!-- Depth: 4 -->

[Skip to content](#sidebar)

# Sidebar [¶](\#sidebar "Permanent link")

## `` marimo.sidebar [¶](\#marimo.sidebar "Permanent link")

Bases: `Html`

Displays content in a sidebar.

This is a special layout component that will display the content in a sidebar
layout, rather than below/above the cell.

This component still needs to be the last expression in the cell,
in order to display correctly.

You may use more than one `mo.sidebar` \- they will be displayed in the
order they are called.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | The content to display in the sidebar.<br>**TYPE:** `object` |
| `footer` | The content to display at the bottom of the sidebar.<br>**TYPE:** `object`**DEFAULT:** `None` |
| `width` | The width of the sidebar when open. Can be any valid CSS width<br>value (e.g. "300px", "20rem"). If not provided, defaults to the standard width.<br>**TYPE:** `str`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

### `` text`property`[¶](\#marimo.sidebar.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.sidebar.batch "Permanent link")

### `` callout [¶](\#marimo.sidebar.callout "Permanent link")

### `` center [¶](\#marimo.sidebar.center "Permanent link")

### `` left [¶](\#marimo.sidebar.left "Permanent link")

### `` right [¶](\#marimo.sidebar.right "Permanent link")

### `` style [¶](\#marimo.sidebar.style "Permanent link")

Back to top

# Sidebar [¶](\#sidebar "Permanent link")

## `` marimo.sidebar [¶](\#marimo.sidebar "Permanent link")

Bases: `Html`

Displays content in a sidebar.

This is a special layout component that will display the content in a sidebar
layout, rather than below/above the cell.

This component still needs to be the last expression in the cell,
in order to display correctly.

You may use more than one `mo.sidebar` \- they will be displayed in the
order they are called.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | The content to display in the sidebar.<br>**TYPE:** `object` |
| `footer` | The content to display at the bottom of the sidebar.<br>**TYPE:** `object`**DEFAULT:** `None` |
| `width` | The width of the sidebar when open. Can be any valid CSS width<br>value (e.g. "300px", "20rem"). If not provided, defaults to the standard width.<br>**TYPE:** `str`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

### `` text`property`[¶](\#marimo.sidebar.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.sidebar.batch "Permanent link")

### `` callout [¶](\#marimo.sidebar.callout "Permanent link")

### `` center [¶](\#marimo.sidebar.center "Permanent link")

### `` left [¶](\#marimo.sidebar.left "Permanent link")

### `` right [¶](\#marimo.sidebar.right "Permanent link")

### `` style [¶](\#marimo.sidebar.style "Permanent link")

 Back to top

# Sidebar [¶](\#sidebar "Permanent link")

## `` marimo.sidebar [¶](\#marimo.sidebar "Permanent link")

Bases: `Html`

Displays content in a sidebar.

This is a special layout component that will display the content in a sidebar
layout, rather than below/above the cell.

This component still needs to be the last expression in the cell,
in order to display correctly.

You may use more than one `mo.sidebar` \- they will be displayed in the
order they are called.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | The content to display in the sidebar.<br>**TYPE:** `object` |
| `footer` | The content to display at the bottom of the sidebar.<br>**TYPE:** `object`**DEFAULT:** `None` |
| `width` | The width of the sidebar when open. Can be any valid CSS width<br>value (e.g. "300px", "20rem"). If not provided, defaults to the standard width.<br>**TYPE:** `str`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

### `` text`property`[¶](\#marimo.sidebar.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.sidebar.batch "Permanent link")

### `` callout [¶](\#marimo.sidebar.callout "Permanent link")

### `` center [¶](\#marimo.sidebar.center "Permanent link")

### `` left [¶](\#marimo.sidebar.left "Permanent link")

### `` right [¶](\#marimo.sidebar.right "Permanent link")

### `` style [¶](\#marimo.sidebar.style "Permanent link")


---

##### Stacks - marimoStacks¶

*Source: [https://docs.marimo.io/api/layouts/stacks/](https://docs.marimo.io/api/layouts/stacks/)*

<!-- Source: https://docs.marimo.io/api/layouts/stacks/ -->
<!-- Title: Stacks - marimoStacks¶ -->
<!-- Depth: 4 -->

[Skip to content](#stacks)

# Stacks [¶](\#stacks "Permanent link")

## `` marimo.hstack [¶](\#marimo.hstack "Permanent link")

Stack items horizontally, in a row.

Combine with `vstack` to build a grid.

Examples:

Build a row of items:

Build a row of items with equal width:

Have one item stretch to fill the available space,
while another fits its content:

Build a grid:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |
| `justify` | Justify items horizontally: start, center, end, space-between, or space-around.<br>Defaults to "space-between".<br>**TYPE:** `Literal['start', 'center', 'end', 'space-between', 'space-around']`**DEFAULT:** `'space-between'` |
| `align` | Align items<br>vertically: start, end, center, or stretch.<br>**TYPE:** `Optional[Literal['start', 'end', 'center', 'stretch']]`**DEFAULT:** `None` |
| `wrap` | Wrap items or not. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `gap` | Gap between items as a float in rem. 1rem is 16px by default.<br>Defaults to 0.5.<br>**TYPE:** `float`**DEFAULT:** `0.5` |
| `widths` | "equal" to give items<br>equal width; or a list of relative widths with same length as `items`,<br>eg, \[1, 2\] means the second item is twice as wide as the first; or None<br>for a sensible default.<br>**TYPE:** `Optional[Literal['equal'] | Sequence[float]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

## `` marimo.vstack [¶](\#marimo.vstack "Permanent link")

Stack items vertically, in a column.

Combine with `hstack` to build a grid of items.

Examples:

Build a column of items:

Build a grid:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |
| `align` | Align items<br>horizontally: start, end, center, or stretch.<br>**TYPE:** `Optional[Literal['start', 'end', 'center', 'stretch']]`**DEFAULT:** `None` |
| `justify` | Justify items vertically: start, center, end, space-between, or space-around.<br>Defaults to "start".<br>**TYPE:** `Literal['start', 'center', 'end', 'space-between', 'space-around']`**DEFAULT:** `'start'` |
| `gap` | Gap between items as a float in rem. 1rem is 16px by default.<br>Defaults to 0.5.<br>**TYPE:** `float`**DEFAULT:** `0.5` |
| `heights` | "equal" to give items<br>equal height; or a list of relative heights with same length as `items`,<br>eg, \[1, 2\] means the second item is twice as tall as the first; or None<br>for a sensible default.<br>**TYPE:** `Optional[Literal['equal'] | Sequence[float]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

Back to top

# Stacks [¶](\#stacks "Permanent link")

## `` marimo.hstack [¶](\#marimo.hstack "Permanent link")

Stack items horizontally, in a row.

Combine with `vstack` to build a grid.

Examples:

Build a row of items:

Build a row of items with equal width:

Have one item stretch to fill the available space,
while another fits its content:

Build a grid:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |
| `justify` | Justify items horizontally: start, center, end, space-between, or space-around.<br>Defaults to "space-between".<br>**TYPE:** `Literal['start', 'center', 'end', 'space-between', 'space-around']`**DEFAULT:** `'space-between'` |
| `align` | Align items<br>vertically: start, end, center, or stretch.<br>**TYPE:** `Optional[Literal['start', 'end', 'center', 'stretch']]`**DEFAULT:** `None` |
| `wrap` | Wrap items or not. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `gap` | Gap between items as a float in rem. 1rem is 16px by default.<br>Defaults to 0.5.<br>**TYPE:** `float`**DEFAULT:** `0.5` |
| `widths` | "equal" to give items<br>equal width; or a list of relative widths with same length as `items`,<br>eg, \[1, 2\] means the second item is twice as wide as the first; or None<br>for a sensible default.<br>**TYPE:** `Optional[Literal['equal'] | Sequence[float]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

## `` marimo.vstack [¶](\#marimo.vstack "Permanent link")

Stack items vertically, in a column.

Combine with `hstack` to build a grid of items.

Examples:

Build a column of items:

Build a grid:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |
| `align` | Align items<br>horizontally: start, end, center, or stretch.<br>**TYPE:** `Optional[Literal['start', 'end', 'center', 'stretch']]`**DEFAULT:** `None` |
| `justify` | Justify items vertically: start, center, end, space-between, or space-around.<br>Defaults to "start".<br>**TYPE:** `Literal['start', 'center', 'end', 'space-between', 'space-around']`**DEFAULT:** `'start'` |
| `gap` | Gap between items as a float in rem. 1rem is 16px by default.<br>Defaults to 0.5.<br>**TYPE:** `float`**DEFAULT:** `0.5` |
| `heights` | "equal" to give items<br>equal height; or a list of relative heights with same length as `items`,<br>eg, \[1, 2\] means the second item is twice as tall as the first; or None<br>for a sensible default.<br>**TYPE:** `Optional[Literal['equal'] | Sequence[float]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

 Back to top

# Stacks [¶](\#stacks "Permanent link")

## `` marimo.hstack [¶](\#marimo.hstack "Permanent link")

Stack items horizontally, in a row.

Combine with `vstack` to build a grid.

Examples:

Build a row of items:

Build a row of items with equal width:

Have one item stretch to fill the available space,
while another fits its content:

Build a grid:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |
| `justify` | Justify items horizontally: start, center, end, space-between, or space-around.<br>Defaults to "space-between".<br>**TYPE:** `Literal['start', 'center', 'end', 'space-between', 'space-around']`**DEFAULT:** `'space-between'` |
| `align` | Align items<br>vertically: start, end, center, or stretch.<br>**TYPE:** `Optional[Literal['start', 'end', 'center', 'stretch']]`**DEFAULT:** `None` |
| `wrap` | Wrap items or not. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `gap` | Gap between items as a float in rem. 1rem is 16px by default.<br>Defaults to 0.5.<br>**TYPE:** `float`**DEFAULT:** `0.5` |
| `widths` | "equal" to give items<br>equal width; or a list of relative widths with same length as `items`,<br>eg, \[1, 2\] means the second item is twice as wide as the first; or None<br>for a sensible default.<br>**TYPE:** `Optional[Literal['equal'] | Sequence[float]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

## `` marimo.vstack [¶](\#marimo.vstack "Permanent link")

Stack items vertically, in a column.

Combine with `hstack` to build a grid of items.

Examples:

Build a column of items:

Build a grid:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |
| `align` | Align items<br>horizontally: start, end, center, or stretch.<br>**TYPE:** `Optional[Literal['start', 'end', 'center', 'stretch']]`**DEFAULT:** `None` |
| `justify` | Justify items vertically: start, center, end, space-between, or space-around.<br>Defaults to "start".<br>**TYPE:** `Literal['start', 'center', 'end', 'space-between', 'space-around']`**DEFAULT:** `'start'` |
| `gap` | Gap between items as a float in rem. 1rem is 16px by default.<br>Defaults to 0.5.<br>**TYPE:** `float`**DEFAULT:** `0.5` |
| `heights` | "equal" to give items<br>equal height; or a list of relative heights with same length as `items`,<br>eg, \[1, 2\] means the second item is twice as tall as the first; or None<br>for a sensible default.<br>**TYPE:** `Optional[Literal['equal'] | Sequence[float]]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |


---

##### Miscellaneous - marimoMiscellaneous¶

*Source: [https://docs.marimo.io/api/miscellaneous/](https://docs.marimo.io/api/miscellaneous/)*

<!-- Source: https://docs.marimo.io/api/miscellaneous/ -->
<!-- Title: Miscellaneous - marimoMiscellaneous¶ -->
<!-- Depth: 4 -->

[Skip to content](#miscellaneous)

# Miscellaneous [¶](\#miscellaneous "Permanent link")

## `` marimo.running\_in\_notebook [¶](\#marimo.running_in_notebook "Permanent link")

Returns True if running in a marimo notebook, False otherwise

## `` marimo.defs [¶](\#marimo.defs "Permanent link")

Get the definitions of the currently executing cell.

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[str, ...]` | tuple\[str, ...\]: A tuple of the currently executing cell's defs. |

## `` marimo.refs [¶](\#marimo.refs "Permanent link")

Get the references of the currently executing cell.

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[str, ...]` | tuple\[str, ...\]: A tuple of the currently executing cell's refs. |

## `` marimo.notebook\_dir [¶](\#marimo.notebook_dir "Permanent link")

Get the directory of the currently executing notebook.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Path | None` | pathlib.Path \| None: A pathlib.Path object representing the directory of the current<br>notebook, or None if the notebook's directory cannot be determined. |

Examples:

## `` marimo.notebook\_location [¶](\#marimo.notebook_location "Permanent link")

Get the location of the currently executing notebook.

In WASM, this is the URL of webpage, for example, `https://my-site.com`.
For nested paths, this is the URL including the origin and pathname.
`https://<my-org>.github.io/<my-repo>/folder`.

In non-WASM, this is the directory of the notebook, which is the same as
`mo.notebook_dir()`.

Examples:

In order to access data both locally and when a notebook runs via
WebAssembly (e.g. hosted on GitHub Pages), you can use this
approach to fetch data from the notebook's location.

| RETURNS | DESCRIPTION |
| --- | --- |
| `PurePath | None` | Path \| None: A Path object representing the URL or directory of the current<br>notebook, or None if the notebook's directory cannot be determined. |

## Inspect [¶](\#inspect "Permanent link")

Use `mo.inspect()` to explore Python objects with a rich, interactive display of their attributes, methods, and documentation.

### Example [¶](\#example "Permanent link")

## `` marimo.inspect [¶](\#marimo.inspect "Permanent link")

Bases: `Html`

Inspect a Python object.

Displays objects with their attributes, methods, and documentation
in a rich HTML format. Useful for exploring objects that lack a rich repr.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `obj` | The object to inspect.<br>**TYPE:** `object` |
| `help` | Show full help text (otherwise just first paragraph).<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `methods` | Show methods.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `docs` | Show documentation for attributes/methods.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `private` | Show private attributes (starting with '\_').<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `dunder` | Show dunder attributes (starting with '\_\_').<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `sort` | Sort attributes alphabetically.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `all` | Show all attributes (methods, private, and dunder).<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `value` | Show the object's value/repr.<br>**TYPE:** `bool`**DEFAULT:** `True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

### `` text`property`[¶](\#marimo.inspect.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.inspect.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.inspect.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.inspect.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` left [¶](\#marimo.inspect.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.inspect.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.inspect.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Miscellaneous [¶](\#miscellaneous "Permanent link")

## `` marimo.running\_in\_notebook [¶](\#marimo.running_in_notebook "Permanent link")

Returns True if running in a marimo notebook, False otherwise

## `` marimo.defs [¶](\#marimo.defs "Permanent link")

Get the definitions of the currently executing cell.

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[str, ...]` | tuple\[str, ...\]: A tuple of the currently executing cell's defs. |

## `` marimo.refs [¶](\#marimo.refs "Permanent link")

Get the references of the currently executing cell.

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[str, ...]` | tuple\[str, ...\]: A tuple of the currently executing cell's refs. |

## `` marimo.notebook\_dir [¶](\#marimo.notebook_dir "Permanent link")

Get the directory of the currently executing notebook.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Path | None` | pathlib.Path \| None: A pathlib.Path object representing the directory of the current<br>notebook, or None if the notebook's directory cannot be determined. |

Examples:

## `` marimo.notebook\_location [¶](\#marimo.notebook_location "Permanent link")

Get the location of the currently executing notebook.

In WASM, this is the URL of webpage, for example, `https://my-site.com`.
For nested paths, this is the URL including the origin and pathname.
`https://<my-org>.github.io/<my-repo>/folder`.

In non-WASM, this is the directory of the notebook, which is the same as
`mo.notebook_dir()`.

Examples:

In order to access data both locally and when a notebook runs via
WebAssembly (e.g. hosted on GitHub Pages), you can use this
approach to fetch data from the notebook's location.

| RETURNS | DESCRIPTION |
| --- | --- |
| `PurePath | None` | Path \| None: A Path object representing the URL or directory of the current<br>notebook, or None if the notebook's directory cannot be determined. |

## Inspect [¶](\#inspect "Permanent link")

Use `mo.inspect()` to explore Python objects with a rich, interactive display of their attributes, methods, and documentation.

### Example [¶](\#example "Permanent link")

## `` marimo.inspect [¶](\#marimo.inspect "Permanent link")

Bases: `Html`

Inspect a Python object.

Displays objects with their attributes, methods, and documentation
in a rich HTML format. Useful for exploring objects that lack a rich repr.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `obj` | The object to inspect.<br>**TYPE:** `object` |
| `help` | Show full help text (otherwise just first paragraph).<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `methods` | Show methods.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `docs` | Show documentation for attributes/methods.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `private` | Show private attributes (starting with '\_').<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `dunder` | Show dunder attributes (starting with '\_\_').<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `sort` | Sort attributes alphabetically.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `all` | Show all attributes (methods, private, and dunder).<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `value` | Show the object's value/repr.<br>**TYPE:** `bool`**DEFAULT:** `True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

### `` text`property`[¶](\#marimo.inspect.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.inspect.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.inspect.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.inspect.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` left [¶](\#marimo.inspect.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.inspect.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.inspect.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Miscellaneous [¶](\#miscellaneous "Permanent link")

## `` marimo.running\_in\_notebook [¶](\#marimo.running_in_notebook "Permanent link")

Returns True if running in a marimo notebook, False otherwise

## `` marimo.defs [¶](\#marimo.defs "Permanent link")

Get the definitions of the currently executing cell.

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[str, ...]` | tuple\[str, ...\]: A tuple of the currently executing cell's defs. |

## `` marimo.refs [¶](\#marimo.refs "Permanent link")

Get the references of the currently executing cell.

| RETURNS | DESCRIPTION |
| --- | --- |
| `tuple[str, ...]` | tuple\[str, ...\]: A tuple of the currently executing cell's refs. |

## `` marimo.notebook\_dir [¶](\#marimo.notebook_dir "Permanent link")

Get the directory of the currently executing notebook.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Path | None` | pathlib.Path \| None: A pathlib.Path object representing the directory of the current<br>notebook, or None if the notebook's directory cannot be determined. |

Examples:

## `` marimo.notebook\_location [¶](\#marimo.notebook_location "Permanent link")

Get the location of the currently executing notebook.

In WASM, this is the URL of webpage, for example, `https://my-site.com`.
For nested paths, this is the URL including the origin and pathname.
`https://<my-org>.github.io/<my-repo>/folder`.

In non-WASM, this is the directory of the notebook, which is the same as
`mo.notebook_dir()`.

Examples:

In order to access data both locally and when a notebook runs via
WebAssembly (e.g. hosted on GitHub Pages), you can use this
approach to fetch data from the notebook's location.

| RETURNS | DESCRIPTION |
| --- | --- |
| `PurePath | None` | Path \| None: A Path object representing the URL or directory of the current<br>notebook, or None if the notebook's directory cannot be determined. |

## Inspect [¶](\#inspect "Permanent link")

Use `mo.inspect()` to explore Python objects with a rich, interactive display of their attributes, methods, and documentation.

### Example [¶](\#example "Permanent link")

## `` marimo.inspect [¶](\#marimo.inspect "Permanent link")

Bases: `Html`

Inspect a Python object.

Displays objects with their attributes, methods, and documentation
in a rich HTML format. Useful for exploring objects that lack a rich repr.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `obj` | The object to inspect.<br>**TYPE:** `object` |
| `help` | Show full help text (otherwise just first paragraph).<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `methods` | Show methods.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `docs` | Show documentation for attributes/methods.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `private` | Show private attributes (starting with '\_').<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `dunder` | Show dunder attributes (starting with '\_\_').<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `sort` | Sort attributes alphabetically.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `all` | Show all attributes (methods, private, and dunder).<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `value` | Show the object's value/repr.<br>**TYPE:** `bool`**DEFAULT:** `True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

### `` text`property`[¶](\#marimo.inspect.text "Permanent link")

A string of HTML representing this element.

### `` batch [¶](\#marimo.inspect.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.inspect.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.inspect.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` left [¶](\#marimo.inspect.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.inspect.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.inspect.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

##### Outputs - marimoOutputs¶

*Source: [https://docs.marimo.io/api/outputs/](https://docs.marimo.io/api/outputs/)*

<!-- Source: https://docs.marimo.io/api/outputs/ -->
<!-- Title: Outputs - marimoOutputs¶ -->
<!-- Depth: 4 -->

[Skip to content](#outputs)

# Outputs [¶](\#outputs "Permanent link")

## Cell outputs [¶](\#cell-outputs "Permanent link")

Cell outputs

Every cell in a marimo notebook can have a visual **output**. When editing,
outputs are displayed above cells. When running a notebook as an app,
its UI is an arrangement of outputs.

A cell's output is by default its last expression. You can also create outputs
programmatically, using `mo.output.replace()` and `mo.output.append()`.

## `` marimo.output.replace [¶](\#marimo.output.replace "Permanent link")

Replace a cell's output with a new one.

Call `mo.output.replace()` to write to a cell's output area, replacing
the existing output, if any.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | object to output<br>**TYPE:** `object` |

**Watch `mo.output.replace` in action**

See a demo of how `mo.output.replace` works in this [short YouTube video](https://youtube.com/shorts/tCMeQb-PqNU?si=7PeFzQJzNvXsLoXN).

## `` marimo.output.append [¶](\#marimo.output.append "Permanent link")

Append a new object to a cell's output.

Call this function to incrementally build a cell's output. Appended
outputs are stacked vertically.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | object to output<br>**TYPE:** `object` |

## `` marimo.output.clear [¶](\#marimo.output.clear "Permanent link")

Clear a cell's output.

## `` marimo.output.replace\_at\_index [¶](\#marimo.output.replace_at_index "Permanent link")

Replace a cell's output at the given index with value.

Call this function to replace an existing object in a cell's output. If idx
is equal to the length of the output, this is equivalent to an append.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | new object to replace an existing object<br>**TYPE:** `object` |
| `idx` | index of output to replace<br>**TYPE:** `int` |

Last expression replaces existing output

Ending a cell with a non- `None` expression is the same as calling
`mo.output.replace()` on it: the last expression replaces any output you may have
already written. Wrap the last expression in `mo.output.append` if you want
to add to an existing output instead of replacing it.

### Display cell code in marimo's app views [¶](\#display-cell-code-in-marimos-app-views "Permanent link")

Use `mo.show_code()` to display the cell's code in the output area, which
will then be visible in all app views.

## `` marimo.show\_code [¶](\#marimo.show_code "Permanent link")

Display an output along with the code of the current cell.

Use `mo.show_code` to show the code of the current cell along with
the cell's output. This is useful if you want a cell's code to
appear in the app preview or when running the notebook as an app
with `marimo run`.

In the displayed code, all occurrences of mo.show\_code(...) will be
replaced with ...

Show code that produces the output `factorial(5)`:

Show code of a cell, without an output:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `output` | the output to display with the cell's code; omit the output<br>to just show the cell's code.<br>**TYPE:** `object`**DEFAULT:** `None` |
| `position` | Where to display the code relative to the output.<br>Use "above" to show code above the output, or "below" (default) to show<br>code below the output.<br>**TYPE:** `Literal['above', 'below']`**DEFAULT:** `'below'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML of the `output` arg displayed with its code. |

## Console outputs [¶](\#console-outputs "Permanent link")

Console outputs

Text written to `stdout`/ `stderr`, including print statements
and logs, shows up in a console output area below a cell.

By default, these console outputs don't appear when running a marimo notebook
as an app. If you do want them to appear in apps, marimo provides utility
functions for capturing console outputs and redirecting them to cell outputs.

## `` marimo.redirect\_stdout [¶](\#marimo.redirect_stdout "Permanent link")

Redirect stdout to a cell's output area.

Examples:

## `` marimo.redirect\_stderr [¶](\#marimo.redirect_stderr "Permanent link")

Redirect `stderr` to a cell's output area.

Examples:

## `` marimo.capture\_stdout [¶](\#marimo.capture_stdout "Permanent link")

Capture standard output.

Use this context manager to capture print statements and
other output sent to standard output.

Examples:

## `` marimo.capture\_stderr [¶](\#marimo.capture_stderr "Permanent link")

Capture standard error.

Use this context manager to capture output sent to standard error.

Examples:

## Object inspection [¶](\#object-inspection "Permanent link")

Use [`mo.inspect`](../miscellaneous/#marimo.inspect "            marimo.inspect") to create a rich display of Python objects
for exploring their attributes, methods, and documentation.

Back to top

# Outputs [¶](\#outputs "Permanent link")

## Cell outputs [¶](\#cell-outputs "Permanent link")

Cell outputs

Every cell in a marimo notebook can have a visual **output**. When editing,
outputs are displayed above cells. When running a notebook as an app,
its UI is an arrangement of outputs.

A cell's output is by default its last expression. You can also create outputs
programmatically, using `mo.output.replace()` and `mo.output.append()`.

## `` marimo.output.replace [¶](\#marimo.output.replace "Permanent link")

Replace a cell's output with a new one.

Call `mo.output.replace()` to write to a cell's output area, replacing
the existing output, if any.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | object to output<br>**TYPE:** `object` |

**Watch `mo.output.replace` in action**

See a demo of how `mo.output.replace` works in this [short YouTube video](https://youtube.com/shorts/tCMeQb-PqNU?si=7PeFzQJzNvXsLoXN).

## `` marimo.output.append [¶](\#marimo.output.append "Permanent link")

Append a new object to a cell's output.

Call this function to incrementally build a cell's output. Appended
outputs are stacked vertically.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | object to output<br>**TYPE:** `object` |

## `` marimo.output.clear [¶](\#marimo.output.clear "Permanent link")

Clear a cell's output.

## `` marimo.output.replace\_at\_index [¶](\#marimo.output.replace_at_index "Permanent link")

Replace a cell's output at the given index with value.

Call this function to replace an existing object in a cell's output. If idx
is equal to the length of the output, this is equivalent to an append.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | new object to replace an existing object<br>**TYPE:** `object` |
| `idx` | index of output to replace<br>**TYPE:** `int` |

Last expression replaces existing output

Ending a cell with a non- `None` expression is the same as calling
`mo.output.replace()` on it: the last expression replaces any output you may have
already written. Wrap the last expression in `mo.output.append` if you want
to add to an existing output instead of replacing it.

### Display cell code in marimo's app views [¶](\#display-cell-code-in-marimos-app-views "Permanent link")

Use `mo.show_code()` to display the cell's code in the output area, which
will then be visible in all app views.

## `` marimo.show\_code [¶](\#marimo.show_code "Permanent link")

Display an output along with the code of the current cell.

Use `mo.show_code` to show the code of the current cell along with
the cell's output. This is useful if you want a cell's code to
appear in the app preview or when running the notebook as an app
with `marimo run`.

In the displayed code, all occurrences of mo.show\_code(...) will be
replaced with ...

Show code that produces the output `factorial(5)`:

Show code of a cell, without an output:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `output` | the output to display with the cell's code; omit the output<br>to just show the cell's code.<br>**TYPE:** `object`**DEFAULT:** `None` |
| `position` | Where to display the code relative to the output.<br>Use "above" to show code above the output, or "below" (default) to show<br>code below the output.<br>**TYPE:** `Literal['above', 'below']`**DEFAULT:** `'below'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML of the `output` arg displayed with its code. |

## Console outputs [¶](\#console-outputs "Permanent link")

Console outputs

Text written to `stdout`/ `stderr`, including print statements
and logs, shows up in a console output area below a cell.

By default, these console outputs don't appear when running a marimo notebook
as an app. If you do want them to appear in apps, marimo provides utility
functions for capturing console outputs and redirecting them to cell outputs.

## `` marimo.redirect\_stdout [¶](\#marimo.redirect_stdout "Permanent link")

Redirect stdout to a cell's output area.

Examples:

## `` marimo.redirect\_stderr [¶](\#marimo.redirect_stderr "Permanent link")

Redirect `stderr` to a cell's output area.

Examples:

## `` marimo.capture\_stdout [¶](\#marimo.capture_stdout "Permanent link")

Capture standard output.

Use this context manager to capture print statements and
other output sent to standard output.

Examples:

## `` marimo.capture\_stderr [¶](\#marimo.capture_stderr "Permanent link")

Capture standard error.

Use this context manager to capture output sent to standard error.

Examples:

## Object inspection [¶](\#object-inspection "Permanent link")

Use [`mo.inspect`](../miscellaneous/#marimo.inspect "            marimo.inspect") to create a rich display of Python objects
for exploring their attributes, methods, and documentation.

 Back to top

# Outputs [¶](\#outputs "Permanent link")

## Cell outputs [¶](\#cell-outputs "Permanent link")

Cell outputs

Every cell in a marimo notebook can have a visual **output**. When editing,
outputs are displayed above cells. When running a notebook as an app,
its UI is an arrangement of outputs.

A cell's output is by default its last expression. You can also create outputs
programmatically, using `mo.output.replace()` and `mo.output.append()`.

## `` marimo.output.replace [¶](\#marimo.output.replace "Permanent link")

Replace a cell's output with a new one.

Call `mo.output.replace()` to write to a cell's output area, replacing
the existing output, if any.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | object to output<br>**TYPE:** `object` |

**Watch `mo.output.replace` in action**

See a demo of how `mo.output.replace` works in this [short YouTube video](https://youtube.com/shorts/tCMeQb-PqNU?si=7PeFzQJzNvXsLoXN).

## `` marimo.output.append [¶](\#marimo.output.append "Permanent link")

Append a new object to a cell's output.

Call this function to incrementally build a cell's output. Appended
outputs are stacked vertically.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | object to output<br>**TYPE:** `object` |

## `` marimo.output.clear [¶](\#marimo.output.clear "Permanent link")

Clear a cell's output.

## `` marimo.output.replace\_at\_index [¶](\#marimo.output.replace_at_index "Permanent link")

Replace a cell's output at the given index with value.

Call this function to replace an existing object in a cell's output. If idx
is equal to the length of the output, this is equivalent to an append.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | new object to replace an existing object<br>**TYPE:** `object` |
| `idx` | index of output to replace<br>**TYPE:** `int` |

Last expression replaces existing output

Ending a cell with a non- `None` expression is the same as calling
`mo.output.replace()` on it: the last expression replaces any output you may have
already written. Wrap the last expression in `mo.output.append` if you want
to add to an existing output instead of replacing it.

### Display cell code in marimo's app views [¶](\#display-cell-code-in-marimos-app-views "Permanent link")

Use `mo.show_code()` to display the cell's code in the output area, which
will then be visible in all app views.

## `` marimo.show\_code [¶](\#marimo.show_code "Permanent link")

Display an output along with the code of the current cell.

Use `mo.show_code` to show the code of the current cell along with
the cell's output. This is useful if you want a cell's code to
appear in the app preview or when running the notebook as an app
with `marimo run`.

In the displayed code, all occurrences of mo.show\_code(...) will be
replaced with ...

Show code that produces the output `factorial(5)`:

Show code of a cell, without an output:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `output` | the output to display with the cell's code; omit the output<br>to just show the cell's code.<br>**TYPE:** `object`**DEFAULT:** `None` |
| `position` | Where to display the code relative to the output.<br>Use "above" to show code above the output, or "below" (default) to show<br>code below the output.<br>**TYPE:** `Literal['above', 'below']`**DEFAULT:** `'below'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML of the `output` arg displayed with its code. |

## Console outputs [¶](\#console-outputs "Permanent link")

Console outputs

Text written to `stdout`/ `stderr`, including print statements
and logs, shows up in a console output area below a cell.

By default, these console outputs don't appear when running a marimo notebook
as an app. If you do want them to appear in apps, marimo provides utility
functions for capturing console outputs and redirecting them to cell outputs.

## `` marimo.redirect\_stdout [¶](\#marimo.redirect_stdout "Permanent link")

Redirect stdout to a cell's output area.

Examples:

## `` marimo.redirect\_stderr [¶](\#marimo.redirect_stderr "Permanent link")

Redirect `stderr` to a cell's output area.

Examples:

## `` marimo.capture\_stdout [¶](\#marimo.capture_stdout "Permanent link")

Capture standard output.

Use this context manager to capture print statements and
other output sent to standard output.

Examples:

## `` marimo.capture\_stderr [¶](\#marimo.capture_stderr "Permanent link")

Capture standard error.

Use this context manager to capture output sent to standard error.

Examples:

## Object inspection [¶](\#object-inspection "Permanent link")

Use [`mo.inspect`](../miscellaneous/#marimo.inspect "            marimo.inspect") to create a rich display of Python objects
for exploring their attributes, methods, and documentation.


---

##### Lint Rules - marimoLint Rules¶

*Source: [https://docs.marimo.io/guides/lint_rules/](https://docs.marimo.io/guides/lint_rules/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/ -->
<!-- Title: Lint Rules - marimoLint Rules¶ -->
<!-- Depth: 4 -->

[Skip to content](#lint-rules)

# Lint Rules [¶](\#lint-rules "Permanent link")

marimo includes a comprehensive linting system that helps you write better, more reliable notebooks. The linter checks for various issues that could prevent your notebook from running correctly or cause confusion.

## How to Use [¶](\#how-to-use "Permanent link")

You can run the linter using the CLI:

## Rule Categories [¶](\#rule-categories "Permanent link")

marimo's lint rules are organized into three main categories based on their severity:

### 🚨 Breaking Rules [¶](\#breaking-rules "Permanent link")

These errors prevent notebook execution.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MB001](rules/unparsable_cells/) | unparsable-cells | Cell contains unparsable code | ❌ |
| [MB002](rules/multiple_definitions/) | multiple-definitions | Multiple cells define the same variable | ❌ |
| [MB003](rules/cycle_dependencies/) | cycle-dependencies | Cells have circular dependencies | ❌ |
| [MB004](rules/setup_cell_dependencies/) | setup-cell-dependencies | Setup cell cannot have dependencies | ❌ |
| [MB005](rules/invalid_syntax/) | invalid-syntax | Cell contains code that throws a SyntaxError on compilation | ❌ |

### ⚠️ Runtime Rules [¶](\#runtime-rules "Permanent link")

These issues may cause runtime problems.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MR001](rules/self_import/) | self-import | Importing a module with the same name as the file | ❌ |

### ✨ Formatting Rules [¶](\#formatting-rules "Permanent link")

These are style and formatting issues.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MF001](rules/general_formatting/) | general-formatting | General formatting issues with the notebook format. | 🛠️ |
| [MF002](rules/parse_stdout/) | parse-stdout | Parse captured stdout during notebook loading | ❌ |
| [MF003](rules/parse_stderr/) | parse-stderr | Parse captured stderr during notebook loading | ❌ |
| [MF004](rules/empty_cells/) | empty-cells | Empty cells that can be safely removed. | ⚠️ |
| [MF005](rules/sql_parse_error/) | sql-parse-error | SQL parsing errors during dependency analysis | ❌ |
| [MF006](rules/misc_log_capture/) | misc-log-capture | Miscellaneous log messages during processing | ❌ |
| [MF007](rules/markdown_indentation/) | markdown-indentation | Markdown cells in `mo.md()` should be dedented. | 🛠️ |

## Legend [¶](\#legend "Permanent link")

- 🛠️ = Automatically fixable with `marimo check --fix`
- ⚠️ = Fixable with `marimo check --fix --unsafe-fixes` (may change code behavior)
- ❌ = Not automatically fixable

## Configuration [¶](\#configuration "Permanent link")

Most lint rules are enabled by default. You can configure the linter behavior through marimo's configuration system.

## Related Documentation [¶](\#related-documentation "Permanent link")

- [Understanding Errors](../understanding_errors/) \- Detailed explanations of common marimo errors
- [CLI Reference](../../cli/) \- Complete CLI documentation including `marimo check`

Back to top

# Lint Rules [¶](\#lint-rules "Permanent link")

marimo includes a comprehensive linting system that helps you write better, more reliable notebooks. The linter checks for various issues that could prevent your notebook from running correctly or cause confusion.

## How to Use [¶](\#how-to-use "Permanent link")

You can run the linter using the CLI:

## Rule Categories [¶](\#rule-categories "Permanent link")

marimo's lint rules are organized into three main categories based on their severity:

### 🚨 Breaking Rules [¶](\#breaking-rules "Permanent link")

These errors prevent notebook execution.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MB001](rules/unparsable_cells/) | unparsable-cells | Cell contains unparsable code | ❌ |
| [MB002](rules/multiple_definitions/) | multiple-definitions | Multiple cells define the same variable | ❌ |
| [MB003](rules/cycle_dependencies/) | cycle-dependencies | Cells have circular dependencies | ❌ |
| [MB004](rules/setup_cell_dependencies/) | setup-cell-dependencies | Setup cell cannot have dependencies | ❌ |
| [MB005](rules/invalid_syntax/) | invalid-syntax | Cell contains code that throws a SyntaxError on compilation | ❌ |

### ⚠️ Runtime Rules [¶](\#runtime-rules "Permanent link")

These issues may cause runtime problems.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MR001](rules/self_import/) | self-import | Importing a module with the same name as the file | ❌ |

### ✨ Formatting Rules [¶](\#formatting-rules "Permanent link")

These are style and formatting issues.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MF001](rules/general_formatting/) | general-formatting | General formatting issues with the notebook format. | 🛠️ |
| [MF002](rules/parse_stdout/) | parse-stdout | Parse captured stdout during notebook loading | ❌ |
| [MF003](rules/parse_stderr/) | parse-stderr | Parse captured stderr during notebook loading | ❌ |
| [MF004](rules/empty_cells/) | empty-cells | Empty cells that can be safely removed. | ⚠️ |
| [MF005](rules/sql_parse_error/) | sql-parse-error | SQL parsing errors during dependency analysis | ❌ |
| [MF006](rules/misc_log_capture/) | misc-log-capture | Miscellaneous log messages during processing | ❌ |
| [MF007](rules/markdown_indentation/) | markdown-indentation | Markdown cells in `mo.md()` should be dedented. | 🛠️ |

## Legend [¶](\#legend "Permanent link")

- 🛠️ = Automatically fixable with `marimo check --fix`
- ⚠️ = Fixable with `marimo check --fix --unsafe-fixes` (may change code behavior)
- ❌ = Not automatically fixable

## Configuration [¶](\#configuration "Permanent link")

Most lint rules are enabled by default. You can configure the linter behavior through marimo's configuration system.

## Related Documentation [¶](\#related-documentation "Permanent link")

- [Understanding Errors](../understanding_errors/) \- Detailed explanations of common marimo errors
- [CLI Reference](../../cli/) \- Complete CLI documentation including `marimo check`

 Back to top

# Lint Rules [¶](\#lint-rules "Permanent link")

marimo includes a comprehensive linting system that helps you write better, more reliable notebooks. The linter checks for various issues that could prevent your notebook from running correctly or cause confusion.

## How to Use [¶](\#how-to-use "Permanent link")

You can run the linter using the CLI:

## Rule Categories [¶](\#rule-categories "Permanent link")

marimo's lint rules are organized into three main categories based on their severity:

### 🚨 Breaking Rules [¶](\#breaking-rules "Permanent link")

These errors prevent notebook execution.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MB001](rules/unparsable_cells/) | unparsable-cells | Cell contains unparsable code | ❌ |
| [MB002](rules/multiple_definitions/) | multiple-definitions | Multiple cells define the same variable | ❌ |
| [MB003](rules/cycle_dependencies/) | cycle-dependencies | Cells have circular dependencies | ❌ |
| [MB004](rules/setup_cell_dependencies/) | setup-cell-dependencies | Setup cell cannot have dependencies | ❌ |
| [MB005](rules/invalid_syntax/) | invalid-syntax | Cell contains code that throws a SyntaxError on compilation | ❌ |

### ⚠️ Runtime Rules [¶](\#runtime-rules "Permanent link")

These issues may cause runtime problems.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MR001](rules/self_import/) | self-import | Importing a module with the same name as the file | ❌ |

### ✨ Formatting Rules [¶](\#formatting-rules "Permanent link")

These are style and formatting issues.

| Code | Name | Description | Fixable |
| --- | --- | --- | --- |
| [MF001](rules/general_formatting/) | general-formatting | General formatting issues with the notebook format. | 🛠️ |
| [MF002](rules/parse_stdout/) | parse-stdout | Parse captured stdout during notebook loading | ❌ |
| [MF003](rules/parse_stderr/) | parse-stderr | Parse captured stderr during notebook loading | ❌ |
| [MF004](rules/empty_cells/) | empty-cells | Empty cells that can be safely removed. | ⚠️ |
| [MF005](rules/sql_parse_error/) | sql-parse-error | SQL parsing errors during dependency analysis | ❌ |
| [MF006](rules/misc_log_capture/) | misc-log-capture | Miscellaneous log messages during processing | ❌ |
| [MF007](rules/markdown_indentation/) | markdown-indentation | Markdown cells in `mo.md()` should be dedented. | 🛠️ |

## Legend [¶](\#legend "Permanent link")

- 🛠️ = Automatically fixable with `marimo check --fix`
- ⚠️ = Fixable with `marimo check --fix --unsafe-fixes` (may change code behavior)
- ❌ = Not automatically fixable

## Configuration [¶](\#configuration "Permanent link")

Most lint rules are enabled by default. You can configure the linter behavior through marimo's configuration system.

## Related Documentation [¶](\#related-documentation "Permanent link")

- [Understanding Errors](../understanding_errors/) \- Detailed explanations of common marimo errors
- [CLI Reference](../../cli/) \- Complete CLI documentation including `marimo check`


---

###### Refresh - marimoRefresh¶

*Source: [https://docs.marimo.io/api/inputs/refresh/](https://docs.marimo.io/api/inputs/refresh/)*

<!-- Source: https://docs.marimo.io/api/inputs/refresh/ -->
<!-- Title: Refresh - marimoRefresh¶ -->
<!-- Depth: 5 -->

[Skip to content](#refresh)

# Refresh [¶](\#refresh "Permanent link")

## `` marimo.ui.refresh [¶](\#marimo.ui.refresh "Permanent link")

Bases: `UIElement[int, int]`

A refresh button that will auto-refresh its descendants for a given interval.

Each option value can either be a number (int or float) in seconds or a
human-readable string (e.g. "1s", "10s", "1m"). You can also combine multiple
time units (e.g. "1m 30s").

Note

The refresh interval may not be exact, as it depends on the
time it takes to render the content and the time it takes to send
the content to the client. Also, due to the buffering of UI element
changes, if the downstream cells take a long time to render, the
refresh interval may be longer than expected.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The time in seconds since the refresh has been activated.<br>**TYPE:** `int` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `options` | The options for the<br>refresh interval, as a list of human-readable strings or numbers (int or float)<br>in seconds. If no options are provided and default\_interval is provided,<br>the options will be generated automatically. If no options are provided and<br>default\_interval is not provided, the refresh button will not be displayed<br>with a dropdown for auto-refresh.<br>**TYPE:** `Optional[list[Union[int, float, str]]]`**DEFAULT:** `None` |
| `default_interval` | The default value<br>of the refresh interval.<br>**TYPE:** `Optional[Union[int, float, str]]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to run<br>when this element's value changes.<br>**TYPE:** `Optional[Callable[[int], None]]`**DEFAULT:** `None` |

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.refresh.name "Permanent link")

### `` text`property`[¶](\#marimo.ui.refresh.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.refresh.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.refresh.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.refresh.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.refresh.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.refresh.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.refresh.from_args "Permanent link")

### `` left [¶](\#marimo.ui.refresh.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.refresh.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.refresh.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Refresh [¶](\#refresh "Permanent link")

## `` marimo.ui.refresh [¶](\#marimo.ui.refresh "Permanent link")

Bases: `UIElement[int, int]`

A refresh button that will auto-refresh its descendants for a given interval.

Each option value can either be a number (int or float) in seconds or a
human-readable string (e.g. "1s", "10s", "1m"). You can also combine multiple
time units (e.g. "1m 30s").

Note

The refresh interval may not be exact, as it depends on the
time it takes to render the content and the time it takes to send
the content to the client. Also, due to the buffering of UI element
changes, if the downstream cells take a long time to render, the
refresh interval may be longer than expected.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The time in seconds since the refresh has been activated.<br>**TYPE:** `int` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `options` | The options for the<br>refresh interval, as a list of human-readable strings or numbers (int or float)<br>in seconds. If no options are provided and default\_interval is provided,<br>the options will be generated automatically. If no options are provided and<br>default\_interval is not provided, the refresh button will not be displayed<br>with a dropdown for auto-refresh.<br>**TYPE:** `Optional[list[Union[int, float, str]]]`**DEFAULT:** `None` |
| `default_interval` | The default value<br>of the refresh interval.<br>**TYPE:** `Optional[Union[int, float, str]]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to run<br>when this element's value changes.<br>**TYPE:** `Optional[Callable[[int], None]]`**DEFAULT:** `None` |

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.refresh.name "Permanent link")

### `` text`property`[¶](\#marimo.ui.refresh.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.refresh.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.refresh.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.refresh.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.refresh.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.refresh.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.refresh.from_args "Permanent link")

### `` left [¶](\#marimo.ui.refresh.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.refresh.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.refresh.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Refresh [¶](\#refresh "Permanent link")

## `` marimo.ui.refresh [¶](\#marimo.ui.refresh "Permanent link")

Bases: `UIElement[int, int]`

A refresh button that will auto-refresh its descendants for a given interval.

Each option value can either be a number (int or float) in seconds or a
human-readable string (e.g. "1s", "10s", "1m"). You can also combine multiple
time units (e.g. "1m 30s").

Note

The refresh interval may not be exact, as it depends on the
time it takes to render the content and the time it takes to send
the content to the client. Also, due to the buffering of UI element
changes, if the downstream cells take a long time to render, the
refresh interval may be longer than expected.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The time in seconds since the refresh has been activated.<br>**TYPE:** `int` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `options` | The options for the<br>refresh interval, as a list of human-readable strings or numbers (int or float)<br>in seconds. If no options are provided and default\_interval is provided,<br>the options will be generated automatically. If no options are provided and<br>default\_interval is not provided, the refresh button will not be displayed<br>with a dropdown for auto-refresh.<br>**TYPE:** `Optional[list[Union[int, float, str]]]`**DEFAULT:** `None` |
| `default_interval` | The default value<br>of the refresh interval.<br>**TYPE:** `Optional[Union[int, float, str]]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to run<br>when this element's value changes.<br>**TYPE:** `Optional[Callable[[int], None]]`**DEFAULT:** `None` |

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.refresh.name "Permanent link")

### `` text`property`[¶](\#marimo.ui.refresh.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.refresh.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.refresh.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.refresh.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.refresh.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.refresh.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.refresh.from_args "Permanent link")

### `` left [¶](\#marimo.ui.refresh.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.refresh.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.refresh.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

###### Run Button - marimoRun Button¶

*Source: [https://docs.marimo.io/api/inputs/run_button/](https://docs.marimo.io/api/inputs/run_button/)*

<!-- Source: https://docs.marimo.io/api/inputs/run_button/ -->
<!-- Title: Run Button - marimoRun Button¶ -->
<!-- Depth: 5 -->

[Skip to content](#run-button)

# Run Button [¶](\#run-button "Permanent link")

Source code for `examples/ui/run_button.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.ui.run\_button [¶](\#marimo.ui.run_button "Permanent link")

Bases: `UIElement[Any, Any]`

A button that can be used to trigger computation.

When clicked, run\_button's value is set to True, and any cells referencing it are run.
After those cells are run, run\_button's value will automatically be set back to False
as long as automatic execution is enabled.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The value of the button; True when clicked, and reset to False after<br>cells referencing the button finish running (when automatic execution is enabled).<br>**TYPE:** `bool` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `kind` | Button style.<br>Defaults to "neutral".<br>**TYPE:** `Literal['neutral', 'success', 'warn', 'danger']`**DEFAULT:** `'neutral'` |
| `disabled` | Whether the button is disabled. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `tooltip` | A tooltip to display for the button. Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "click to run".<br>**TYPE:** `str`**DEFAULT:** `'click to run'` |
| `on_change` | Optional callback to run when this<br>element's value changes.<br>**TYPE:** `Callable[[Any], None]`**DEFAULT:** `None` |
| `full_width` | Whether the input should take up the full width of its<br>container. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `keyboard_shortcut` | Keyboard shortcut to trigger the button<br>(e.g. 'Ctrl-L'). Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |

### `` text`property`[¶](\#marimo.ui.run_button.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.run_button.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.run_button.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.run_button.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.run_button.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.run_button.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.run_button.from_args "Permanent link")

### `` left [¶](\#marimo.ui.run_button.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.run_button.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.run_button.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Run Button [¶](\#run-button "Permanent link")

Source code for `examples/ui/run_button.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.ui.run\_button [¶](\#marimo.ui.run_button "Permanent link")

Bases: `UIElement[Any, Any]`

A button that can be used to trigger computation.

When clicked, run\_button's value is set to True, and any cells referencing it are run.
After those cells are run, run\_button's value will automatically be set back to False
as long as automatic execution is enabled.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The value of the button; True when clicked, and reset to False after<br>cells referencing the button finish running (when automatic execution is enabled).<br>**TYPE:** `bool` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `kind` | Button style.<br>Defaults to "neutral".<br>**TYPE:** `Literal['neutral', 'success', 'warn', 'danger']`**DEFAULT:** `'neutral'` |
| `disabled` | Whether the button is disabled. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `tooltip` | A tooltip to display for the button. Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "click to run".<br>**TYPE:** `str`**DEFAULT:** `'click to run'` |
| `on_change` | Optional callback to run when this<br>element's value changes.<br>**TYPE:** `Callable[[Any], None]`**DEFAULT:** `None` |
| `full_width` | Whether the input should take up the full width of its<br>container. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `keyboard_shortcut` | Keyboard shortcut to trigger the button<br>(e.g. 'Ctrl-L'). Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |

### `` text`property`[¶](\#marimo.ui.run_button.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.run_button.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.run_button.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.run_button.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.run_button.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.run_button.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.run_button.from_args "Permanent link")

### `` left [¶](\#marimo.ui.run_button.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.run_button.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.run_button.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Run Button [¶](\#run-button "Permanent link")

Source code for `examples/ui/run_button.py`

Tip: paste this code into an empty cell, and the marimo editor will create cells for you

## `` marimo.ui.run\_button [¶](\#marimo.ui.run_button "Permanent link")

Bases: `UIElement[Any, Any]`

A button that can be used to trigger computation.

When clicked, run\_button's value is set to True, and any cells referencing it are run.
After those cells are run, run\_button's value will automatically be set back to False
as long as automatic execution is enabled.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The value of the button; True when clicked, and reset to False after<br>cells referencing the button finish running (when automatic execution is enabled).<br>**TYPE:** `bool` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `kind` | Button style.<br>Defaults to "neutral".<br>**TYPE:** `Literal['neutral', 'success', 'warn', 'danger']`**DEFAULT:** `'neutral'` |
| `disabled` | Whether the button is disabled. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `tooltip` | A tooltip to display for the button. Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "click to run".<br>**TYPE:** `str`**DEFAULT:** `'click to run'` |
| `on_change` | Optional callback to run when this<br>element's value changes.<br>**TYPE:** `Callable[[Any], None]`**DEFAULT:** `None` |
| `full_width` | Whether the input should take up the full width of its<br>container. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `keyboard_shortcut` | Keyboard shortcut to trigger the button<br>(e.g. 'Ctrl-L'). Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |

### `` text`property`[¶](\#marimo.ui.run_button.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.run_button.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.run_button.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.run_button.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.run_button.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.run_button.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.run_button.from_args "Permanent link")

### `` left [¶](\#marimo.ui.run_button.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.run_button.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.run_button.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

###### Callout - marimoCallout¶

*Source: [https://docs.marimo.io/api/layouts/callout/](https://docs.marimo.io/api/layouts/callout/)*

<!-- Source: https://docs.marimo.io/api/layouts/callout/ -->
<!-- Title: Callout - marimoCallout¶ -->
<!-- Depth: 5 -->

[Skip to content](#callout)

# Callout [¶](\#callout "Permanent link")

## `` marimo.callout [¶](\#marimo.callout "Permanent link")

Build a callout output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | A value to render in the callout<br>**TYPE:** `object` |
| `kind` | The kind of callout (affects styling).<br>**TYPE:** `Literal['neutral', 'warn', 'success', 'info', 'danger']`**DEFAULT:** `'neutral'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An HTML object.<br>**TYPE:** `Html` |

Back to top

# Callout [¶](\#callout "Permanent link")

## `` marimo.callout [¶](\#marimo.callout "Permanent link")

Build a callout output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | A value to render in the callout<br>**TYPE:** `object` |
| `kind` | The kind of callout (affects styling).<br>**TYPE:** `Literal['neutral', 'warn', 'success', 'info', 'danger']`**DEFAULT:** `'neutral'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An HTML object.<br>**TYPE:** `Html` |

 Back to top

# Callout [¶](\#callout "Permanent link")

## `` marimo.callout [¶](\#marimo.callout "Permanent link")

Build a callout output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | A value to render in the callout<br>**TYPE:** `object` |
| `kind` | The kind of callout (affects styling).<br>**TYPE:** `Literal['neutral', 'warn', 'success', 'info', 'danger']`**DEFAULT:** `'neutral'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An HTML object.<br>**TYPE:** `Html` |


---

###### Carousel - marimoCarousel¶

*Source: [https://docs.marimo.io/api/layouts/carousel/](https://docs.marimo.io/api/layouts/carousel/)*

<!-- Source: https://docs.marimo.io/api/layouts/carousel/ -->
<!-- Title: Carousel - marimoCarousel¶ -->
<!-- Depth: 5 -->

[Skip to content](#carousel)

# Carousel [¶](\#carousel "Permanent link")

## `` marimo.carousel [¶](\#marimo.carousel "Permanent link")

Create a carousel of items.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

Back to top

# Carousel [¶](\#carousel "Permanent link")

## `` marimo.carousel [¶](\#marimo.carousel "Permanent link")

Create a carousel of items.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example

 Back to top

# Carousel [¶](\#carousel "Permanent link")

## `` marimo.carousel [¶](\#marimo.carousel "Permanent link")

Create a carousel of items.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | A list of items.<br>**TYPE:** `Sequence[object]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

Example


---

###### Json - marimoJson¶

*Source: [https://docs.marimo.io/api/layouts/json/](https://docs.marimo.io/api/layouts/json/)*

<!-- Source: https://docs.marimo.io/api/layouts/json/ -->
<!-- Title: Json - marimoJson¶ -->
<!-- Depth: 5 -->

[Skip to content](#json)

# Json [¶](\#json "Permanent link")

## `` marimo.json [¶](\#marimo.json "Permanent link")

Render a JSON with tree.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | JSON string or JSON-compatible Python object(s) to render<br>**TYPE:** `str | dict[str, Any] | list[Any]` |
| `label` | optional text label for the tree<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | `Html` object<br>**TYPE:** `Html` |

Back to top

# Json [¶](\#json "Permanent link")

## `` marimo.json [¶](\#marimo.json "Permanent link")

Render a JSON with tree.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | JSON string or JSON-compatible Python object(s) to render<br>**TYPE:** `str | dict[str, Any] | list[Any]` |
| `label` | optional text label for the tree<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | `Html` object<br>**TYPE:** `Html` |

 Back to top

# Json [¶](\#json "Permanent link")

## `` marimo.json [¶](\#marimo.json "Permanent link")

Render a JSON with tree.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | JSON string or JSON-compatible Python object(s) to render<br>**TYPE:** `str | dict[str, Any] | list[Any]` |
| `label` | optional text label for the tree<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | `Html` object<br>**TYPE:** `Html` |


---

###### Justify - marimoJustify¶

*Source: [https://docs.marimo.io/api/layouts/justify/](https://docs.marimo.io/api/layouts/justify/)*

<!-- Source: https://docs.marimo.io/api/layouts/justify/ -->
<!-- Title: Justify - marimoJustify¶ -->
<!-- Depth: 5 -->

[Skip to content](#justify)

# Justify [¶](\#justify "Permanent link")

## `` marimo.center [¶](\#marimo.center "Permanent link")

Center an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to center.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A centered `Html` object. |

## `` marimo.left [¶](\#marimo.left "Permanent link")

Left-justify an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to left-justify.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A left-justified `Html` object. |

## `` marimo.right [¶](\#marimo.right "Permanent link")

Right-justify an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to right-justify.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A right-justified `Html` object. |

Back to top

# Justify [¶](\#justify "Permanent link")

## `` marimo.center [¶](\#marimo.center "Permanent link")

Center an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to center.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A centered `Html` object. |

## `` marimo.left [¶](\#marimo.left "Permanent link")

Left-justify an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to left-justify.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A left-justified `Html` object. |

## `` marimo.right [¶](\#marimo.right "Permanent link")

Right-justify an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to right-justify.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A right-justified `Html` object. |

 Back to top

# Justify [¶](\#justify "Permanent link")

## `` marimo.center [¶](\#marimo.center "Permanent link")

Center an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to center.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A centered `Html` object. |

## `` marimo.left [¶](\#marimo.left "Permanent link")

Left-justify an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to left-justify.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A left-justified `Html` object. |

## `` marimo.right [¶](\#marimo.right "Permanent link")

Right-justify an item.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `item` | object to right-justify.<br>**TYPE:** `object` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | A right-justified `Html` object. |


---

###### Lazy - marimoLazy¶

*Source: [https://docs.marimo.io/api/layouts/lazy/](https://docs.marimo.io/api/layouts/lazy/)*

<!-- Source: https://docs.marimo.io/api/layouts/lazy/ -->
<!-- Title: Lazy - marimoLazy¶ -->
<!-- Depth: 5 -->

[Skip to content](#lazy)

# Lazy [¶](\#lazy "Permanent link")

## `` marimo.lazy [¶](\#marimo.lazy "Permanent link")

Bases: `UIElement[bool, bool]`

Lazy load a component until it is visible.

Use `mo.lazy` to defer rendering of an item until it's visible. This is
useful for loading expensive components only when they are needed, e.g.,
only when an accordion or tab is opened.

The argument to `mo.lazy` can be an object to render lazily, or a function
that returns the object to render (that is, functions are lazily
evaluated). The function can be synchronous or asynchronous.
Using a function is useful when the item to render is
the result of a database query or some other expensive operation.

Examples:

Create a lazy-loaded tab:

Create a lazy-loaded accordion:

Usage with async functions:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `element` | Object or callable that returns content to be lazily loaded.<br>**TYPE:** `Union[Callable[[], object], object, Callable[[], Coroutine[None, None, object]]]` |
| `show_loading_indicator` | Whether to show a loading indicator<br>while the content is being loaded. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

### `` text`property`[¶](\#marimo.lazy.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.lazy.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.lazy.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.lazy.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.lazy.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.lazy.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.lazy.from_args "Permanent link")

### `` left [¶](\#marimo.lazy.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.lazy.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.lazy.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Lazy [¶](\#lazy "Permanent link")

## `` marimo.lazy [¶](\#marimo.lazy "Permanent link")

Bases: `UIElement[bool, bool]`

Lazy load a component until it is visible.

Use `mo.lazy` to defer rendering of an item until it's visible. This is
useful for loading expensive components only when they are needed, e.g.,
only when an accordion or tab is opened.

The argument to `mo.lazy` can be an object to render lazily, or a function
that returns the object to render (that is, functions are lazily
evaluated). The function can be synchronous or asynchronous.
Using a function is useful when the item to render is
the result of a database query or some other expensive operation.

Examples:

Create a lazy-loaded tab:

Create a lazy-loaded accordion:

Usage with async functions:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `element` | Object or callable that returns content to be lazily loaded.<br>**TYPE:** `Union[Callable[[], object], object, Callable[[], Coroutine[None, None, object]]]` |
| `show_loading_indicator` | Whether to show a loading indicator<br>while the content is being loaded. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

### `` text`property`[¶](\#marimo.lazy.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.lazy.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.lazy.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.lazy.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.lazy.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.lazy.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.lazy.from_args "Permanent link")

### `` left [¶](\#marimo.lazy.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.lazy.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.lazy.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Lazy [¶](\#lazy "Permanent link")

## `` marimo.lazy [¶](\#marimo.lazy "Permanent link")

Bases: `UIElement[bool, bool]`

Lazy load a component until it is visible.

Use `mo.lazy` to defer rendering of an item until it's visible. This is
useful for loading expensive components only when they are needed, e.g.,
only when an accordion or tab is opened.

The argument to `mo.lazy` can be an object to render lazily, or a function
that returns the object to render (that is, functions are lazily
evaluated). The function can be synchronous or asynchronous.
Using a function is useful when the item to render is
the result of a database query or some other expensive operation.

Examples:

Create a lazy-loaded tab:

Create a lazy-loaded accordion:

Usage with async functions:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `element` | Object or callable that returns content to be lazily loaded.<br>**TYPE:** `Union[Callable[[], object], object, Callable[[], Coroutine[None, None, object]]]` |
| `show_loading_indicator` | Whether to show a loading indicator<br>while the content is being loaded. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |

### `` text`property`[¶](\#marimo.lazy.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.lazy.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.lazy.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.lazy.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.lazy.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.lazy.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.lazy.from_args "Permanent link")

### `` left [¶](\#marimo.lazy.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.lazy.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.lazy.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

###### Outline - marimoOutline¶

*Source: [https://docs.marimo.io/api/layouts/outline/](https://docs.marimo.io/api/layouts/outline/)*

<!-- Source: https://docs.marimo.io/api/layouts/outline/ -->
<!-- Title: Outline - marimoOutline¶ -->
<!-- Depth: 5 -->

[Skip to content](#outline)

# Outline [¶](\#outline "Permanent link")

## `` marimo.outline [¶](\#marimo.outline "Permanent link")

Display a table of contents outline showing all markdown headers in the notebook.

The outline automatically extracts all markdown headers from executed cells
and displays them in a hierarchical structure with clickable navigation.

Examples:

Basic outline:

With custom label:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A descriptive label for the outline. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An HTML object that renders the outline component.<br>**TYPE:** `Html` |

Back to top

# Outline [¶](\#outline "Permanent link")

## `` marimo.outline [¶](\#marimo.outline "Permanent link")

Display a table of contents outline showing all markdown headers in the notebook.

The outline automatically extracts all markdown headers from executed cells
and displays them in a hierarchical structure with clickable navigation.

Examples:

Basic outline:

With custom label:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A descriptive label for the outline. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An HTML object that renders the outline component.<br>**TYPE:** `Html` |

 Back to top

# Outline [¶](\#outline "Permanent link")

## `` marimo.outline [¶](\#marimo.outline "Permanent link")

Display a table of contents outline showing all markdown headers in the notebook.

The outline automatically extracts all markdown headers from executed cells
and displays them in a hierarchical structure with clickable navigation.

Examples:

Basic outline:

With custom label:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A descriptive label for the outline. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An HTML object that renders the outline component.<br>**TYPE:** `Html` |


---

###### Plain - marimoPlain¶

*Source: [https://docs.marimo.io/api/layouts/plain/](https://docs.marimo.io/api/layouts/plain/)*

<!-- Source: https://docs.marimo.io/api/layouts/plain/ -->
<!-- Title: Plain - marimoPlain¶ -->
<!-- Depth: 5 -->

[Skip to content](#plain)

# Plain [¶](\#plain "Permanent link")

## `` marimo.plain [¶](\#marimo.plain "Permanent link")

Wrap a value to indicate that it should be displayed without any opinionated formatting.

This is the best way to opt out of marimo's default dataframe rendering.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | Any value<br>**TYPE:** `Any` |

Back to top

# Plain [¶](\#plain "Permanent link")

## `` marimo.plain [¶](\#marimo.plain "Permanent link")

Wrap a value to indicate that it should be displayed without any opinionated formatting.

This is the best way to opt out of marimo's default dataframe rendering.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | Any value<br>**TYPE:** `Any` |

 Back to top

# Plain [¶](\#plain "Permanent link")

## `` marimo.plain [¶](\#marimo.plain "Permanent link")

Wrap a value to indicate that it should be displayed without any opinionated formatting.

This is the best way to opt out of marimo's default dataframe rendering.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | Any value<br>**TYPE:** `Any` |


---

###### Routes - marimoRoutes¶

*Source: [https://docs.marimo.io/api/layouts/routes/](https://docs.marimo.io/api/layouts/routes/)*

<!-- Source: https://docs.marimo.io/api/layouts/routes/ -->
<!-- Title: Routes - marimoRoutes¶ -->
<!-- Depth: 5 -->

[Skip to content](#routes)

# Routes [¶](\#routes "Permanent link")

## `` marimo.routes [¶](\#marimo.routes "Permanent link")

Bases: `UIElement[str, str]`

Renders a list of routes that are switched based on the URL path.

Routes currently don't support nested routes, or dynamic routes (e.g. `#/user/:id`).
If you'd like to see these features, please let us know on GitHub:
[https://github.com/marimo-team/marimo/issues](https://github.com/marimo-team/marimo/issues)

For a simple-page-application (SPA) experience, you should use hash-based routing.
For example, prefix your routes with `#/`.

If you are using a multi-page-application (MPA) with `marimo.create_asgi_app`,
you should use path-based routing. For example, prefix your routes with `/`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `routes` | A dictionary of routes, where the key is the URL path and the value is a function<br>that returns the content to display.<br>**TYPE:** `dict[str, Union[Callable[[], object], Callable[[], Coroutine[None, None, object]], object]]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

### `` CATCH\_ALL`class-attribute``instance-attribute`[¶](\#marimo.routes.CATCH_ALL "Permanent link")

### `` DEFAULT`class-attribute``instance-attribute`[¶](\#marimo.routes.DEFAULT "Permanent link")

### `` text`property`[¶](\#marimo.routes.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.routes.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.routes.batch "Permanent link")

### `` callout [¶](\#marimo.routes.callout "Permanent link")

### `` center [¶](\#marimo.routes.center "Permanent link")

### `` form [¶](\#marimo.routes.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.routes.from_args "Permanent link")

### `` left [¶](\#marimo.routes.left "Permanent link")

### `` right [¶](\#marimo.routes.right "Permanent link")

### `` style [¶](\#marimo.routes.style "Permanent link")

python
import marimo

app = marimo.App()

@app.cell
def \_\_():
import marimo as mo
return

@app.cell
def \_\_():
mo.sidebar(
\[
mo.md("# marimo"),
mo.nav\_menu(
{
"#/": f"{mo.icon('lucide:home')} Home",
"#/about": f"{mo.icon('lucide:user')} About",
"#/contact": f"{mo.icon('lucide:phone')} Contact",
"Links": {
" [https://twitter.com/marimo\_io](https://twitter.com/marimo_io)": "Twitter",
" [https://github.com/marimo-team/marimo](https://github.com/marimo-team/marimo)": "GitHub",
},
},
orientation="vertical",
),
\]
)
return

@app.cell
def \_\_():
mo.routes({
"#/": mo.md("# Home"),
"#/about": mo.md("# About"),
"#/contact": mo.md("# Contact"),
mo.routes.CATCH\_ALL: mo.md("# Home"),
})
return

Back to top

# Routes [¶](\#routes "Permanent link")

## `` marimo.routes [¶](\#marimo.routes "Permanent link")

Bases: `UIElement[str, str]`

Renders a list of routes that are switched based on the URL path.

Routes currently don't support nested routes, or dynamic routes (e.g. `#/user/:id`).
If you'd like to see these features, please let us know on GitHub:
[https://github.com/marimo-team/marimo/issues](https://github.com/marimo-team/marimo/issues)

For a simple-page-application (SPA) experience, you should use hash-based routing.
For example, prefix your routes with `#/`.

If you are using a multi-page-application (MPA) with `marimo.create_asgi_app`,
you should use path-based routing. For example, prefix your routes with `/`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `routes` | A dictionary of routes, where the key is the URL path and the value is a function<br>that returns the content to display.<br>**TYPE:** `dict[str, Union[Callable[[], object], Callable[[], Coroutine[None, None, object]], object]]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

### `` CATCH\_ALL`class-attribute``instance-attribute`[¶](\#marimo.routes.CATCH_ALL "Permanent link")

### `` DEFAULT`class-attribute``instance-attribute`[¶](\#marimo.routes.DEFAULT "Permanent link")

### `` text`property`[¶](\#marimo.routes.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.routes.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.routes.batch "Permanent link")

### `` callout [¶](\#marimo.routes.callout "Permanent link")

### `` center [¶](\#marimo.routes.center "Permanent link")

### `` form [¶](\#marimo.routes.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.routes.from_args "Permanent link")

### `` left [¶](\#marimo.routes.left "Permanent link")

### `` right [¶](\#marimo.routes.right "Permanent link")

### `` style [¶](\#marimo.routes.style "Permanent link")

python
import marimo

app = marimo.App()

@app.cell
def \_\_():
import marimo as mo
return

@app.cell
def \_\_():
mo.sidebar(
\[
mo.md("# marimo"),
mo.nav\_menu(
{
"#/": f"{mo.icon('lucide:home')} Home",
"#/about": f"{mo.icon('lucide:user')} About",
"#/contact": f"{mo.icon('lucide:phone')} Contact",
"Links": {
" [https://twitter.com/marimo\_io](https://twitter.com/marimo_io)": "Twitter",
" [https://github.com/marimo-team/marimo](https://github.com/marimo-team/marimo)": "GitHub",
},
},
orientation="vertical",
),
\]
)
return

@app.cell
def \_\_():
mo.routes({
"#/": mo.md("# Home"),
"#/about": mo.md("# About"),
"#/contact": mo.md("# Contact"),
mo.routes.CATCH\_ALL: mo.md("# Home"),
})
return

 Back to top

# Routes [¶](\#routes "Permanent link")

## `` marimo.routes [¶](\#marimo.routes "Permanent link")

Bases: `UIElement[str, str]`

Renders a list of routes that are switched based on the URL path.

Routes currently don't support nested routes, or dynamic routes (e.g. `#/user/:id`).
If you'd like to see these features, please let us know on GitHub:
[https://github.com/marimo-team/marimo/issues](https://github.com/marimo-team/marimo/issues)

For a simple-page-application (SPA) experience, you should use hash-based routing.
For example, prefix your routes with `#/`.

If you are using a multi-page-application (MPA) with `marimo.create_asgi_app`,
you should use path-based routing. For example, prefix your routes with `/`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `routes` | A dictionary of routes, where the key is the URL path and the value is a function<br>that returns the content to display.<br>**TYPE:** `dict[str, Union[Callable[[], object], Callable[[], Coroutine[None, None, object]], object]]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An Html object.<br>**TYPE:** `Html` |

### `` CATCH\_ALL`class-attribute``instance-attribute`[¶](\#marimo.routes.CATCH_ALL "Permanent link")

### `` DEFAULT`class-attribute``instance-attribute`[¶](\#marimo.routes.DEFAULT "Permanent link")

### `` text`property`[¶](\#marimo.routes.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.routes.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.routes.batch "Permanent link")

### `` callout [¶](\#marimo.routes.callout "Permanent link")

### `` center [¶](\#marimo.routes.center "Permanent link")

### `` form [¶](\#marimo.routes.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.routes.from_args "Permanent link")

### `` left [¶](\#marimo.routes.left "Permanent link")

### `` right [¶](\#marimo.routes.right "Permanent link")

### `` style [¶](\#marimo.routes.style "Permanent link")

python
import marimo

app = marimo.App()

@app.cell
def \_\_():
import marimo as mo
return

@app.cell
def \_\_():
mo.sidebar(
\[
mo.md("# marimo"),
mo.nav\_menu(
{
"#/": f"{mo.icon('lucide:home')} Home",
"#/about": f"{mo.icon('lucide:user')} About",
"#/contact": f"{mo.icon('lucide:phone')} Contact",
"Links": {
" [https://twitter.com/marimo\_io](https://twitter.com/marimo_io)": "Twitter",
" [https://github.com/marimo-team/marimo](https://github.com/marimo-team/marimo)": "GitHub",
},
},
orientation="vertical",
),
\]
)
return

@app.cell
def \_\_():
mo.routes({
"#/": mo.md("# Home"),
"#/about": mo.md("# About"),
"#/contact": mo.md("# Contact"),
mo.routes.CATCH\_ALL: mo.md("# Home"),
})
return


---

###### Stat - marimoStat¶

*Source: [https://docs.marimo.io/api/layouts/stat/](https://docs.marimo.io/api/layouts/stat/)*

<!-- Source: https://docs.marimo.io/api/layouts/stat/ -->
<!-- Title: Stat - marimoStat¶ -->
<!-- Depth: 5 -->

[Skip to content](#stat)

# Stat [¶](\#stat "Permanent link")

## `` marimo.stat [¶](\#marimo.stat "Permanent link")

Display a statistic.

Optionally include a label, caption, and direction.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | the value to display<br>**TYPE:** `Union[str, int, float]` |
| `label` | the label to display<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `caption` | the caption to display<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `direction` | the direction of the statistic,<br>either `increase` or `decrease`<br>**TYPE:** `Optional[Literal['increase', 'decrease']]`**DEFAULT:** `None` |
| `bordered` | whether to display a border around the statistic<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object representing the statistic. |

Back to top

# Stat [¶](\#stat "Permanent link")

## `` marimo.stat [¶](\#marimo.stat "Permanent link")

Display a statistic.

Optionally include a label, caption, and direction.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | the value to display<br>**TYPE:** `Union[str, int, float]` |
| `label` | the label to display<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `caption` | the caption to display<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `direction` | the direction of the statistic,<br>either `increase` or `decrease`<br>**TYPE:** `Optional[Literal['increase', 'decrease']]`**DEFAULT:** `None` |
| `bordered` | whether to display a border around the statistic<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object representing the statistic. |

 Back to top

# Stat [¶](\#stat "Permanent link")

## `` marimo.stat [¶](\#marimo.stat "Permanent link")

Display a statistic.

Optionally include a label, caption, and direction.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `value` | the value to display<br>**TYPE:** `Union[str, int, float]` |
| `label` | the label to display<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `caption` | the caption to display<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `direction` | the direction of the statistic,<br>either `increase` or `decrease`<br>**TYPE:** `Optional[Literal['increase', 'decrease']]`**DEFAULT:** `None` |
| `bordered` | whether to display a border around the statistic<br>**TYPE:** `bool`**DEFAULT:** `False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object representing the statistic. |


---

###### Tree - marimoTree¶

*Source: [https://docs.marimo.io/api/layouts/tree/](https://docs.marimo.io/api/layouts/tree/)*

<!-- Source: https://docs.marimo.io/api/layouts/tree/ -->
<!-- Title: Tree - marimoTree¶ -->
<!-- Depth: 5 -->

[Skip to content](#tree)

# Tree [¶](\#tree "Permanent link")

## `` marimo.tree [¶](\#marimo.tree "Permanent link")

Render a nested structure of lists, tuples, or dicts as a tree.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | nested structure of lists, tuples, or dicts<br>**TYPE:** `list[Any] | tuple[Any] | dict[Any, Any]` |
| `label` | optional text label for the tree<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | `Html` object<br>**TYPE:** `Html` |

Back to top

# Tree [¶](\#tree "Permanent link")

## `` marimo.tree [¶](\#marimo.tree "Permanent link")

Render a nested structure of lists, tuples, or dicts as a tree.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | nested structure of lists, tuples, or dicts<br>**TYPE:** `list[Any] | tuple[Any] | dict[Any, Any]` |
| `label` | optional text label for the tree<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | `Html` object<br>**TYPE:** `Html` |

 Back to top

# Tree [¶](\#tree "Permanent link")

## `` marimo.tree [¶](\#marimo.tree "Permanent link")

Render a nested structure of lists, tuples, or dicts as a tree.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `items` | nested structure of lists, tuples, or dicts<br>**TYPE:** `list[Any] | tuple[Any] | dict[Any, Any]` |
| `label` | optional text label for the tree<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | `Html` object<br>**TYPE:** `Html` |


---

###### Commands - marimomarimo¶

*Source: [https://docs.marimo.io/cli/](https://docs.marimo.io/cli/)*

<!-- Source: https://docs.marimo.io/cli/ -->
<!-- Title: Commands - marimomarimo¶ -->
<!-- Depth: 5 -->

[Skip to content](#marimo)

# marimo [¶](\#marimo "Permanent link")

Welcome to marimo!
Getting started:

- marimo tutorial intro

Example usage:

- marimo edit create or edit notebooks
- marimo edit notebook.py create or edit a notebook called notebook.py
- marimo run notebook.py run a notebook as a read-only app
- marimo tutorial --help list tutorials

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--version` | boolean | Show the version and exit. | `False` |
| `-l`, `--log-level` | choice ( `DEBUG` \| `INFO` \| `WARN` \| `ERROR` \| `CRITICAL`) | Choose logging level. | `WARN` |
| `-q`, `--quiet` | boolean | Suppress standard out. | `False` |
| `-y`, `--yes` | boolean | Automatic yes to prompts, running non-interactively. | `False` |
| `-d`, `--development-mode` | boolean | Run in development mode; enables debug logs and server autoreload. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo check [¶](\#marimo-check "Permanent link")

Check and format marimo files.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--fix` | boolean | Whether to in place update files. | `False` |
| `--strict` | boolean | Whether warnings return a non-zero exit code. | `False` |
| `-v`, `--verbose` / `-q`, `--quiet` | boolean | Whether to print detailed messages. | `True` |
| `--unsafe-fixes` | boolean | Enable fixes that may change code behavior (e.g., removing empty cells). | `False` |
| `--ignore-scripts` | boolean | Ignore files that are not recognizable as marimo notebooks. | `False` |
| `--format` | choice ( `full` \| `json`) | Output format for diagnostics. | `full` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo config [¶](\#marimo-config "Permanent link")

Various commands for the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo config describe [¶](\#marimo-config-describe "Permanent link")

Describe the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo config show [¶](\#marimo-config-show "Permanent link")

Show the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo convert [¶](\#marimo-convert "Permanent link")

Convert a Jupyter notebook, Markdown file, or Python script to a marimo notebook.

Supported input formats:
\- `.ipynb` (local or GitHub-hosted)
\- `.md` files with `{python}` code fences
\- `.py` scripts in py:percent format

Behavior:
\- Jupyter notebooks: outputs are stripped.

- Markdown files: only `{python}` fenced code blocks are converted.

Example:

\- Python scripts:
\- If already a valid marimo notebook, no conversion is performed.
\- Otherwise, marimo attempts to convert using py:percent formatting,
preserving top-level comments and docstrings.

Example usage:

or

or

After conversion:

Note:
Since marimo's reactive execution differs from traditional notebooks,
you may need to refactor code that mutates variables across cells
(e.g., modifying a dataframe in multiple cells), which can lead to
unexpected behavior.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-o`, `--output` | path | Output file to save the converted notebook to. If not provided, the converted notebook will be printed to stdout. | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo edit [¶](\#marimo-edit "Permanent link")

Create or edit notebooks.

- marimo edit Start the marimo notebook server
- marimo edit notebook.py Create or edit notebook.py

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--allow-origins` | text | Allowed origins for CORS. Can be repeated. Use \* for all origins. | None |
| `--skip-update-check` | boolean | Don't check if a new version of marimo is available for download. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--watch` | boolean | Watch the file for changes and reload the code when saved in another editor. | `False` |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--timeout` | float | Enable a global timeout to shut down the server after specified number of minutes of no connection | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo env [¶](\#marimo-env "Permanent link")

Print out environment information for debugging purposes.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo export [¶](\#marimo-export "Permanent link")

Export a notebook to various formats.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export html [¶](\#marimo-export-html "Permanent link")

Run a notebook and export it as an HTML file.

Example:

Optionally pass CLI args to the notebook:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--include-code` / `--no-include-code` | boolean | Include notebook code in the exported HTML file. | `True` |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the HTML to. If not provided, the HTML will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export html-wasm [¶](\#marimo-export-html-wasm "Permanent link")

Export a notebook as a WASM-powered standalone HTML file.

Example:

The exported HTML file will run the notebook using WebAssembly, making it
completely self-contained and executable in the browser. This lets you
share interactive notebooks on the web without setting up
infrastructure to run Python code.

The exported notebook runs using Pyodide, which supports most
but not all Python packages. To learn more, see the Pyodide
documentation.

In order for this file to be able to run, it must be served over HTTP,
and cannot be opened directly from the file system (e.g. file://).

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-o`, `--output` | path | Output directory to save the HTML to. | `Sentinel.UNSET` |
| `--mode` | choice ( `edit` \| `run`) | Whether the notebook code should be editable or readonly. | `run` |
| `--watch` / `--no-watch` | boolean | Whether to watch the original file and export upon change | `False` |
| `--show-code` / `--no-show-code` | boolean | Whether to show code by default in the exported HTML file; only relevant for run mode. | `False` |
| `--include-cloudflare` / `--no-include-cloudflare` | boolean | Whether to include Cloudflare Worker configuration files (index.js and wrangler.jsonc) for easy deployment. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export ipynb [¶](\#marimo-export-ipynb "Permanent link")

Export a marimo notebook as a Jupyter notebook in topological order.

Example:

Watch for changes and regenerate the script on modification:

Requires nbformat to be installed.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--sort` | choice ( `top-down` \| `topological`) | Sort cells top-down or in topological order. | `topological` |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the ipynb file to. If not provided, the ipynb contents will be printed to stdout. | None |
| `--include-outputs` / `--no-include-outputs` | boolean | Run the notebook and include outputs in the exported ipynb file. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export md [¶](\#marimo-export-md "Permanent link")

Export a marimo notebook as a code fenced Markdown file.

Example:

Watch for changes and regenerate the script on modification:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the markdown to. If not provided, markdown will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export script [¶](\#marimo-export-script "Permanent link")

Export a marimo notebook as a flat script, in topological order.

Example:

Watch for changes and regenerate the script on modification:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the script to. If not provided, the script will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo new [¶](\#marimo-new "Permanent link")

Create an empty notebook, or generate from a prompt with AI

- marimo new Create an empty notebook
- marimo new "Plot an interactive 3D surface with matplotlib." Generate a notebook from a prompt.
- marimo new prompt.txt Generate a notebook from a file containing a prompt.

Visit [https://marimo.app/ai](https://marimo.app/ai) for more prompt examples.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--timeout` | float | Enable a global timeout to shut down the server after specified number of minutes of no connection | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo recover [¶](\#marimo-recover "Permanent link")

Recover a marimo notebook from JSON.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo run [¶](\#marimo-run "Permanent link")

Run a notebook as an app in read-only mode.

If NAME is a url, the notebook will be downloaded to a temporary file.

Example:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `False` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--include-code` | boolean | Include notebook code in the app. | `False` |
| `--session-ttl` | integer | Seconds to wait before closing a session on websocket disconnect. | `120` |
| `--watch` | boolean | Watch the file for changes and reload the app. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--allow-origins` | text | Allowed origins for CORS. Can be repeated. | None |
| `--redirect-console-to-browser` | boolean | Redirect console logs to the browser console. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--check` / `--no-check` | boolean | Disable a static check of the notebook before running. | `True` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo shell-completion [¶](\#marimo-shell-completion "Permanent link")

Install shell completions for marimo. Supports bash, zsh, and fish.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo tutorial [¶](\#marimo-tutorial "Permanent link")

Open a tutorial.

marimo is a powerful library for making reactive notebooks
and apps. To get the most out of marimo, get started with a few
tutorials, starting with the intro:

Recommended sequence:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

Back to top

# marimo [¶](\#marimo "Permanent link")

Welcome to marimo!
Getting started:

- marimo tutorial intro

Example usage:

- marimo edit create or edit notebooks
- marimo edit notebook.py create or edit a notebook called notebook.py
- marimo run notebook.py run a notebook as a read-only app
- marimo tutorial --help list tutorials

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--version` | boolean | Show the version and exit. | `False` |
| `-l`, `--log-level` | choice ( `DEBUG` \| `INFO` \| `WARN` \| `ERROR` \| `CRITICAL`) | Choose logging level. | `WARN` |
| `-q`, `--quiet` | boolean | Suppress standard out. | `False` |
| `-y`, `--yes` | boolean | Automatic yes to prompts, running non-interactively. | `False` |
| `-d`, `--development-mode` | boolean | Run in development mode; enables debug logs and server autoreload. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo check [¶](\#marimo-check "Permanent link")

Check and format marimo files.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--fix` | boolean | Whether to in place update files. | `False` |
| `--strict` | boolean | Whether warnings return a non-zero exit code. | `False` |
| `-v`, `--verbose` / `-q`, `--quiet` | boolean | Whether to print detailed messages. | `True` |
| `--unsafe-fixes` | boolean | Enable fixes that may change code behavior (e.g., removing empty cells). | `False` |
| `--ignore-scripts` | boolean | Ignore files that are not recognizable as marimo notebooks. | `False` |
| `--format` | choice ( `full` \| `json`) | Output format for diagnostics. | `full` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo config [¶](\#marimo-config "Permanent link")

Various commands for the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo config describe [¶](\#marimo-config-describe "Permanent link")

Describe the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo config show [¶](\#marimo-config-show "Permanent link")

Show the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo convert [¶](\#marimo-convert "Permanent link")

Convert a Jupyter notebook, Markdown file, or Python script to a marimo notebook.

Supported input formats:
\- `.ipynb` (local or GitHub-hosted)
\- `.md` files with `{python}` code fences
\- `.py` scripts in py:percent format

Behavior:
\- Jupyter notebooks: outputs are stripped.

- Markdown files: only `{python}` fenced code blocks are converted.

Example:

\- Python scripts:
\- If already a valid marimo notebook, no conversion is performed.
\- Otherwise, marimo attempts to convert using py:percent formatting,
preserving top-level comments and docstrings.

Example usage:

or

or

After conversion:

Note:
Since marimo's reactive execution differs from traditional notebooks,
you may need to refactor code that mutates variables across cells
(e.g., modifying a dataframe in multiple cells), which can lead to
unexpected behavior.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-o`, `--output` | path | Output file to save the converted notebook to. If not provided, the converted notebook will be printed to stdout. | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo edit [¶](\#marimo-edit "Permanent link")

Create or edit notebooks.

- marimo edit Start the marimo notebook server
- marimo edit notebook.py Create or edit notebook.py

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--allow-origins` | text | Allowed origins for CORS. Can be repeated. Use \* for all origins. | None |
| `--skip-update-check` | boolean | Don't check if a new version of marimo is available for download. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--watch` | boolean | Watch the file for changes and reload the code when saved in another editor. | `False` |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--timeout` | float | Enable a global timeout to shut down the server after specified number of minutes of no connection | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo env [¶](\#marimo-env "Permanent link")

Print out environment information for debugging purposes.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo export [¶](\#marimo-export "Permanent link")

Export a notebook to various formats.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export html [¶](\#marimo-export-html "Permanent link")

Run a notebook and export it as an HTML file.

Example:

Optionally pass CLI args to the notebook:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--include-code` / `--no-include-code` | boolean | Include notebook code in the exported HTML file. | `True` |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the HTML to. If not provided, the HTML will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export html-wasm [¶](\#marimo-export-html-wasm "Permanent link")

Export a notebook as a WASM-powered standalone HTML file.

Example:

The exported HTML file will run the notebook using WebAssembly, making it
completely self-contained and executable in the browser. This lets you
share interactive notebooks on the web without setting up
infrastructure to run Python code.

The exported notebook runs using Pyodide, which supports most
but not all Python packages. To learn more, see the Pyodide
documentation.

In order for this file to be able to run, it must be served over HTTP,
and cannot be opened directly from the file system (e.g. file://).

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-o`, `--output` | path | Output directory to save the HTML to. | `Sentinel.UNSET` |
| `--mode` | choice ( `edit` \| `run`) | Whether the notebook code should be editable or readonly. | `run` |
| `--watch` / `--no-watch` | boolean | Whether to watch the original file and export upon change | `False` |
| `--show-code` / `--no-show-code` | boolean | Whether to show code by default in the exported HTML file; only relevant for run mode. | `False` |
| `--include-cloudflare` / `--no-include-cloudflare` | boolean | Whether to include Cloudflare Worker configuration files (index.js and wrangler.jsonc) for easy deployment. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export ipynb [¶](\#marimo-export-ipynb "Permanent link")

Export a marimo notebook as a Jupyter notebook in topological order.

Example:

Watch for changes and regenerate the script on modification:

Requires nbformat to be installed.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--sort` | choice ( `top-down` \| `topological`) | Sort cells top-down or in topological order. | `topological` |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the ipynb file to. If not provided, the ipynb contents will be printed to stdout. | None |
| `--include-outputs` / `--no-include-outputs` | boolean | Run the notebook and include outputs in the exported ipynb file. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export md [¶](\#marimo-export-md "Permanent link")

Export a marimo notebook as a code fenced Markdown file.

Example:

Watch for changes and regenerate the script on modification:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the markdown to. If not provided, markdown will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export script [¶](\#marimo-export-script "Permanent link")

Export a marimo notebook as a flat script, in topological order.

Example:

Watch for changes and regenerate the script on modification:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the script to. If not provided, the script will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo new [¶](\#marimo-new "Permanent link")

Create an empty notebook, or generate from a prompt with AI

- marimo new Create an empty notebook
- marimo new "Plot an interactive 3D surface with matplotlib." Generate a notebook from a prompt.
- marimo new prompt.txt Generate a notebook from a file containing a prompt.

Visit [https://marimo.app/ai](https://marimo.app/ai) for more prompt examples.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--timeout` | float | Enable a global timeout to shut down the server after specified number of minutes of no connection | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo recover [¶](\#marimo-recover "Permanent link")

Recover a marimo notebook from JSON.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo run [¶](\#marimo-run "Permanent link")

Run a notebook as an app in read-only mode.

If NAME is a url, the notebook will be downloaded to a temporary file.

Example:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `False` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--include-code` | boolean | Include notebook code in the app. | `False` |
| `--session-ttl` | integer | Seconds to wait before closing a session on websocket disconnect. | `120` |
| `--watch` | boolean | Watch the file for changes and reload the app. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--allow-origins` | text | Allowed origins for CORS. Can be repeated. | None |
| `--redirect-console-to-browser` | boolean | Redirect console logs to the browser console. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--check` / `--no-check` | boolean | Disable a static check of the notebook before running. | `True` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo shell-completion [¶](\#marimo-shell-completion "Permanent link")

Install shell completions for marimo. Supports bash, zsh, and fish.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo tutorial [¶](\#marimo-tutorial "Permanent link")

Open a tutorial.

marimo is a powerful library for making reactive notebooks
and apps. To get the most out of marimo, get started with a few
tutorials, starting with the intro:

Recommended sequence:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

 Back to top

# marimo [¶](\#marimo "Permanent link")

Welcome to marimo!
Getting started:

- marimo tutorial intro

Example usage:

- marimo edit create or edit notebooks
- marimo edit notebook.py create or edit a notebook called notebook.py
- marimo run notebook.py run a notebook as a read-only app
- marimo tutorial --help list tutorials

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--version` | boolean | Show the version and exit. | `False` |
| `-l`, `--log-level` | choice ( `DEBUG` \| `INFO` \| `WARN` \| `ERROR` \| `CRITICAL`) | Choose logging level. | `WARN` |
| `-q`, `--quiet` | boolean | Suppress standard out. | `False` |
| `-y`, `--yes` | boolean | Automatic yes to prompts, running non-interactively. | `False` |
| `-d`, `--development-mode` | boolean | Run in development mode; enables debug logs and server autoreload. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo check [¶](\#marimo-check "Permanent link")

Check and format marimo files.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--fix` | boolean | Whether to in place update files. | `False` |
| `--strict` | boolean | Whether warnings return a non-zero exit code. | `False` |
| `-v`, `--verbose` / `-q`, `--quiet` | boolean | Whether to print detailed messages. | `True` |
| `--unsafe-fixes` | boolean | Enable fixes that may change code behavior (e.g., removing empty cells). | `False` |
| `--ignore-scripts` | boolean | Ignore files that are not recognizable as marimo notebooks. | `False` |
| `--format` | choice ( `full` \| `json`) | Output format for diagnostics. | `full` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo config [¶](\#marimo-config "Permanent link")

Various commands for the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo config describe [¶](\#marimo-config-describe "Permanent link")

Describe the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo config show [¶](\#marimo-config-show "Permanent link")

Show the marimo config.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo convert [¶](\#marimo-convert "Permanent link")

Convert a Jupyter notebook, Markdown file, or Python script to a marimo notebook.

Supported input formats:
\- `.ipynb` (local or GitHub-hosted)
\- `.md` files with `{python}` code fences
\- `.py` scripts in py:percent format

Behavior:
\- Jupyter notebooks: outputs are stripped.

- Markdown files: only `{python}` fenced code blocks are converted.

Example:

\- Python scripts:
 \- If already a valid marimo notebook, no conversion is performed.
 \- Otherwise, marimo attempts to convert using py:percent formatting,
 preserving top-level comments and docstrings.

Example usage:

or

or

After conversion:

Note:
Since marimo's reactive execution differs from traditional notebooks,
you may need to refactor code that mutates variables across cells
(e.g., modifying a dataframe in multiple cells), which can lead to
unexpected behavior.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-o`, `--output` | path | Output file to save the converted notebook to. If not provided, the converted notebook will be printed to stdout. | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo edit [¶](\#marimo-edit "Permanent link")

Create or edit notebooks.

- marimo edit Start the marimo notebook server
- marimo edit notebook.py Create or edit notebook.py

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--allow-origins` | text | Allowed origins for CORS. Can be repeated. Use \* for all origins. | None |
| `--skip-update-check` | boolean | Don't check if a new version of marimo is available for download. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--watch` | boolean | Watch the file for changes and reload the code when saved in another editor. | `False` |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--timeout` | float | Enable a global timeout to shut down the server after specified number of minutes of no connection | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo env [¶](\#marimo-env "Permanent link")

Print out environment information for debugging purposes.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo export [¶](\#marimo-export "Permanent link")

Export a notebook to various formats.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export html [¶](\#marimo-export-html "Permanent link")

Run a notebook and export it as an HTML file.

Example:

Optionally pass CLI args to the notebook:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--include-code` / `--no-include-code` | boolean | Include notebook code in the exported HTML file. | `True` |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the HTML to. If not provided, the HTML will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export html-wasm [¶](\#marimo-export-html-wasm "Permanent link")

Export a notebook as a WASM-powered standalone HTML file.

Example:

The exported HTML file will run the notebook using WebAssembly, making it
completely self-contained and executable in the browser. This lets you
share interactive notebooks on the web without setting up
infrastructure to run Python code.

The exported notebook runs using Pyodide, which supports most
but not all Python packages. To learn more, see the Pyodide
documentation.

In order for this file to be able to run, it must be served over HTTP,
and cannot be opened directly from the file system (e.g. file://).

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-o`, `--output` | path | Output directory to save the HTML to. | `Sentinel.UNSET` |
| `--mode` | choice ( `edit` \| `run`) | Whether the notebook code should be editable or readonly. | `run` |
| `--watch` / `--no-watch` | boolean | Whether to watch the original file and export upon change | `False` |
| `--show-code` / `--no-show-code` | boolean | Whether to show code by default in the exported HTML file; only relevant for run mode. | `False` |
| `--include-cloudflare` / `--no-include-cloudflare` | boolean | Whether to include Cloudflare Worker configuration files (index.js and wrangler.jsonc) for easy deployment. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export ipynb [¶](\#marimo-export-ipynb "Permanent link")

Export a marimo notebook as a Jupyter notebook in topological order.

Example:

Watch for changes and regenerate the script on modification:

Requires nbformat to be installed.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--sort` | choice ( `top-down` \| `topological`) | Sort cells top-down or in topological order. | `topological` |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the ipynb file to. If not provided, the ipynb contents will be printed to stdout. | None |
| `--include-outputs` / `--no-include-outputs` | boolean | Run the notebook and include outputs in the exported ipynb file. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export md [¶](\#marimo-export-md "Permanent link")

Export a marimo notebook as a code fenced Markdown file.

Example:

Watch for changes and regenerate the script on modification:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the markdown to. If not provided, markdown will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

### marimo export script [¶](\#marimo-export-script "Permanent link")

Export a marimo notebook as a flat script, in topological order.

Example:

Watch for changes and regenerate the script on modification:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--watch` / `--no-watch` | boolean | Watch notebook for changes and regenerate the output on modification. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `-o`, `--output` | path | Output file to save the script to. If not provided, the script will be printed to stdout. | None |
| `--sandbox` / `--no-sandbox` | boolean | Run the command in an isolated virtual environment using `uv run --isolated`. Requires `uv`. | None |
| `-f`, `--force` | boolean | Force overwrite of the output file if it already exists. | `False` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo new [¶](\#marimo-new "Permanent link")

Create an empty notebook, or generate from a prompt with AI

- marimo new Create an empty notebook
- marimo new "Plot an interactive 3D surface with matplotlib." Generate a notebook from a prompt.
- marimo new prompt.txt Generate a notebook from a file containing a prompt.

Visit [https://marimo.app/ai](https://marimo.app/ai) for more prompt examples.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--timeout` | float | Enable a global timeout to shut down the server after specified number of minutes of no connection | None |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo recover [¶](\#marimo-recover "Permanent link")

Recover a marimo notebook from JSON.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo run [¶](\#marimo-run "Permanent link")

Run a notebook as an app in read-only mode.

If NAME is a url, the notebook will be downloaded to a temporary file.

Example:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `False` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--include-code` | boolean | Include notebook code in the app. | `False` |
| `--session-ttl` | integer | Seconds to wait before closing a session on websocket disconnect. | `120` |
| `--watch` | boolean | Watch the file for changes and reload the app. If watchdog is installed, it will be used to watch the file. Otherwise, file watcher will poll the file every 1s. | `False` |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `--base-url` | text | Base URL for the server. Should start with a /. | \`\` |
| `--allow-origins` | text | Allowed origins for CORS. Can be repeated. | None |
| `--redirect-console-to-browser` | boolean | Redirect console logs to the browser console. | `False` |
| `--sandbox` / `--no-sandbox` | boolean | Run the notebook in an isolated environment, with dependencies tracked via PEP 723 inline metadata. If already declared, dependencies will install automatically. Requires uv. | None |
| `--check` / `--no-check` | boolean | Disable a static check of the notebook before running. | `True` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo shell-completion [¶](\#marimo-shell-completion "Permanent link")

Install shell completions for marimo. Supports bash, zsh, and fish.

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |

## marimo tutorial [¶](\#marimo-tutorial "Permanent link")

Open a tutorial.

marimo is a powerful library for making reactive notebooks
and apps. To get the most out of marimo, get started with a few
tutorials, starting with the intro:

Recommended sequence:

**Usage:**

**Options:**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `-p`, `--port` | integer | Port to attach to. | None |
| `--host` | text | Host to attach to. | `127.0.0.1` |
| `--proxy` | text | Address of reverse proxy. | None |
| `--headless` | boolean | Don't launch a browser. | `False` |
| `--token` / `--no-token` | boolean | Use a token for authentication. This enables session-based authentication. A random token will be generated if --token-password is not set. If --no-token is set, session-based authentication will not be used. | `True` |
| `--token-password` | text | Use a specific token for authentication. This enables session-based authentication. A random token will be generated if not set. | None |
| `--token-password-file` | text | Path to file containing token password, or '-' for stdin. Mutually exclusive with --token-password. | None |
| `--skew-protection` / `--no-skew-protection` | boolean | Enable skew protection middleware to prevent version mismatch issues. | `True` |
| `-h`, `--help` | boolean | Show this message and exit. | `False` |


---

###### Programmatic - marimoRunning the marimo backend programmatically¶

*Source: [https://docs.marimo.io/guides/deploying/programmatically/](https://docs.marimo.io/guides/deploying/programmatically/)*

<!-- Source: https://docs.marimo.io/guides/deploying/programmatically/ -->
<!-- Title: Programmatic - marimoRunning the marimo backend programmatically¶ -->
<!-- Depth: 5 -->

[Skip to content](#running-the-marimo-backend-programmatically)

# Running the marimo backend programmatically [¶](\#running-the-marimo-backend-programmatically "Permanent link")

marimo can be run programmatically using the `marimo` module. This is useful when you want to run marimo as part of a larger application or when you need to customize the behavior of marimo (e.g. middleware, custom error handling, authentication, routing, etc).

## FastAPI Example [¶](\#fastapi-example "Permanent link")

Here's an example of how you can run a marimo application programmatically using FastAPI:

For a more complete example, see the [FastAPI example](https://github.com/marimo-team/marimo/tree/main/examples/frameworks/fastapi).

Note that when run in this mode, marimo will serve its static assets under the name of the notebook (in the example above, that would be `http://hostname/<dashboard|sales>/assets/<assetname.css|js|...>`). If you are using custom authorization middleware, skip authentication for these assets to avoid server round-trips. There are _many_ of them.

## Dynamic directory [¶](\#dynamic-directory "Permanent link")

If you'd like to create a server to dynamically load marimo notebooks from a directory, you can use the `with_dynamic_directory` method. This is useful if the contents of the directory change often, such as a directory of notebooks for a dashboard, without restarting the server.

If the notebooks in the directory are expected to be static, it is better to use the `with_app` method and loop through the directory contents.

## Accessing Request Data [¶](\#accessing-request-data "Permanent link")

Inside your marimo notebooks, you can access the current request data using `mo.app_meta().request`. This is particularly useful when implementing authentication or accessing user data.

### Authentication Middleware Example [¶](\#authentication-middleware-example "Permanent link")

Here's an example of how to implement authentication middleware that populates `request.user`:

The `request` object provides access to:

- `request.headers`: Request headers
- `request.cookies`: Request cookies
- `request.query_params`: Query parameters
- `request.path_params`: Path parameters
- `request.user`: User data added by authentication middleware
- `request.url`: URL information including path, query parameters
- `request.meta`: Metadata added by your custom middleware

### Documenting and Validating Query Parameters [¶](\#documenting-and-validating-query-parameters "Permanent link")

When mounted apps accept [query parameters](../../../api/query_params/), it can be helpful to declare, validate, and document them with the help of a [Pydantic model](https://fastapi.tiangolo.com/tutorial/query-param-models/).

If a marimo app called `notebooks/items.py` is mounted to `/items`, declaring an endpoint with the same route will take the query parameters through Pydantic model validation first, then redirect to the marimo endpoint.

## Under the Hood [¶](\#under-the-hood "Permanent link")

Behind the scenes, in this mode, marimo is spinning up a new computational kernel
in a separate sub-thread (same process) for each new session / app created.
There are a few implications of this from a performance and reliability perspective:

- If you are running multiple instances of this same server for load balancing, you will need to use sticky sessions in your load balancer to ensure that the same client gets the same kernel each time.
- Similarly, attempting to run multiple instances of the same FastAPI
process (a common approach with Python web services) on the same node will not work reliably, since only one of them will actually be running the kernel.

In summary, there are limitations to how far the approach described here can horizontally scale, so we recommend scaling vertically first. In other words, increase the container CPU/Memory specs before increasing the number of container instances.

Back to top

# Running the marimo backend programmatically [¶](\#running-the-marimo-backend-programmatically "Permanent link")

marimo can be run programmatically using the `marimo` module. This is useful when you want to run marimo as part of a larger application or when you need to customize the behavior of marimo (e.g. middleware, custom error handling, authentication, routing, etc).

## FastAPI Example [¶](\#fastapi-example "Permanent link")

Here's an example of how you can run a marimo application programmatically using FastAPI:

For a more complete example, see the [FastAPI example](https://github.com/marimo-team/marimo/tree/main/examples/frameworks/fastapi).

Note that when run in this mode, marimo will serve its static assets under the name of the notebook (in the example above, that would be `http://hostname/<dashboard|sales>/assets/<assetname.css|js|...>`). If you are using custom authorization middleware, skip authentication for these assets to avoid server round-trips. There are _many_ of them.

## Dynamic directory [¶](\#dynamic-directory "Permanent link")

If you'd like to create a server to dynamically load marimo notebooks from a directory, you can use the `with_dynamic_directory` method. This is useful if the contents of the directory change often, such as a directory of notebooks for a dashboard, without restarting the server.

If the notebooks in the directory are expected to be static, it is better to use the `with_app` method and loop through the directory contents.

## Accessing Request Data [¶](\#accessing-request-data "Permanent link")

Inside your marimo notebooks, you can access the current request data using `mo.app_meta().request`. This is particularly useful when implementing authentication or accessing user data.

### Authentication Middleware Example [¶](\#authentication-middleware-example "Permanent link")

Here's an example of how to implement authentication middleware that populates `request.user`:

The `request` object provides access to:

- `request.headers`: Request headers
- `request.cookies`: Request cookies
- `request.query_params`: Query parameters
- `request.path_params`: Path parameters
- `request.user`: User data added by authentication middleware
- `request.url`: URL information including path, query parameters
- `request.meta`: Metadata added by your custom middleware

### Documenting and Validating Query Parameters [¶](\#documenting-and-validating-query-parameters "Permanent link")

When mounted apps accept [query parameters](../../../api/query_params/), it can be helpful to declare, validate, and document them with the help of a [Pydantic model](https://fastapi.tiangolo.com/tutorial/query-param-models/).

If a marimo app called `notebooks/items.py` is mounted to `/items`, declaring an endpoint with the same route will take the query parameters through Pydantic model validation first, then redirect to the marimo endpoint.

## Under the Hood [¶](\#under-the-hood "Permanent link")

Behind the scenes, in this mode, marimo is spinning up a new computational kernel
in a separate sub-thread (same process) for each new session / app created.
There are a few implications of this from a performance and reliability perspective:

- If you are running multiple instances of this same server for load balancing, you will need to use sticky sessions in your load balancer to ensure that the same client gets the same kernel each time.
- Similarly, attempting to run multiple instances of the same FastAPI
process (a common approach with Python web services) on the same node will not work reliably, since only one of them will actually be running the kernel.

In summary, there are limitations to how far the approach described here can horizontally scale, so we recommend scaling vertically first. In other words, increase the container CPU/Memory specs before increasing the number of container instances.

 Back to top

# Running the marimo backend programmatically [¶](\#running-the-marimo-backend-programmatically "Permanent link")

marimo can be run programmatically using the `marimo` module. This is useful when you want to run marimo as part of a larger application or when you need to customize the behavior of marimo (e.g. middleware, custom error handling, authentication, routing, etc).

## FastAPI Example [¶](\#fastapi-example "Permanent link")

Here's an example of how you can run a marimo application programmatically using FastAPI:

For a more complete example, see the [FastAPI example](https://github.com/marimo-team/marimo/tree/main/examples/frameworks/fastapi).

Note that when run in this mode, marimo will serve its static assets under the name of the notebook (in the example above, that would be `http://hostname/<dashboard|sales>/assets/<assetname.css|js|...>`). If you are using custom authorization middleware, skip authentication for these assets to avoid server round-trips. There are _many_ of them.

## Dynamic directory [¶](\#dynamic-directory "Permanent link")

If you'd like to create a server to dynamically load marimo notebooks from a directory, you can use the `with_dynamic_directory` method. This is useful if the contents of the directory change often, such as a directory of notebooks for a dashboard, without restarting the server.

If the notebooks in the directory are expected to be static, it is better to use the `with_app` method and loop through the directory contents.

## Accessing Request Data [¶](\#accessing-request-data "Permanent link")

Inside your marimo notebooks, you can access the current request data using `mo.app_meta().request`. This is particularly useful when implementing authentication or accessing user data.

### Authentication Middleware Example [¶](\#authentication-middleware-example "Permanent link")

Here's an example of how to implement authentication middleware that populates `request.user`:

The `request` object provides access to:

- `request.headers`: Request headers
- `request.cookies`: Request cookies
- `request.query_params`: Query parameters
- `request.path_params`: Path parameters
- `request.user`: User data added by authentication middleware
- `request.url`: URL information including path, query parameters
- `request.meta`: Metadata added by your custom middleware

### Documenting and Validating Query Parameters [¶](\#documenting-and-validating-query-parameters "Permanent link")

When mounted apps accept [query parameters](../../../api/query_params/), it can be helpful to declare, validate, and document them with the help of a [Pydantic model](https://fastapi.tiangolo.com/tutorial/query-param-models/).

If a marimo app called `notebooks/items.py` is mounted to `/items`, declaring an endpoint with the same route will take the query parameters through Pydantic model validation first, then redirect to the marimo endpoint.

## Under the Hood [¶](\#under-the-hood "Permanent link")

Behind the scenes, in this mode, marimo is spinning up a new computational kernel
in a separate sub-thread (same process) for each new session / app created.
There are a few implications of this from a performance and reliability perspective:

- If you are running multiple instances of this same server for load balancing, you will need to use sticky sessions in your load balancer to ensure that the same client gets the same kernel each time.
- Similarly, attempting to run multiple instances of the same FastAPI
   process (a common approach with Python web services) on the same node will not work reliably, since only one of them will actually be running the kernel.

In summary, there are limitations to how far the approach described here can horizontally scale, so we recommend scaling vertically first. In other words, increase the container CPU/Memory specs before increasing the number of container instances.


---

###### Expensive notebooks - marimoWorking with expensive notebooks¶

*Source: [https://docs.marimo.io/guides/expensive_notebooks/](https://docs.marimo.io/guides/expensive_notebooks/)*

<!-- Source: https://docs.marimo.io/guides/expensive_notebooks/ -->
<!-- Title: Expensive notebooks - marimoWorking with expensive notebooks¶ -->
<!-- Depth: 5 -->

[Skip to content](#working-with-expensive-notebooks)

# Working with expensive notebooks [¶](\#working-with-expensive-notebooks "Permanent link")

marimo provides tools to control when cells run. Use these tools to
prevent expensive cells, which may call APIs or take a long time to run, from
accidentally running.

## Stop execution with `mo.stop` [¶](\#stop-execution-with-mostop "Permanent link")

Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to stop a cell from executing if a condition
is met:

Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") with
[`mo.ui.run_button()`](../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to require a button press for
expensive cells:

## Configure how marimo runs cells [¶](\#configure-how-marimo-runs-cells "Permanent link")

### Disable cell autorun [¶](\#disable-cell-autorun "Permanent link")

If you habitually work with very expensive notebooks, you can
[disable automatic\
execution](../configuration/runtime_configuration/#disable-autorun-on-cell-change-lazy-execution). When
automatic execution is disabled, when you run a cell, marimo
marks dependent cells as stale instead of running them automatically.

### Disable autorun on startup [¶](\#disable-autorun-on-startup "Permanent link")

marimo autoruns notebooks on startup, with `marimo edit notebook.py` behaving
analogously to `python notebook.py`. This can also be disabled through the
[notebook settings](../configuration/runtime_configuration/#on-startup).

### Disable individual cells [¶](\#disable-individual-cells "Permanent link")

marimo lets you temporarily disable cells from automatically running. This is
helpful when you want to edit one part of a notebook without triggering
execution of other parts. See the
[reactivity guide](../reactivity/#disabling-cells) for more info.

## Manage memory [¶](\#manage-memory "Permanent link")

Here are a few tips for managing the memory consumption of your notebooks,
on host or GPU.

### Wrap intermediate computations in functions [¶](\#wrap-intermediate-computations-in-functions "Permanent link")

By default, global variables live in the kernel memory. Intermediate variables
that are defined in functions are cleaned up automatically.

For example, if `X` is a temporary:

**Do this:**

**Don't do this:**

### Use `del` to remove variables from kernel memory [¶](\#use-del-to-remove-variables-from-kernel-memory "Permanent link")

Use the `del` operator to remove variables from kernel memory.

**In a single cell.** Prefer deleting variables in the cell they were defined
in. For example,
if `X` is a temporary that you don't need after computing `Y`:

**In another cell.** Sometimes, computations are spread across multiple cells,
and you only realize later on that you need to free memory that you've already
allocated. In such cases you can still use the `del` keyword. For example:

marimo inserts control dependences to make sure that variables are not deleted
before they are used. When `del` is used to delete a variable that was defined
in a another cell, the cell where `del` was used becomes a child of all other
cells that reference that variable. In this case, that means marimo knows to
run the third cell after the second cell, since the second cell references
`data` and the third cell deletes it. However, once `data` is deleted,
attempting to manually run the second cell will raise a `NameError`, and you'll
need to re-run the defining cell in order to get your notebook back to a
consistent state.

## Automatically snapshot outputs as HTML or IPYNB [¶](\#automatically-snapshot-outputs-as-html-or-ipynb "Permanent link")

To keep a record of your cell outputs while working on your
notebook, you can configure notebooks to automatically save as HTML or ipynb
through the notebook menu (these files are saved in addition to the
notebook's `.py` file). Snapshots are saved to a folder called
`__marimo__` in the notebook directory.

Learn more about exporting notebooks in our [exporting guide](../exporting/).

## Cache expensive computations [¶](\#cache-expensive-computations "Permanent link")

marimo provides two decorators to cache the return values of expensive functions:

1. In-memory caching with [`mo.cache`](../../api/caching/#marimo.cache "            marimo.cache")
2. Disk caching with [`mo.persistent_cache`](../../api/caching/#marimo.persistent_cache "            marimo.persistent_cache")

Both utilities can be used as decorators or context managers.

`mo.cache` `mo.persistent_cache`

See our [guide on caching](../../api/caching/) for details, including how the cache
key is constructed, and limitations.

## Lazy-load expensive UIs [¶](\#lazy-load-expensive-uis "Permanent link")

Lazily render UI elements that are expensive to compute using
`marimo.lazy`.

For example,

In this example, `mo.ui.table(data)` will not be rendered on the frontend until is it in the viewport.
For example, an element can be out of the viewport due to scroll, inside a tab that is not selected, or inside an accordion that is not open.

However, in this example, data is eagerly computed, while only the rendering of the table is lazy. It is possible to lazily compute the data as well: see the next example.

In this example, we pass a function to `mo.lazy` instead of a component. This
function will only be called when the user opens the accordion. In this way,
`expensive_component` lazily computed and we only query the database when the
user needs to see the data. This can be useful when the data is expensive to
compute and the user may not need to see it immediately.

Back to top

# Working with expensive notebooks [¶](\#working-with-expensive-notebooks "Permanent link")

marimo provides tools to control when cells run. Use these tools to
prevent expensive cells, which may call APIs or take a long time to run, from
accidentally running.

## Stop execution with `mo.stop` [¶](\#stop-execution-with-mostop "Permanent link")

Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to stop a cell from executing if a condition
is met:

Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") with
[`mo.ui.run_button()`](../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to require a button press for
expensive cells:

## Configure how marimo runs cells [¶](\#configure-how-marimo-runs-cells "Permanent link")

### Disable cell autorun [¶](\#disable-cell-autorun "Permanent link")

If you habitually work with very expensive notebooks, you can
[disable automatic\
execution](../configuration/runtime_configuration/#disable-autorun-on-cell-change-lazy-execution). When
automatic execution is disabled, when you run a cell, marimo
marks dependent cells as stale instead of running them automatically.

### Disable autorun on startup [¶](\#disable-autorun-on-startup "Permanent link")

marimo autoruns notebooks on startup, with `marimo edit notebook.py` behaving
analogously to `python notebook.py`. This can also be disabled through the
[notebook settings](../configuration/runtime_configuration/#on-startup).

### Disable individual cells [¶](\#disable-individual-cells "Permanent link")

marimo lets you temporarily disable cells from automatically running. This is
helpful when you want to edit one part of a notebook without triggering
execution of other parts. See the
[reactivity guide](../reactivity/#disabling-cells) for more info.

## Manage memory [¶](\#manage-memory "Permanent link")

Here are a few tips for managing the memory consumption of your notebooks,
on host or GPU.

### Wrap intermediate computations in functions [¶](\#wrap-intermediate-computations-in-functions "Permanent link")

By default, global variables live in the kernel memory. Intermediate variables
that are defined in functions are cleaned up automatically.

For example, if `X` is a temporary:

**Do this:**

**Don't do this:**

### Use `del` to remove variables from kernel memory [¶](\#use-del-to-remove-variables-from-kernel-memory "Permanent link")

Use the `del` operator to remove variables from kernel memory.

**In a single cell.** Prefer deleting variables in the cell they were defined
in. For example,
if `X` is a temporary that you don't need after computing `Y`:

**In another cell.** Sometimes, computations are spread across multiple cells,
and you only realize later on that you need to free memory that you've already
allocated. In such cases you can still use the `del` keyword. For example:

marimo inserts control dependences to make sure that variables are not deleted
before they are used. When `del` is used to delete a variable that was defined
in a another cell, the cell where `del` was used becomes a child of all other
cells that reference that variable. In this case, that means marimo knows to
run the third cell after the second cell, since the second cell references
`data` and the third cell deletes it. However, once `data` is deleted,
attempting to manually run the second cell will raise a `NameError`, and you'll
need to re-run the defining cell in order to get your notebook back to a
consistent state.

## Automatically snapshot outputs as HTML or IPYNB [¶](\#automatically-snapshot-outputs-as-html-or-ipynb "Permanent link")

To keep a record of your cell outputs while working on your
notebook, you can configure notebooks to automatically save as HTML or ipynb
through the notebook menu (these files are saved in addition to the
notebook's `.py` file). Snapshots are saved to a folder called
`__marimo__` in the notebook directory.

Learn more about exporting notebooks in our [exporting guide](../exporting/).

## Cache expensive computations [¶](\#cache-expensive-computations "Permanent link")

marimo provides two decorators to cache the return values of expensive functions:

1. In-memory caching with [`mo.cache`](../../api/caching/#marimo.cache "            marimo.cache")
2. Disk caching with [`mo.persistent_cache`](../../api/caching/#marimo.persistent_cache "            marimo.persistent_cache")

Both utilities can be used as decorators or context managers.

`mo.cache` `mo.persistent_cache`

See our [guide on caching](../../api/caching/) for details, including how the cache
key is constructed, and limitations.

## Lazy-load expensive UIs [¶](\#lazy-load-expensive-uis "Permanent link")

Lazily render UI elements that are expensive to compute using
`marimo.lazy`.

For example,

In this example, `mo.ui.table(data)` will not be rendered on the frontend until is it in the viewport.
For example, an element can be out of the viewport due to scroll, inside a tab that is not selected, or inside an accordion that is not open.

However, in this example, data is eagerly computed, while only the rendering of the table is lazy. It is possible to lazily compute the data as well: see the next example.

In this example, we pass a function to `mo.lazy` instead of a component. This
function will only be called when the user opens the accordion. In this way,
`expensive_component` lazily computed and we only query the database when the
user needs to see the data. This can be useful when the data is expensive to
compute and the user may not need to see it immediately.

 Back to top

# Working with expensive notebooks [¶](\#working-with-expensive-notebooks "Permanent link")

marimo provides tools to control when cells run. Use these tools to
prevent expensive cells, which may call APIs or take a long time to run, from
accidentally running.

## Stop execution with `mo.stop` [¶](\#stop-execution-with-mostop "Permanent link")

Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to stop a cell from executing if a condition
is met:

Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") with
[`mo.ui.run_button()`](../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to require a button press for
expensive cells:

## Configure how marimo runs cells [¶](\#configure-how-marimo-runs-cells "Permanent link")

### Disable cell autorun [¶](\#disable-cell-autorun "Permanent link")

If you habitually work with very expensive notebooks, you can
[disable automatic\
execution](../configuration/runtime_configuration/#disable-autorun-on-cell-change-lazy-execution). When
automatic execution is disabled, when you run a cell, marimo
marks dependent cells as stale instead of running them automatically.

### Disable autorun on startup [¶](\#disable-autorun-on-startup "Permanent link")

marimo autoruns notebooks on startup, with `marimo edit notebook.py` behaving
analogously to `python notebook.py`. This can also be disabled through the
[notebook settings](../configuration/runtime_configuration/#on-startup).

### Disable individual cells [¶](\#disable-individual-cells "Permanent link")

marimo lets you temporarily disable cells from automatically running. This is
helpful when you want to edit one part of a notebook without triggering
execution of other parts. See the
[reactivity guide](../reactivity/#disabling-cells) for more info.

## Manage memory [¶](\#manage-memory "Permanent link")

Here are a few tips for managing the memory consumption of your notebooks,
on host or GPU.

### Wrap intermediate computations in functions [¶](\#wrap-intermediate-computations-in-functions "Permanent link")

By default, global variables live in the kernel memory. Intermediate variables
that are defined in functions are cleaned up automatically.

For example, if `X` is a temporary:

**Do this:**

**Don't do this:**

### Use `del` to remove variables from kernel memory [¶](\#use-del-to-remove-variables-from-kernel-memory "Permanent link")

Use the `del` operator to remove variables from kernel memory.

**In a single cell.** Prefer deleting variables in the cell they were defined
in. For example,
if `X` is a temporary that you don't need after computing `Y`:

**In another cell.** Sometimes, computations are spread across multiple cells,
and you only realize later on that you need to free memory that you've already
allocated. In such cases you can still use the `del` keyword. For example:

marimo inserts control dependences to make sure that variables are not deleted
before they are used. When `del` is used to delete a variable that was defined
in a another cell, the cell where `del` was used becomes a child of all other
cells that reference that variable. In this case, that means marimo knows to
run the third cell after the second cell, since the second cell references
`data` and the third cell deletes it. However, once `data` is deleted,
attempting to manually run the second cell will raise a `NameError`, and you'll
need to re-run the defining cell in order to get your notebook back to a
consistent state.

## Automatically snapshot outputs as HTML or IPYNB [¶](\#automatically-snapshot-outputs-as-html-or-ipynb "Permanent link")

To keep a record of your cell outputs while working on your
notebook, you can configure notebooks to automatically save as HTML or ipynb
through the notebook menu (these files are saved in addition to the
notebook's `.py` file). Snapshots are saved to a folder called
`__marimo__` in the notebook directory.

Learn more about exporting notebooks in our [exporting guide](../exporting/).

## Cache expensive computations [¶](\#cache-expensive-computations "Permanent link")

marimo provides two decorators to cache the return values of expensive functions:

1. In-memory caching with [`mo.cache`](../../api/caching/#marimo.cache "            marimo.cache")
2. Disk caching with [`mo.persistent_cache`](../../api/caching/#marimo.persistent_cache "            marimo.persistent_cache")

Both utilities can be used as decorators or context managers.

`mo.cache` `mo.persistent_cache`

See our [guide on caching](../../api/caching/) for details, including how the cache
key is constructed, and limitations.

## Lazy-load expensive UIs [¶](\#lazy-load-expensive-uis "Permanent link")

Lazily render UI elements that are expensive to compute using
`marimo.lazy`.

For example,

In this example, `mo.ui.table(data)` will not be rendered on the frontend until is it in the viewport.
For example, an element can be out of the viewport due to scroll, inside a tab that is not selected, or inside an accordion that is not open.

However, in this example, data is eagerly computed, while only the rendering of the table is lazy. It is possible to lazily compute the data as well: see the next example.

In this example, we pass a function to `mo.lazy` instead of a component. This
function will only be called when the user opens the accordion. In this way,
`expensive_component` lazily computed and we only query the database when the
user needs to see the data. This can be useful when the data is expensive to
compute and the user may not need to see it immediately.


---

###### Cycle dependencies - marimoMB003: cycle-dependencies¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/cycle_dependencies/](https://docs.marimo.io/guides/lint_rules/rules/cycle_dependencies/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/cycle_dependencies/ -->
<!-- Title: Cycle dependencies - marimoMB003: cycle-dependencies¶ -->
<!-- Depth: 5 -->

[Skip to content](#mb003-cycle-dependencies)

# MB003: cycle-dependencies [¶](\#mb003-cycle-dependencies "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB003: Cells have circular dependencies.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes the dependency graph to detect circular references between cells,
where cells depend on each other in a way that creates an impossible
execution order.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Circular dependencies prevent marimo from:
\- Determining a valid execution order
\- Running notebooks reproducibly
\- Executing notebooks as scripts
\- Providing reliable reactive updates

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Cycles Guide](https://docs.marimo.io/guides/understanding_errors/cycles/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

Back to top

# MB003: cycle-dependencies [¶](\#mb003-cycle-dependencies "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB003: Cells have circular dependencies.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes the dependency graph to detect circular references between cells,
where cells depend on each other in a way that creates an impossible
execution order.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Circular dependencies prevent marimo from:
\- Determining a valid execution order
\- Running notebooks reproducibly
\- Executing notebooks as scripts
\- Providing reliable reactive updates

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Cycles Guide](https://docs.marimo.io/guides/understanding_errors/cycles/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

 Back to top

# MB003: cycle-dependencies [¶](\#mb003-cycle-dependencies "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB003: Cells have circular dependencies.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes the dependency graph to detect circular references between cells,
where cells depend on each other in a way that creates an impossible
execution order.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Circular dependencies prevent marimo from:
\- Determining a valid execution order
\- Running notebooks reproducibly
\- Executing notebooks as scripts
\- Providing reliable reactive updates

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Cycles Guide](https://docs.marimo.io/guides/understanding_errors/cycles/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)


---

###### Empty cells - marimoMF004: empty-cells¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/empty_cells/](https://docs.marimo.io/guides/lint_rules/rules/empty_cells/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/empty_cells/ -->
<!-- Title: Empty cells - marimoMF004: empty-cells¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf004-empty-cells)

# MF004: empty-cells [¶](\#mf004-empty-cells "Permanent link")

✨ **Formatting** ⚠️ Unsafe Fixable

MF004: Empty cells that can be safely removed.

## What it does [¶](\#what-it-does "Permanent link")

Detects cells that contain only:
\- Whitespace characters (spaces, tabs, newlines)
\- Comments (lines starting with #)
\- Pass statements ( `pass`)
\- Any combination of the above

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Empty cells can:
\- Create clutter in notebook structure
\- Add unnecessary complexity to the execution graph
\- Make notebooks harder to read and maintain
\- Increase file size without adding value

While not functionally breaking, removing empty cells improves code
clarity and reduces visual noise.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Problematic:**

**Problematic:**

**Note:** This fix requires `--unsafe-fixes` because removing cells changes
the notebook structure, and potentially removes user-intended content.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Best Practices](https://docs.marimo.io/guides/best_practices/)

Back to top

# MF004: empty-cells [¶](\#mf004-empty-cells "Permanent link")

✨ **Formatting** ⚠️ Unsafe Fixable

MF004: Empty cells that can be safely removed.

## What it does [¶](\#what-it-does "Permanent link")

Detects cells that contain only:
\- Whitespace characters (spaces, tabs, newlines)
\- Comments (lines starting with #)
\- Pass statements ( `pass`)
\- Any combination of the above

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Empty cells can:
\- Create clutter in notebook structure
\- Add unnecessary complexity to the execution graph
\- Make notebooks harder to read and maintain
\- Increase file size without adding value

While not functionally breaking, removing empty cells improves code
clarity and reduces visual noise.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Problematic:**

**Problematic:**

**Note:** This fix requires `--unsafe-fixes` because removing cells changes
the notebook structure, and potentially removes user-intended content.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Best Practices](https://docs.marimo.io/guides/best_practices/)

 Back to top

# MF004: empty-cells [¶](\#mf004-empty-cells "Permanent link")

✨ **Formatting** ⚠️ Unsafe Fixable

MF004: Empty cells that can be safely removed.

## What it does [¶](\#what-it-does "Permanent link")

Detects cells that contain only:
\- Whitespace characters (spaces, tabs, newlines)
\- Comments (lines starting with #)
\- Pass statements ( `pass`)
\- Any combination of the above

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Empty cells can:
\- Create clutter in notebook structure
\- Add unnecessary complexity to the execution graph
\- Make notebooks harder to read and maintain
\- Increase file size without adding value

While not functionally breaking, removing empty cells improves code
clarity and reduces visual noise.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Problematic:**

**Problematic:**

**Note:** This fix requires `--unsafe-fixes` because removing cells changes
the notebook structure, and potentially removes user-intended content.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Best Practices](https://docs.marimo.io/guides/best_practices/)


---

###### General formatting - marimoMF001: general-formatting¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/general_formatting/](https://docs.marimo.io/guides/lint_rules/rules/general_formatting/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/general_formatting/ -->
<!-- Title: General formatting - marimoMF001: general-formatting¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf001-general-formatting)

# MF001: general-formatting [¶](\#mf001-general-formatting "Permanent link")

✨ **Formatting** 🛠️ Fixable

MF001: General formatting issues with the notebook format.

## What it does [¶](\#what-it-does "Permanent link")

Examines the notebook serialization for structural violations such as:
\- Missing or incorrect marimo import statements
\- Improperly formatted cell definitions
\- Missing app initialization code
\- Incorrect file generation metadata

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Format violations can prevent marimo from properly loading or executing
notebooks. While these don't affect the Python code logic, formatting errors
mark a deviation in the expected script structure, which can lead to
unexpected behavior when run as a script, or when loading the notebook.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

**Note:** Most format issues are automatically fixable with `marimo check --fix`.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [File Format Documentation](https://docs.marimo.io/guides/coming_from/jupyter/#marimo-file-format)

Back to top

# MF001: general-formatting [¶](\#mf001-general-formatting "Permanent link")

✨ **Formatting** 🛠️ Fixable

MF001: General formatting issues with the notebook format.

## What it does [¶](\#what-it-does "Permanent link")

Examines the notebook serialization for structural violations such as:
\- Missing or incorrect marimo import statements
\- Improperly formatted cell definitions
\- Missing app initialization code
\- Incorrect file generation metadata

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Format violations can prevent marimo from properly loading or executing
notebooks. While these don't affect the Python code logic, formatting errors
mark a deviation in the expected script structure, which can lead to
unexpected behavior when run as a script, or when loading the notebook.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

**Note:** Most format issues are automatically fixable with `marimo check --fix`.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [File Format Documentation](https://docs.marimo.io/guides/coming_from/jupyter/#marimo-file-format)

 Back to top

# MF001: general-formatting [¶](\#mf001-general-formatting "Permanent link")

✨ **Formatting** 🛠️ Fixable

MF001: General formatting issues with the notebook format.

## What it does [¶](\#what-it-does "Permanent link")

Examines the notebook serialization for structural violations such as:
\- Missing or incorrect marimo import statements
\- Improperly formatted cell definitions
\- Missing app initialization code
\- Incorrect file generation metadata

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Format violations can prevent marimo from properly loading or executing
notebooks. While these don't affect the Python code logic, formatting errors
mark a deviation in the expected script structure, which can lead to
unexpected behavior when run as a script, or when loading the notebook.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

**Note:** Most format issues are automatically fixable with `marimo check --fix`.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [File Format Documentation](https://docs.marimo.io/guides/coming_from/jupyter/#marimo-file-format)


---

###### Syntax error - marimoMB005: invalid-syntax¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/invalid_syntax/](https://docs.marimo.io/guides/lint_rules/rules/invalid_syntax/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/invalid_syntax/ -->
<!-- Title: Syntax error - marimoMB005: invalid-syntax¶ -->
<!-- Depth: 5 -->

[Skip to content](#mb005-invalid-syntax)

# MB005: invalid-syntax [¶](\#mb005-invalid-syntax "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB005: Cell contains code that throws a SyntaxError on compilation.

## What it does [¶](\#what-it-does "Permanent link")

Attempts to compile each cell using marimo's internal compiler and catches any
SyntaxError exceptions that occur during the compilation process.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Cells with syntax errors cannot be executed, making the notebook non-functional.
SyntaxErrors prevent marimo from creating the dependency graph and running the
reactive execution system, breaking the core functionality of the notebook.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Problematic:**

**Solution:**

**Solution:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python SyntaxError Documentation](https://docs.python.org/3/tutorial/errors.html#syntax-errors)

Back to top

# MB005: invalid-syntax [¶](\#mb005-invalid-syntax "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB005: Cell contains code that throws a SyntaxError on compilation.

## What it does [¶](\#what-it-does "Permanent link")

Attempts to compile each cell using marimo's internal compiler and catches any
SyntaxError exceptions that occur during the compilation process.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Cells with syntax errors cannot be executed, making the notebook non-functional.
SyntaxErrors prevent marimo from creating the dependency graph and running the
reactive execution system, breaking the core functionality of the notebook.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Problematic:**

**Solution:**

**Solution:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python SyntaxError Documentation](https://docs.python.org/3/tutorial/errors.html#syntax-errors)

 Back to top

# MB005: invalid-syntax [¶](\#mb005-invalid-syntax "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB005: Cell contains code that throws a SyntaxError on compilation.

## What it does [¶](\#what-it-does "Permanent link")

Attempts to compile each cell using marimo's internal compiler and catches any
SyntaxError exceptions that occur during the compilation process.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Cells with syntax errors cannot be executed, making the notebook non-functional.
SyntaxErrors prevent marimo from creating the dependency graph and running the
reactive execution system, breaking the core functionality of the notebook.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Problematic:**

**Solution:**

**Solution:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python SyntaxError Documentation](https://docs.python.org/3/tutorial/errors.html#syntax-errors)


---

###### Markdown Indentation - marimoMF007: markdown-indentation¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/markdown_indentation/](https://docs.marimo.io/guides/lint_rules/rules/markdown_indentation/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/markdown_indentation/ -->
<!-- Title: Markdown Indentation - marimoMF007: markdown-indentation¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf007-markdown-indentation)

# MF007: markdown-indentation [¶](\#mf007-markdown-indentation "Permanent link")

✨ **Formatting** 🛠️ Fixable

MF007: Markdown strings in `mo.md()` should be properly indented.

## What it does [¶](\#what-it-does "Permanent link")

Checks cells containing `mo.md()` calls to see if the markdown string
content has unnecessary leading whitespace that should be removed.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Indented markdown strings:
\- Are harder to read when viewing the source code
\- Produce larger diffs when making changes
\- Don't match the standard marimo formatting style
\- Can be confusing when the indentation doesn't reflect the markdown structure

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

**Note:** This fix is automatically applied with `marimo check --fix`.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Best Practices](https://docs.marimo.io/guides/best_practices/)

Back to top

# MF007: markdown-indentation [¶](\#mf007-markdown-indentation "Permanent link")

✨ **Formatting** 🛠️ Fixable

MF007: Markdown strings in `mo.md()` should be properly indented.

## What it does [¶](\#what-it-does "Permanent link")

Checks cells containing `mo.md()` calls to see if the markdown string
content has unnecessary leading whitespace that should be removed.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Indented markdown strings:
\- Are harder to read when viewing the source code
\- Produce larger diffs when making changes
\- Don't match the standard marimo formatting style
\- Can be confusing when the indentation doesn't reflect the markdown structure

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

**Note:** This fix is automatically applied with `marimo check --fix`.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Best Practices](https://docs.marimo.io/guides/best_practices/)

 Back to top

# MF007: markdown-indentation [¶](\#mf007-markdown-indentation "Permanent link")

✨ **Formatting** 🛠️ Fixable

MF007: Markdown strings in `mo.md()` should be properly indented.

## What it does [¶](\#what-it-does "Permanent link")

Checks cells containing `mo.md()` calls to see if the markdown string
content has unnecessary leading whitespace that should be removed.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Indented markdown strings:
\- Are harder to read when viewing the source code
\- Produce larger diffs when making changes
\- Don't match the standard marimo formatting style
\- Can be confusing when the indentation doesn't reflect the markdown structure

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

**Note:** This fix is automatically applied with `marimo check --fix`.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Best Practices](https://docs.marimo.io/guides/best_practices/)


---

###### Misc parse log - marimoMF006: misc-log-capture¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/misc_log_capture/](https://docs.marimo.io/guides/lint_rules/rules/misc_log_capture/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/misc_log_capture/ -->
<!-- Title: Misc parse log - marimoMF006: misc-log-capture¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf006-misc-log-capture)

# MF006: misc-log-capture [¶](\#mf006-misc-log-capture "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF006: Miscellaneous log messages during processing.

## What it does [¶](\#what-it-does "Permanent link")

Captures warning and error level log messages that aren't handled by
other specific log rules and creates diagnostics to surface them.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Unhandled log messages may indicate:
\- Unexpected issues during notebook processing
\- Configuration problems
\- Library warnings that affect execution
\- Performance or resource issues

## Examples [¶](\#examples "Permanent link")

**Triggered by:**
\- General warnings from imported libraries
\- Configuration issues
\- Unexpected errors during processing

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

Back to top

# MF006: misc-log-capture [¶](\#mf006-misc-log-capture "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF006: Miscellaneous log messages during processing.

## What it does [¶](\#what-it-does "Permanent link")

Captures warning and error level log messages that aren't handled by
other specific log rules and creates diagnostics to surface them.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Unhandled log messages may indicate:
\- Unexpected issues during notebook processing
\- Configuration problems
\- Library warnings that affect execution
\- Performance or resource issues

## Examples [¶](\#examples "Permanent link")

**Triggered by:**
\- General warnings from imported libraries
\- Configuration issues
\- Unexpected errors during processing

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

 Back to top

# MF006: misc-log-capture [¶](\#mf006-misc-log-capture "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF006: Miscellaneous log messages during processing.

## What it does [¶](\#what-it-does "Permanent link")

Captures warning and error level log messages that aren't handled by
other specific log rules and creates diagnostics to surface them.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Unhandled log messages may indicate:
\- Unexpected issues during notebook processing
\- Configuration problems
\- Library warnings that affect execution
\- Performance or resource issues

## Examples [¶](\#examples "Permanent link")

**Triggered by:**
\- General warnings from imported libraries
\- Configuration issues
\- Unexpected errors during processing

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)


---

###### Multiple definitions - marimoMB002: multiple-definitions¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/multiple_definitions/](https://docs.marimo.io/guides/lint_rules/rules/multiple_definitions/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/multiple_definitions/ -->
<!-- Title: Multiple definitions - marimoMB002: multiple-definitions¶ -->
<!-- Depth: 5 -->

[Skip to content](#mb002-multiple-definitions)

# MB002: multiple-definitions [¶](\#mb002-multiple-definitions "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB002: Multiple cells define the same variable.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes the dependency graph to detect variables that are defined in more
than one cell, which violates marimo's fundamental constraint for reactive execution.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Multiple definitions prevent marimo from:
\- Determining the correct execution order
\- Creating a reliable dependency graph
\- Running notebooks as scripts
\- Providing consistent reactive updates

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Multiple Definitions Guide](https://docs.marimo.io/guides/understanding_errors/multiple_definitions/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

Back to top

# MB002: multiple-definitions [¶](\#mb002-multiple-definitions "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB002: Multiple cells define the same variable.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes the dependency graph to detect variables that are defined in more
than one cell, which violates marimo's fundamental constraint for reactive execution.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Multiple definitions prevent marimo from:
\- Determining the correct execution order
\- Creating a reliable dependency graph
\- Running notebooks as scripts
\- Providing consistent reactive updates

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Multiple Definitions Guide](https://docs.marimo.io/guides/understanding_errors/multiple_definitions/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

 Back to top

# MB002: multiple-definitions [¶](\#mb002-multiple-definitions "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB002: Multiple cells define the same variable.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes the dependency graph to detect variables that are defined in more
than one cell, which violates marimo's fundamental constraint for reactive execution.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Multiple definitions prevent marimo from:
\- Determining the correct execution order
\- Creating a reliable dependency graph
\- Running notebooks as scripts
\- Providing consistent reactive updates

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Multiple Definitions Guide](https://docs.marimo.io/guides/understanding_errors/multiple_definitions/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)


---

###### Parse stderr - marimoMF003: parse-stderr¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/parse_stderr/](https://docs.marimo.io/guides/lint_rules/rules/parse_stderr/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/parse_stderr/ -->
<!-- Title: Parse stderr - marimoMF003: parse-stderr¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf003-parse-stderr)

# MF003: parse-stderr [¶](\#mf003-parse-stderr "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF003: Parse captured stderr during notebook loading.

## What it does [¶](\#what-it-does "Permanent link")

Captures stderr output during notebook loading and creates diagnostics
from any error messages or warnings. This helps identify potential
issues that don't prevent parsing but may affect runtime behavior.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Stderr output during parsing often indicates:
\- Syntax warnings (like invalid escape sequences)
\- Import warnings or errors
\- Deprecation notices from libraries
\- Configuration issues that might affect execution

While these don't break the notebook, they can lead to unexpected
behavior or indicate code that needs updating.

## Examples [¶](\#examples "Permanent link")

**Captured stderr:**

**Result:** Creates a diagnostic pointing to line 68 about the invalid escape sequence.

**Common issues:**
\- Raw strings needed: `r"\path\to\file"` instead of `"\path\to\file"`
\- Deprecated library usage
\- Missing import dependencies

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python Warning Categories](https://docs.python.org/3/library/warnings.html#warning-categories)

Back to top

# MF003: parse-stderr [¶](\#mf003-parse-stderr "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF003: Parse captured stderr during notebook loading.

## What it does [¶](\#what-it-does "Permanent link")

Captures stderr output during notebook loading and creates diagnostics
from any error messages or warnings. This helps identify potential
issues that don't prevent parsing but may affect runtime behavior.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Stderr output during parsing often indicates:
\- Syntax warnings (like invalid escape sequences)
\- Import warnings or errors
\- Deprecation notices from libraries
\- Configuration issues that might affect execution

While these don't break the notebook, they can lead to unexpected
behavior or indicate code that needs updating.

## Examples [¶](\#examples "Permanent link")

**Captured stderr:**

**Result:** Creates a diagnostic pointing to line 68 about the invalid escape sequence.

**Common issues:**
\- Raw strings needed: `r"\path\to\file"` instead of `"\path\to\file"`
\- Deprecated library usage
\- Missing import dependencies

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python Warning Categories](https://docs.python.org/3/library/warnings.html#warning-categories)

 Back to top

# MF003: parse-stderr [¶](\#mf003-parse-stderr "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF003: Parse captured stderr during notebook loading.

## What it does [¶](\#what-it-does "Permanent link")

Captures stderr output during notebook loading and creates diagnostics
from any error messages or warnings. This helps identify potential
issues that don't prevent parsing but may affect runtime behavior.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Stderr output during parsing often indicates:
\- Syntax warnings (like invalid escape sequences)
\- Import warnings or errors
\- Deprecation notices from libraries
\- Configuration issues that might affect execution

While these don't break the notebook, they can lead to unexpected
behavior or indicate code that needs updating.

## Examples [¶](\#examples "Permanent link")

**Captured stderr:**

**Result:** Creates a diagnostic pointing to line 68 about the invalid escape sequence.

**Common issues:**
\- Raw strings needed: `r"\path\to\file"` instead of `"\path\to\file"`
\- Deprecated library usage
\- Missing import dependencies

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python Warning Categories](https://docs.python.org/3/library/warnings.html#warning-categories)


---

###### Parse stdout - marimoMF002: parse-stdout¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/parse_stdout/](https://docs.marimo.io/guides/lint_rules/rules/parse_stdout/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/parse_stdout/ -->
<!-- Title: Parse stdout - marimoMF002: parse-stdout¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf002-parse-stdout)

# MF002: parse-stdout [¶](\#mf002-parse-stdout "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF002: Parse captured stdout during notebook loading.

## What it does [¶](\#what-it-does "Permanent link")

Captures and parses stdout output during notebook loading, looking for
structured warning messages that include file and line number references.
Creates diagnostics from any warnings or messages found.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

While stdout output doesn't prevent execution, it often indicates:
\- Deprecation warnings from imported libraries
\- Configuration issues
\- Potential compatibility problems
\- Code that produces unexpected side effects during import

## Examples [¶](\#examples "Permanent link")

**Captured stdout:**

**Result:** Creates a diagnostic pointing to line 15 with the deprecation warning.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

Back to top

# MF002: parse-stdout [¶](\#mf002-parse-stdout "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF002: Parse captured stdout during notebook loading.

## What it does [¶](\#what-it-does "Permanent link")

Captures and parses stdout output during notebook loading, looking for
structured warning messages that include file and line number references.
Creates diagnostics from any warnings or messages found.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

While stdout output doesn't prevent execution, it often indicates:
\- Deprecation warnings from imported libraries
\- Configuration issues
\- Potential compatibility problems
\- Code that produces unexpected side effects during import

## Examples [¶](\#examples "Permanent link")

**Captured stdout:**

**Result:** Creates a diagnostic pointing to line 15 with the deprecation warning.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

 Back to top

# MF002: parse-stdout [¶](\#mf002-parse-stdout "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF002: Parse captured stdout during notebook loading.

## What it does [¶](\#what-it-does "Permanent link")

Captures and parses stdout output during notebook loading, looking for
structured warning messages that include file and line number references.
Creates diagnostics from any warnings or messages found.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

While stdout output doesn't prevent execution, it often indicates:
\- Deprecation warnings from imported libraries
\- Configuration issues
\- Potential compatibility problems
\- Code that produces unexpected side effects during import

## Examples [¶](\#examples "Permanent link")

**Captured stdout:**

**Result:** Creates a diagnostic pointing to line 15 with the deprecation warning.

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)


---

###### Self Import - marimoMR001: self-import¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/self_import/](https://docs.marimo.io/guides/lint_rules/rules/self_import/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/self_import/ -->
<!-- Title: Self Import - marimoMR001: self-import¶ -->
<!-- Depth: 5 -->

[Skip to content](#mr001-self-import)

# MR001: self-import [¶](\#mr001-self-import "Permanent link")

⚠️ **Runtime** ❌ Not Fixable

MR001: Importing a module with the same name as the file.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes import statements in each cell to detect cases where the imported
module name matches the current file's name (without the .py extension).

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Importing a module with the same name as the file causes several issues:
\- Python may attempt to import the current file instead of the intended module
\- This can lead to circular import errors or unexpected behavior
\- It makes the code confusing and hard to debug
\- It can prevent the notebook from running correctly

This is a runtime issue because it can cause import confusion and unexpected behavior.

## Examples [¶](\#examples "Permanent link")

**Problematic (in a file named `requests.py`):**

**Problematic (in a file named `math.py`):**

**Solution:**

**Alternative Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python Import System](https://docs.python.org/3/reference/import.html)

Back to top

# MR001: self-import [¶](\#mr001-self-import "Permanent link")

⚠️ **Runtime** ❌ Not Fixable

MR001: Importing a module with the same name as the file.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes import statements in each cell to detect cases where the imported
module name matches the current file's name (without the .py extension).

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Importing a module with the same name as the file causes several issues:
\- Python may attempt to import the current file instead of the intended module
\- This can lead to circular import errors or unexpected behavior
\- It makes the code confusing and hard to debug
\- It can prevent the notebook from running correctly

This is a runtime issue because it can cause import confusion and unexpected behavior.

## Examples [¶](\#examples "Permanent link")

**Problematic (in a file named `requests.py`):**

**Problematic (in a file named `math.py`):**

**Solution:**

**Alternative Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python Import System](https://docs.python.org/3/reference/import.html)

 Back to top

# MR001: self-import [¶](\#mr001-self-import "Permanent link")

⚠️ **Runtime** ❌ Not Fixable

MR001: Importing a module with the same name as the file.

## What it does [¶](\#what-it-does "Permanent link")

Analyzes import statements in each cell to detect cases where the imported
module name matches the current file's name (without the .py extension).

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Importing a module with the same name as the file causes several issues:
\- Python may attempt to import the current file instead of the intended module
\- This can lead to circular import errors or unexpected behavior
\- It makes the code confusing and hard to debug
\- It can prevent the notebook from running correctly

This is a runtime issue because it can cause import confusion and unexpected behavior.

## Examples [¶](\#examples "Permanent link")

**Problematic (in a file named `requests.py`):**

**Problematic (in a file named `math.py`):**

**Solution:**

**Alternative Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [Python Import System](https://docs.python.org/3/reference/import.html)


---

###### Setup cell dependencies - marimoMB004: setup-cell-dependencies¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/setup_cell_dependencies/](https://docs.marimo.io/guides/lint_rules/rules/setup_cell_dependencies/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/setup_cell_dependencies/ -->
<!-- Title: Setup cell dependencies - marimoMB004: setup-cell-dependencies¶ -->
<!-- Depth: 5 -->

[Skip to content](#mb004-setup-cell-dependencies)

# MB004: setup-cell-dependencies [¶](\#mb004-setup-cell-dependencies "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB004: Setup cell cannot have dependencies.

## What it does [¶](\#what-it-does "Permanent link")

Validates that the setup cell (if present) does not depend on variables
defined in other cells, ensuring proper execution order.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Setup cell dependencies break marimo's execution model because:
\- The setup cell must run first to initialize the notebook
\- Dependencies on other cells would create impossible execution order
\- It violates the setup cell's purpose as initialization code

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Setup References Guide](https://docs.marimo.io/guides/understanding_errors/setup/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

Back to top

# MB004: setup-cell-dependencies [¶](\#mb004-setup-cell-dependencies "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB004: Setup cell cannot have dependencies.

## What it does [¶](\#what-it-does "Permanent link")

Validates that the setup cell (if present) does not depend on variables
defined in other cells, ensuring proper execution order.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Setup cell dependencies break marimo's execution model because:
\- The setup cell must run first to initialize the notebook
\- Dependencies on other cells would create impossible execution order
\- It violates the setup cell's purpose as initialization code

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Setup References Guide](https://docs.marimo.io/guides/understanding_errors/setup/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

 Back to top

# MB004: setup-cell-dependencies [¶](\#mb004-setup-cell-dependencies "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB004: Setup cell cannot have dependencies.

## What it does [¶](\#what-it-does "Permanent link")

Validates that the setup cell (if present) does not depend on variables
defined in other cells, ensuring proper execution order.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Setup cell dependencies break marimo's execution model because:
\- The setup cell must run first to initialize the notebook
\- Dependencies on other cells would create impossible execution order
\- It violates the setup cell's purpose as initialization code

This is a breaking error because it makes the notebook non-executable.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Setup References Guide](https://docs.marimo.io/guides/understanding_errors/setup/)
- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)


---

###### SQL parse error - marimoMF005: sql-parse-error¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/sql_parse_error/](https://docs.marimo.io/guides/lint_rules/rules/sql_parse_error/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/sql_parse_error/ -->
<!-- Title: SQL parse error - marimoMF005: sql-parse-error¶ -->
<!-- Depth: 5 -->

[Skip to content](#mf005-sql-parse-error)

# MF005: sql-parse-error [¶](\#mf005-sql-parse-error "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF005: SQL parsing errors during dependency analysis.

## What it does [¶](\#what-it-does "Permanent link")

Captures SQL parsing error logs and creates diagnostics pointing to
problematic SQL statements in cells.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

SQL parsing failures can lead to:
\- Incorrect dependency analysis for SQL-using cells
\- Missing dataframe references in dependency graph
\- Reduced effectiveness of reactive execution
\- Potential runtime errors when SQL is executed

## Examples [¶](\#examples "Permanent link")

**Triggered by:**
\- Invalid SQL syntax in cell code
\- Unsupported SQL dialects or extensions
\- Complex SQL that exceeds parser capabilities

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [SQL Support](https://docs.marimo.io/guides/sql/)

Back to top

# MF005: sql-parse-error [¶](\#mf005-sql-parse-error "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF005: SQL parsing errors during dependency analysis.

## What it does [¶](\#what-it-does "Permanent link")

Captures SQL parsing error logs and creates diagnostics pointing to
problematic SQL statements in cells.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

SQL parsing failures can lead to:
\- Incorrect dependency analysis for SQL-using cells
\- Missing dataframe references in dependency graph
\- Reduced effectiveness of reactive execution
\- Potential runtime errors when SQL is executed

## Examples [¶](\#examples "Permanent link")

**Triggered by:**
\- Invalid SQL syntax in cell code
\- Unsupported SQL dialects or extensions
\- Complex SQL that exceeds parser capabilities

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [SQL Support](https://docs.marimo.io/guides/sql/)

 Back to top

# MF005: sql-parse-error [¶](\#mf005-sql-parse-error "Permanent link")

✨ **Formatting** ❌ Not Fixable

MF005: SQL parsing errors during dependency analysis.

## What it does [¶](\#what-it-does "Permanent link")

Captures SQL parsing error logs and creates diagnostics pointing to
problematic SQL statements in cells.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

SQL parsing failures can lead to:
\- Incorrect dependency analysis for SQL-using cells
\- Missing dataframe references in dependency graph
\- Reduced effectiveness of reactive execution
\- Potential runtime errors when SQL is executed

## Examples [¶](\#examples "Permanent link")

**Triggered by:**
\- Invalid SQL syntax in cell code
\- Unsupported SQL dialects or extensions
\- Complex SQL that exceeds parser capabilities

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)
- [SQL Support](https://docs.marimo.io/guides/sql/)


---

###### Unparsable cells - marimoMB001: unparsable-cells¶

*Source: [https://docs.marimo.io/guides/lint_rules/rules/unparsable_cells/](https://docs.marimo.io/guides/lint_rules/rules/unparsable_cells/)*

<!-- Source: https://docs.marimo.io/guides/lint_rules/rules/unparsable_cells/ -->
<!-- Title: Unparsable cells - marimoMB001: unparsable-cells¶ -->
<!-- Depth: 5 -->

[Skip to content](#mb001-unparsable-cells)

# MB001: unparsable-cells [¶](\#mb001-unparsable-cells "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB001: Cell contains unparsable code.

## What it does [¶](\#what-it-does "Permanent link")

Identifies cells that cannot be parsed into valid Python AST nodes, indicating
fundamental syntax or encoding problems that prevent the notebook from being loaded.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Unparsable cells prevent the notebook from running as a script and will throw
errors when executed in notebook mode. While marimo can still open the notebook,
these cells cannot be run until the parsing issues are resolved.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

Back to top

# MB001: unparsable-cells [¶](\#mb001-unparsable-cells "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB001: Cell contains unparsable code.

## What it does [¶](\#what-it-does "Permanent link")

Identifies cells that cannot be parsed into valid Python AST nodes, indicating
fundamental syntax or encoding problems that prevent the notebook from being loaded.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Unparsable cells prevent the notebook from running as a script and will throw
errors when executed in notebook mode. While marimo can still open the notebook,
these cells cannot be run until the parsing issues are resolved.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)

 Back to top

# MB001: unparsable-cells [¶](\#mb001-unparsable-cells "Permanent link")

🚨 **Breaking** ❌ Not Fixable

MB001: Cell contains unparsable code.

## What it does [¶](\#what-it-does "Permanent link")

Identifies cells that cannot be parsed into valid Python AST nodes, indicating
fundamental syntax or encoding problems that prevent the notebook from being loaded.

## Why is this bad? [¶](\#why-is-this-bad "Permanent link")

Unparsable cells prevent the notebook from running as a script and will throw
errors when executed in notebook mode. While marimo can still open the notebook,
these cells cannot be run until the parsing issues are resolved.

## Examples [¶](\#examples "Permanent link")

**Problematic:**

**Problematic:**

**Solution:**

## References [¶](\#references "Permanent link")

- [Understanding Errors](https://docs.marimo.io/guides/understanding_errors/)


---

###### Query Parameters - marimoQuery Parameters¶

*Source: [https://docs.marimo.io/api/query_params/](https://docs.marimo.io/api/query_params/)*

<!-- Source: https://docs.marimo.io/api/query_params/ -->
<!-- Title: Query Parameters - marimoQuery Parameters¶ -->
<!-- Depth: 6 -->

[Skip to content](#query-parameters)

# Query Parameters [¶](\#query-parameters "Permanent link")

Query parameters are key-value pairs appended to the end of a URL to pass data to the server or customize a request.

Use `mo.query_params` to access query parameters passed to the notebook. You
can also use `mo.query_params` to set query parameters in order to keep track
of state in the URL. This is useful for bookmarking or sharing a particular
state of the notebook while running as an application with `marimo run`.

## `` marimo.query\_params [¶](\#marimo.query_params "Permanent link")

Get the query parameters of a marimo app.

Examples:

Keep the text input in sync with the URL query parameters:

You can also set the query parameters reactively:

| RETURNS | DESCRIPTION |
| --- | --- |
| `QueryParams` | A QueryParams object containing the query parameters.<br>You can directly interact with this object like a dictionary.<br>If you mutate this object, changes will be persisted to the frontend<br>query parameters and any other cells referencing the query parameters<br>will automatically re-run.<br>**TYPE:** `QueryParams` |

## Pydantic Models for Query Parameters [¶](\#pydantic-models-for-query-parameters "Permanent link")

One of the use cases for URL query parameters is to set the initial state of UI elements.

Passing query parameters into a Pydantic model helps document and validate the parameters.

In the next cell: (see [live](https://marimo.app/l/03egkc?g=255))

When using [marimo apps mounted to FastAPI](../../guides/deploying/programmatically/), the Pydantic model can be passed into the API documentation for the main app.

CLI arguments

You can also access command-line arguments passed to the notebook using
`mo.cli_args`. This allows you to pass arguments to the notebook that are not controllable by the user.

Back to top

# Query Parameters [¶](\#query-parameters "Permanent link")

Query parameters are key-value pairs appended to the end of a URL to pass data to the server or customize a request.

Use `mo.query_params` to access query parameters passed to the notebook. You
can also use `mo.query_params` to set query parameters in order to keep track
of state in the URL. This is useful for bookmarking or sharing a particular
state of the notebook while running as an application with `marimo run`.

## `` marimo.query\_params [¶](\#marimo.query_params "Permanent link")

Get the query parameters of a marimo app.

Examples:

Keep the text input in sync with the URL query parameters:

You can also set the query parameters reactively:

| RETURNS | DESCRIPTION |
| --- | --- |
| `QueryParams` | A QueryParams object containing the query parameters.<br>You can directly interact with this object like a dictionary.<br>If you mutate this object, changes will be persisted to the frontend<br>query parameters and any other cells referencing the query parameters<br>will automatically re-run.<br>**TYPE:** `QueryParams` |

## Pydantic Models for Query Parameters [¶](\#pydantic-models-for-query-parameters "Permanent link")

One of the use cases for URL query parameters is to set the initial state of UI elements.

Passing query parameters into a Pydantic model helps document and validate the parameters.

In the next cell: (see [live](https://marimo.app/l/03egkc?g=255))

When using [marimo apps mounted to FastAPI](../../guides/deploying/programmatically/), the Pydantic model can be passed into the API documentation for the main app.

CLI arguments

You can also access command-line arguments passed to the notebook using
`mo.cli_args`. This allows you to pass arguments to the notebook that are not controllable by the user.

 Back to top

# Query Parameters [¶](\#query-parameters "Permanent link")

Query parameters are key-value pairs appended to the end of a URL to pass data to the server or customize a request.

Use `mo.query_params` to access query parameters passed to the notebook. You
can also use `mo.query_params` to set query parameters in order to keep track
of state in the URL. This is useful for bookmarking or sharing a particular
state of the notebook while running as an application with `marimo run`.

## `` marimo.query\_params [¶](\#marimo.query_params "Permanent link")

Get the query parameters of a marimo app.

Examples:

Keep the text input in sync with the URL query parameters:

You can also set the query parameters reactively:

| RETURNS | DESCRIPTION |
| --- | --- |
| `QueryParams` | A QueryParams object containing the query parameters.<br>You can directly interact with this object like a dictionary.<br>If you mutate this object, changes will be persisted to the frontend<br>query parameters and any other cells referencing the query parameters<br>will automatically re-run.<br>**TYPE:** `QueryParams` |

## Pydantic Models for Query Parameters [¶](\#pydantic-models-for-query-parameters "Permanent link")

One of the use cases for URL query parameters is to set the initial state of UI elements.

Passing query parameters into a Pydantic model helps document and validate the parameters.

In the next cell: (see [live](https://marimo.app/l/03egkc?g=255))

When using [marimo apps mounted to FastAPI](../../guides/deploying/programmatically/), the Pydantic model can be passed into the API documentation for the main app.

CLI arguments

You can also access command-line arguments passed to the notebook using
`mo.cli_args`. This allows you to pass arguments to the notebook that are not controllable by the user.


---

###### Best practices - marimoBest practices¶

*Source: [https://docs.marimo.io/guides/best_practices/](https://docs.marimo.io/guides/best_practices/)*

<!-- Source: https://docs.marimo.io/guides/best_practices/ -->
<!-- Title: Best practices - marimoBest practices¶ -->
<!-- Depth: 6 -->

[Skip to content](#best-practices)

# Best practices [¶](\#best-practices "Permanent link")

Here are best practices for writing marimo notebooks.

**Use global variables sparingly.** Keep the number of global variables in your
program small to avoid name collisions. If you have intermediate variables,
encapsulate them in functions or prefix them with an underscore ( `_tmp = ...`) to
make them local to a cell.

**Use descriptive names.** Use descriptive variable names, especially for
global variables. This will help you minimize name clashes, and will also
result in better code.

**Use functions.** Encapsulate logic into functions to avoid polluting the
global namespace with
temporary or intermediate variables, and to avoid code duplication.

**Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to stop execution.** Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop")
to stop a cell from running when a condition is true; this is helpful
when working with expensive notebooks. For example, prevent a cell from running
until a button is clicked using [`mo.ui.run_button`](../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") and
[`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop").

Expensive notebooks

For more tips on working with expensive notebooks, see the
associated [guide](../expensive_notebooks/).

**Use Python modules.** If your notebook gets too long, split complex logic
into helper Python modules and import them into your notebook. Use marimo's
built-in [module\
reloading](../configuration/runtime_configuration/#on-module-change)
to automatically bring changes from your modules into your notebook.

**Minimize mutations.** marimo does not track mutations to objects. Try to
only mutate an object in the cell that creates it, or create new objects
instead of mutating existing ones.

Example

_Don't_ split up declarations and mutations over multiple cells. For example, _don't_
_do this:_

Instead, _do_ **declare and mutate in the same cell**:

or, if working in multiple cells, **declare a new variable based on the old**
**one**:

**Don't use state and `on_change` handlers.** Don't use `on_change` handlers
to react to UI interactions. Instead, use marimo's built-in [reactive execution\
for interactive elements](../interactivity/).

**Write idempotent cells.**
Write cells whose outputs and behavior are the same
when given the same inputs (references); such cells are called idempotent. This
will help you avoid bugs and cache expensive intermediate computations.

Back to top

# Best practices [¶](\#best-practices "Permanent link")

Here are best practices for writing marimo notebooks.

**Use global variables sparingly.** Keep the number of global variables in your
program small to avoid name collisions. If you have intermediate variables,
encapsulate them in functions or prefix them with an underscore ( `_tmp = ...`) to
make them local to a cell.

**Use descriptive names.** Use descriptive variable names, especially for
global variables. This will help you minimize name clashes, and will also
result in better code.

**Use functions.** Encapsulate logic into functions to avoid polluting the
global namespace with
temporary or intermediate variables, and to avoid code duplication.

**Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to stop execution.** Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop")
to stop a cell from running when a condition is true; this is helpful
when working with expensive notebooks. For example, prevent a cell from running
until a button is clicked using [`mo.ui.run_button`](../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") and
[`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop").

Expensive notebooks

For more tips on working with expensive notebooks, see the
associated [guide](../expensive_notebooks/).

**Use Python modules.** If your notebook gets too long, split complex logic
into helper Python modules and import them into your notebook. Use marimo's
built-in [module\
reloading](../configuration/runtime_configuration/#on-module-change)
to automatically bring changes from your modules into your notebook.

**Minimize mutations.** marimo does not track mutations to objects. Try to
only mutate an object in the cell that creates it, or create new objects
instead of mutating existing ones.

Example

_Don't_ split up declarations and mutations over multiple cells. For example, _don't_
_do this:_

Instead, _do_ **declare and mutate in the same cell**:

or, if working in multiple cells, **declare a new variable based on the old**
**one**:

**Don't use state and `on_change` handlers.** Don't use `on_change` handlers
to react to UI interactions. Instead, use marimo's built-in [reactive execution\
for interactive elements](../interactivity/).

**Write idempotent cells.**
Write cells whose outputs and behavior are the same
when given the same inputs (references); such cells are called idempotent. This
will help you avoid bugs and cache expensive intermediate computations.

 Back to top

# Best practices [¶](\#best-practices "Permanent link")

Here are best practices for writing marimo notebooks.

**Use global variables sparingly.** Keep the number of global variables in your
program small to avoid name collisions. If you have intermediate variables,
encapsulate them in functions or prefix them with an underscore ( `_tmp = ...`) to
make them local to a cell.

**Use descriptive names.** Use descriptive variable names, especially for
global variables. This will help you minimize name clashes, and will also
result in better code.

**Use functions.** Encapsulate logic into functions to avoid polluting the
global namespace with
temporary or intermediate variables, and to avoid code duplication.

**Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to stop execution.** Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop")
to stop a cell from running when a condition is true; this is helpful
when working with expensive notebooks. For example, prevent a cell from running
until a button is clicked using [`mo.ui.run_button`](../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") and
[`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop").

Expensive notebooks

For more tips on working with expensive notebooks, see the
associated [guide](../expensive_notebooks/).

**Use Python modules.** If your notebook gets too long, split complex logic
into helper Python modules and import them into your notebook. Use marimo's
built-in [module\
reloading](../configuration/runtime_configuration/#on-module-change)
to automatically bring changes from your modules into your notebook.

**Minimize mutations.** marimo does not track mutations to objects. Try to
only mutate an object in the cell that creates it, or create new objects
instead of mutating existing ones.

Example

_Don't_ split up declarations and mutations over multiple cells. For example, _don't_
_do this:_

Instead, _do_ **declare and mutate in the same cell**:

or, if working in multiple cells, **declare a new variable based on the old**
**one**:

**Don't use state and `on_change` handlers.** Don't use `on_change` handlers
to react to UI interactions. Instead, use marimo's built-in [reactive execution\
for interactive elements](../interactivity/).

**Write idempotent cells.**
Write cells whose outputs and behavior are the same
when given the same inputs (references); such cells are called idempotent. This
will help you avoid bugs and cache expensive intermediate computations.


---

###### Runtime configuration - marimoRuntime configuration¶

*Source: [https://docs.marimo.io/guides/configuration/runtime_configuration/](https://docs.marimo.io/guides/configuration/runtime_configuration/)*

<!-- Source: https://docs.marimo.io/guides/configuration/runtime_configuration/ -->
<!-- Title: Runtime configuration - marimoRuntime configuration¶ -->
<!-- Depth: 6 -->

[Skip to content](#runtime-configuration)

# Runtime configuration [¶](\#runtime-configuration "Permanent link")

Through the notebook settings menu, you can configure how and when marimo runs
cells.

## Disable autorun on startup [¶](\#on-startup "Permanent link")

By default, marimo notebooks run automatically on startup; just how the command

executes a script,

executes the notebook.

Disable this behavior by unchecking "Autorun on startup".

_When sharing a notebook as an app with `marimo run`, this setting has_
_no effect._

## Disable autorun on cell change (lazy execution) [¶](\#disable-autorun-on-cell-change-lazy-execution "Permanent link")

By default, when a cell is run or a UI element is interacted with, marimo
automatically runs cells that reference any of its variables. **You can disable**
**automatic execution of cell's descendants in the notebook settings menu by**
**setting `"On cell change"` to `"lazy"`.**

When the runtime is lazy, running a cell marks affected cells as stale but
doesn't automatically run them. Lazy evaluation means cells are only run when
their outputs are needed. If you run a cell that has stale ancestors, those
ancestors will also run to make sure your cell doesn't use stale inputs. You
can always click the notebook run button or use the keyboard shortcut to run
all stale cells.

**When should I use lazy evaluation?** Choosing the lazy runtime can be helpful
when working on notebooks with expensive cells.

Tip: speed up expensive notebooks with marimo's smart caching

In addition to runtime configuration, marimo also provides [opt-in caching](../../../api/caching/)
to help you work with expensive or side-effectful notebooks. marimo's
can cache expensive functions in memory and expensive blocks of code to disk,
letting you skip entire sections of your code and automatically loading
variables in memory on notebook startup. Read our [caching\
guide](../../../api/caching/) to learn more.

_When sharing a notebook as an app with `marimo run`, this setting has_
_no effect._

## On module change [¶](\#on-module-change "Permanent link")

When module autoreloading is enabled, marimo automatically runs cells when you
edit Python files. Based on static analysis, the reloader only runs cells
affected by your edits. The reloader is recursive, meaning that marimo tracks
modifications for modules imported by your notebook's imported modules too.

Why autoreload?

Autoreloading enables a workflow that many developers find
productive: develop complex logic in Python modules, and use the marimo
notebook as a DAG or main script that orchestrates your logic.

Autoreloading comes in two types:

1. **autorun**: automatically re-runs cells affected by module modification.

When set to autorun, marimo's reloader automatically run cells when you edit Python files.

1. **lazy**: marks cells affected by module modifications as stale, letting you know which cells need to be re-run.

When set to lazy, marimo's reloader marks cells as stale when you edit Python files.

## Python path [¶](\#python-path "Permanent link")

By default, marimo will not add any additional directories to the Python path.
This keeps the behavior of `marimo edit nb.py` and `python nb.py` consistent.

You can add directories to the Python path by setting the `pythonpath` key in
the runtime configuration. These directories will be added to the head of
`sys.path`, similar to how the `PYTHONPATH` environment variable works. This
allows Python to find and import modules from these directories.

Consider alternatives to path manipulation

When possible, it's preferred to avoid path manipulation. If you want to
work on a module in a separate directory alongside your notebooks, we
recommend creating a _package_ and including marimo as a project dependency.

This will make it easier to share your library code. For multiple packages,
consider configuring [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/).

See our guide on [notebooks in existing\
projects](../../package_management/notebooks_in_projects/) for more details.

## Environment variables [¶](\#environment-variables "Permanent link")

### .env files [¶](\#env-files "Permanent link")

marimo supports loading environment variables from `.env` files. This is useful for managing configuration that should not be committed to version control, such as API keys or database credentials.

The `.env` next to your `pyproject.toml` is loaded by default. To configure multiple or a different location, you can specify them in your configuration:

Environment variables from your `dotenv` will be surfaced in the UI when creating databases.

Back to top

# Runtime configuration [¶](\#runtime-configuration "Permanent link")

Through the notebook settings menu, you can configure how and when marimo runs
cells.

## Disable autorun on startup [¶](\#on-startup "Permanent link")

By default, marimo notebooks run automatically on startup; just how the command

executes a script,

executes the notebook.

Disable this behavior by unchecking "Autorun on startup".

_When sharing a notebook as an app with `marimo run`, this setting has_
_no effect._

## Disable autorun on cell change (lazy execution) [¶](\#disable-autorun-on-cell-change-lazy-execution "Permanent link")

By default, when a cell is run or a UI element is interacted with, marimo
automatically runs cells that reference any of its variables. **You can disable**
**automatic execution of cell's descendants in the notebook settings menu by**
**setting `"On cell change"` to `"lazy"`.**

When the runtime is lazy, running a cell marks affected cells as stale but
doesn't automatically run them. Lazy evaluation means cells are only run when
their outputs are needed. If you run a cell that has stale ancestors, those
ancestors will also run to make sure your cell doesn't use stale inputs. You
can always click the notebook run button or use the keyboard shortcut to run
all stale cells.

**When should I use lazy evaluation?** Choosing the lazy runtime can be helpful
when working on notebooks with expensive cells.

Tip: speed up expensive notebooks with marimo's smart caching

In addition to runtime configuration, marimo also provides [opt-in caching](../../../api/caching/)
to help you work with expensive or side-effectful notebooks. marimo's
can cache expensive functions in memory and expensive blocks of code to disk,
letting you skip entire sections of your code and automatically loading
variables in memory on notebook startup. Read our [caching\
guide](../../../api/caching/) to learn more.

_When sharing a notebook as an app with `marimo run`, this setting has_
_no effect._

## On module change [¶](\#on-module-change "Permanent link")

When module autoreloading is enabled, marimo automatically runs cells when you
edit Python files. Based on static analysis, the reloader only runs cells
affected by your edits. The reloader is recursive, meaning that marimo tracks
modifications for modules imported by your notebook's imported modules too.

Why autoreload?

Autoreloading enables a workflow that many developers find
productive: develop complex logic in Python modules, and use the marimo
notebook as a DAG or main script that orchestrates your logic.

Autoreloading comes in two types:

1. **autorun**: automatically re-runs cells affected by module modification.

When set to autorun, marimo's reloader automatically run cells when you edit Python files.

1. **lazy**: marks cells affected by module modifications as stale, letting you know which cells need to be re-run.

When set to lazy, marimo's reloader marks cells as stale when you edit Python files.

## Python path [¶](\#python-path "Permanent link")

By default, marimo will not add any additional directories to the Python path.
This keeps the behavior of `marimo edit nb.py` and `python nb.py` consistent.

You can add directories to the Python path by setting the `pythonpath` key in
the runtime configuration. These directories will be added to the head of
`sys.path`, similar to how the `PYTHONPATH` environment variable works. This
allows Python to find and import modules from these directories.

Consider alternatives to path manipulation

When possible, it's preferred to avoid path manipulation. If you want to
work on a module in a separate directory alongside your notebooks, we
recommend creating a _package_ and including marimo as a project dependency.

This will make it easier to share your library code. For multiple packages,
consider configuring [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/).

See our guide on [notebooks in existing\
projects](../../package_management/notebooks_in_projects/) for more details.

## Environment variables [¶](\#environment-variables "Permanent link")

### .env files [¶](\#env-files "Permanent link")

marimo supports loading environment variables from `.env` files. This is useful for managing configuration that should not be committed to version control, such as API keys or database credentials.

The `.env` next to your `pyproject.toml` is loaded by default. To configure multiple or a different location, you can specify them in your configuration:

Environment variables from your `dotenv` will be surfaced in the UI when creating databases.

 Back to top

# Runtime configuration [¶](\#runtime-configuration "Permanent link")

Through the notebook settings menu, you can configure how and when marimo runs
cells.

## Disable autorun on startup [¶](\#on-startup "Permanent link")

By default, marimo notebooks run automatically on startup; just how the command

executes a script,

executes the notebook.

Disable this behavior by unchecking "Autorun on startup".

_When sharing a notebook as an app with `marimo run`, this setting has_
_no effect._

## Disable autorun on cell change (lazy execution) [¶](\#disable-autorun-on-cell-change-lazy-execution "Permanent link")

By default, when a cell is run or a UI element is interacted with, marimo
automatically runs cells that reference any of its variables. **You can disable**
**automatic execution of cell's descendants in the notebook settings menu by**
**setting `"On cell change"` to `"lazy"`.**

When the runtime is lazy, running a cell marks affected cells as stale but
doesn't automatically run them. Lazy evaluation means cells are only run when
their outputs are needed. If you run a cell that has stale ancestors, those
ancestors will also run to make sure your cell doesn't use stale inputs. You
can always click the notebook run button or use the keyboard shortcut to run
all stale cells.

**When should I use lazy evaluation?** Choosing the lazy runtime can be helpful
when working on notebooks with expensive cells.

Tip: speed up expensive notebooks with marimo's smart caching

In addition to runtime configuration, marimo also provides [opt-in caching](../../../api/caching/)
to help you work with expensive or side-effectful notebooks. marimo's
can cache expensive functions in memory and expensive blocks of code to disk,
letting you skip entire sections of your code and automatically loading
variables in memory on notebook startup. Read our [caching\
guide](../../../api/caching/) to learn more.

_When sharing a notebook as an app with `marimo run`, this setting has_
_no effect._

## On module change [¶](\#on-module-change "Permanent link")

When module autoreloading is enabled, marimo automatically runs cells when you
edit Python files. Based on static analysis, the reloader only runs cells
affected by your edits. The reloader is recursive, meaning that marimo tracks
modifications for modules imported by your notebook's imported modules too.

Why autoreload?

Autoreloading enables a workflow that many developers find
productive: develop complex logic in Python modules, and use the marimo
notebook as a DAG or main script that orchestrates your logic.

Autoreloading comes in two types:

1. **autorun**: automatically re-runs cells affected by module modification.

When set to autorun, marimo's reloader automatically run cells when you edit Python files.

1. **lazy**: marks cells affected by module modifications as stale, letting you know which cells need to be re-run.

When set to lazy, marimo's reloader marks cells as stale when you edit Python files.

## Python path [¶](\#python-path "Permanent link")

By default, marimo will not add any additional directories to the Python path.
This keeps the behavior of `marimo edit nb.py` and `python nb.py` consistent.

You can add directories to the Python path by setting the `pythonpath` key in
the runtime configuration. These directories will be added to the head of
`sys.path`, similar to how the `PYTHONPATH` environment variable works. This
allows Python to find and import modules from these directories.

Consider alternatives to path manipulation

When possible, it's preferred to avoid path manipulation. If you want to
work on a module in a separate directory alongside your notebooks, we
recommend creating a _package_ and including marimo as a project dependency.

This will make it easier to share your library code. For multiple packages,
consider configuring [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/).

See our guide on [notebooks in existing\
projects](../../package_management/notebooks_in_projects/) for more details.

## Environment variables [¶](\#environment-variables "Permanent link")

### .env files [¶](\#env-files "Permanent link")

marimo supports loading environment variables from `.env` files. This is useful for managing configuration that should not be committed to version control, such as API keys or database credentials.

The `.env` next to your `pyproject.toml` is loaded by default. To configure multiple or a different location, you can specify them in your configuration:

Environment variables from your `dotenv` will be surfaced in the UI when creating databases.


---

###### Export to other formats - marimoExporting to HTML and other formats¶

*Source: [https://docs.marimo.io/guides/exporting/](https://docs.marimo.io/guides/exporting/)*

<!-- Source: https://docs.marimo.io/guides/exporting/ -->
<!-- Title: Export to other formats - marimoExporting to HTML and other formats¶ -->
<!-- Depth: 6 -->

[Skip to content](#exporting-to-html-and-other-formats)

# Exporting to HTML and other formats [¶](\#exporting-to-html-and-other-formats "Permanent link")

Export marimo notebooks to other file formats at the command line using

## Export to static HTML [¶](\#export-to-static-html "Permanent link")

### Export from a running notebook [¶](\#export-from-a-running-notebook "Permanent link")

Export the current view your notebook to static HTML via the notebook
menu:

![](/_static/docs-html-export.png)Download as static HTML.

Additionally, you can configure individual notebooks to automatically
save as HTML through the notebook menu. These automatic snapshots are
saved to a folder called `__marimo__` in the notebook directory.

### Export from the command line [¶](\#export-from-the-command-line "Permanent link")

Export to HTML at the command line:

or watch the notebook for changes and automatically export to HTML:

When you export from the command line, marimo runs your notebook to produce
its visual outputs before saving as HTML.

Note

If any cells error during the export process, the status code will be non-zero. However, the export result may still be generated, with the error included in the output.
Errors can be ignored by appending `|| true` to the command, e.g. `marimo export html notebook.py || true`.

### Pre-render HTML exports [¶](\#pre-render-html-exports "Permanent link")

Static marimo exports execute Javascript to render the notebook source code as HTML at browser runtime. If you would like to directly serve the HTML representation of your notebook, you can run the following post-processing script and serve the resulting file instead.

## Export to a Python script [¶](\#export-to-a-python-script "Permanent link")

Export to a flat Python script in topological order, so the cells adhere to
their dependency graph.

Top-level await not supported

Exporting to a flat Python script does not support top-level await. If you have
top-level await in your notebook, you can still execute the notebook as a
script with `python notebook.py`.

## Export to markdown [¶](\#export-to-markdown "Permanent link")

Export to markdown notebook in top to bottom order, so the cells are in the
order as they appear in the notebook.

This can be useful to plug into other tools that read markdown, such as [Quarto](https://quarto.org/) or [MyST](https://myst-parser.readthedocs.io/).

marimo can directly open markdown files as notebooks

Learn more with `marimo tutorial markdown-format` at the command line.

You can also convert the markdown back to a marimo notebook:

## Export to Jupyter notebook [¶](\#export-to-jupyter-notebook "Permanent link")

Export to Jupyter notebook in topological order, so the cells adhere to
their dependency graph.

## Exporting to PDF, slides, or rst [¶](\#exporting-to-pdf-slides-or-rst "Permanent link")

The marimo [Quarto](https://www.github.com/marimo-team/quarto-marimo) plugin
enables exporting to PDF and other formats with Pandoc. See this [publishing](../publishing/quarto/) for more details.

However, if you export to a Jupyter notebook, you can leverage various other Jupyter ecosystem tools. For PDFs, you will
need to have [Pandoc](https://nbconvert.readthedocs.io/en/latest/install.html#installing-pandoc) and [Tex](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex) installed. The examples below use `uvx`, which you can obtain by [installing `uv`](https://docs.astral.sh/uv/getting-started/installation/).

You can also use other tools that work with Jupyter notebooks:

- [Quarto](https://quarto.org) \- Create beautiful documents, websites, presentations
- [nbgrader](https://nbgrader.readthedocs.io/) \- Grade notebook assignments

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../wasm/):

The exported HTML file will run your notebook using WebAssembly, making it completely self-contained and executable in the browser. This means users can interact with your notebook without needing Python or marimo installed.

Options:

- `--mode`: Choose between `run` (read-only) or `edit` (allows editing)
- `--output`: Directory to save the HTML and required assets
- `--show-code/--no-show-code`: Whether to initially show or hide the code in the notebook
- `--watch/--no-watch`: Watch the notebook for changes and automatically export
- `--include-cloudflare`: Write configuration files necessary for deploying to Cloudflare

Note that WebAssembly notebooks have [limitations](../wasm/#limitations); in particular,
[many but not all packages work](../wasm/#packages).

Note

The exported file must be served over HTTP to function correctly - it
cannot be opened directly from the filesystem ( `file://`). Your server must
also serve the assets in the `assets` directory, next to the HTML file. For
a simpler publishing experience, publish to [GitHub\
Pages](../publishing/github_pages/), [Cloudflare](../publishing/cloudflare/) or use the [online\
playground](../publishing/playground/).

Deploying to Cloudflare

You can include `--include-cloudflare` for deploying to Cloudflare. For example:

To run locally, run:

To deploy to Cloudflare, run:

### Testing the export [¶](\#testing-the-export "Permanent link")

You can test the export by running the following command in the directory containing your notebook:

### Including data files [¶](\#including-data-files "Permanent link")

See the docs for [mo.notebook\_location](../../api/miscellaneous/#marimo.notebook_location "            marimo.notebook_location") to learn how
to include data files in exported WASM HTML notebooks.

### Publishing to GitHub Pages [¶](\#publishing-to-github-pages "Permanent link")

After exporting your notebook to WASM HTML, you can publish it to
[GitHub Pages](https://pages.github.com/) for free. See our [guide on\
GitHub Pages](../publishing/github_pages/) to learn more.

### Exporting multiple notebooks [¶](\#exporting-multiple-notebooks "Permanent link")

In order to export multiple notebooks under the same folder, you can use the following snippet:

Optionally, you can create an `index.html` file in the public directory:

## 🏝️ Embed marimo outputs in HTML using Islands [¶](\#embed-marimo-outputs-in-html-using-islands "Permanent link")

Preview

Islands are an early feature. While the API likely won't change, there are some improvements we'd like to make before we consider them stable.
Please let us know on [GitHub](https://github.com/marimo-team/marimo/issues) if you run into any issues or have any feedback!

marimo islands are a way to embed marimo outputs and/or python code in your HTML that will become interactive when the page is loaded. This is useful for creating interactive blog posts, tutorials, and educational materials, all powered by marimo's reactive runtime.

Check out an [example island-powered document](../island_example/).

### Generating islands [¶](\#generating-islands "Permanent link")

Use `MarimoIslandGenerator` to generate HTML for islands

Example

From code blocksFrom notebook files

Any relevant `.html` that gets generated can be run through the [`development.md`](https://github.com/marimo-team/marimo/blob/main/frontend/islands/development.md) file instructions.

### Islands in action [¶](\#islands-in-action "Permanent link")

Advanced topic!

Islands are an advanced concept that is meant to be a building block for creating integrations with existing tools such as static site generators or documentation tools.

In order to use marimo islands, you need to import the necessary JS/CSS headers in your HTML file, and use our custom HTML tags to define the islands.

## `` marimo.MarimoIslandGenerator [¶](\#marimo.MarimoIslandGenerator "Permanent link")

Generates Marimo islands for embedding in other pages.

This is a great way to use another SSG framework that converts
Python code to HTML using marimo-islands.

Generally you will want to:

1. Find all the code snippets and add them to the generator.
2. Build the app.
3. Replace all code snippets with the rendered HTML.
4. Include the header in the tag.

Examples:
Using the MarimoIslandGenerator class:

You can also create the generator from a file:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `app_id` | The optional identifier of the app, defaults to `main`.<br>**TYPE:** `str`**DEFAULT:** `'main'` |

### `` has\_run`instance-attribute`[¶](\#marimo.MarimoIslandGenerator.has_run "Permanent link")

### `` add\_code [¶](\#marimo.MarimoIslandGenerator.add_code "Permanent link")

Add a code cell to the app.

_Args:_

- code (str): The code to add to the app.
- display\_code (bool): Whether to display the code in the HTML.
- display\_output (bool): Whether to display the output in the HTML.
- is\_raw (bool): Whether to handled the code without formatting.
- is\_reactive (bool): Whether this code block will run with pyodide.

### `` build`async`[¶](\#marimo.MarimoIslandGenerator.build "Permanent link")

Build the app. This should be called after adding all the code cells.

_Returns:_

- App: The built app.

### `` from\_file`staticmethod`[¶](\#marimo.MarimoIslandGenerator.from_file "Permanent link")

Create a MarimoIslandGenerator and populate MarimoIslandStubs
using code cells from a marimo \*.py file.

_Args:_

- filename (str): Marimo .py filename to convert to reactive HTML.
- display\_code (bool): Whether to display the code in HTML snippets.

### `` render\_body [¶](\#marimo.MarimoIslandGenerator.render_body "Permanent link")

Render the body for the app.
This should be included in the tag of the page.

_Args:_
\- include\_init\_island (bool): If True, adds initialization loader.
\- max\_width (str): CSS style max\_width property.
\- margin (str): CSS style margin property.
\- style (str): CSS style. Overrides max\_width and margin.

### `` render\_head [¶](\#marimo.MarimoIslandGenerator.render_head "Permanent link")

Render the header for the app.
This should be included in the tag of the page.

_Args:_

- version\_override (str): Marimo version to use for loaded js/css.
- \_development\_url (str): If True, uses local marimo islands js.

### `` render\_html [¶](\#marimo.MarimoIslandGenerator.render_html "Permanent link")

Render reactive html for the app.

_Args:_

- version\_override (str): Marimo version to use for loaded js/css.
- \_development\_url (str): If True, uses local marimo islands js.
- include\_init\_island (bool): If True, adds initialization loader.
- max\_width (str): CSS style max\_width property.
- margin (str): CSS style margin property.
- style (str): CSS style. Overrides max\_width and margin.

### `` render\_init\_island [¶](\#marimo.MarimoIslandGenerator.render_init_island "Permanent link")

Renders a static html MarimoIsland str which displays a spinning
initialization loader while Pyodide loads and disappears once
the kernel is ready to use.

Back to top

# Exporting to HTML and other formats [¶](\#exporting-to-html-and-other-formats "Permanent link")

Export marimo notebooks to other file formats at the command line using

## Export to static HTML [¶](\#export-to-static-html "Permanent link")

### Export from a running notebook [¶](\#export-from-a-running-notebook "Permanent link")

Export the current view your notebook to static HTML via the notebook
menu:

![](/_static/docs-html-export.png)Download as static HTML.

Additionally, you can configure individual notebooks to automatically
save as HTML through the notebook menu. These automatic snapshots are
saved to a folder called `__marimo__` in the notebook directory.

### Export from the command line [¶](\#export-from-the-command-line "Permanent link")

Export to HTML at the command line:

or watch the notebook for changes and automatically export to HTML:

When you export from the command line, marimo runs your notebook to produce
its visual outputs before saving as HTML.

Note

If any cells error during the export process, the status code will be non-zero. However, the export result may still be generated, with the error included in the output.
Errors can be ignored by appending `|| true` to the command, e.g. `marimo export html notebook.py || true`.

### Pre-render HTML exports [¶](\#pre-render-html-exports "Permanent link")

Static marimo exports execute Javascript to render the notebook source code as HTML at browser runtime. If you would like to directly serve the HTML representation of your notebook, you can run the following post-processing script and serve the resulting file instead.

## Export to a Python script [¶](\#export-to-a-python-script "Permanent link")

Export to a flat Python script in topological order, so the cells adhere to
their dependency graph.

Top-level await not supported

Exporting to a flat Python script does not support top-level await. If you have
top-level await in your notebook, you can still execute the notebook as a
script with `python notebook.py`.

## Export to markdown [¶](\#export-to-markdown "Permanent link")

Export to markdown notebook in top to bottom order, so the cells are in the
order as they appear in the notebook.

This can be useful to plug into other tools that read markdown, such as [Quarto](https://quarto.org/) or [MyST](https://myst-parser.readthedocs.io/).

marimo can directly open markdown files as notebooks

Learn more with `marimo tutorial markdown-format` at the command line.

You can also convert the markdown back to a marimo notebook:

## Export to Jupyter notebook [¶](\#export-to-jupyter-notebook "Permanent link")

Export to Jupyter notebook in topological order, so the cells adhere to
their dependency graph.

## Exporting to PDF, slides, or rst [¶](\#exporting-to-pdf-slides-or-rst "Permanent link")

The marimo [Quarto](https://www.github.com/marimo-team/quarto-marimo) plugin
enables exporting to PDF and other formats with Pandoc. See this [publishing](../publishing/quarto/) for more details.

However, if you export to a Jupyter notebook, you can leverage various other Jupyter ecosystem tools. For PDFs, you will
need to have [Pandoc](https://nbconvert.readthedocs.io/en/latest/install.html#installing-pandoc) and [Tex](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex) installed. The examples below use `uvx`, which you can obtain by [installing `uv`](https://docs.astral.sh/uv/getting-started/installation/).

You can also use other tools that work with Jupyter notebooks:

- [Quarto](https://quarto.org) \- Create beautiful documents, websites, presentations
- [nbgrader](https://nbgrader.readthedocs.io/) \- Grade notebook assignments

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../wasm/):

The exported HTML file will run your notebook using WebAssembly, making it completely self-contained and executable in the browser. This means users can interact with your notebook without needing Python or marimo installed.

Options:

- `--mode`: Choose between `run` (read-only) or `edit` (allows editing)
- `--output`: Directory to save the HTML and required assets
- `--show-code/--no-show-code`: Whether to initially show or hide the code in the notebook
- `--watch/--no-watch`: Watch the notebook for changes and automatically export
- `--include-cloudflare`: Write configuration files necessary for deploying to Cloudflare

Note that WebAssembly notebooks have [limitations](../wasm/#limitations); in particular,
[many but not all packages work](../wasm/#packages).

Note

The exported file must be served over HTTP to function correctly - it
cannot be opened directly from the filesystem ( `file://`). Your server must
also serve the assets in the `assets` directory, next to the HTML file. For
a simpler publishing experience, publish to [GitHub\
Pages](../publishing/github_pages/), [Cloudflare](../publishing/cloudflare/) or use the [online\
playground](../publishing/playground/).

Deploying to Cloudflare

You can include `--include-cloudflare` for deploying to Cloudflare. For example:

To run locally, run:

To deploy to Cloudflare, run:

### Testing the export [¶](\#testing-the-export "Permanent link")

You can test the export by running the following command in the directory containing your notebook:

### Including data files [¶](\#including-data-files "Permanent link")

See the docs for [mo.notebook\_location](../../api/miscellaneous/#marimo.notebook_location "            marimo.notebook_location") to learn how
to include data files in exported WASM HTML notebooks.

### Publishing to GitHub Pages [¶](\#publishing-to-github-pages "Permanent link")

After exporting your notebook to WASM HTML, you can publish it to
[GitHub Pages](https://pages.github.com/) for free. See our [guide on\
GitHub Pages](../publishing/github_pages/) to learn more.

### Exporting multiple notebooks [¶](\#exporting-multiple-notebooks "Permanent link")

In order to export multiple notebooks under the same folder, you can use the following snippet:

Optionally, you can create an `index.html` file in the public directory:

## 🏝️ Embed marimo outputs in HTML using Islands [¶](\#embed-marimo-outputs-in-html-using-islands "Permanent link")

Preview

Islands are an early feature. While the API likely won't change, there are some improvements we'd like to make before we consider them stable.
Please let us know on [GitHub](https://github.com/marimo-team/marimo/issues) if you run into any issues or have any feedback!

marimo islands are a way to embed marimo outputs and/or python code in your HTML that will become interactive when the page is loaded. This is useful for creating interactive blog posts, tutorials, and educational materials, all powered by marimo's reactive runtime.

Check out an [example island-powered document](../island_example/).

### Generating islands [¶](\#generating-islands "Permanent link")

Use `MarimoIslandGenerator` to generate HTML for islands

Example

From code blocksFrom notebook files

Any relevant `.html` that gets generated can be run through the [`development.md`](https://github.com/marimo-team/marimo/blob/main/frontend/islands/development.md) file instructions.

### Islands in action [¶](\#islands-in-action "Permanent link")

Advanced topic!

Islands are an advanced concept that is meant to be a building block for creating integrations with existing tools such as static site generators or documentation tools.

In order to use marimo islands, you need to import the necessary JS/CSS headers in your HTML file, and use our custom HTML tags to define the islands.

## `` marimo.MarimoIslandGenerator [¶](\#marimo.MarimoIslandGenerator "Permanent link")

Generates Marimo islands for embedding in other pages.

This is a great way to use another SSG framework that converts
Python code to HTML using marimo-islands.

Generally you will want to:

1. Find all the code snippets and add them to the generator.
2. Build the app.
3. Replace all code snippets with the rendered HTML.
4. Include the header in the tag.

Examples:
Using the MarimoIslandGenerator class:

You can also create the generator from a file:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `app_id` | The optional identifier of the app, defaults to `main`.<br>**TYPE:** `str`**DEFAULT:** `'main'` |

### `` has\_run`instance-attribute`[¶](\#marimo.MarimoIslandGenerator.has_run "Permanent link")

### `` add\_code [¶](\#marimo.MarimoIslandGenerator.add_code "Permanent link")

Add a code cell to the app.

_Args:_

- code (str): The code to add to the app.
- display\_code (bool): Whether to display the code in the HTML.
- display\_output (bool): Whether to display the output in the HTML.
- is\_raw (bool): Whether to handled the code without formatting.
- is\_reactive (bool): Whether this code block will run with pyodide.

### `` build`async`[¶](\#marimo.MarimoIslandGenerator.build "Permanent link")

Build the app. This should be called after adding all the code cells.

_Returns:_

- App: The built app.

### `` from\_file`staticmethod`[¶](\#marimo.MarimoIslandGenerator.from_file "Permanent link")

Create a MarimoIslandGenerator and populate MarimoIslandStubs
using code cells from a marimo \*.py file.

_Args:_

- filename (str): Marimo .py filename to convert to reactive HTML.
- display\_code (bool): Whether to display the code in HTML snippets.

### `` render\_body [¶](\#marimo.MarimoIslandGenerator.render_body "Permanent link")

Render the body for the app.
This should be included in the tag of the page.

_Args:_
\- include\_init\_island (bool): If True, adds initialization loader.
\- max\_width (str): CSS style max\_width property.
\- margin (str): CSS style margin property.
\- style (str): CSS style. Overrides max\_width and margin.

### `` render\_head [¶](\#marimo.MarimoIslandGenerator.render_head "Permanent link")

Render the header for the app.
This should be included in the tag of the page.

_Args:_

- version\_override (str): Marimo version to use for loaded js/css.
- \_development\_url (str): If True, uses local marimo islands js.

### `` render\_html [¶](\#marimo.MarimoIslandGenerator.render_html "Permanent link")

Render reactive html for the app.

_Args:_

- version\_override (str): Marimo version to use for loaded js/css.
- \_development\_url (str): If True, uses local marimo islands js.
- include\_init\_island (bool): If True, adds initialization loader.
- max\_width (str): CSS style max\_width property.
- margin (str): CSS style margin property.
- style (str): CSS style. Overrides max\_width and margin.

### `` render\_init\_island [¶](\#marimo.MarimoIslandGenerator.render_init_island "Permanent link")

Renders a static html MarimoIsland str which displays a spinning
initialization loader while Pyodide loads and disappears once
the kernel is ready to use.

 Back to top

# Exporting to HTML and other formats [¶](\#exporting-to-html-and-other-formats "Permanent link")

Export marimo notebooks to other file formats at the command line using

## Export to static HTML [¶](\#export-to-static-html "Permanent link")

### Export from a running notebook [¶](\#export-from-a-running-notebook "Permanent link")

Export the current view your notebook to static HTML via the notebook
menu:

![](/_static/docs-html-export.png)Download as static HTML.

Additionally, you can configure individual notebooks to automatically
save as HTML through the notebook menu. These automatic snapshots are
saved to a folder called `__marimo__` in the notebook directory.

### Export from the command line [¶](\#export-from-the-command-line "Permanent link")

Export to HTML at the command line:

or watch the notebook for changes and automatically export to HTML:

When you export from the command line, marimo runs your notebook to produce
its visual outputs before saving as HTML.

Note

If any cells error during the export process, the status code will be non-zero. However, the export result may still be generated, with the error included in the output.
Errors can be ignored by appending `|| true` to the command, e.g. `marimo export html notebook.py || true`.

### Pre-render HTML exports [¶](\#pre-render-html-exports "Permanent link")

Static marimo exports execute Javascript to render the notebook source code as HTML at browser runtime. If you would like to directly serve the HTML representation of your notebook, you can run the following post-processing script and serve the resulting file instead.

## Export to a Python script [¶](\#export-to-a-python-script "Permanent link")

Export to a flat Python script in topological order, so the cells adhere to
their dependency graph.

Top-level await not supported

Exporting to a flat Python script does not support top-level await. If you have
top-level await in your notebook, you can still execute the notebook as a
script with `python notebook.py`.

## Export to markdown [¶](\#export-to-markdown "Permanent link")

Export to markdown notebook in top to bottom order, so the cells are in the
order as they appear in the notebook.

This can be useful to plug into other tools that read markdown, such as [Quarto](https://quarto.org/) or [MyST](https://myst-parser.readthedocs.io/).

marimo can directly open markdown files as notebooks

Learn more with `marimo tutorial markdown-format` at the command line.

You can also convert the markdown back to a marimo notebook:

## Export to Jupyter notebook [¶](\#export-to-jupyter-notebook "Permanent link")

Export to Jupyter notebook in topological order, so the cells adhere to
their dependency graph.

## Exporting to PDF, slides, or rst [¶](\#exporting-to-pdf-slides-or-rst "Permanent link")

The marimo [Quarto](https://www.github.com/marimo-team/quarto-marimo) plugin
enables exporting to PDF and other formats with Pandoc. See this [publishing](../publishing/quarto/) for more details.

However, if you export to a Jupyter notebook, you can leverage various other Jupyter ecosystem tools. For PDFs, you will
need to have [Pandoc](https://nbconvert.readthedocs.io/en/latest/install.html#installing-pandoc) and [Tex](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex) installed. The examples below use `uvx`, which you can obtain by [installing `uv`](https://docs.astral.sh/uv/getting-started/installation/).

You can also use other tools that work with Jupyter notebooks:

- [Quarto](https://quarto.org) \- Create beautiful documents, websites, presentations
- [nbgrader](https://nbgrader.readthedocs.io/) \- Grade notebook assignments

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../wasm/):

The exported HTML file will run your notebook using WebAssembly, making it completely self-contained and executable in the browser. This means users can interact with your notebook without needing Python or marimo installed.

Options:

- `--mode`: Choose between `run` (read-only) or `edit` (allows editing)
- `--output`: Directory to save the HTML and required assets
- `--show-code/--no-show-code`: Whether to initially show or hide the code in the notebook
- `--watch/--no-watch`: Watch the notebook for changes and automatically export
- `--include-cloudflare`: Write configuration files necessary for deploying to Cloudflare

Note that WebAssembly notebooks have [limitations](../wasm/#limitations); in particular,
[many but not all packages work](../wasm/#packages).

Note

The exported file must be served over HTTP to function correctly - it
cannot be opened directly from the filesystem ( `file://`). Your server must
also serve the assets in the `assets` directory, next to the HTML file. For
a simpler publishing experience, publish to [GitHub\
Pages](../publishing/github_pages/), [Cloudflare](../publishing/cloudflare/) or use the [online\
playground](../publishing/playground/).

Deploying to Cloudflare

You can include `--include-cloudflare` for deploying to Cloudflare. For example:

To run locally, run:

To deploy to Cloudflare, run:

### Testing the export [¶](\#testing-the-export "Permanent link")

You can test the export by running the following command in the directory containing your notebook:

### Including data files [¶](\#including-data-files "Permanent link")

See the docs for [mo.notebook\_location](../../api/miscellaneous/#marimo.notebook_location "            marimo.notebook_location") to learn how
to include data files in exported WASM HTML notebooks.

### Publishing to GitHub Pages [¶](\#publishing-to-github-pages "Permanent link")

After exporting your notebook to WASM HTML, you can publish it to
[GitHub Pages](https://pages.github.com/) for free. See our [guide on\
GitHub Pages](../publishing/github_pages/) to learn more.

### Exporting multiple notebooks [¶](\#exporting-multiple-notebooks "Permanent link")

In order to export multiple notebooks under the same folder, you can use the following snippet:

Optionally, you can create an `index.html` file in the public directory:

## 🏝️ Embed marimo outputs in HTML using Islands [¶](\#embed-marimo-outputs-in-html-using-islands "Permanent link")

Preview

Islands are an early feature. While the API likely won't change, there are some improvements we'd like to make before we consider them stable.
Please let us know on [GitHub](https://github.com/marimo-team/marimo/issues) if you run into any issues or have any feedback!

marimo islands are a way to embed marimo outputs and/or python code in your HTML that will become interactive when the page is loaded. This is useful for creating interactive blog posts, tutorials, and educational materials, all powered by marimo's reactive runtime.

Check out an [example island-powered document](../island_example/).

### Generating islands [¶](\#generating-islands "Permanent link")

Use `MarimoIslandGenerator` to generate HTML for islands

Example

From code blocksFrom notebook files

Any relevant `.html` that gets generated can be run through the [`development.md`](https://github.com/marimo-team/marimo/blob/main/frontend/islands/development.md) file instructions.

### Islands in action [¶](\#islands-in-action "Permanent link")

Advanced topic!

Islands are an advanced concept that is meant to be a building block for creating integrations with existing tools such as static site generators or documentation tools.

In order to use marimo islands, you need to import the necessary JS/CSS headers in your HTML file, and use our custom HTML tags to define the islands.

## `` marimo.MarimoIslandGenerator [¶](\#marimo.MarimoIslandGenerator "Permanent link")

Generates Marimo islands for embedding in other pages.

This is a great way to use another SSG framework that converts
Python code to HTML using marimo-islands.

Generally you will want to:

1. Find all the code snippets and add them to the generator.
2. Build the app.
3. Replace all code snippets with the rendered HTML.
4. Include the header in the tag.

Examples:
Using the MarimoIslandGenerator class:

You can also create the generator from a file:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `app_id` | The optional identifier of the app, defaults to `main`.<br>**TYPE:** `str`**DEFAULT:** `'main'` |

### `` has\_run`instance-attribute`[¶](\#marimo.MarimoIslandGenerator.has_run "Permanent link")

### `` add\_code [¶](\#marimo.MarimoIslandGenerator.add_code "Permanent link")

Add a code cell to the app.

_Args:_

- code (str): The code to add to the app.
- display\_code (bool): Whether to display the code in the HTML.
- display\_output (bool): Whether to display the output in the HTML.
- is\_raw (bool): Whether to handled the code without formatting.
- is\_reactive (bool): Whether this code block will run with pyodide.

### `` build`async`[¶](\#marimo.MarimoIslandGenerator.build "Permanent link")

Build the app. This should be called after adding all the code cells.

_Returns:_

- App: The built app.

### `` from\_file`staticmethod`[¶](\#marimo.MarimoIslandGenerator.from_file "Permanent link")

Create a MarimoIslandGenerator and populate MarimoIslandStubs
using code cells from a marimo \*.py file.

_Args:_

- filename (str): Marimo .py filename to convert to reactive HTML.
- display\_code (bool): Whether to display the code in HTML snippets.

### `` render\_body [¶](\#marimo.MarimoIslandGenerator.render_body "Permanent link")

Render the body for the app.
This should be included in the tag of the page.

_Args:_
\- include\_init\_island (bool): If True, adds initialization loader.
\- max\_width (str): CSS style max\_width property.
\- margin (str): CSS style margin property.
\- style (str): CSS style. Overrides max\_width and margin.

### `` render\_head [¶](\#marimo.MarimoIslandGenerator.render_head "Permanent link")

Render the header for the app.
This should be included in the tag of the page.

_Args:_

- version\_override (str): Marimo version to use for loaded js/css.
- \_development\_url (str): If True, uses local marimo islands js.

### `` render\_html [¶](\#marimo.MarimoIslandGenerator.render_html "Permanent link")

Render reactive html for the app.

_Args:_

- version\_override (str): Marimo version to use for loaded js/css.
- \_development\_url (str): If True, uses local marimo islands js.
- include\_init\_island (bool): If True, adds initialization loader.
- max\_width (str): CSS style max\_width property.
- margin (str): CSS style margin property.
- style (str): CSS style. Overrides max\_width and margin.

### `` render\_init\_island [¶](\#marimo.MarimoIslandGenerator.render_init_island "Permanent link")

Renders a static html MarimoIsland str which displays a spinning
initialization loader while Pyodide loads and disappears once
the kernel is ready to use.


---

###### Running cells - marimoRunning cells¶

*Source: [https://docs.marimo.io/guides/reactivity/](https://docs.marimo.io/guides/reactivity/)*

<!-- Source: https://docs.marimo.io/guides/reactivity/ -->
<!-- Title: Running cells - marimoRunning cells¶ -->
<!-- Depth: 6 -->

[Skip to content](#running-cells)

# Running cells [¶](\#running-cells "Permanent link")

marimo _reacts_ to your code changes: run a cell, and all other cells that
refer to the variables it defines are automatically run with the latest data.
This keeps your code and outputs consistent, and eliminates bugs before they
happen.

Why run cells reactively?

marimo's "reactive" execution model makes your notebooks more reproducible
by eliminating hidden state and providing a deterministic execution order.
It also powers marimo's support for [interactive\
elements](../interactivity/), for running as apps, and executing as
scripts.

How marimo runs cells is one of the biggest differences between marimo and
traditional notebooks like Jupyter. Learn more at our
[FAQ](../../faq/#faq-jupyter).

Working with expensive notebooks

marimo provides tools for working with expensive notebooks, in which cells
might take a long time to run or have side-effects.

- The [runtime can be configured](../configuration/runtime_configuration/)
to be **lazy** instead of
automatic, marking cells as stale instead of running them.
- Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to conditionally
stop execution at runtime.

See [the expensive notebooks guide](../expensive_notebooks/) for more tips.

## How marimo runs cells [¶](\#how-marimo-runs-cells "Permanent link")

marimo statically analyzes each cell (i.e., without running it) to determine
its

- references, the global variables it reads but doesn't define;
- definitions, the global variables it defines.

It then forms a directed acyclic graph (DAG) on cells, with an edge from
one cell to another if the latter references any of the definitions of the
former. When a cell is run, its descendants are marked for execution.

Visualizing the DAG

marimo provides several tools to help you visualize and navigate this dependency graph,
including a [dependency explorer](../editor_features/dataflow/#dependency-explorer),
[minimap](../editor_features/dataflow/#minimap),
and [reactive reference highlighting](../editor_features/dataflow/#reactive-reference-highlighting).

See the [understanding dataflow](../editor_features/dataflow/) guide for details.

Runtime Rule

When a cell is run, marimo automatically runs all other cells that
**reference** any of the global variables it **defines**.

marimo [does not track mutations](#variable-mutations-are-not-tracked) to
variables, nor assignments to attributes. That means that if you assign an
attribute like `foo.bar = 10`, other cells referencing `foo.bar` will _not_ be
run.

### Execution order [¶](\#execution-order "Permanent link")

The order cells are executed in is determined by the relationships between
cells and their variables, not by the order of cells on the page (similar
to a spreadsheet). This lets you organize your code in whatever way makes the
most sense to you. For example, you can put helper functions at the bottom of
your notebook.

### Deleting a cell deletes its variables [¶](\#deleting-a-cell-deletes-its-variables "Permanent link")

In marimo, _deleting a cell deletes its global variables from program memory_.
Cells that previously referenced these variables are automatically re-run and
invalidated (or marked as stale, depending on your [runtime\
configuration](../configuration/runtime_configuration/)). In this way, marimo
eliminates a common cause of bugs in traditional notebooks like Jupyter.

### Variable mutations are not tracked [¶](\#variable-mutations-are-not-tracked "Permanent link")

marimo does not track mutations to objects, _e.g._, mutations like
`my_list.append(42)` or `my_object.value = 42` don't trigger reactive re-runs of
other cells. **Avoid defining a variable in one cell and**
**mutating it in another**.

Why not track mutations?

Tracking mutations reliably is impossible in Python. Reacting to mutations
could result in surprising re-runs of notebook cells.

If you need to mutate a variable (such as adding a new column to a dataframe),
you should perform the mutation in the same cell as the one that defines it,
or try creating a new variable instead.

Create new variables, don't mutate existing ones

Do this ...... not this

Mutate variables in the cells that define them

Do this ...... not this

## Global variable names must be unique [¶](\#global-variable-names-must-be-unique "Permanent link")

**marimo requires that every global variable be defined by only one cell.**
This lets marimo keep code and outputs consistent.

Global variables

A variable can refer to any Python object. Functions, classes, and imported
names are all variables.

This rule encourages you to keep the number of global variables in your
program small, which is generally considered good practice.

### Creating temporary variables [¶](\#creating-temporary-variables "Permanent link")

marimo provides two ways to define temporary variables, which can
help keep the number of global variables in your notebook small.

#### Creating local variables [¶](\#creating-local-variables "Permanent link")

Variables prefixed with an underscore ( _e.g._, `_x`) are "local" to a
cell: they can't be read by other cells. Multiple cells can reuse the same
local variables names.

#### Encapsulating code in functions [¶](\#encapsulating-code-in-functions "Permanent link")

If you want most or all the variables in a cell to be temporary, prefixing each
variable with an underscore to make it local may feel inconvenient. In these
situations we recommend encapsulating the temporary variables in a function.

For example, if you find yourself copy-pasting the same plotting code across
multiple cells and only tweaking a few parameters, try the following pattern:

Here, the variables `plt`, `fig`, and `ax` aren't added to the globals.

### Managing memory [¶](\#managing-memory "Permanent link")

Because variable names must be unique, you cannot reassign variables as a means
of freeing memory. Instead, manage memory by encapsulating code in functions or
using the `del` operator. See our guide on [expensive\
notebooks](../expensive_notebooks/#manage-memory) to learn more.

## Configuring how marimo runs cells [¶](\#configuring-how-marimo-runs-cells "Permanent link")

Through the notebook settings menu, you can configure how and when marimo runs
cells. In particular, you can disable autorun on startup, disable autorun
on cell execution, and enable a module autoreloader. Read our
[runtime configuration guide](../configuration/runtime_configuration/) to learn more.

## Disabling cells [¶](\#disabling-cells "Permanent link")

Sometimes, you may want to edit one part of a notebook without triggering
automatic execution of its dependent cells. For example, the dependent cells
may take a long time to execute, and you only want to iterate on the first part
of a multi-cell computation.

For cases like this, marimo lets you **disable** cells: when a cell is
disabled, it and its dependents are blocked from running.

Disabling a cell blocks it from running.

When you re-enable a cell, if any of the cell's ancestors ran while it was
disabled, marimo will automatically run it.

Enable a cell through the context menu. Stale cells run
automatically.

Back to top

# Running cells [¶](\#running-cells "Permanent link")

marimo _reacts_ to your code changes: run a cell, and all other cells that
refer to the variables it defines are automatically run with the latest data.
This keeps your code and outputs consistent, and eliminates bugs before they
happen.

Why run cells reactively?

marimo's "reactive" execution model makes your notebooks more reproducible
by eliminating hidden state and providing a deterministic execution order.
It also powers marimo's support for [interactive\
elements](../interactivity/), for running as apps, and executing as
scripts.

How marimo runs cells is one of the biggest differences between marimo and
traditional notebooks like Jupyter. Learn more at our
[FAQ](../../faq/#faq-jupyter).

Working with expensive notebooks

marimo provides tools for working with expensive notebooks, in which cells
might take a long time to run or have side-effects.

- The [runtime can be configured](../configuration/runtime_configuration/)
to be **lazy** instead of
automatic, marking cells as stale instead of running them.
- Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to conditionally
stop execution at runtime.

See [the expensive notebooks guide](../expensive_notebooks/) for more tips.

## How marimo runs cells [¶](\#how-marimo-runs-cells "Permanent link")

marimo statically analyzes each cell (i.e., without running it) to determine
its

- references, the global variables it reads but doesn't define;
- definitions, the global variables it defines.

It then forms a directed acyclic graph (DAG) on cells, with an edge from
one cell to another if the latter references any of the definitions of the
former. When a cell is run, its descendants are marked for execution.

Visualizing the DAG

marimo provides several tools to help you visualize and navigate this dependency graph,
including a [dependency explorer](../editor_features/dataflow/#dependency-explorer),
[minimap](../editor_features/dataflow/#minimap),
and [reactive reference highlighting](../editor_features/dataflow/#reactive-reference-highlighting).

See the [understanding dataflow](../editor_features/dataflow/) guide for details.

Runtime Rule

When a cell is run, marimo automatically runs all other cells that
**reference** any of the global variables it **defines**.

marimo [does not track mutations](#variable-mutations-are-not-tracked) to
variables, nor assignments to attributes. That means that if you assign an
attribute like `foo.bar = 10`, other cells referencing `foo.bar` will _not_ be
run.

### Execution order [¶](\#execution-order "Permanent link")

The order cells are executed in is determined by the relationships between
cells and their variables, not by the order of cells on the page (similar
to a spreadsheet). This lets you organize your code in whatever way makes the
most sense to you. For example, you can put helper functions at the bottom of
your notebook.

### Deleting a cell deletes its variables [¶](\#deleting-a-cell-deletes-its-variables "Permanent link")

In marimo, _deleting a cell deletes its global variables from program memory_.
Cells that previously referenced these variables are automatically re-run and
invalidated (or marked as stale, depending on your [runtime\
configuration](../configuration/runtime_configuration/)). In this way, marimo
eliminates a common cause of bugs in traditional notebooks like Jupyter.

### Variable mutations are not tracked [¶](\#variable-mutations-are-not-tracked "Permanent link")

marimo does not track mutations to objects, _e.g._, mutations like
`my_list.append(42)` or `my_object.value = 42` don't trigger reactive re-runs of
other cells. **Avoid defining a variable in one cell and**
**mutating it in another**.

Why not track mutations?

Tracking mutations reliably is impossible in Python. Reacting to mutations
could result in surprising re-runs of notebook cells.

If you need to mutate a variable (such as adding a new column to a dataframe),
you should perform the mutation in the same cell as the one that defines it,
or try creating a new variable instead.

Create new variables, don't mutate existing ones

Do this ...... not this

Mutate variables in the cells that define them

Do this ...... not this

## Global variable names must be unique [¶](\#global-variable-names-must-be-unique "Permanent link")

**marimo requires that every global variable be defined by only one cell.**
This lets marimo keep code and outputs consistent.

Global variables

A variable can refer to any Python object. Functions, classes, and imported
names are all variables.

This rule encourages you to keep the number of global variables in your
program small, which is generally considered good practice.

### Creating temporary variables [¶](\#creating-temporary-variables "Permanent link")

marimo provides two ways to define temporary variables, which can
help keep the number of global variables in your notebook small.

#### Creating local variables [¶](\#creating-local-variables "Permanent link")

Variables prefixed with an underscore ( _e.g._, `_x`) are "local" to a
cell: they can't be read by other cells. Multiple cells can reuse the same
local variables names.

#### Encapsulating code in functions [¶](\#encapsulating-code-in-functions "Permanent link")

If you want most or all the variables in a cell to be temporary, prefixing each
variable with an underscore to make it local may feel inconvenient. In these
situations we recommend encapsulating the temporary variables in a function.

For example, if you find yourself copy-pasting the same plotting code across
multiple cells and only tweaking a few parameters, try the following pattern:

Here, the variables `plt`, `fig`, and `ax` aren't added to the globals.

### Managing memory [¶](\#managing-memory "Permanent link")

Because variable names must be unique, you cannot reassign variables as a means
of freeing memory. Instead, manage memory by encapsulating code in functions or
using the `del` operator. See our guide on [expensive\
notebooks](../expensive_notebooks/#manage-memory) to learn more.

## Configuring how marimo runs cells [¶](\#configuring-how-marimo-runs-cells "Permanent link")

Through the notebook settings menu, you can configure how and when marimo runs
cells. In particular, you can disable autorun on startup, disable autorun
on cell execution, and enable a module autoreloader. Read our
[runtime configuration guide](../configuration/runtime_configuration/) to learn more.

## Disabling cells [¶](\#disabling-cells "Permanent link")

Sometimes, you may want to edit one part of a notebook without triggering
automatic execution of its dependent cells. For example, the dependent cells
may take a long time to execute, and you only want to iterate on the first part
of a multi-cell computation.

For cases like this, marimo lets you **disable** cells: when a cell is
disabled, it and its dependents are blocked from running.

Disabling a cell blocks it from running.

When you re-enable a cell, if any of the cell's ancestors ran while it was
disabled, marimo will automatically run it.

Enable a cell through the context menu. Stale cells run
automatically.

 Back to top

# Running cells [¶](\#running-cells "Permanent link")

marimo _reacts_ to your code changes: run a cell, and all other cells that
refer to the variables it defines are automatically run with the latest data.
This keeps your code and outputs consistent, and eliminates bugs before they
happen.

Why run cells reactively?

marimo's "reactive" execution model makes your notebooks more reproducible
by eliminating hidden state and providing a deterministic execution order.
It also powers marimo's support for [interactive\
elements](../interactivity/), for running as apps, and executing as
scripts.

How marimo runs cells is one of the biggest differences between marimo and
traditional notebooks like Jupyter. Learn more at our
[FAQ](../../faq/#faq-jupyter).

Working with expensive notebooks

marimo provides tools for working with expensive notebooks, in which cells
might take a long time to run or have side-effects.

- The [runtime can be configured](../configuration/runtime_configuration/)
to be **lazy** instead of
automatic, marking cells as stale instead of running them.
- Use [`mo.stop`](../../api/control_flow/#marimo.stop "            marimo.stop") to conditionally
stop execution at runtime.

See [the expensive notebooks guide](../expensive_notebooks/) for more tips.

## How marimo runs cells [¶](\#how-marimo-runs-cells "Permanent link")

marimo statically analyzes each cell (i.e., without running it) to determine
its

- references, the global variables it reads but doesn't define;
- definitions, the global variables it defines.

It then forms a directed acyclic graph (DAG) on cells, with an edge from
one cell to another if the latter references any of the definitions of the
former. When a cell is run, its descendants are marked for execution.

Visualizing the DAG

marimo provides several tools to help you visualize and navigate this dependency graph,
including a [dependency explorer](../editor_features/dataflow/#dependency-explorer),
[minimap](../editor_features/dataflow/#minimap),
and [reactive reference highlighting](../editor_features/dataflow/#reactive-reference-highlighting).

See the [understanding dataflow](../editor_features/dataflow/) guide for details.

Runtime Rule

When a cell is run, marimo automatically runs all other cells that
**reference** any of the global variables it **defines**.

marimo [does not track mutations](#variable-mutations-are-not-tracked) to
variables, nor assignments to attributes. That means that if you assign an
attribute like `foo.bar = 10`, other cells referencing `foo.bar` will _not_ be
run.

### Execution order [¶](\#execution-order "Permanent link")

The order cells are executed in is determined by the relationships between
cells and their variables, not by the order of cells on the page (similar
to a spreadsheet). This lets you organize your code in whatever way makes the
most sense to you. For example, you can put helper functions at the bottom of
your notebook.

### Deleting a cell deletes its variables [¶](\#deleting-a-cell-deletes-its-variables "Permanent link")

In marimo, _deleting a cell deletes its global variables from program memory_.
Cells that previously referenced these variables are automatically re-run and
invalidated (or marked as stale, depending on your [runtime\
configuration](../configuration/runtime_configuration/)). In this way, marimo
eliminates a common cause of bugs in traditional notebooks like Jupyter.

### Variable mutations are not tracked [¶](\#variable-mutations-are-not-tracked "Permanent link")

marimo does not track mutations to objects, _e.g._, mutations like
`my_list.append(42)` or `my_object.value = 42` don't trigger reactive re-runs of
other cells. **Avoid defining a variable in one cell and**
**mutating it in another**.

Why not track mutations?

Tracking mutations reliably is impossible in Python. Reacting to mutations
could result in surprising re-runs of notebook cells.

If you need to mutate a variable (such as adding a new column to a dataframe),
you should perform the mutation in the same cell as the one that defines it,
or try creating a new variable instead.

Create new variables, don't mutate existing ones

Do this ...... not this

Mutate variables in the cells that define them

Do this ...... not this

## Global variable names must be unique [¶](\#global-variable-names-must-be-unique "Permanent link")

**marimo requires that every global variable be defined by only one cell.**
This lets marimo keep code and outputs consistent.

Global variables

A variable can refer to any Python object. Functions, classes, and imported
names are all variables.

This rule encourages you to keep the number of global variables in your
program small, which is generally considered good practice.

### Creating temporary variables [¶](\#creating-temporary-variables "Permanent link")

marimo provides two ways to define temporary variables, which can
help keep the number of global variables in your notebook small.

#### Creating local variables [¶](\#creating-local-variables "Permanent link")

Variables prefixed with an underscore ( _e.g._, `_x`) are "local" to a
cell: they can't be read by other cells. Multiple cells can reuse the same
local variables names.

#### Encapsulating code in functions [¶](\#encapsulating-code-in-functions "Permanent link")

If you want most or all the variables in a cell to be temporary, prefixing each
variable with an underscore to make it local may feel inconvenient. In these
situations we recommend encapsulating the temporary variables in a function.

For example, if you find yourself copy-pasting the same plotting code across
multiple cells and only tweaking a few parameters, try the following pattern:

Here, the variables `plt`, `fig`, and `ax` aren't added to the globals.

### Managing memory [¶](\#managing-memory "Permanent link")

Because variable names must be unique, you cannot reassign variables as a means
of freeing memory. Instead, manage memory by encapsulating code in functions or
using the `del` operator. See our guide on [expensive\
notebooks](../expensive_notebooks/#manage-memory) to learn more.

## Configuring how marimo runs cells [¶](\#configuring-how-marimo-runs-cells "Permanent link")

Through the notebook settings menu, you can configure how and when marimo runs
cells. In particular, you can disable autorun on startup, disable autorun
on cell execution, and enable a module autoreloader. Read our
[runtime configuration guide](../configuration/runtime_configuration/) to learn more.

## Disabling cells [¶](\#disabling-cells "Permanent link")

Sometimes, you may want to edit one part of a notebook without triggering
automatic execution of its dependent cells. For example, the dependent cells
may take a long time to execute, and you only want to iterate on the first part
of a multi-cell computation.

For cases like this, marimo lets you **disable** cells: when a cell is
disabled, it and its dependents are blocked from running.

Disabling a cell blocks it from running.

When you re-enable a cell, if any of the cell's ancestors ran while it was
disabled, marimo will automatically run it.

Enable a cell through the context menu. Stale cells run
automatically.


---

###### Understanding errors - marimoUnderstanding errors¶

*Source: [https://docs.marimo.io/guides/understanding_errors/](https://docs.marimo.io/guides/understanding_errors/)*

<!-- Source: https://docs.marimo.io/guides/understanding_errors/ -->
<!-- Title: Understanding errors - marimoUnderstanding errors¶ -->
<!-- Depth: 6 -->

[Skip to content](#understanding-errors)

# Understanding errors [¶](\#understanding-errors "Permanent link")

marimo imposes a few constraints on your notebook code:

- no multiply defined variables: each variable can be defined in only one cell
- no cycles: if one cell declares variable `a` and reads `b`, then another cannot declare `b` and read `a`.
- no `import *`: importing all symbols from a library is not allowed

Why these constraints?

These constraints let marimo work its magic, making your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

When a cell violates any of these constraints, marimo doesn't run it and instead
reports an error. In these guides, we explain these errors and provide tips for
how to work around them.

These errors might be surprising at first, but spend just a
bit of time with marimo and adhering to these constraints will become second nature — and you'll get used to writing error-free code by default.

## Lint Rules [¶](\#lint-rules "Permanent link")

marimo includes a comprehensive linting system that automatically detects these constraints and other code quality issues. The linter can help you identify and fix problems before they cause runtime errors.

Run the linter using:

See the [Lint Rules](../lint_rules/) guide for a complete list of rules and detailed explanations.

| Guide | Description |
| --- | --- |
| [Multiple definitions](multiple_definitions/) | How to deal with variables defined in multiple cells |
| [Import `*`](import_star/) | Why you can't use `import *` |
| [Cycles](cycles/) | How to resolve cycle errors |
| [setup](setup/) | How to enable top level definitions |

Back to top

# Understanding errors [¶](\#understanding-errors "Permanent link")

marimo imposes a few constraints on your notebook code:

- no multiply defined variables: each variable can be defined in only one cell
- no cycles: if one cell declares variable `a` and reads `b`, then another cannot declare `b` and read `a`.
- no `import *`: importing all symbols from a library is not allowed

Why these constraints?

These constraints let marimo work its magic, making your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

When a cell violates any of these constraints, marimo doesn't run it and instead
reports an error. In these guides, we explain these errors and provide tips for
how to work around them.

These errors might be surprising at first, but spend just a
bit of time with marimo and adhering to these constraints will become second nature — and you'll get used to writing error-free code by default.

## Lint Rules [¶](\#lint-rules "Permanent link")

marimo includes a comprehensive linting system that automatically detects these constraints and other code quality issues. The linter can help you identify and fix problems before they cause runtime errors.

Run the linter using:

See the [Lint Rules](../lint_rules/) guide for a complete list of rules and detailed explanations.

| Guide | Description |
| --- | --- |
| [Multiple definitions](multiple_definitions/) | How to deal with variables defined in multiple cells |
| [Import `*`](import_star/) | Why you can't use `import *` |
| [Cycles](cycles/) | How to resolve cycle errors |
| [setup](setup/) | How to enable top level definitions |

 Back to top

# Understanding errors [¶](\#understanding-errors "Permanent link")

marimo imposes a few constraints on your notebook code:

- no multiply defined variables: each variable can be defined in only one cell
- no cycles: if one cell declares variable `a` and reads `b`, then another cannot declare `b` and read `a`.
- no `import *`: importing all symbols from a library is not allowed

Why these constraints?

These constraints let marimo work its magic, making your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

When a cell violates any of these constraints, marimo doesn't run it and instead
reports an error. In these guides, we explain these errors and provide tips for
how to work around them.

These errors might be surprising at first, but spend just a
bit of time with marimo and adhering to these constraints will become second nature — and you'll get used to writing error-free code by default.

## Lint Rules [¶](\#lint-rules "Permanent link")

marimo includes a comprehensive linting system that automatically detects these constraints and other code quality issues. The linter can help you identify and fix problems before they cause runtime errors.

Run the linter using:

See the [Lint Rules](../lint_rules/) guide for a complete list of rules and detailed explanations.

| Guide | Description |
| --- | --- |
| [Multiple definitions](multiple_definitions/) | How to deal with variables defined in multiple cells |
| [Import `*`](import_star/) | Why you can't use `import *` |
| [Cycles](cycles/) | How to resolve cycle errors |
| [setup](setup/) | How to enable top level definitions |


---

###### Caching - marimoCaching¶

*Source: [https://docs.marimo.io/api/caching/](https://docs.marimo.io/api/caching/)*

<!-- Source: https://docs.marimo.io/api/caching/ -->
<!-- Title: Caching - marimoCaching¶ -->
<!-- Depth: 7 -->

[Skip to content](#caching)

# Caching [¶](\#caching "Permanent link")

marimo comes with utilities to cache intermediate computations. These
utilities can be applied as decorators to functions to cache their
returned values; you can choose between saving caches in memory or to disk.

## Basic usage [¶](\#basic-usage "Permanent link")

marimo provides two decorators for caching the return values of expensive
functions:

1. [`mo.cache`](#marimo.cache "            marimo.cache"), which saves cached values to memory;
2. [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache"), which saves cached values to disk.

`mo.cache` `mo.persistent_cache` `mo.cache` (async) `mo.persistent_cache` (async)

Roughly speaking, the first time a cached function is called with a particular
sequence of arguments, the function will run and its return value will be
cached. The next time it is called with the same sequence of arguments (on
cache hit), the function body will be skipped and the return value will be
retrieved from cache instead.

The in-memory cache ( [`mo.cache`](#marimo.cache "            marimo.cache")) is faster and doesn't consume disk space, but
it is lost on notebook restart. The disk cache ( [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache")) is
slower and consumes space on disk, but it persists across notebook runs,
letting you pick up where you left off.

(For an in-memory cache of bounded size, use [`mo.lru_cache`](#marimo.lru_cache "            marimo.lru_cache").)

Async functions are fully supported

All cache decorators ( `mo.cache`, `mo.lru_cache`, `mo.persistent_cache`) work
seamlessly with both synchronous and asynchronous functions. When multiple
concurrent calls are made to a cached async function with the same arguments,
only one execution occurs—the rest await the result. This prevents race conditions
and duplicate work.

Where persistent caches are stored

By default, persistent caches are stored in `__marimo__/cache/`, in the directory of the
current notebook. For projects versioned with `git`, consider adding
`**/__marimo__/cache/` to your `.gitignore`.

Caches are preserved even when a cell is re-run

If a cell defining a cached function is re-run, the cache will be preserved unless
its source code (or the source code of the cell's ancestors) has changed.

### Persistent cache context manager [¶](\#persistent-cache-context-manager "Permanent link")

You can also use marimo's [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache") as a context manager:

The next time this block of code is run, if marimo detects a cache hit, the
code will be skipped and your variables will be loaded into memory. The cache
key for the context manager is computed in the same way as it is computed for
decorated functions.

## Cache key [¶](\#cache-key "Permanent link")

Both `mo.cache` and `mo.persistent_cache` use the same mechanism for creating a
cache key, differing only in where the cache is stored. The cache key is based
on function arguments and closed-over variables.

### Function arguments [¶](\#function-arguments "Permanent link")

**Arguments must either be primitive, marimo UI elements, array-like, or pickleable:**

1. Primitive types (strings, bytes, numbers, None) are hashed.
2. marimo UI elements are hashed based on their value.
3. [Array-like objects](https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__) are introspected, with their values being hashed.
4. All other objects are pickled.

### Closed-over variables [¶](\#closed-over-variables "Permanent link")

Syntactically closing over variables provides another way to parametrize
functions. In this example, the variable `x` is "closed over":

Closed-over variables are processed in the following way:

- marimo first attempts to hash or pickle the closed-over variables, just as it
does for arguments.
- If a closed-over variables cannot be hashed or pickled, then marimo uses the
source code that defines the value as part of the cache key; in particular, marimo
hashes the cell that defines the variable as well as the source code of that
cell's ancestors. This assumes that the variable's value is a deterministic
function of the source code that defines it, although certain side-effects
(specifically, if a cell raised an exception or loaded from another cache)
are taken into account.

Because marimo's cache key construction can fall back to source code for
closed-over variables, closing over variables lets you cache functions even in
the presence of non-hashable and non-pickleable arguments.

## Limitations [¶](\#limitations "Permanent link")

marimo's cache has some limitations:

- Side effects are not cached. This means that on a cache hit, side effects like printing, file I/O, or network requests will not occur.
- The source code of imported modules is not used when computing the cache key.
  - By setting `pin_modules` to `True`, you can ensure that the cache is invalidated when module versions change ( `e.g.`, update when the module's `__version__` attribute changes).
  - This limitation does not apply if the external module is a marimo notebook.
- The return values of persistently cached functions must be serializable with pickle.

Don't mutate variables

marimo works best when you don't mutate variables across cells. The same
is true for caching, since the cache key may not always be able to take
mutations into account.

### Decorators defined in other Python modules [¶](\#decorators-defined-in-other-python-modules "Permanent link")

Decorators defined in other Python modules that do not use `functools.wraps`
cannot be correctly cached. This can lead to confusing bugs like the example
below:

The fix is to make sure the decorator uses `functools.wraps`:

In this instance, the cache will work as expected because the decorated function has the same signature and metadata as the original function.

## Comparison with `functools.cache` [¶](\#comparison-with-functoolscache "Permanent link")

Here is a table comparing marimo's cache with `functools.cache`:

| Feature | marimo cache | `functools.cache` |
| --- | --- | --- |
| Cache return values in memory? | ✅ | ✅ |
| Cache return values to disk? | ✅ | ❌ |
| Preserved on cell re-runs? | ✅ | ❌ |
| Tracks closed-over variables | ✅ | ❌ |
| Allows unhashable arguments? | ✅ | ❌ |
| Allows Array-like arguments? | ✅ | ❌ |
| Supports async functions? | ✅ | ❌ |
| Suitable for lightweight functions (microseconds)? | ❌ | ✅ |

When to use `functools.cache`

Prefer `functools.cache` for extremely lightweight functions
(that execute in less than a millisecond). Using memoization to calculate
the Fibonacci sequence is a classic example of using `functools.cache`
effectively. On a basic macbook in pure python, `fib(35)` takes 1 second to
compute; with `mo.cache` it takes 0.000229 seconds; with `functools.cache`, it
takes 0.000025 seconds (x9 faster!!). Although relatively small, the additional overhead of
`mo.cache` (and more so `mo.persistent_cache`) is larger than
`functools.cache`. If your function takes more than a few milliseconds to
compute, the difference is negligible.

## Tips [¶](\#tips "Permanent link")

### Isolate cached code blocks to their own cells [¶](\#isolate-cached-code-blocks-to-their-own-cells "Permanent link")

Isolating cached functions in separate cells improves cache reliability. When
dependencies and cached functions are in the same cell, any change to the cell
invalidates the cache, even if the cached function itself hasn't changed.
Separating them ensures the cache is only invalidated when the function
actually changes.

**Don't do this:**

**Do this instead:**

### Close over unhashable or un-pickleable arguments [¶](\#close-over-unhashable-or-un-pickleable-arguments "Permanent link")

The [cache key](#cache-key) is constructed in part by hashing or pickle-ing function arguments. When you call a cached function with arguments that cannot be processed in this way, an exception will be raised. To parametrize cached functions with unhashable or un-pickleable arguments, syntactically close over them instead.

**You can't do this:**

Instead, you can close over `my_database_engine`:

**Do this:**

### Close-over low-memory-footprint variables [¶](\#close-over-low-memory-footprint-variables "Permanent link")

Non-primitive closed-over variables are serialized for cache key generation.
When possible, compute derived values (like length) outside the cache block and
only use the small values inside.

**Don't do this:**

**Do this instead:**

### Use `mo.watch.file` when working with files [¶](\#use-mowatchfile-when-working-with-files "Permanent link")

**Don't do this:**

**Do this instead:**

## API [¶](\#api "Permanent link")

## `` marimo.cache [¶](\#marimo.cache "Permanent link")

#### Cache the value of a function based on args and closed-over variables. [¶](\#marimo.cache--cache-the-value-of-a-function-based-on-args-and-closed-over-variables "Permanent link")

Decorating a function with `@mo.cache` will cache its value based on
the function's arguments, closed-over values, and the notebook code.

Examples:

`mo.cache` is similar to `functools.cache`, but with three key benefits:

1. `mo.cache` persists its cache even if the cell defining the
    cached function is re-run, as long as the code defining the function
    (excluding comments and formatting) has not changed.
2. `mo.cache` keys on closed-over values in addition to function arguments,
    preventing accumulation of hidden state associated with
    `functools.cache`.
3. `mo.cache` does not require its arguments to be
    hashable (only pickleable), meaning it can work with lists, sets, NumPy
    arrays, PyTorch tensors, and more.

`mo.cache` obtains these benefits at the cost of slightly higher overhead
than `functools.cache`, so it is best used for expensive functions.

Like `functools.cache`, `mo.cache` is thread-safe.

The cache has an unlimited maximum size. To limit the cache size, use
`@mo.lru_cache`. `mo.cache` is slightly faster than `mo.lru_cache`, but in
most applications the difference is negligible.

Note, `mo.cache` can also be used as a drop in replacement for context block
caching like `mo.persistent_cache`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |

#### Context manager to cache the return value of a block of code. [¶](\#marimo.cache--context-manager-to-cache-the-return-value-of-a-block-of-code "Permanent link")

The `mo.cache` context manager lets you delimit a block of code in which
variables will be cached to memory when they are first computed.

By default, the cache is stored in memory and is not persisted across kernel
runs, for that functionality, refer to `mo.persistent_cache`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | the name of the cache, used to set saving path- to manually<br>invalidate the cache, change the name.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `loader` | the loader to use for the cache, defaults to `MemoryLoader`.<br>**TYPE:** `Optional[Union[LoaderPartial, Loader]]`**DEFAULT:** `None` |
| `**kwargs` | keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |
| `*args` | positional arguments<br>**TYPE:** `Any`**DEFAULT:** `()` |

## `` marimo.lru\_cache [¶](\#marimo.lru_cache "Permanent link")

Decorator for LRU caching the return value of a function.

`mo.lru_cache` is a version of `mo.cache` with a bounded cache size. As an
LRU (Least Recently Used) cache, only the last used `maxsize` values are
retained, with the oldest values being discarded. For more information,
see the documentation of `mo.cache`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `maxsize` | the maximum number of entries in the cache; defaults to 128.<br>Setting to -1 disables cache limits.<br>**TYPE:** `int`**DEFAULT:** `128` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |

#### Context manager for LRU caching the return value of a block of code. [¶](\#marimo.lru_cache--context-manager-for-lru-caching-the-return-value-of-a-block-of-code "Permanent link")

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | Namespace key for the cache.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `maxsize` | the maximum number of entries in the cache; defaults to 128.<br>Setting to -1 disables cache limits.<br>**TYPE:** `int`**DEFAULT:** `128` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `**kwargs` | keyword arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `{}` |
| `*args` | positional arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `()` |

## `` marimo.persistent\_cache [¶](\#marimo.persistent_cache "Permanent link")

#### Context manager to save variables to disk and restore them thereafter. [¶](\#marimo.persistent_cache--context-manager-to-save-variables-to-disk-and-restore-them-thereafter "Permanent link")

The `mo.persistent_cache` context manager lets you delimit a block of code
in which variables will be cached to disk when they are first computed. On
subsequent runs of the cell, if marimo determines that this block of code
hasn't changed and neither has its ancestors, it will restore the variables
from disk instead of re-computing them, skipping execution of the block
entirely.

Restoration happens even across notebook runs, meaning you can use
`mo.persistent_cache` to make notebooks start _instantly_, with variables
that would otherwise be expensive to compute already materialized in
memory.

Examples:

In this example, `variable` will be cached the first time the block
is executed, and restored on subsequent runs of the block. If cache
conditions are hit, the contents of `with` block will be skipped on
execution. This means that side-effects such as writing to stdout and
stderr will be skipped on cache hits.

Note that `mo.state` and `UIElement` changes will also trigger cache
invalidation, and be accordingly updated.

**Warning.** Since context abuses sys frame trace, this may conflict with
debugging tools or libraries that also use `sys.settrace`.

#### Decorator for persistently caching the return value of a function. [¶](\#marimo.persistent_cache--decorator-for-persistently-caching-the-return-value-of-a-function "Permanent link")

`persistent_cache` can also be used as a drop in function-level memoization
for `@mo.cache` or `@mo.lru_cache`. This is much slower than cache, but
can be useful for saving function values between kernel restarts. For more
details, refer to `mo.cache`.

**Usage.**

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | the name of the cache, used to set saving path- to manually<br>invalidate the cache, change the name.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `save_path` | the folder in which to save the cache, defaults to<br>`__marimo__/cache` in the directory of the notebook file<br>**TYPE:** `str | None`**DEFAULT:** `None` |
| `method` | the serialization method to use, current options are "json",<br>and "pickle" (default).<br>**TYPE:** `LoaderKey`**DEFAULT:** `'pickle'` |
| `store` | optional store.<br>**TYPE:** `Optional[Store]`**DEFAULT:** `None` |
| `fn` | the wrapped function if no settings are passed.<br>**TYPE:** `Optional[Callable[..., Any]]`**DEFAULT:** `None` |
| `*args` | positional arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `()` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ between runs, defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `**kwargs` | keyword arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Caching [¶](\#caching "Permanent link")

marimo comes with utilities to cache intermediate computations. These
utilities can be applied as decorators to functions to cache their
returned values; you can choose between saving caches in memory or to disk.

## Basic usage [¶](\#basic-usage "Permanent link")

marimo provides two decorators for caching the return values of expensive
functions:

1. [`mo.cache`](#marimo.cache "            marimo.cache"), which saves cached values to memory;
2. [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache"), which saves cached values to disk.

`mo.cache` `mo.persistent_cache` `mo.cache` (async) `mo.persistent_cache` (async)

Roughly speaking, the first time a cached function is called with a particular
sequence of arguments, the function will run and its return value will be
cached. The next time it is called with the same sequence of arguments (on
cache hit), the function body will be skipped and the return value will be
retrieved from cache instead.

The in-memory cache ( [`mo.cache`](#marimo.cache "            marimo.cache")) is faster and doesn't consume disk space, but
it is lost on notebook restart. The disk cache ( [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache")) is
slower and consumes space on disk, but it persists across notebook runs,
letting you pick up where you left off.

(For an in-memory cache of bounded size, use [`mo.lru_cache`](#marimo.lru_cache "            marimo.lru_cache").)

Async functions are fully supported

All cache decorators ( `mo.cache`, `mo.lru_cache`, `mo.persistent_cache`) work
seamlessly with both synchronous and asynchronous functions. When multiple
concurrent calls are made to a cached async function with the same arguments,
only one execution occurs—the rest await the result. This prevents race conditions
and duplicate work.

Where persistent caches are stored

By default, persistent caches are stored in `__marimo__/cache/`, in the directory of the
current notebook. For projects versioned with `git`, consider adding
`**/__marimo__/cache/` to your `.gitignore`.

Caches are preserved even when a cell is re-run

If a cell defining a cached function is re-run, the cache will be preserved unless
its source code (or the source code of the cell's ancestors) has changed.

### Persistent cache context manager [¶](\#persistent-cache-context-manager "Permanent link")

You can also use marimo's [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache") as a context manager:

The next time this block of code is run, if marimo detects a cache hit, the
code will be skipped and your variables will be loaded into memory. The cache
key for the context manager is computed in the same way as it is computed for
decorated functions.

## Cache key [¶](\#cache-key "Permanent link")

Both `mo.cache` and `mo.persistent_cache` use the same mechanism for creating a
cache key, differing only in where the cache is stored. The cache key is based
on function arguments and closed-over variables.

### Function arguments [¶](\#function-arguments "Permanent link")

**Arguments must either be primitive, marimo UI elements, array-like, or pickleable:**

1. Primitive types (strings, bytes, numbers, None) are hashed.
2. marimo UI elements are hashed based on their value.
3. [Array-like objects](https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__) are introspected, with their values being hashed.
4. All other objects are pickled.

### Closed-over variables [¶](\#closed-over-variables "Permanent link")

Syntactically closing over variables provides another way to parametrize
functions. In this example, the variable `x` is "closed over":

Closed-over variables are processed in the following way:

- marimo first attempts to hash or pickle the closed-over variables, just as it
does for arguments.
- If a closed-over variables cannot be hashed or pickled, then marimo uses the
source code that defines the value as part of the cache key; in particular, marimo
hashes the cell that defines the variable as well as the source code of that
cell's ancestors. This assumes that the variable's value is a deterministic
function of the source code that defines it, although certain side-effects
(specifically, if a cell raised an exception or loaded from another cache)
are taken into account.

Because marimo's cache key construction can fall back to source code for
closed-over variables, closing over variables lets you cache functions even in
the presence of non-hashable and non-pickleable arguments.

## Limitations [¶](\#limitations "Permanent link")

marimo's cache has some limitations:

- Side effects are not cached. This means that on a cache hit, side effects like printing, file I/O, or network requests will not occur.
- The source code of imported modules is not used when computing the cache key.
  - By setting `pin_modules` to `True`, you can ensure that the cache is invalidated when module versions change ( `e.g.`, update when the module's `__version__` attribute changes).
  - This limitation does not apply if the external module is a marimo notebook.
- The return values of persistently cached functions must be serializable with pickle.

Don't mutate variables

marimo works best when you don't mutate variables across cells. The same
is true for caching, since the cache key may not always be able to take
mutations into account.

### Decorators defined in other Python modules [¶](\#decorators-defined-in-other-python-modules "Permanent link")

Decorators defined in other Python modules that do not use `functools.wraps`
cannot be correctly cached. This can lead to confusing bugs like the example
below:

The fix is to make sure the decorator uses `functools.wraps`:

In this instance, the cache will work as expected because the decorated function has the same signature and metadata as the original function.

## Comparison with `functools.cache` [¶](\#comparison-with-functoolscache "Permanent link")

Here is a table comparing marimo's cache with `functools.cache`:

| Feature | marimo cache | `functools.cache` |
| --- | --- | --- |
| Cache return values in memory? | ✅ | ✅ |
| Cache return values to disk? | ✅ | ❌ |
| Preserved on cell re-runs? | ✅ | ❌ |
| Tracks closed-over variables | ✅ | ❌ |
| Allows unhashable arguments? | ✅ | ❌ |
| Allows Array-like arguments? | ✅ | ❌ |
| Supports async functions? | ✅ | ❌ |
| Suitable for lightweight functions (microseconds)? | ❌ | ✅ |

When to use `functools.cache`

Prefer `functools.cache` for extremely lightweight functions
(that execute in less than a millisecond). Using memoization to calculate
the Fibonacci sequence is a classic example of using `functools.cache`
effectively. On a basic macbook in pure python, `fib(35)` takes 1 second to
compute; with `mo.cache` it takes 0.000229 seconds; with `functools.cache`, it
takes 0.000025 seconds (x9 faster!!). Although relatively small, the additional overhead of
`mo.cache` (and more so `mo.persistent_cache`) is larger than
`functools.cache`. If your function takes more than a few milliseconds to
compute, the difference is negligible.

## Tips [¶](\#tips "Permanent link")

### Isolate cached code blocks to their own cells [¶](\#isolate-cached-code-blocks-to-their-own-cells "Permanent link")

Isolating cached functions in separate cells improves cache reliability. When
dependencies and cached functions are in the same cell, any change to the cell
invalidates the cache, even if the cached function itself hasn't changed.
Separating them ensures the cache is only invalidated when the function
actually changes.

**Don't do this:**

**Do this instead:**

### Close over unhashable or un-pickleable arguments [¶](\#close-over-unhashable-or-un-pickleable-arguments "Permanent link")

The [cache key](#cache-key) is constructed in part by hashing or pickle-ing function arguments. When you call a cached function with arguments that cannot be processed in this way, an exception will be raised. To parametrize cached functions with unhashable or un-pickleable arguments, syntactically close over them instead.

**You can't do this:**

Instead, you can close over `my_database_engine`:

**Do this:**

### Close-over low-memory-footprint variables [¶](\#close-over-low-memory-footprint-variables "Permanent link")

Non-primitive closed-over variables are serialized for cache key generation.
When possible, compute derived values (like length) outside the cache block and
only use the small values inside.

**Don't do this:**

**Do this instead:**

### Use `mo.watch.file` when working with files [¶](\#use-mowatchfile-when-working-with-files "Permanent link")

**Don't do this:**

**Do this instead:**

## API [¶](\#api "Permanent link")

## `` marimo.cache [¶](\#marimo.cache "Permanent link")

#### Cache the value of a function based on args and closed-over variables. [¶](\#marimo.cache--cache-the-value-of-a-function-based-on-args-and-closed-over-variables "Permanent link")

Decorating a function with `@mo.cache` will cache its value based on
the function's arguments, closed-over values, and the notebook code.

Examples:

`mo.cache` is similar to `functools.cache`, but with three key benefits:

1. `mo.cache` persists its cache even if the cell defining the
    cached function is re-run, as long as the code defining the function
    (excluding comments and formatting) has not changed.
2. `mo.cache` keys on closed-over values in addition to function arguments,
    preventing accumulation of hidden state associated with
    `functools.cache`.
3. `mo.cache` does not require its arguments to be
    hashable (only pickleable), meaning it can work with lists, sets, NumPy
    arrays, PyTorch tensors, and more.

`mo.cache` obtains these benefits at the cost of slightly higher overhead
than `functools.cache`, so it is best used for expensive functions.

Like `functools.cache`, `mo.cache` is thread-safe.

The cache has an unlimited maximum size. To limit the cache size, use
`@mo.lru_cache`. `mo.cache` is slightly faster than `mo.lru_cache`, but in
most applications the difference is negligible.

Note, `mo.cache` can also be used as a drop in replacement for context block
caching like `mo.persistent_cache`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |

#### Context manager to cache the return value of a block of code. [¶](\#marimo.cache--context-manager-to-cache-the-return-value-of-a-block-of-code "Permanent link")

The `mo.cache` context manager lets you delimit a block of code in which
variables will be cached to memory when they are first computed.

By default, the cache is stored in memory and is not persisted across kernel
runs, for that functionality, refer to `mo.persistent_cache`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | the name of the cache, used to set saving path- to manually<br>invalidate the cache, change the name.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `loader` | the loader to use for the cache, defaults to `MemoryLoader`.<br>**TYPE:** `Optional[Union[LoaderPartial, Loader]]`**DEFAULT:** `None` |
| `**kwargs` | keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |
| `*args` | positional arguments<br>**TYPE:** `Any`**DEFAULT:** `()` |

## `` marimo.lru\_cache [¶](\#marimo.lru_cache "Permanent link")

Decorator for LRU caching the return value of a function.

`mo.lru_cache` is a version of `mo.cache` with a bounded cache size. As an
LRU (Least Recently Used) cache, only the last used `maxsize` values are
retained, with the oldest values being discarded. For more information,
see the documentation of `mo.cache`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `maxsize` | the maximum number of entries in the cache; defaults to 128.<br>Setting to -1 disables cache limits.<br>**TYPE:** `int`**DEFAULT:** `128` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |

#### Context manager for LRU caching the return value of a block of code. [¶](\#marimo.lru_cache--context-manager-for-lru-caching-the-return-value-of-a-block-of-code "Permanent link")

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | Namespace key for the cache.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `maxsize` | the maximum number of entries in the cache; defaults to 128.<br>Setting to -1 disables cache limits.<br>**TYPE:** `int`**DEFAULT:** `128` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `**kwargs` | keyword arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `{}` |
| `*args` | positional arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `()` |

## `` marimo.persistent\_cache [¶](\#marimo.persistent_cache "Permanent link")

#### Context manager to save variables to disk and restore them thereafter. [¶](\#marimo.persistent_cache--context-manager-to-save-variables-to-disk-and-restore-them-thereafter "Permanent link")

The `mo.persistent_cache` context manager lets you delimit a block of code
in which variables will be cached to disk when they are first computed. On
subsequent runs of the cell, if marimo determines that this block of code
hasn't changed and neither has its ancestors, it will restore the variables
from disk instead of re-computing them, skipping execution of the block
entirely.

Restoration happens even across notebook runs, meaning you can use
`mo.persistent_cache` to make notebooks start _instantly_, with variables
that would otherwise be expensive to compute already materialized in
memory.

Examples:

In this example, `variable` will be cached the first time the block
is executed, and restored on subsequent runs of the block. If cache
conditions are hit, the contents of `with` block will be skipped on
execution. This means that side-effects such as writing to stdout and
stderr will be skipped on cache hits.

Note that `mo.state` and `UIElement` changes will also trigger cache
invalidation, and be accordingly updated.

**Warning.** Since context abuses sys frame trace, this may conflict with
debugging tools or libraries that also use `sys.settrace`.

#### Decorator for persistently caching the return value of a function. [¶](\#marimo.persistent_cache--decorator-for-persistently-caching-the-return-value-of-a-function "Permanent link")

`persistent_cache` can also be used as a drop in function-level memoization
for `@mo.cache` or `@mo.lru_cache`. This is much slower than cache, but
can be useful for saving function values between kernel restarts. For more
details, refer to `mo.cache`.

**Usage.**

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | the name of the cache, used to set saving path- to manually<br>invalidate the cache, change the name.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `save_path` | the folder in which to save the cache, defaults to<br>`__marimo__/cache` in the directory of the notebook file<br>**TYPE:** `str | None`**DEFAULT:** `None` |
| `method` | the serialization method to use, current options are "json",<br>and "pickle" (default).<br>**TYPE:** `LoaderKey`**DEFAULT:** `'pickle'` |
| `store` | optional store.<br>**TYPE:** `Optional[Store]`**DEFAULT:** `None` |
| `fn` | the wrapped function if no settings are passed.<br>**TYPE:** `Optional[Callable[..., Any]]`**DEFAULT:** `None` |
| `*args` | positional arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `()` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ between runs, defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `**kwargs` | keyword arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Caching [¶](\#caching "Permanent link")

marimo comes with utilities to cache intermediate computations. These
utilities can be applied as decorators to functions to cache their
returned values; you can choose between saving caches in memory or to disk.

## Basic usage [¶](\#basic-usage "Permanent link")

marimo provides two decorators for caching the return values of expensive
functions:

1. [`mo.cache`](#marimo.cache "            marimo.cache"), which saves cached values to memory;
2. [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache"), which saves cached values to disk.

`mo.cache` `mo.persistent_cache` `mo.cache` (async) `mo.persistent_cache` (async)

Roughly speaking, the first time a cached function is called with a particular
sequence of arguments, the function will run and its return value will be
cached. The next time it is called with the same sequence of arguments (on
cache hit), the function body will be skipped and the return value will be
retrieved from cache instead.

The in-memory cache ( [`mo.cache`](#marimo.cache "            marimo.cache")) is faster and doesn't consume disk space, but
it is lost on notebook restart. The disk cache ( [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache")) is
slower and consumes space on disk, but it persists across notebook runs,
letting you pick up where you left off.

(For an in-memory cache of bounded size, use [`mo.lru_cache`](#marimo.lru_cache "            marimo.lru_cache").)

Async functions are fully supported

All cache decorators ( `mo.cache`, `mo.lru_cache`, `mo.persistent_cache`) work
seamlessly with both synchronous and asynchronous functions. When multiple
concurrent calls are made to a cached async function with the same arguments,
only one execution occurs—the rest await the result. This prevents race conditions
and duplicate work.

Where persistent caches are stored

By default, persistent caches are stored in `__marimo__/cache/`, in the directory of the
current notebook. For projects versioned with `git`, consider adding
`**/__marimo__/cache/` to your `.gitignore`.

Caches are preserved even when a cell is re-run

If a cell defining a cached function is re-run, the cache will be preserved unless
its source code (or the source code of the cell's ancestors) has changed.

### Persistent cache context manager [¶](\#persistent-cache-context-manager "Permanent link")

You can also use marimo's [`mo.persistent_cache`](#marimo.persistent_cache "            marimo.persistent_cache") as a context manager:

The next time this block of code is run, if marimo detects a cache hit, the
code will be skipped and your variables will be loaded into memory. The cache
key for the context manager is computed in the same way as it is computed for
decorated functions.

## Cache key [¶](\#cache-key "Permanent link")

Both `mo.cache` and `mo.persistent_cache` use the same mechanism for creating a
cache key, differing only in where the cache is stored. The cache key is based
on function arguments and closed-over variables.

### Function arguments [¶](\#function-arguments "Permanent link")

**Arguments must either be primitive, marimo UI elements, array-like, or pickleable:**

1. Primitive types (strings, bytes, numbers, None) are hashed.
2. marimo UI elements are hashed based on their value.
3. [Array-like objects](https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__) are introspected, with their values being hashed.
4. All other objects are pickled.

### Closed-over variables [¶](\#closed-over-variables "Permanent link")

Syntactically closing over variables provides another way to parametrize
functions. In this example, the variable `x` is "closed over":

Closed-over variables are processed in the following way:

- marimo first attempts to hash or pickle the closed-over variables, just as it
  does for arguments.
- If a closed-over variables cannot be hashed or pickled, then marimo uses the
  source code that defines the value as part of the cache key; in particular, marimo
  hashes the cell that defines the variable as well as the source code of that
  cell's ancestors. This assumes that the variable's value is a deterministic
  function of the source code that defines it, although certain side-effects
  (specifically, if a cell raised an exception or loaded from another cache)
  are taken into account.

Because marimo's cache key construction can fall back to source code for
closed-over variables, closing over variables lets you cache functions even in
the presence of non-hashable and non-pickleable arguments.

## Limitations [¶](\#limitations "Permanent link")

marimo's cache has some limitations:

- Side effects are not cached. This means that on a cache hit, side effects like printing, file I/O, or network requests will not occur.
- The source code of imported modules is not used when computing the cache key.
  - By setting `pin_modules` to `True`, you can ensure that the cache is invalidated when module versions change ( `e.g.`, update when the module's `__version__` attribute changes).
  - This limitation does not apply if the external module is a marimo notebook.
- The return values of persistently cached functions must be serializable with pickle.

Don't mutate variables

marimo works best when you don't mutate variables across cells. The same
is true for caching, since the cache key may not always be able to take
mutations into account.

### Decorators defined in other Python modules [¶](\#decorators-defined-in-other-python-modules "Permanent link")

Decorators defined in other Python modules that do not use `functools.wraps`
cannot be correctly cached. This can lead to confusing bugs like the example
below:

The fix is to make sure the decorator uses `functools.wraps`:

In this instance, the cache will work as expected because the decorated function has the same signature and metadata as the original function.

## Comparison with `functools.cache` [¶](\#comparison-with-functoolscache "Permanent link")

Here is a table comparing marimo's cache with `functools.cache`:

| Feature | marimo cache | `functools.cache` |
| --- | --- | --- |
| Cache return values in memory? | ✅ | ✅ |
| Cache return values to disk? | ✅ | ❌ |
| Preserved on cell re-runs? | ✅ | ❌ |
| Tracks closed-over variables | ✅ | ❌ |
| Allows unhashable arguments? | ✅ | ❌ |
| Allows Array-like arguments? | ✅ | ❌ |
| Supports async functions? | ✅ | ❌ |
| Suitable for lightweight functions (microseconds)? | ❌ | ✅ |

When to use `functools.cache`

Prefer `functools.cache` for extremely lightweight functions
(that execute in less than a millisecond). Using memoization to calculate
the Fibonacci sequence is a classic example of using `functools.cache`
effectively. On a basic macbook in pure python, `fib(35)` takes 1 second to
compute; with `mo.cache` it takes 0.000229 seconds; with `functools.cache`, it
takes 0.000025 seconds (x9 faster!!). Although relatively small, the additional overhead of
`mo.cache` (and more so `mo.persistent_cache`) is larger than
`functools.cache`. If your function takes more than a few milliseconds to
compute, the difference is negligible.

## Tips [¶](\#tips "Permanent link")

### Isolate cached code blocks to their own cells [¶](\#isolate-cached-code-blocks-to-their-own-cells "Permanent link")

Isolating cached functions in separate cells improves cache reliability. When
dependencies and cached functions are in the same cell, any change to the cell
invalidates the cache, even if the cached function itself hasn't changed.
Separating them ensures the cache is only invalidated when the function
actually changes.

**Don't do this:**

**Do this instead:**

### Close over unhashable or un-pickleable arguments [¶](\#close-over-unhashable-or-un-pickleable-arguments "Permanent link")

The [cache key](#cache-key) is constructed in part by hashing or pickle-ing function arguments. When you call a cached function with arguments that cannot be processed in this way, an exception will be raised. To parametrize cached functions with unhashable or un-pickleable arguments, syntactically close over them instead.

**You can't do this:**

Instead, you can close over `my_database_engine`:

**Do this:**

### Close-over low-memory-footprint variables [¶](\#close-over-low-memory-footprint-variables "Permanent link")

Non-primitive closed-over variables are serialized for cache key generation.
When possible, compute derived values (like length) outside the cache block and
only use the small values inside.

**Don't do this:**

**Do this instead:**

### Use `mo.watch.file` when working with files [¶](\#use-mowatchfile-when-working-with-files "Permanent link")

**Don't do this:**

**Do this instead:**

## API [¶](\#api "Permanent link")

## `` marimo.cache [¶](\#marimo.cache "Permanent link")

#### Cache the value of a function based on args and closed-over variables. [¶](\#marimo.cache--cache-the-value-of-a-function-based-on-args-and-closed-over-variables "Permanent link")

Decorating a function with `@mo.cache` will cache its value based on
the function's arguments, closed-over values, and the notebook code.

Examples:

`mo.cache` is similar to `functools.cache`, but with three key benefits:

1. `mo.cache` persists its cache even if the cell defining the
    cached function is re-run, as long as the code defining the function
    (excluding comments and formatting) has not changed.
2. `mo.cache` keys on closed-over values in addition to function arguments,
    preventing accumulation of hidden state associated with
    `functools.cache`.
3. `mo.cache` does not require its arguments to be
    hashable (only pickleable), meaning it can work with lists, sets, NumPy
    arrays, PyTorch tensors, and more.

`mo.cache` obtains these benefits at the cost of slightly higher overhead
than `functools.cache`, so it is best used for expensive functions.

Like `functools.cache`, `mo.cache` is thread-safe.

The cache has an unlimited maximum size. To limit the cache size, use
`@mo.lru_cache`. `mo.cache` is slightly faster than `mo.lru_cache`, but in
most applications the difference is negligible.

Note, `mo.cache` can also be used as a drop in replacement for context block
caching like `mo.persistent_cache`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |

#### Context manager to cache the return value of a block of code. [¶](\#marimo.cache--context-manager-to-cache-the-return-value-of-a-block-of-code "Permanent link")

The `mo.cache` context manager lets you delimit a block of code in which
variables will be cached to memory when they are first computed.

By default, the cache is stored in memory and is not persisted across kernel
runs, for that functionality, refer to `mo.persistent_cache`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | the name of the cache, used to set saving path- to manually<br>invalidate the cache, change the name.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `loader` | the loader to use for the cache, defaults to `MemoryLoader`.<br>**TYPE:** `Optional[Union[LoaderPartial, Loader]]`**DEFAULT:** `None` |
| `**kwargs` | keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |
| `*args` | positional arguments<br>**TYPE:** `Any`**DEFAULT:** `()` |

## `` marimo.lru\_cache [¶](\#marimo.lru_cache "Permanent link")

Decorator for LRU caching the return value of a function.

`mo.lru_cache` is a version of `mo.cache` with a bounded cache size. As an
LRU (Least Recently Used) cache, only the last used `maxsize` values are
retained, with the oldest values being discarded. For more information,
see the documentation of `mo.cache`.

Examples:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `maxsize` | the maximum number of entries in the cache; defaults to 128.<br>Setting to -1 disables cache limits.<br>**TYPE:** `int`**DEFAULT:** `128` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |

#### Context manager for LRU caching the return value of a block of code. [¶](\#marimo.lru_cache--context-manager-for-lru-caching-the-return-value-of-a-block-of-code "Permanent link")

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | Namespace key for the cache.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `maxsize` | the maximum number of entries in the cache; defaults to 128.<br>Setting to -1 disables cache limits.<br>**TYPE:** `int`**DEFAULT:** `128` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `**kwargs` | keyword arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `{}` |
| `*args` | positional arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `()` |

## `` marimo.persistent\_cache [¶](\#marimo.persistent_cache "Permanent link")

#### Context manager to save variables to disk and restore them thereafter. [¶](\#marimo.persistent_cache--context-manager-to-save-variables-to-disk-and-restore-them-thereafter "Permanent link")

The `mo.persistent_cache` context manager lets you delimit a block of code
in which variables will be cached to disk when they are first computed. On
subsequent runs of the cell, if marimo determines that this block of code
hasn't changed and neither has its ancestors, it will restore the variables
from disk instead of re-computing them, skipping execution of the block
entirely.

Restoration happens even across notebook runs, meaning you can use
`mo.persistent_cache` to make notebooks start _instantly_, with variables
that would otherwise be expensive to compute already materialized in
memory.

Examples:

In this example, `variable` will be cached the first time the block
is executed, and restored on subsequent runs of the block. If cache
conditions are hit, the contents of `with` block will be skipped on
execution. This means that side-effects such as writing to stdout and
stderr will be skipped on cache hits.

Note that `mo.state` and `UIElement` changes will also trigger cache
invalidation, and be accordingly updated.

**Warning.** Since context abuses sys frame trace, this may conflict with
debugging tools or libraries that also use `sys.settrace`.

#### Decorator for persistently caching the return value of a function. [¶](\#marimo.persistent_cache--decorator-for-persistently-caching-the-return-value-of-a-function "Permanent link")

`persistent_cache` can also be used as a drop in function-level memoization
for `@mo.cache` or `@mo.lru_cache`. This is much slower than cache, but
can be useful for saving function values between kernel restarts. For more
details, refer to `mo.cache`.

**Usage.**

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | the name of the cache, used to set saving path- to manually<br>invalidate the cache, change the name.<br>**TYPE:** `Union[str, Optional[Callable[..., Any]]]`**DEFAULT:** `None` |
| `save_path` | the folder in which to save the cache, defaults to<br>`__marimo__/cache` in the directory of the notebook file<br>**TYPE:** `str | None`**DEFAULT:** `None` |
| `method` | the serialization method to use, current options are "json",<br>and "pickle" (default).<br>**TYPE:** `LoaderKey`**DEFAULT:** `'pickle'` |
| `store` | optional store.<br>**TYPE:** `Optional[Store]`**DEFAULT:** `None` |
| `fn` | the wrapped function if no settings are passed.<br>**TYPE:** `Optional[Callable[..., Any]]`**DEFAULT:** `None` |
| `*args` | positional arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `()` |
| `pin_modules` | if True, the cache will be invalidated if module versions<br>differ between runs, defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `**kwargs` | keyword arguments passed to `cache()`<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

###### FAQ - marimoFAQ

*Source: [https://docs.marimo.io/faq/](https://docs.marimo.io/faq/)*

<!-- Source: https://docs.marimo.io/faq/ -->
<!-- Title: FAQ - marimoFAQ -->
<!-- Depth: 7 -->

[Skip to content](#choosing-marimo)

# FAQ

## Choosing marimo [¶](\#choosing-marimo "Permanent link")

### How is marimo different from Jupyter? [¶](\#how-is-marimo-different-from-jupyter "Permanent link")

marimo is a reinvention of the Python notebook as a reproducible, interactive,
and shareable Python program that can be executed as scripts or deployed as
interactive web apps.

**Consistent state.** In marimo, your notebook code, outputs, and program state
are guaranteed to be consistent. Run a cell and marimo reacts by automatically
running the cells that reference its variables. Delete a cell and marimo scrubs
its variables from program memory, eliminating hidden state.

**Built-in interactivity.** marimo also comes with [UI\
elements](../guides/interactivity/) like sliders, a dataframe transformer, and
interactive plots that are automatically synchronized with Python. Interact
with an element and the cells that use it are automatically re-run with its
latest value.

**Pure Python programs.** Unlike Jupyter notebooks, marimo notebooks are stored
as pure Python files that can be executed as scripts, deployed as interactive
web apps, and versioned easily with Git.

### What problems does marimo solve? [¶](\#what-problems-does-marimo-solve "Permanent link")

marimo solves problems in reproducibility, maintainability, interactivity,
reusability, and shareability of notebooks.

**Reproducibility.**
In Jupyter notebooks, the code you see doesn't necessarily match the outputs on
the page or the program state. If you
delete a cell, its variables stay in memory, which other cells may still
reference; users can execute cells in arbitrary order. This leads to
widespread reproducibility issues. [One study](https://blog.jetbrains.com/datalore/2020/12/17/we-downloaded-10-000-000-jupyter-notebooks-from-github-this-is-what-we-learned/#consistency-of-notebooks) analyzed 10 million Jupyter
notebooks and found that 36% of them weren't reproducible.

In contrast, marimo guarantees that your code, outputs, and program state are
consistent, eliminating hidden state and making your notebook reproducible.
marimo achieves this by intelligently analyzing your code and understanding the
relationships between cells, and automatically re-running cells as needed.

In addition, marimo notebooks can serialize package requirements inline;
marimo runs these "sandboxed" notebooks in temporary virtual environments,
making them [reproducible down to the packages](../guides/editor_features/package_management/).

**Maintainability.**
marimo notebooks are stored as pure Python programs ( `.py` files). This lets you
version them with Git; in contrast, Jupyter notebooks are stored as JSON and
require extra steps to version.

**Interactivity.**
marimo notebooks come with [UI elements](../guides/interactivity/) that are
automatically synchronized with Python (like sliders, dropdowns); _eg_, scrub a
slider and all cells that reference it are automatically re-run with the new
value. This is difficult to get working in Jupyter notebooks.

**Reusability.**
marimo notebooks can be executed as Python scripts from the command-line (since
they're stored as `.py` files). In contrast, this requires extra steps to
do for Jupyter, such as copying and pasting the code out or using external
frameworks. We also let you import symbols (functions, classes) defined in a
marimo notebook into other Python programs/notebooks, something you can't
easily do with Jupyter.

**Shareability.**
Every marimo notebook can double as an interactive web app, complete with UI
elements, which you can serve using the `marimo run` command. This isn't
possible in Jupyter without substantial extra effort.

_To learn more about problems with traditional notebooks,_
_see these references_
_[\[1\]](https://austinhenley.com/pubs/Chattopadhyay2020CHI_NotebookPainpoints.pdf) [\[2\]](https://www.youtube.com/watch?v=7jiPeIFXb6U&t=1s)._

### How is `marimo.ui` different from Jupyter widgets? [¶](\#how-is-marimoui-different-from-jupyter-widgets "Permanent link")

Unlike Jupyter widgets, marimo's interactive elements are automatically
synchronized with the Python kernel: no callbacks, no observers, no manually
re-running cells.

## Using marimo [¶](\#using-marimo "Permanent link")

### Is marimo a notebook or a library? [¶](\#is-marimo-a-notebook-or-a-library "Permanent link")

marimo is both a notebook and a library.

- Create _marimo notebooks_ with the editor that opens in your
browser when you run `marimo edit`.
- Use the _marimo library_ ( `import marimo as mo`) in
marimo notebooks. Write markdown with `mo.md(...)`,
create stateful interactive elements with `mo.ui` ( `mo.ui.slider(...)`), and
more. See the docs for an [API reference](../api/).

### What's the difference between a marimo notebook and a marimo app? [¶](\#whats-the-difference-between-a-marimo-notebook-and-a-marimo-app "Permanent link")

marimo programs are notebooks, apps, or both, depending on how you use them.

There are two ways to interact with a marimo program:

1. open it as a computational _notebook_ with `marimo edit`
2. run it as an interactive _app_ with `marimo run`

All marimo programs start as notebooks, since they are created with `marimo
edit`. Because marimo notebooks are reactive and have built-in interactive
elements, many can easily be made into useful and beautiful apps by simply
hiding the notebook code: this is what `marimo run` does.

Not every notebook needs to be run as an app — marimo notebooks are useful in
and of themselves for rapidly exploring data and doing reproducible science.
And not every app is improved by interacting with the notebook. In some
settings, such as collaborative research, education, and technical
presentations, going back and forth between the notebook view and app view
(which you can do from `marimo edit`) can be useful!

### How does marimo know what cells to run? [¶](\#how-does-marimo-know-what-cells-to-run "Permanent link")

marimo reads each cell once to determine what global names it defines and what
global names it reads. When a cell is run, marimo runs all other cells that
read any of the global names it defines. A global name can refer to a variable,
class, function, or import.

In other words, marimo uses _static analysis_ to make a dataflow graph out of
your cells. Each cell is a node in the graph across which global
variables "flow". Whenever a cell is run, either because you changed its
code or interacted with a UI element it reads, all its descendants run in turn.

### Does marimo slow my code down? [¶](\#does-marimo-slow-my-code-down "Permanent link")

No, marimo doesn't slow your code down. marimo determines the dependencies
among cells by reading your code, not running or tracing it, so there's
zero runtime overhead.

### How do I prevent automatic execution from running expensive cells? [¶](\#how-do-i-prevent-automatic-execution-from-running-expensive-cells "Permanent link")

Reactive (automatic) execution ensures your code and outputs are always
in sync, improving reproducibility by eliminating hidden state and
out-of-order execution; marimo also takes care to run only the minimal set of
cells needed to keep your notebook up to date. But when some cells take a long
time to run, it's understandable to be concerned that automatic execution will
kick off expensive cells before you're ready to run them.

_Here are some tips to avoid accidental execution of expensive cells:_

- [Disable expensive cells](../guides/reactivity/#disabling-cells). When a cell
is disabled, it and its descendants are blocked from running.
- Wrap UI elements in a [form](../api/inputs/form/#marimo.ui.form "            marimo.ui.form").
- Use [`mo.stop`](../api/control_flow/#marimo.stop "            marimo.stop") to conditionally stop
execution of a cell and its descendants.
- Decorate functions with marimo's [`mo.cache`](../api/caching/#marimo.cache "            marimo.cache") to cache
expensive intermediate computations.
- Use [`mo.persistent_cache`](../api/caching/#marimo.persistent_cache "            marimo.persistent_cache") to cache variables to
disk; on re-run, marimo will read values from disk instead of recalculating
them as long as the cell is not stale.
- Disable automatic execution in the [runtime configuration](../guides/configuration/runtime_configuration/).

### How do I disable automatic execution? [¶](\#how-do-i-disable-automatic-execution "Permanent link")

You can disable automatic execution through the notebook runtime settings;
see the [guide on runtime configuration](../guides/configuration/runtime_configuration/).

When automatic execution is disabled, marimo still gives you guarantees on
your notebook state and automatically marks cells as stale when appropriate.

### How do I use sliders and other interactive elements? [¶](\#how-do-i-use-sliders-and-other-interactive-elements "Permanent link")

Interactive UI elements like sliders are available in `marimo.ui`.

- Assign the UI element to a global variable ( `slider = mo.ui.slider(0, 100)`)
- Include it in the last expression of a cell to display it ( `slider` or `mo.md(f"Choose a value: {slider}")`)
- Read its current value in another cell via its `value` attribute ( `slider.value`)

_When a UI element bound to a global variable is interacted with, all cells_
_referencing the global variable are run automatically_.

If you have many UI elements or don't know the elements
you'll create until runtime, use `marimo.ui.array` and `marimo.ui.dictionary`
to create UI elements that wrap other UI elements ( `sliders =
mo.ui.array([slider(1, 100) for _ in range(n_sliders)])`).

All this and more is explained in the UI tutorial. Run it with

at the command line.

### How do I add a submit button to UI elements? [¶](\#how-do-i-add-a-submit-button-to-ui-elements "Permanent link")

Use the `form` method to add a submit button to a UI element. For
example,

When wrapped in a form, the
text area's value will only be sent to Python when you click the submit button.
Access the last submitted value of the text area with `form.value`.

### How do I write markdown? [¶](\#how-do-i-write-markdown "Permanent link")

Import `marimo` (as `mo`) in a notebook, and use the `mo.md` function.
Learn more in the [outputs guide](../guides/outputs/#markdown)
or by running `marimo tutorial markdown`.

### How do I display plots? [¶](\#how-do-i-display-plots "Permanent link")

Include plots in the last expression of a cell to display them, just like all
other outputs. If you're using matplotlib, you can display the `Figure` object
(get the current figure with `plt.gcf()`). For examples, run the plots tutorial:

Also see the [plotting API reference](../api/plotting/).

### How do I prevent matplotlib plots from being cut off? [¶](\#how-do-i-prevent-matplotlib-plots-from-being-cut-off "Permanent link")

If your legend or axes labels are cut off, try calling `plt.tight_layout()`
before outputting your plot:

### How do I display interactive matplotlib plots? [¶](\#how-do-i-display-interactive-matplotlib-plots "Permanent link")

Use [`marimo.mpl.interactive`](../api/plotting/#marimo.mpl.interactive "            marimo.mpl.interactive").

### How do I display objects in rows and columns? [¶](\#how-do-i-display-objects-in-rows-and-columns "Permanent link")

Use `marimo.hstack` and `marimo.vstack`. See the layout tutorial for details:

### How do I show cell code in the app view? [¶](\#how-do-i-show-cell-code-in-the-app-view "Permanent link")

Use [`mo.show_code`](../api/outputs/#marimo.show_code "            marimo.show_code").

### How do I create an output with a dynamic number of UI elements? [¶](\#how-do-i-create-an-output-with-a-dynamic-number-of-ui-elements "Permanent link")

Use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"),
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary"), or
[`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch") to create a UI element
that wraps a dynamic number of other UI elements.

If you need custom
formatting, use [`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch"), otherwise
use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array") or
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary").

For usage examples, see the
[recipes for grouping UI elements together](../recipes/#grouping-ui-elements-together).

### How do I restart a notebook? [¶](\#how-do-i-restart-a-notebook "Permanent link")

To clear all program memory and restart the notebook from scratch, open the
notebook menu in the top right and click "Restart kernel".

### How do I reload modules? [¶](\#how-do-i-reload-modules "Permanent link")

Enable automatic reloading of modules via the runtime settings in your
marimo installation's user configuration. (Click the "gear" icon in the
top right of a marimo notebook).

When enabled, marimo will automatically hot-reload modified modules
before executing a cell.

### Why aren't my `on_change`/ `on_click` handlers being called? [¶](\#why-arent-my-on_changeon_click-handlers-being-called "Permanent link")

A UI Element's `on_change` (or for buttons, `on_click`) handlers are only
called if the element is bound to a global variable. For example, this won't work

In such cases (when you want to output a dynamic number of UI elements),
you need to use
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"),
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary"), or
[`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch").

See the
[recipes for grouping UI elements together](../recipes/#grouping-ui-elements-together)
for example code.

### Why are my `on_change` handlers in an array all referencing the last element? [¶](\#why-are-my-on_change-handlers-in-an-array-all-referencing-the-last-element "Permanent link")

**Don't do this**: In the below snippet, every `on_change` will print `9`!.

**Instead, do this**: Explicitly bind `i` to the current loop value:

This is necessary because [in Python, closures are late-binding](https://docs.python-guide.org/writing/gotchas/#late-binding-closures).

### Why aren't my SQL brackets working? [¶](\#why-arent-my-sql-brackets-working "Permanent link")

Our "SQL" cells are really just Python under the hood to keep notebooks as pure Python scripts. By default, we use `f-strings` for SQL strings, which allows for parameterized SQL like `SELECT * from table where value < {min}`.

To escape real `{` / `}` that you don't want parameterized, use double `\{\{...\}\}`:

### How does marimo treat type annotations? [¶](\#how-does-marimo-treat-type-annotations "Permanent link")

Type annotations are registered as references of a cell, unless they
are explicitly written as strings. This helps ensure correctness of code that
depends on type annotations at runtime ( _e.g._, Pydantic), while still
providing a way to omit annotations from affecting dataflow graph.

For example, in

`A` is treated as a reference, used in determining the dataflow graph, but
in

`A` isn't made a reference.

For Python 3.12+, marimo additionally implements annotation scoping.

### How do I use dotenv? [¶](\#how-do-i-use-dotenv "Permanent link")

The package `dotenv`'s `loadenv()` function does not work out-of-the box in
marimo. Instead, use `dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))`.

### What packages can I use? [¶](\#what-packages-can-i-use "Permanent link")

You can use any Python package. marimo cells run arbitrary Python code.

### How do I use marimo on a remote server? [¶](\#how-do-i-use-marimo-on-a-remote-server "Permanent link")

Use SSH port-forwarding to run marimo on a remote server
and connect to it from a browser on your local machine. Make sure
to pass the `--headless` flag when starting marimo on remote; on the remote
machine, we also recommend using a port other than marimo's default port, such
as 8080:

_On the remote machine, run:_

or, if you want to set a custom host:

_On local, run:_

Then open `localhost:3718` in your browser.

### How do I make marimo accessible on all network interfaces? [¶](\#how-do-i-make-marimo-accessible-on-all-network-interfaces "Permanent link")

Use `--host 0.0.0.0` with `marimo edit`, `marimo run`, or `marimo tutorial`:

### How do I use marimo behind JupyterHub? [¶](\#how-do-i-use-marimo-behind-jupyterhub "Permanent link")

JupyterHub can be configured to launch marimo using the [`jupyter-marimo-proxy`\
package](https://github.com/jyio/jupyter-marimo-proxy).

### How do I use marimo with JupyterBook? [¶](\#how-do-i-use-marimo-with-jupyterbook "Permanent link")

[JupyterBook](https://jupyterbook.org/en/stable/intro.html) makes it easy
to create static websites with markdown and Jupyter notebooks.

To include a marimo notebook in a JupyterBook, you can either export your
notebook to an `ipynb` file, or export to `HTML`:

1. export to ipynb: `marimo export ipynb my_notebook.py -o my_notebook.ipynb --include-outputs`
2. export to HTML: `marimo export html my_notebook.py -o my_notebook.html`

### How do I deploy apps? [¶](\#how-do-i-deploy-apps "Permanent link")

Use the marimo CLI's `run` command to serve a notebook as an app:

If you are running marimo inside a Docker container, you may want to run under a different host and port:

### Is marimo free? [¶](\#is-marimo-free "Permanent link")

Yes!

Back to top

# FAQ

## Choosing marimo [¶](\#choosing-marimo "Permanent link")

### How is marimo different from Jupyter? [¶](\#how-is-marimo-different-from-jupyter "Permanent link")

marimo is a reinvention of the Python notebook as a reproducible, interactive,
and shareable Python program that can be executed as scripts or deployed as
interactive web apps.

**Consistent state.** In marimo, your notebook code, outputs, and program state
are guaranteed to be consistent. Run a cell and marimo reacts by automatically
running the cells that reference its variables. Delete a cell and marimo scrubs
its variables from program memory, eliminating hidden state.

**Built-in interactivity.** marimo also comes with [UI\
elements](../guides/interactivity/) like sliders, a dataframe transformer, and
interactive plots that are automatically synchronized with Python. Interact
with an element and the cells that use it are automatically re-run with its
latest value.

**Pure Python programs.** Unlike Jupyter notebooks, marimo notebooks are stored
as pure Python files that can be executed as scripts, deployed as interactive
web apps, and versioned easily with Git.

### What problems does marimo solve? [¶](\#what-problems-does-marimo-solve "Permanent link")

marimo solves problems in reproducibility, maintainability, interactivity,
reusability, and shareability of notebooks.

**Reproducibility.**
In Jupyter notebooks, the code you see doesn't necessarily match the outputs on
the page or the program state. If you
delete a cell, its variables stay in memory, which other cells may still
reference; users can execute cells in arbitrary order. This leads to
widespread reproducibility issues. [One study](https://blog.jetbrains.com/datalore/2020/12/17/we-downloaded-10-000-000-jupyter-notebooks-from-github-this-is-what-we-learned/#consistency-of-notebooks) analyzed 10 million Jupyter
notebooks and found that 36% of them weren't reproducible.

In contrast, marimo guarantees that your code, outputs, and program state are
consistent, eliminating hidden state and making your notebook reproducible.
marimo achieves this by intelligently analyzing your code and understanding the
relationships between cells, and automatically re-running cells as needed.

In addition, marimo notebooks can serialize package requirements inline;
marimo runs these "sandboxed" notebooks in temporary virtual environments,
making them [reproducible down to the packages](../guides/editor_features/package_management/).

**Maintainability.**
marimo notebooks are stored as pure Python programs ( `.py` files). This lets you
version them with Git; in contrast, Jupyter notebooks are stored as JSON and
require extra steps to version.

**Interactivity.**
marimo notebooks come with [UI elements](../guides/interactivity/) that are
automatically synchronized with Python (like sliders, dropdowns); _eg_, scrub a
slider and all cells that reference it are automatically re-run with the new
value. This is difficult to get working in Jupyter notebooks.

**Reusability.**
marimo notebooks can be executed as Python scripts from the command-line (since
they're stored as `.py` files). In contrast, this requires extra steps to
do for Jupyter, such as copying and pasting the code out or using external
frameworks. We also let you import symbols (functions, classes) defined in a
marimo notebook into other Python programs/notebooks, something you can't
easily do with Jupyter.

**Shareability.**
Every marimo notebook can double as an interactive web app, complete with UI
elements, which you can serve using the `marimo run` command. This isn't
possible in Jupyter without substantial extra effort.

_To learn more about problems with traditional notebooks,_
_see these references_
_[\[1\]](https://austinhenley.com/pubs/Chattopadhyay2020CHI_NotebookPainpoints.pdf) [\[2\]](https://www.youtube.com/watch?v=7jiPeIFXb6U&t=1s)._

### How is `marimo.ui` different from Jupyter widgets? [¶](\#how-is-marimoui-different-from-jupyter-widgets "Permanent link")

Unlike Jupyter widgets, marimo's interactive elements are automatically
synchronized with the Python kernel: no callbacks, no observers, no manually
re-running cells.

## Using marimo [¶](\#using-marimo "Permanent link")

### Is marimo a notebook or a library? [¶](\#is-marimo-a-notebook-or-a-library "Permanent link")

marimo is both a notebook and a library.

- Create _marimo notebooks_ with the editor that opens in your
browser when you run `marimo edit`.
- Use the _marimo library_ ( `import marimo as mo`) in
marimo notebooks. Write markdown with `mo.md(...)`,
create stateful interactive elements with `mo.ui` ( `mo.ui.slider(...)`), and
more. See the docs for an [API reference](../api/).

### What's the difference between a marimo notebook and a marimo app? [¶](\#whats-the-difference-between-a-marimo-notebook-and-a-marimo-app "Permanent link")

marimo programs are notebooks, apps, or both, depending on how you use them.

There are two ways to interact with a marimo program:

1. open it as a computational _notebook_ with `marimo edit`
2. run it as an interactive _app_ with `marimo run`

All marimo programs start as notebooks, since they are created with `marimo
edit`. Because marimo notebooks are reactive and have built-in interactive
elements, many can easily be made into useful and beautiful apps by simply
hiding the notebook code: this is what `marimo run` does.

Not every notebook needs to be run as an app — marimo notebooks are useful in
and of themselves for rapidly exploring data and doing reproducible science.
And not every app is improved by interacting with the notebook. In some
settings, such as collaborative research, education, and technical
presentations, going back and forth between the notebook view and app view
(which you can do from `marimo edit`) can be useful!

### How does marimo know what cells to run? [¶](\#how-does-marimo-know-what-cells-to-run "Permanent link")

marimo reads each cell once to determine what global names it defines and what
global names it reads. When a cell is run, marimo runs all other cells that
read any of the global names it defines. A global name can refer to a variable,
class, function, or import.

In other words, marimo uses _static analysis_ to make a dataflow graph out of
your cells. Each cell is a node in the graph across which global
variables "flow". Whenever a cell is run, either because you changed its
code or interacted with a UI element it reads, all its descendants run in turn.

### Does marimo slow my code down? [¶](\#does-marimo-slow-my-code-down "Permanent link")

No, marimo doesn't slow your code down. marimo determines the dependencies
among cells by reading your code, not running or tracing it, so there's
zero runtime overhead.

### How do I prevent automatic execution from running expensive cells? [¶](\#how-do-i-prevent-automatic-execution-from-running-expensive-cells "Permanent link")

Reactive (automatic) execution ensures your code and outputs are always
in sync, improving reproducibility by eliminating hidden state and
out-of-order execution; marimo also takes care to run only the minimal set of
cells needed to keep your notebook up to date. But when some cells take a long
time to run, it's understandable to be concerned that automatic execution will
kick off expensive cells before you're ready to run them.

_Here are some tips to avoid accidental execution of expensive cells:_

- [Disable expensive cells](../guides/reactivity/#disabling-cells). When a cell
is disabled, it and its descendants are blocked from running.
- Wrap UI elements in a [form](../api/inputs/form/#marimo.ui.form "            marimo.ui.form").
- Use [`mo.stop`](../api/control_flow/#marimo.stop "            marimo.stop") to conditionally stop
execution of a cell and its descendants.
- Decorate functions with marimo's [`mo.cache`](../api/caching/#marimo.cache "            marimo.cache") to cache
expensive intermediate computations.
- Use [`mo.persistent_cache`](../api/caching/#marimo.persistent_cache "            marimo.persistent_cache") to cache variables to
disk; on re-run, marimo will read values from disk instead of recalculating
them as long as the cell is not stale.
- Disable automatic execution in the [runtime configuration](../guides/configuration/runtime_configuration/).

### How do I disable automatic execution? [¶](\#how-do-i-disable-automatic-execution "Permanent link")

You can disable automatic execution through the notebook runtime settings;
see the [guide on runtime configuration](../guides/configuration/runtime_configuration/).

When automatic execution is disabled, marimo still gives you guarantees on
your notebook state and automatically marks cells as stale when appropriate.

### How do I use sliders and other interactive elements? [¶](\#how-do-i-use-sliders-and-other-interactive-elements "Permanent link")

Interactive UI elements like sliders are available in `marimo.ui`.

- Assign the UI element to a global variable ( `slider = mo.ui.slider(0, 100)`)
- Include it in the last expression of a cell to display it ( `slider` or `mo.md(f"Choose a value: {slider}")`)
- Read its current value in another cell via its `value` attribute ( `slider.value`)

_When a UI element bound to a global variable is interacted with, all cells_
_referencing the global variable are run automatically_.

If you have many UI elements or don't know the elements
you'll create until runtime, use `marimo.ui.array` and `marimo.ui.dictionary`
to create UI elements that wrap other UI elements ( `sliders =
mo.ui.array([slider(1, 100) for _ in range(n_sliders)])`).

All this and more is explained in the UI tutorial. Run it with

at the command line.

### How do I add a submit button to UI elements? [¶](\#how-do-i-add-a-submit-button-to-ui-elements "Permanent link")

Use the `form` method to add a submit button to a UI element. For
example,

When wrapped in a form, the
text area's value will only be sent to Python when you click the submit button.
Access the last submitted value of the text area with `form.value`.

### How do I write markdown? [¶](\#how-do-i-write-markdown "Permanent link")

Import `marimo` (as `mo`) in a notebook, and use the `mo.md` function.
Learn more in the [outputs guide](../guides/outputs/#markdown)
or by running `marimo tutorial markdown`.

### How do I display plots? [¶](\#how-do-i-display-plots "Permanent link")

Include plots in the last expression of a cell to display them, just like all
other outputs. If you're using matplotlib, you can display the `Figure` object
(get the current figure with `plt.gcf()`). For examples, run the plots tutorial:

Also see the [plotting API reference](../api/plotting/).

### How do I prevent matplotlib plots from being cut off? [¶](\#how-do-i-prevent-matplotlib-plots-from-being-cut-off "Permanent link")

If your legend or axes labels are cut off, try calling `plt.tight_layout()`
before outputting your plot:

### How do I display interactive matplotlib plots? [¶](\#how-do-i-display-interactive-matplotlib-plots "Permanent link")

Use [`marimo.mpl.interactive`](../api/plotting/#marimo.mpl.interactive "            marimo.mpl.interactive").

### How do I display objects in rows and columns? [¶](\#how-do-i-display-objects-in-rows-and-columns "Permanent link")

Use `marimo.hstack` and `marimo.vstack`. See the layout tutorial for details:

### How do I show cell code in the app view? [¶](\#how-do-i-show-cell-code-in-the-app-view "Permanent link")

Use [`mo.show_code`](../api/outputs/#marimo.show_code "            marimo.show_code").

### How do I create an output with a dynamic number of UI elements? [¶](\#how-do-i-create-an-output-with-a-dynamic-number-of-ui-elements "Permanent link")

Use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"),
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary"), or
[`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch") to create a UI element
that wraps a dynamic number of other UI elements.

If you need custom
formatting, use [`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch"), otherwise
use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array") or
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary").

For usage examples, see the
[recipes for grouping UI elements together](../recipes/#grouping-ui-elements-together).

### How do I restart a notebook? [¶](\#how-do-i-restart-a-notebook "Permanent link")

To clear all program memory and restart the notebook from scratch, open the
notebook menu in the top right and click "Restart kernel".

### How do I reload modules? [¶](\#how-do-i-reload-modules "Permanent link")

Enable automatic reloading of modules via the runtime settings in your
marimo installation's user configuration. (Click the "gear" icon in the
top right of a marimo notebook).

When enabled, marimo will automatically hot-reload modified modules
before executing a cell.

### Why aren't my `on_change`/ `on_click` handlers being called? [¶](\#why-arent-my-on_changeon_click-handlers-being-called "Permanent link")

A UI Element's `on_change` (or for buttons, `on_click`) handlers are only
called if the element is bound to a global variable. For example, this won't work

In such cases (when you want to output a dynamic number of UI elements),
you need to use
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"),
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary"), or
[`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch").

See the
[recipes for grouping UI elements together](../recipes/#grouping-ui-elements-together)
for example code.

### Why are my `on_change` handlers in an array all referencing the last element? [¶](\#why-are-my-on_change-handlers-in-an-array-all-referencing-the-last-element "Permanent link")

**Don't do this**: In the below snippet, every `on_change` will print `9`!.

**Instead, do this**: Explicitly bind `i` to the current loop value:

This is necessary because [in Python, closures are late-binding](https://docs.python-guide.org/writing/gotchas/#late-binding-closures).

### Why aren't my SQL brackets working? [¶](\#why-arent-my-sql-brackets-working "Permanent link")

Our "SQL" cells are really just Python under the hood to keep notebooks as pure Python scripts. By default, we use `f-strings` for SQL strings, which allows for parameterized SQL like `SELECT * from table where value < {min}`.

To escape real `{` / `}` that you don't want parameterized, use double `\{\{...\}\}`:

### How does marimo treat type annotations? [¶](\#how-does-marimo-treat-type-annotations "Permanent link")

Type annotations are registered as references of a cell, unless they
are explicitly written as strings. This helps ensure correctness of code that
depends on type annotations at runtime ( _e.g._, Pydantic), while still
providing a way to omit annotations from affecting dataflow graph.

For example, in

`A` is treated as a reference, used in determining the dataflow graph, but
in

`A` isn't made a reference.

For Python 3.12+, marimo additionally implements annotation scoping.

### How do I use dotenv? [¶](\#how-do-i-use-dotenv "Permanent link")

The package `dotenv`'s `loadenv()` function does not work out-of-the box in
marimo. Instead, use `dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))`.

### What packages can I use? [¶](\#what-packages-can-i-use "Permanent link")

You can use any Python package. marimo cells run arbitrary Python code.

### How do I use marimo on a remote server? [¶](\#how-do-i-use-marimo-on-a-remote-server "Permanent link")

Use SSH port-forwarding to run marimo on a remote server
and connect to it from a browser on your local machine. Make sure
to pass the `--headless` flag when starting marimo on remote; on the remote
machine, we also recommend using a port other than marimo's default port, such
as 8080:

_On the remote machine, run:_

or, if you want to set a custom host:

_On local, run:_

Then open `localhost:3718` in your browser.

### How do I make marimo accessible on all network interfaces? [¶](\#how-do-i-make-marimo-accessible-on-all-network-interfaces "Permanent link")

Use `--host 0.0.0.0` with `marimo edit`, `marimo run`, or `marimo tutorial`:

### How do I use marimo behind JupyterHub? [¶](\#how-do-i-use-marimo-behind-jupyterhub "Permanent link")

JupyterHub can be configured to launch marimo using the [`jupyter-marimo-proxy`\
package](https://github.com/jyio/jupyter-marimo-proxy).

### How do I use marimo with JupyterBook? [¶](\#how-do-i-use-marimo-with-jupyterbook "Permanent link")

[JupyterBook](https://jupyterbook.org/en/stable/intro.html) makes it easy
to create static websites with markdown and Jupyter notebooks.

To include a marimo notebook in a JupyterBook, you can either export your
notebook to an `ipynb` file, or export to `HTML`:

1. export to ipynb: `marimo export ipynb my_notebook.py -o my_notebook.ipynb --include-outputs`
2. export to HTML: `marimo export html my_notebook.py -o my_notebook.html`

### How do I deploy apps? [¶](\#how-do-i-deploy-apps "Permanent link")

Use the marimo CLI's `run` command to serve a notebook as an app:

If you are running marimo inside a Docker container, you may want to run under a different host and port:

### Is marimo free? [¶](\#is-marimo-free "Permanent link")

Yes!

 Back to top

# FAQ

## Choosing marimo [¶](\#choosing-marimo "Permanent link")

### How is marimo different from Jupyter? [¶](\#how-is-marimo-different-from-jupyter "Permanent link")

marimo is a reinvention of the Python notebook as a reproducible, interactive,
and shareable Python program that can be executed as scripts or deployed as
interactive web apps.

**Consistent state.** In marimo, your notebook code, outputs, and program state
are guaranteed to be consistent. Run a cell and marimo reacts by automatically
running the cells that reference its variables. Delete a cell and marimo scrubs
its variables from program memory, eliminating hidden state.

**Built-in interactivity.** marimo also comes with [UI\
elements](../guides/interactivity/) like sliders, a dataframe transformer, and
interactive plots that are automatically synchronized with Python. Interact
with an element and the cells that use it are automatically re-run with its
latest value.

**Pure Python programs.** Unlike Jupyter notebooks, marimo notebooks are stored
as pure Python files that can be executed as scripts, deployed as interactive
web apps, and versioned easily with Git.

### What problems does marimo solve? [¶](\#what-problems-does-marimo-solve "Permanent link")

marimo solves problems in reproducibility, maintainability, interactivity,
reusability, and shareability of notebooks.

**Reproducibility.**
In Jupyter notebooks, the code you see doesn't necessarily match the outputs on
the page or the program state. If you
delete a cell, its variables stay in memory, which other cells may still
reference; users can execute cells in arbitrary order. This leads to
widespread reproducibility issues. [One study](https://blog.jetbrains.com/datalore/2020/12/17/we-downloaded-10-000-000-jupyter-notebooks-from-github-this-is-what-we-learned/#consistency-of-notebooks) analyzed 10 million Jupyter
notebooks and found that 36% of them weren't reproducible.

In contrast, marimo guarantees that your code, outputs, and program state are
consistent, eliminating hidden state and making your notebook reproducible.
marimo achieves this by intelligently analyzing your code and understanding the
relationships between cells, and automatically re-running cells as needed.

In addition, marimo notebooks can serialize package requirements inline;
marimo runs these "sandboxed" notebooks in temporary virtual environments,
making them [reproducible down to the packages](../guides/editor_features/package_management/).

**Maintainability.**
marimo notebooks are stored as pure Python programs ( `.py` files). This lets you
version them with Git; in contrast, Jupyter notebooks are stored as JSON and
require extra steps to version.

**Interactivity.**
marimo notebooks come with [UI elements](../guides/interactivity/) that are
automatically synchronized with Python (like sliders, dropdowns); _eg_, scrub a
slider and all cells that reference it are automatically re-run with the new
value. This is difficult to get working in Jupyter notebooks.

**Reusability.**
marimo notebooks can be executed as Python scripts from the command-line (since
they're stored as `.py` files). In contrast, this requires extra steps to
do for Jupyter, such as copying and pasting the code out or using external
frameworks. We also let you import symbols (functions, classes) defined in a
marimo notebook into other Python programs/notebooks, something you can't
easily do with Jupyter.

**Shareability.**
Every marimo notebook can double as an interactive web app, complete with UI
elements, which you can serve using the `marimo run` command. This isn't
possible in Jupyter without substantial extra effort.

_To learn more about problems with traditional notebooks,_
_see these references_
_[\[1\]](https://austinhenley.com/pubs/Chattopadhyay2020CHI_NotebookPainpoints.pdf) [\[2\]](https://www.youtube.com/watch?v=7jiPeIFXb6U&t=1s)._

### How is `marimo.ui` different from Jupyter widgets? [¶](\#how-is-marimoui-different-from-jupyter-widgets "Permanent link")

Unlike Jupyter widgets, marimo's interactive elements are automatically
synchronized with the Python kernel: no callbacks, no observers, no manually
re-running cells.

## Using marimo [¶](\#using-marimo "Permanent link")

### Is marimo a notebook or a library? [¶](\#is-marimo-a-notebook-or-a-library "Permanent link")

marimo is both a notebook and a library.

- Create _marimo notebooks_ with the editor that opens in your
   browser when you run `marimo edit`.
- Use the _marimo library_ ( `import marimo as mo`) in
   marimo notebooks. Write markdown with `mo.md(...)`,
   create stateful interactive elements with `mo.ui` ( `mo.ui.slider(...)`), and
   more. See the docs for an [API reference](../api/).

### What's the difference between a marimo notebook and a marimo app? [¶](\#whats-the-difference-between-a-marimo-notebook-and-a-marimo-app "Permanent link")

marimo programs are notebooks, apps, or both, depending on how you use them.

There are two ways to interact with a marimo program:

1. open it as a computational _notebook_ with `marimo edit`
2. run it as an interactive _app_ with `marimo run`

All marimo programs start as notebooks, since they are created with `marimo
edit`. Because marimo notebooks are reactive and have built-in interactive
elements, many can easily be made into useful and beautiful apps by simply
hiding the notebook code: this is what `marimo run` does.

Not every notebook needs to be run as an app — marimo notebooks are useful in
and of themselves for rapidly exploring data and doing reproducible science.
And not every app is improved by interacting with the notebook. In some
settings, such as collaborative research, education, and technical
presentations, going back and forth between the notebook view and app view
(which you can do from `marimo edit`) can be useful!

### How does marimo know what cells to run? [¶](\#how-does-marimo-know-what-cells-to-run "Permanent link")

marimo reads each cell once to determine what global names it defines and what
global names it reads. When a cell is run, marimo runs all other cells that
read any of the global names it defines. A global name can refer to a variable,
class, function, or import.

In other words, marimo uses _static analysis_ to make a dataflow graph out of
your cells. Each cell is a node in the graph across which global
variables "flow". Whenever a cell is run, either because you changed its
code or interacted with a UI element it reads, all its descendants run in turn.

### Does marimo slow my code down? [¶](\#does-marimo-slow-my-code-down "Permanent link")

No, marimo doesn't slow your code down. marimo determines the dependencies
among cells by reading your code, not running or tracing it, so there's
zero runtime overhead.

### How do I prevent automatic execution from running expensive cells? [¶](\#how-do-i-prevent-automatic-execution-from-running-expensive-cells "Permanent link")

Reactive (automatic) execution ensures your code and outputs are always
in sync, improving reproducibility by eliminating hidden state and
out-of-order execution; marimo also takes care to run only the minimal set of
cells needed to keep your notebook up to date. But when some cells take a long
time to run, it's understandable to be concerned that automatic execution will
kick off expensive cells before you're ready to run them.

_Here are some tips to avoid accidental execution of expensive cells:_

- [Disable expensive cells](../guides/reactivity/#disabling-cells). When a cell
   is disabled, it and its descendants are blocked from running.
- Wrap UI elements in a [form](../api/inputs/form/#marimo.ui.form "            marimo.ui.form").
- Use [`mo.stop`](../api/control_flow/#marimo.stop "            marimo.stop") to conditionally stop
   execution of a cell and its descendants.
- Decorate functions with marimo's [`mo.cache`](../api/caching/#marimo.cache "            marimo.cache") to cache
   expensive intermediate computations.
- Use [`mo.persistent_cache`](../api/caching/#marimo.persistent_cache "            marimo.persistent_cache") to cache variables to
   disk; on re-run, marimo will read values from disk instead of recalculating
   them as long as the cell is not stale.
- Disable automatic execution in the [runtime configuration](../guides/configuration/runtime_configuration/).

### How do I disable automatic execution? [¶](\#how-do-i-disable-automatic-execution "Permanent link")

You can disable automatic execution through the notebook runtime settings;
see the [guide on runtime configuration](../guides/configuration/runtime_configuration/).

When automatic execution is disabled, marimo still gives you guarantees on
your notebook state and automatically marks cells as stale when appropriate.

### How do I use sliders and other interactive elements? [¶](\#how-do-i-use-sliders-and-other-interactive-elements "Permanent link")

Interactive UI elements like sliders are available in `marimo.ui`.

- Assign the UI element to a global variable ( `slider = mo.ui.slider(0, 100)`)
- Include it in the last expression of a cell to display it ( `slider` or `mo.md(f"Choose a value: {slider}")`)
- Read its current value in another cell via its `value` attribute ( `slider.value`)

_When a UI element bound to a global variable is interacted with, all cells_
_referencing the global variable are run automatically_.

If you have many UI elements or don't know the elements
you'll create until runtime, use `marimo.ui.array` and `marimo.ui.dictionary`
to create UI elements that wrap other UI elements ( `sliders =
mo.ui.array([slider(1, 100) for _ in range(n_sliders)])`).

All this and more is explained in the UI tutorial. Run it with

at the command line.

### How do I add a submit button to UI elements? [¶](\#how-do-i-add-a-submit-button-to-ui-elements "Permanent link")

Use the `form` method to add a submit button to a UI element. For
example,

When wrapped in a form, the
text area's value will only be sent to Python when you click the submit button.
Access the last submitted value of the text area with `form.value`.

### How do I write markdown? [¶](\#how-do-i-write-markdown "Permanent link")

Import `marimo` (as `mo`) in a notebook, and use the `mo.md` function.
Learn more in the [outputs guide](../guides/outputs/#markdown)
or by running `marimo tutorial markdown`.

### How do I display plots? [¶](\#how-do-i-display-plots "Permanent link")

Include plots in the last expression of a cell to display them, just like all
other outputs. If you're using matplotlib, you can display the `Figure` object
(get the current figure with `plt.gcf()`). For examples, run the plots tutorial:

Also see the [plotting API reference](../api/plotting/).

### How do I prevent matplotlib plots from being cut off? [¶](\#how-do-i-prevent-matplotlib-plots-from-being-cut-off "Permanent link")

If your legend or axes labels are cut off, try calling `plt.tight_layout()`
before outputting your plot:

### How do I display interactive matplotlib plots? [¶](\#how-do-i-display-interactive-matplotlib-plots "Permanent link")

Use [`marimo.mpl.interactive`](../api/plotting/#marimo.mpl.interactive "            marimo.mpl.interactive").

### How do I display objects in rows and columns? [¶](\#how-do-i-display-objects-in-rows-and-columns "Permanent link")

Use `marimo.hstack` and `marimo.vstack`. See the layout tutorial for details:

### How do I show cell code in the app view? [¶](\#how-do-i-show-cell-code-in-the-app-view "Permanent link")

Use [`mo.show_code`](../api/outputs/#marimo.show_code "            marimo.show_code").

### How do I create an output with a dynamic number of UI elements? [¶](\#how-do-i-create-an-output-with-a-dynamic-number-of-ui-elements "Permanent link")

Use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"),
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary"), or
[`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch") to create a UI element
that wraps a dynamic number of other UI elements.

If you need custom
formatting, use [`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch"), otherwise
use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array") or
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary").

For usage examples, see the
[recipes for grouping UI elements together](../recipes/#grouping-ui-elements-together).

### How do I restart a notebook? [¶](\#how-do-i-restart-a-notebook "Permanent link")

To clear all program memory and restart the notebook from scratch, open the
notebook menu in the top right and click "Restart kernel".

### How do I reload modules? [¶](\#how-do-i-reload-modules "Permanent link")

Enable automatic reloading of modules via the runtime settings in your
marimo installation's user configuration. (Click the "gear" icon in the
top right of a marimo notebook).

When enabled, marimo will automatically hot-reload modified modules
before executing a cell.

### Why aren't my `on_change`/ `on_click` handlers being called? [¶](\#why-arent-my-on_changeon_click-handlers-being-called "Permanent link")

A UI Element's `on_change` (or for buttons, `on_click`) handlers are only
called if the element is bound to a global variable. For example, this won't work

In such cases (when you want to output a dynamic number of UI elements),
you need to use
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"),
[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary"), or
[`mo.ui.batch`](../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch").

See the
[recipes for grouping UI elements together](../recipes/#grouping-ui-elements-together)
for example code.

### Why are my `on_change` handlers in an array all referencing the last element? [¶](\#why-are-my-on_change-handlers-in-an-array-all-referencing-the-last-element "Permanent link")

**Don't do this**: In the below snippet, every `on_change` will print `9`!.

**Instead, do this**: Explicitly bind `i` to the current loop value:

This is necessary because [in Python, closures are late-binding](https://docs.python-guide.org/writing/gotchas/#late-binding-closures).

### Why aren't my SQL brackets working? [¶](\#why-arent-my-sql-brackets-working "Permanent link")

Our "SQL" cells are really just Python under the hood to keep notebooks as pure Python scripts. By default, we use `f-strings` for SQL strings, which allows for parameterized SQL like `SELECT * from table where value < {min}`.

To escape real `{` / `}` that you don't want parameterized, use double `\{\{...\}\}`:

### How does marimo treat type annotations? [¶](\#how-does-marimo-treat-type-annotations "Permanent link")

Type annotations are registered as references of a cell, unless they
are explicitly written as strings. This helps ensure correctness of code that
depends on type annotations at runtime ( _e.g._, Pydantic), while still
providing a way to omit annotations from affecting dataflow graph.

For example, in

`A` is treated as a reference, used in determining the dataflow graph, but
in

`A` isn't made a reference.

For Python 3.12+, marimo additionally implements annotation scoping.

### How do I use dotenv? [¶](\#how-do-i-use-dotenv "Permanent link")

The package `dotenv`'s `loadenv()` function does not work out-of-the box in
marimo. Instead, use `dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))`.

### What packages can I use? [¶](\#what-packages-can-i-use "Permanent link")

You can use any Python package. marimo cells run arbitrary Python code.

### How do I use marimo on a remote server? [¶](\#how-do-i-use-marimo-on-a-remote-server "Permanent link")

Use SSH port-forwarding to run marimo on a remote server
and connect to it from a browser on your local machine. Make sure
to pass the `--headless` flag when starting marimo on remote; on the remote
machine, we also recommend using a port other than marimo's default port, such
as 8080:

_On the remote machine, run:_

or, if you want to set a custom host:

_On local, run:_

Then open `localhost:3718` in your browser.

### How do I make marimo accessible on all network interfaces? [¶](\#how-do-i-make-marimo-accessible-on-all-network-interfaces "Permanent link")

Use `--host 0.0.0.0` with `marimo edit`, `marimo run`, or `marimo tutorial`:

### How do I use marimo behind JupyterHub? [¶](\#how-do-i-use-marimo-behind-jupyterhub "Permanent link")

JupyterHub can be configured to launch marimo using the [`jupyter-marimo-proxy`\
package](https://github.com/jyio/jupyter-marimo-proxy).

### How do I use marimo with JupyterBook? [¶](\#how-do-i-use-marimo-with-jupyterbook "Permanent link")

[JupyterBook](https://jupyterbook.org/en/stable/intro.html) makes it easy
to create static websites with markdown and Jupyter notebooks.

To include a marimo notebook in a JupyterBook, you can either export your
notebook to an `ipynb` file, or export to `HTML`:

1. export to ipynb: `marimo export ipynb my_notebook.py -o my_notebook.ipynb --include-outputs`
2. export to HTML: `marimo export html my_notebook.py -o my_notebook.html`

### How do I deploy apps? [¶](\#how-do-i-deploy-apps "Permanent link")

Use the marimo CLI's `run` command to serve a notebook as an app:

If you are running marimo inside a Docker container, you may want to run under a different host and port:

### Is marimo free? [¶](\#is-marimo-free "Permanent link")

Yes!


---

###### Interactive elements - marimoInteractive elements¶

*Source: [https://docs.marimo.io/guides/interactivity/](https://docs.marimo.io/guides/interactivity/)*

<!-- Source: https://docs.marimo.io/guides/interactivity/ -->
<!-- Title: Interactive elements - marimoInteractive elements¶ -->
<!-- Depth: 7 -->

[Skip to content](#interactive-elements)

# Interactive elements [¶](\#interactive-elements "Permanent link")

One of marimo's most powerful features is its first-class support for
interactive user interface (UI) elements, or "widgets", created using
[`marimo.ui`](../../api/inputs/). **Interacting with a UI element bound to a**
**global variable automatically runs all cells that reference it.**

Examples

See the [API reference](../../api/inputs/) or our [GitHub\
repo](https://github.com/marimo-team/marimo/tree/main/examples/ui) for
bite-sized examples on using input elements.

## How interactions run cells [¶](\#how-interactions-run-cells "Permanent link")

Every UI element you make using [`marimo.ui`](../../api/inputs/) has a value, accessible via its
`value` attribute. When you interact with a UI element bound to a global
variable, its value is sent back to Python. A single rule determines what
happens next:

Interaction rule

When a UI element assigned to a global variable is interacted with, marimo
automatically runs all cells that reference the variable (but don't define it).

In the clip at the top of this page, interacting with the slider in the
second cell re-runs the third cell (which outputs markdown) because it
references the slider variable `x`. It doesn't re-run the second cell, because
that cell defines `x`.

**For interactions on a UI element to have any effect, the element must be**
**assigned to a global variable.**

## Displaying UI elements [¶](\#displaying-ui-elements "Permanent link")

Display UI elements in the output area above a cell by including them in the
last expression, just like any other object. You can also embed elements
in [markdown](../../api/markdown/#marimo.md "            marimo.md") using Python f-strings, like so:

## Composite elements [¶](\#composite-elements "Permanent link")

Composite elements are advanced elements let you build UI elements out of other
UI elements. The following composite elements are available:

- [`mo.ui.array`](../../api/inputs/array/#marimo.ui.array "            marimo.ui.array")
- [`mo.ui.dictionary`](../../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary")
- [`mo.ui.batch`](../../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch")
- [`mo.ui.form`](../../api/inputs/form/#marimo.ui.form "            marimo.ui.form")

**Arrays and dictionaries.**
Use [`mo.ui.array`](../../api/inputs/array/#marimo.ui.array "            marimo.ui.array") and
[`mo.ui.dictionary`](../../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") to logically group together related
elements. These elements are especially useful when a set of UI elements is
only known at runtime (so you can't assign each to a global variable
individually, but can assign them to an array or dictionary).

You can access the elements contained in an array or dictionary using
Pythonic syntax, and embed these elements in other outputs. See their docstrings
for code examples.

**Batch and form.**
Use these powerful elements to group together multiple UI elements into a
single element with custom formatting, and gate the sending of an element's
value on form submission.

Use a form to gate value updates on submission

![](/_static/array.png)Use an array to group together elements or create a collection of elements that is determined at runtime

## Building custom UI elements using our plugin API [¶](\#building-custom-ui-elements-using-our-plugin-api "Permanent link")

You can build your own reactive and interactive UI elements using
[anywidget](https://github.com/manzt/anywidget). See [our docs on\
building custom UI elements](../integrating_with_marimo/custom_ui_plugins/) to learn more.

Back to top

# Interactive elements [¶](\#interactive-elements "Permanent link")

One of marimo's most powerful features is its first-class support for
interactive user interface (UI) elements, or "widgets", created using
[`marimo.ui`](../../api/inputs/). **Interacting with a UI element bound to a**
**global variable automatically runs all cells that reference it.**

Examples

See the [API reference](../../api/inputs/) or our [GitHub\
repo](https://github.com/marimo-team/marimo/tree/main/examples/ui) for
bite-sized examples on using input elements.

## How interactions run cells [¶](\#how-interactions-run-cells "Permanent link")

Every UI element you make using [`marimo.ui`](../../api/inputs/) has a value, accessible via its
`value` attribute. When you interact with a UI element bound to a global
variable, its value is sent back to Python. A single rule determines what
happens next:

Interaction rule

When a UI element assigned to a global variable is interacted with, marimo
automatically runs all cells that reference the variable (but don't define it).

In the clip at the top of this page, interacting with the slider in the
second cell re-runs the third cell (which outputs markdown) because it
references the slider variable `x`. It doesn't re-run the second cell, because
that cell defines `x`.

**For interactions on a UI element to have any effect, the element must be**
**assigned to a global variable.**

## Displaying UI elements [¶](\#displaying-ui-elements "Permanent link")

Display UI elements in the output area above a cell by including them in the
last expression, just like any other object. You can also embed elements
in [markdown](../../api/markdown/#marimo.md "            marimo.md") using Python f-strings, like so:

## Composite elements [¶](\#composite-elements "Permanent link")

Composite elements are advanced elements let you build UI elements out of other
UI elements. The following composite elements are available:

- [`mo.ui.array`](../../api/inputs/array/#marimo.ui.array "            marimo.ui.array")
- [`mo.ui.dictionary`](../../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary")
- [`mo.ui.batch`](../../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch")
- [`mo.ui.form`](../../api/inputs/form/#marimo.ui.form "            marimo.ui.form")

**Arrays and dictionaries.**
Use [`mo.ui.array`](../../api/inputs/array/#marimo.ui.array "            marimo.ui.array") and
[`mo.ui.dictionary`](../../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") to logically group together related
elements. These elements are especially useful when a set of UI elements is
only known at runtime (so you can't assign each to a global variable
individually, but can assign them to an array or dictionary).

You can access the elements contained in an array or dictionary using
Pythonic syntax, and embed these elements in other outputs. See their docstrings
for code examples.

**Batch and form.**
Use these powerful elements to group together multiple UI elements into a
single element with custom formatting, and gate the sending of an element's
value on form submission.

Use a form to gate value updates on submission

![](/_static/array.png)Use an array to group together elements or create a collection of elements that is determined at runtime

## Building custom UI elements using our plugin API [¶](\#building-custom-ui-elements-using-our-plugin-api "Permanent link")

You can build your own reactive and interactive UI elements using
[anywidget](https://github.com/manzt/anywidget). See [our docs on\
building custom UI elements](../integrating_with_marimo/custom_ui_plugins/) to learn more.

 Back to top

# Interactive elements [¶](\#interactive-elements "Permanent link")

One of marimo's most powerful features is its first-class support for
interactive user interface (UI) elements, or "widgets", created using
[`marimo.ui`](../../api/inputs/). **Interacting with a UI element bound to a**
**global variable automatically runs all cells that reference it.**

Examples

See the [API reference](../../api/inputs/) or our [GitHub\
repo](https://github.com/marimo-team/marimo/tree/main/examples/ui) for
bite-sized examples on using input elements.

## How interactions run cells [¶](\#how-interactions-run-cells "Permanent link")

Every UI element you make using [`marimo.ui`](../../api/inputs/) has a value, accessible via its
`value` attribute. When you interact with a UI element bound to a global
variable, its value is sent back to Python. A single rule determines what
happens next:

Interaction rule

When a UI element assigned to a global variable is interacted with, marimo
automatically runs all cells that reference the variable (but don't define it).

In the clip at the top of this page, interacting with the slider in the
second cell re-runs the third cell (which outputs markdown) because it
references the slider variable `x`. It doesn't re-run the second cell, because
that cell defines `x`.

**For interactions on a UI element to have any effect, the element must be**
**assigned to a global variable.**

## Displaying UI elements [¶](\#displaying-ui-elements "Permanent link")

Display UI elements in the output area above a cell by including them in the
last expression, just like any other object. You can also embed elements
in [markdown](../../api/markdown/#marimo.md "            marimo.md") using Python f-strings, like so:

## Composite elements [¶](\#composite-elements "Permanent link")

Composite elements are advanced elements let you build UI elements out of other
UI elements. The following composite elements are available:

- [`mo.ui.array`](../../api/inputs/array/#marimo.ui.array "            marimo.ui.array")
- [`mo.ui.dictionary`](../../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary")
- [`mo.ui.batch`](../../api/inputs/batch/#marimo.ui.batch "            marimo.ui.batch")
- [`mo.ui.form`](../../api/inputs/form/#marimo.ui.form "            marimo.ui.form")

**Arrays and dictionaries.**
Use [`mo.ui.array`](../../api/inputs/array/#marimo.ui.array "            marimo.ui.array") and
[`mo.ui.dictionary`](../../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") to logically group together related
elements. These elements are especially useful when a set of UI elements is
only known at runtime (so you can't assign each to a global variable
individually, but can assign them to an array or dictionary).

You can access the elements contained in an array or dictionary using
Pythonic syntax, and embed these elements in other outputs. See their docstrings
for code examples.

**Batch and form.**
Use these powerful elements to group together multiple UI elements into a
single element with custom formatting, and gate the sending of an element's
value on form submission.

Use a form to gate value updates on submission

![](/_static/array.png)Use an array to group together elements or create a collection of elements that is determined at runtime

## Building custom UI elements using our plugin API [¶](\#building-custom-ui-elements-using-our-plugin-api "Permanent link")

You can build your own reactive and interactive UI elements using
[anywidget](https://github.com/manzt/anywidget). See [our docs on\
building custom UI elements](../integrating_with_marimo/custom_ui_plugins/) to learn more.


---

###### marimo islands 🏝️ - marimomarimo islands 🏝️¶

*Source: [https://docs.marimo.io/guides/island_example/](https://docs.marimo.io/guides/island_example/)*

<!-- Source: https://docs.marimo.io/guides/island_example/ -->
<!-- Title: marimo islands 🏝️ - marimomarimo islands 🏝️¶ -->
<!-- Depth: 7 -->

[Skip to content](#marimo-islands)

# marimo islands 🏝️ [¶](\#marimo-islands "Permanent link")

Preview

Islands are an early feature. While the API likely won't change, there are some improvements we'd like to make before we consider them stable.
Please let us know on [GitHub](https://github.com/marimo-team/marimo/issues) if you run into any issues or have any feedback!

> This content below is powered by marimo's reactive runtime. It will become interactive after initializing the marimo runtime.

* * *

import%20marimo%20as%20mo

slider%20%3D%20mo.ui.slider(0%2C%2010,value=2)%3B%20slider

Hello, islands! 🏝️🏝️

* * *

See the HTML

Back to top

# marimo islands 🏝️ [¶](\#marimo-islands "Permanent link")

Preview

Islands are an early feature. While the API likely won't change, there are some improvements we'd like to make before we consider them stable.
Please let us know on [GitHub](https://github.com/marimo-team/marimo/issues) if you run into any issues or have any feedback!

> This content below is powered by marimo's reactive runtime. It will become interactive after initializing the marimo runtime.

* * *

import%20marimo%20as%20mo

slider%20%3D%20mo.ui.slider(0%2C%2010,value=2)%3B%20slider

Hello, islands! 🏝️🏝️

* * *

See the HTML

 Back to top

# marimo islands 🏝️ [¶](\#marimo-islands "Permanent link")

Preview

Islands are an early feature. While the API likely won't change, there are some improvements we'd like to make before we consider them stable.
Please let us know on [GitHub](https://github.com/marimo-team/marimo/issues) if you run into any issues or have any feedback!

> This content below is powered by marimo's reactive runtime. It will become interactive after initializing the marimo runtime.

* * *

import%20marimo%20as%20mo

slider%20%3D%20mo.ui.slider(0%2C%2010,value=2)%3B%20slider

Hello, islands! 🏝️🏝️

* * *

See the HTML


---

###### Notebooks in existing projects - marimoNotebooks in existing projects¶

*Source: [https://docs.marimo.io/guides/package_management/notebooks_in_projects/](https://docs.marimo.io/guides/package_management/notebooks_in_projects/)*

<!-- Source: https://docs.marimo.io/guides/package_management/notebooks_in_projects/ -->
<!-- Title: Notebooks in existing projects - marimoNotebooks in existing projects¶ -->
<!-- Depth: 7 -->

[Skip to content](#notebooks-in-existing-projects)

# Notebooks in existing projects [¶](\#notebooks-in-existing-projects "Permanent link")

When working with notebooks in existing projects, there are two main approaches
depending on your needs:

1. **Sandbox notebooks** \- Self-contained notebooks with isolated dependencies
2. **Project notebooks** \- Notebooks that are part of your project's environment

marimo uses [PEP 723](https://peps.python.org/pep-0723/) inline script metadata
for sandboxing, managed by uv. While sandboxing is currently exclusive to the
uv package manager, other package managers may be supported in the future.

For project notebooks, marimo can be added as a project dependency where all
notebooks share the same environment defined in `pyproject.toml`. This approach
works with uv and other package managers (Poetry, Pixi, Hatch, etc.).

## Sandbox notebooks (recommended for libraries) [¶](\#sandbox-notebooks-recommended-for-libraries "Permanent link")

Sandbox notebooks use inline script metadata ( [PEP\
723](https://peps.python.org/pep-0723/)) to create isolated environments. This
is ideal when:

- Building examples for a library that users can run independently
- Creating notebooks that don't share dependencies with your main project
- Sharing self-contained notebooks that work anywhere

### Basic sandbox notebook [¶](\#basic-sandbox-notebook "Permanent link")

Sandbox notebooks can be created with:

When working in the notebook, marimo will automatically manage PEP 723 metadata
for you. This metadata makes the notebook self-contained, meaning you can
either come back later with marimo or run the notebook as a script directly
with uv:

### Developing against your local package [¶](\#developing-against-your-local-package "Permanent link")

When developing library examples, tutorials, or exploratory code, it's often
useful to have notebooks that _use_ your library. In these cases, you can
create and version sandboxed notebooks with your library as an _editable_
install.

This approach lets you:

- Test your library changes immediately without reinstalling
- Add notebook-specific dependencies (like visualization or data processing tools) without polluting your library's requirements
- Create self-contained examples that users can run without your development dependencies

To add your library to a notebook, use uv:

This will produce a header that looks like:

## Project notebooks [¶](\#project-notebooks "Permanent link")

For notebooks that are integral to your project, you can manage everything
through your project's `pyproject.toml`. This approach uses a single
environment shared between your project and notebooks.

### Adding marimo to your project [¶](\#adding-marimo-to-your-project "Permanent link")

Then work with notebooks using your project's environment:

This approach:
\- Uses a single environment for everything
\- Shares dependencies between notebooks and your project
\- Follows standard Python project practices

Importing from other directories

If your notebooks need to import modules from directories outside your project, marimo supports configuring the Python path via `pyproject.toml`. However, when possible, it's preferred to avoid path manipulation. We recommend creating a package ( `uv init --lib`) and including marimo as a development dependency. For multiple packages, consider configuring [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/). See the [runtime configuration guide](../../configuration/runtime_configuration/#python-path) for details.

## Examples [¶](\#examples "Permanent link")

### Library with example notebooks [¶](\#library-with-example-notebooks "Permanent link")

When building a library, use sandbox notebooks for examples that users can run
independently:

Create example notebooks:

### Data science project [¶](\#data-science-project "Permanent link")

When notebooks are part of your analysis workflow, use project notebooks:

Set up the project:

## Related guides [¶](\#related-guides "Permanent link")

- [Using uv](../using_uv/) \- Detailed guide on uv with marimo
- [Inlining dependencies](../inlining_dependencies/) \- More on self-contained notebooks
- [Package management overview](../) \- General package management in marimo

Back to top

# Notebooks in existing projects [¶](\#notebooks-in-existing-projects "Permanent link")

When working with notebooks in existing projects, there are two main approaches
depending on your needs:

1. **Sandbox notebooks** \- Self-contained notebooks with isolated dependencies
2. **Project notebooks** \- Notebooks that are part of your project's environment

marimo uses [PEP 723](https://peps.python.org/pep-0723/) inline script metadata
for sandboxing, managed by uv. While sandboxing is currently exclusive to the
uv package manager, other package managers may be supported in the future.

For project notebooks, marimo can be added as a project dependency where all
notebooks share the same environment defined in `pyproject.toml`. This approach
works with uv and other package managers (Poetry, Pixi, Hatch, etc.).

## Sandbox notebooks (recommended for libraries) [¶](\#sandbox-notebooks-recommended-for-libraries "Permanent link")

Sandbox notebooks use inline script metadata ( [PEP\
723](https://peps.python.org/pep-0723/)) to create isolated environments. This
is ideal when:

- Building examples for a library that users can run independently
- Creating notebooks that don't share dependencies with your main project
- Sharing self-contained notebooks that work anywhere

### Basic sandbox notebook [¶](\#basic-sandbox-notebook "Permanent link")

Sandbox notebooks can be created with:

When working in the notebook, marimo will automatically manage PEP 723 metadata
for you. This metadata makes the notebook self-contained, meaning you can
either come back later with marimo or run the notebook as a script directly
with uv:

### Developing against your local package [¶](\#developing-against-your-local-package "Permanent link")

When developing library examples, tutorials, or exploratory code, it's often
useful to have notebooks that _use_ your library. In these cases, you can
create and version sandboxed notebooks with your library as an _editable_
install.

This approach lets you:

- Test your library changes immediately without reinstalling
- Add notebook-specific dependencies (like visualization or data processing tools) without polluting your library's requirements
- Create self-contained examples that users can run without your development dependencies

To add your library to a notebook, use uv:

This will produce a header that looks like:

## Project notebooks [¶](\#project-notebooks "Permanent link")

For notebooks that are integral to your project, you can manage everything
through your project's `pyproject.toml`. This approach uses a single
environment shared between your project and notebooks.

### Adding marimo to your project [¶](\#adding-marimo-to-your-project "Permanent link")

Then work with notebooks using your project's environment:

This approach:
\- Uses a single environment for everything
\- Shares dependencies between notebooks and your project
\- Follows standard Python project practices

Importing from other directories

If your notebooks need to import modules from directories outside your project, marimo supports configuring the Python path via `pyproject.toml`. However, when possible, it's preferred to avoid path manipulation. We recommend creating a package ( `uv init --lib`) and including marimo as a development dependency. For multiple packages, consider configuring [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/). See the [runtime configuration guide](../../configuration/runtime_configuration/#python-path) for details.

## Examples [¶](\#examples "Permanent link")

### Library with example notebooks [¶](\#library-with-example-notebooks "Permanent link")

When building a library, use sandbox notebooks for examples that users can run
independently:

Create example notebooks:

### Data science project [¶](\#data-science-project "Permanent link")

When notebooks are part of your analysis workflow, use project notebooks:

Set up the project:

## Related guides [¶](\#related-guides "Permanent link")

- [Using uv](../using_uv/) \- Detailed guide on uv with marimo
- [Inlining dependencies](../inlining_dependencies/) \- More on self-contained notebooks
- [Package management overview](../) \- General package management in marimo

 Back to top

# Notebooks in existing projects [¶](\#notebooks-in-existing-projects "Permanent link")

When working with notebooks in existing projects, there are two main approaches
depending on your needs:

1. **Sandbox notebooks** \- Self-contained notebooks with isolated dependencies
2. **Project notebooks** \- Notebooks that are part of your project's environment

marimo uses [PEP 723](https://peps.python.org/pep-0723/) inline script metadata
for sandboxing, managed by uv. While sandboxing is currently exclusive to the
uv package manager, other package managers may be supported in the future.

For project notebooks, marimo can be added as a project dependency where all
notebooks share the same environment defined in `pyproject.toml`. This approach
works with uv and other package managers (Poetry, Pixi, Hatch, etc.).

## Sandbox notebooks (recommended for libraries) [¶](\#sandbox-notebooks-recommended-for-libraries "Permanent link")

Sandbox notebooks use inline script metadata ( [PEP\
723](https://peps.python.org/pep-0723/)) to create isolated environments. This
is ideal when:

- Building examples for a library that users can run independently
- Creating notebooks that don't share dependencies with your main project
- Sharing self-contained notebooks that work anywhere

### Basic sandbox notebook [¶](\#basic-sandbox-notebook "Permanent link")

Sandbox notebooks can be created with:

When working in the notebook, marimo will automatically manage PEP 723 metadata
for you. This metadata makes the notebook self-contained, meaning you can
either come back later with marimo or run the notebook as a script directly
with uv:

### Developing against your local package [¶](\#developing-against-your-local-package "Permanent link")

When developing library examples, tutorials, or exploratory code, it's often
useful to have notebooks that _use_ your library. In these cases, you can
create and version sandboxed notebooks with your library as an _editable_
install.

This approach lets you:

- Test your library changes immediately without reinstalling
- Add notebook-specific dependencies (like visualization or data processing tools) without polluting your library's requirements
- Create self-contained examples that users can run without your development dependencies

To add your library to a notebook, use uv:

This will produce a header that looks like:

## Project notebooks [¶](\#project-notebooks "Permanent link")

For notebooks that are integral to your project, you can manage everything
through your project's `pyproject.toml`. This approach uses a single
environment shared between your project and notebooks.

### Adding marimo to your project [¶](\#adding-marimo-to-your-project "Permanent link")

Then work with notebooks using your project's environment:

This approach:
\- Uses a single environment for everything
\- Shares dependencies between notebooks and your project
\- Follows standard Python project practices

Importing from other directories

If your notebooks need to import modules from directories outside your project, marimo supports configuring the Python path via `pyproject.toml`. However, when possible, it's preferred to avoid path manipulation. We recommend creating a package ( `uv init --lib`) and including marimo as a development dependency. For multiple packages, consider configuring [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/). See the [runtime configuration guide](../../configuration/runtime_configuration/#python-path) for details.

## Examples [¶](\#examples "Permanent link")

### Library with example notebooks [¶](\#library-with-example-notebooks "Permanent link")

When building a library, use sandbox notebooks for examples that users can run
independently:

Create example notebooks:

### Data science project [¶](\#data-science-project "Permanent link")

When notebooks are part of your analysis workflow, use project notebooks:

Set up the project:

## Related guides [¶](\#related-guides "Permanent link")

- [Using uv](../using_uv/) \- Detailed guide on uv with marimo
- [Inlining dependencies](../inlining_dependencies/) \- More on self-contained notebooks
- [Package management overview](../) \- General package management in marimo


---

###### Cloudflare - marimoPublish to Cloudflare¶

*Source: [https://docs.marimo.io/guides/publishing/cloudflare/](https://docs.marimo.io/guides/publishing/cloudflare/)*

<!-- Source: https://docs.marimo.io/guides/publishing/cloudflare/ -->
<!-- Title: Cloudflare - marimoPublish to Cloudflare¶ -->
<!-- Depth: 7 -->

[Skip to content](#publish-to-cloudflare)

# Publish to Cloudflare [¶](\#publish-to-cloudflare "Permanent link")

You can publish executable notebooks to [Cloudflare Workers](https://workers.cloudflare.com/)
for free, after exporting your notebook to a WebAssembly notebook.

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../../wasm/) with the flag `--include-cloudflare`:

Export as a readonly appExport as an editable notebook

See our [exporting guide](../../exporting/#export-to-wasm-powered-html) for
the full documentation.

## Publish to a Cloudflare Worker [¶](\#publish-to-a-cloudflare-worker "Permanent link")

When you use the `--include-cloudflare` flag, marimo creates two additional files in the parent directory of your output directory:

- `index.js`: A simple Cloudflare Worker script that serves your static assets
- `wrangler.jsonc`: Configuration for Cloudflare's Wrangler CLI

To run locally, run:

To deploy to Cloudflare, run:

Need authentication or custom endpoints?

You can modify the `index.js` to include authentication or custom endpoints. This allows you to:

- Add authentication logic to protect your notebook
- Create API endpoints that serve data from the same domain, avoiding CORS issues

## Publish to Cloudflare Pages using GitHub [¶](\#publish-to-cloudflare-pages-using-github "Permanent link")

As an alternative to Cloudflare Workers, you can publish to Cloudflare Pages. To get started, create a new GitHub repository by visiting [repo.new](https://repo.new/) . After creating a new repository, go to your newly created project directory to prepare and push your local application to GitHub by running the following commands in your terminal:

To deploy your site to Pages:

1. Log in to the Cloudflare [Dashboard](https://dash.cloudflare.com) and select your account.
2. In Account Home, select Workers & Pages > Create application > Pages > Connect to Git.
3. Select the new GitHub repository that you created and, in the Set up builds and deployments section, provide the following information:

1. Save and Deploy

## Publish to Cloudflare Pages Manually [¶](\#publish-to-cloudflare-pages-manually "Permanent link")

To deploy your site to Pages:

1. Create zip of the folder "output\_dir"
2. Log in to the Cloudflare [Dashboard](https://dash.cloudflare.com) and select your account.
3. In Account Home, select Workers & Pages > Create application > Pages > Upload asset.
4. Enter a project name then click Upload and select output\_dir.zip .
5. Save and Deploy

Back to top

# Publish to Cloudflare [¶](\#publish-to-cloudflare "Permanent link")

You can publish executable notebooks to [Cloudflare Workers](https://workers.cloudflare.com/)
for free, after exporting your notebook to a WebAssembly notebook.

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../../wasm/) with the flag `--include-cloudflare`:

Export as a readonly appExport as an editable notebook

See our [exporting guide](../../exporting/#export-to-wasm-powered-html) for
the full documentation.

## Publish to a Cloudflare Worker [¶](\#publish-to-a-cloudflare-worker "Permanent link")

When you use the `--include-cloudflare` flag, marimo creates two additional files in the parent directory of your output directory:

- `index.js`: A simple Cloudflare Worker script that serves your static assets
- `wrangler.jsonc`: Configuration for Cloudflare's Wrangler CLI

To run locally, run:

To deploy to Cloudflare, run:

Need authentication or custom endpoints?

You can modify the `index.js` to include authentication or custom endpoints. This allows you to:

- Add authentication logic to protect your notebook
- Create API endpoints that serve data from the same domain, avoiding CORS issues

## Publish to Cloudflare Pages using GitHub [¶](\#publish-to-cloudflare-pages-using-github "Permanent link")

As an alternative to Cloudflare Workers, you can publish to Cloudflare Pages. To get started, create a new GitHub repository by visiting [repo.new](https://repo.new/) . After creating a new repository, go to your newly created project directory to prepare and push your local application to GitHub by running the following commands in your terminal:

To deploy your site to Pages:

1. Log in to the Cloudflare [Dashboard](https://dash.cloudflare.com) and select your account.
2. In Account Home, select Workers & Pages > Create application > Pages > Connect to Git.
3. Select the new GitHub repository that you created and, in the Set up builds and deployments section, provide the following information:

1. Save and Deploy

## Publish to Cloudflare Pages Manually [¶](\#publish-to-cloudflare-pages-manually "Permanent link")

To deploy your site to Pages:

1. Create zip of the folder "output\_dir"
2. Log in to the Cloudflare [Dashboard](https://dash.cloudflare.com) and select your account.
3. In Account Home, select Workers & Pages > Create application > Pages > Upload asset.
4. Enter a project name then click Upload and select output\_dir.zip .
5. Save and Deploy

 Back to top

# Publish to Cloudflare [¶](\#publish-to-cloudflare "Permanent link")

You can publish executable notebooks to [Cloudflare Workers](https://workers.cloudflare.com/)
for free, after exporting your notebook to a WebAssembly notebook.

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../../wasm/) with the flag `--include-cloudflare`:

Export as a readonly appExport as an editable notebook

See our [exporting guide](../../exporting/#export-to-wasm-powered-html) for
the full documentation.

## Publish to a Cloudflare Worker [¶](\#publish-to-a-cloudflare-worker "Permanent link")

When you use the `--include-cloudflare` flag, marimo creates two additional files in the parent directory of your output directory:

- `index.js`: A simple Cloudflare Worker script that serves your static assets
- `wrangler.jsonc`: Configuration for Cloudflare's Wrangler CLI

To run locally, run:

To deploy to Cloudflare, run:

Need authentication or custom endpoints?

You can modify the `index.js` to include authentication or custom endpoints. This allows you to:

- Add authentication logic to protect your notebook
- Create API endpoints that serve data from the same domain, avoiding CORS issues

## Publish to Cloudflare Pages using GitHub [¶](\#publish-to-cloudflare-pages-using-github "Permanent link")

As an alternative to Cloudflare Workers, you can publish to Cloudflare Pages. To get started, create a new GitHub repository by visiting [repo.new](https://repo.new/) . After creating a new repository, go to your newly created project directory to prepare and push your local application to GitHub by running the following commands in your terminal:

To deploy your site to Pages:

1. Log in to the Cloudflare [Dashboard](https://dash.cloudflare.com) and select your account.
2. In Account Home, select Workers & Pages > Create application > Pages > Connect to Git.
3. Select the new GitHub repository that you created and, in the Set up builds and deployments section, provide the following information:

1. Save and Deploy

## Publish to Cloudflare Pages Manually [¶](\#publish-to-cloudflare-pages-manually "Permanent link")

To deploy your site to Pages:

1. Create zip of the folder "output\_dir"
2. Log in to the Cloudflare [Dashboard](https://dash.cloudflare.com) and select your account.
3. In Account Home, select Workers & Pages > Create application > Pages > Upload asset.
4. Enter a project name then click Upload and select output\_dir.zip .
5. Save and Deploy


---

###### Publishing with Quarto - marimoPublishing with Quarto¶

*Source: [https://docs.marimo.io/guides/publishing/quarto/](https://docs.marimo.io/guides/publishing/quarto/)*

<!-- Source: https://docs.marimo.io/guides/publishing/quarto/ -->
<!-- Title: Publishing with Quarto - marimoPublishing with Quarto¶ -->
<!-- Depth: 7 -->

[Skip to content](#publishing-with-quarto)

# Publishing with Quarto [¶](\#publishing-with-quarto "Permanent link")

[Quarto](https://quarto.org/) is a commonly
used open-source scientific and technical
publishing system that can be used to create
documents, presentations, websites, and more.
Quarto supports a variety of output formats,
including HTML, PDF, and Word.

marimo's [Quarto extension](https://github.com/marimo-team/quarto-marimo) allows you to utilize Quarto on marimo's markdown format to produce interactive, reactive, webpages.

Try the markdown file format tutorial

Learn more with `marimo tutorial markdown-format` at the command line.

To encourage quarto support, `.qmd` and `.md` files are interchangeable to marimo as an editor.

Alternatively, see our [MkDocs guide](../mkdocs/) for other methods of publishing marimo's markdown format.

Back to top

# Publishing with Quarto [¶](\#publishing-with-quarto "Permanent link")

[Quarto](https://quarto.org/) is a commonly
used open-source scientific and technical
publishing system that can be used to create
documents, presentations, websites, and more.
Quarto supports a variety of output formats,
including HTML, PDF, and Word.

marimo's [Quarto extension](https://github.com/marimo-team/quarto-marimo) allows you to utilize Quarto on marimo's markdown format to produce interactive, reactive, webpages.

Try the markdown file format tutorial

Learn more with `marimo tutorial markdown-format` at the command line.

To encourage quarto support, `.qmd` and `.md` files are interchangeable to marimo as an editor.

Alternatively, see our [MkDocs guide](../mkdocs/) for other methods of publishing marimo's markdown format.

 Back to top

# Publishing with Quarto [¶](\#publishing-with-quarto "Permanent link")

[Quarto](https://quarto.org/) is a commonly
used open-source scientific and technical
publishing system that can be used to create
documents, presentations, websites, and more.
Quarto supports a variety of output formats,
including HTML, PDF, and Word.

marimo's [Quarto extension](https://github.com/marimo-team/quarto-marimo) allows you to utilize Quarto on marimo's markdown format to produce interactive, reactive, webpages.

Try the markdown file format tutorial

Learn more with `marimo tutorial markdown-format` at the command line.

To encourage quarto support, `.qmd` and `.md` files are interchangeable to marimo as an editor.

Alternatively, see our [MkDocs guide](../mkdocs/) for other methods of publishing marimo's markdown format.


---

###### Cycles - marimoCycles¶

*Source: [https://docs.marimo.io/guides/understanding_errors/cycles/](https://docs.marimo.io/guides/understanding_errors/cycles/)*

<!-- Source: https://docs.marimo.io/guides/understanding_errors/cycles/ -->
<!-- Title: Cycles - marimoCycles¶ -->
<!-- Depth: 7 -->

[Skip to content](#cycles)

# Cycles [¶](\#cycles "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_cycles_error.png)

marimo raises this error when a cell is involved in a cycle on variables. In
this example, the first cell declares `a` and reads `b`, while the second cell
declares `b` and reads `a`.

## Why can't I have cycles? [¶](\#why-cant-i-have-cycles "Permanent link")

marimo parses your cells to understand the order in which they run:
run a cell, and cells that refer to its defined variables need to run afterward.
With a cycle, the execution order becomes ambiguous, while also introducing
an infinite loop.

**What do I get in return?**

By accepting this constraint on variables, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

**How do I read the error message?** In the error message, each line says which
cell defines a variable and which cell reads a variable. For example, `cell-0
-> a` means `cell-0` defines `a`, and `a -> cell-1` means `cell-1` reads `a`.
Similarly, `cell-1 -> b` means `cell-1` defines `b`, and `b -> cell-0` means
`cell-0` reads `b`. This creates the cycle from `cell-0 -> cell-1 -> cell-0`.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Cycles usually indicate that your notebook has a bug. Still, you can
fix the error by merging the cells involved in the cycle into a single cell:

Back to top

# Cycles [¶](\#cycles "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_cycles_error.png)

marimo raises this error when a cell is involved in a cycle on variables. In
this example, the first cell declares `a` and reads `b`, while the second cell
declares `b` and reads `a`.

## Why can't I have cycles? [¶](\#why-cant-i-have-cycles "Permanent link")

marimo parses your cells to understand the order in which they run:
run a cell, and cells that refer to its defined variables need to run afterward.
With a cycle, the execution order becomes ambiguous, while also introducing
an infinite loop.

**What do I get in return?**

By accepting this constraint on variables, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

**How do I read the error message?** In the error message, each line says which
cell defines a variable and which cell reads a variable. For example, `cell-0
-> a` means `cell-0` defines `a`, and `a -> cell-1` means `cell-1` reads `a`.
Similarly, `cell-1 -> b` means `cell-1` defines `b`, and `b -> cell-0` means
`cell-0` reads `b`. This creates the cycle from `cell-0 -> cell-1 -> cell-0`.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Cycles usually indicate that your notebook has a bug. Still, you can
fix the error by merging the cells involved in the cycle into a single cell:

 Back to top

# Cycles [¶](\#cycles "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_cycles_error.png)

marimo raises this error when a cell is involved in a cycle on variables. In
this example, the first cell declares `a` and reads `b`, while the second cell
declares `b` and reads `a`.

## Why can't I have cycles? [¶](\#why-cant-i-have-cycles "Permanent link")

marimo parses your cells to understand the order in which they run:
run a cell, and cells that refer to its defined variables need to run afterward.
With a cycle, the execution order becomes ambiguous, while also introducing
an infinite loop.

**What do I get in return?**

By accepting this constraint on variables, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

**How do I read the error message?** In the error message, each line says which
cell defines a variable and which cell reads a variable. For example, `cell-0
-> a` means `cell-0` defines `a`, and `a -> cell-1` means `cell-1` reads `a`.
Similarly, `cell-1 -> b` means `cell-1` defines `b`, and `b -> cell-0` means
`cell-0` reads `b`. This creates the cycle from `cell-0 -> cell-1 -> cell-0`.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Cycles usually indicate that your notebook has a bug. Still, you can
fix the error by merging the cells involved in the cycle into a single cell:


---

###### Import star - marimoStar imports¶

*Source: [https://docs.marimo.io/guides/understanding_errors/import_star/](https://docs.marimo.io/guides/understanding_errors/import_star/)*

<!-- Source: https://docs.marimo.io/guides/understanding_errors/import_star/ -->
<!-- Title: Import star - marimoStar imports¶ -->
<!-- Depth: 7 -->

[Skip to content](#star-imports)

# Star imports [¶](\#star-imports "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_import_star_error.png)

marimo raises this error you attempt to use `import *`.

In this example, `x` was already defined, and the subsequent cell raised an
error when we tried to run it. In your case, perhaps your variable is a loop
variable ( `i`), a dataframe ( `df`), or a plot ( `fig`, `ax`).

## Why can't I use `*` imports? [¶](\#why-cant-i-use-imports "Permanent link")

Star imports are incompatible with marimo's git-friendly file format and reproducible reactive execution:

- marimo's Python file format stores code in functions, so notebooks can be imported as regular Python modules without executing all their code. But Python disallows `import *` everywhere except at the top-level of a module.
- Star imports would also silently add names to globals, which would be
incompatible with [reactive execution](../../reactivity/).

Even Python's [official style guide](https://peps.python.org/pep-0008/) discourages the use of `import *`, writing:

> Wildcard imports (from import \*) should be avoided, as they make it unclear which names are present in the namespace, confusing both readers and many automated tools.

**What do I get in return?**

By accepting this constraint on imports, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Fixing this error is simple: just import the module, and use `.` notation
to access its members.

instead of

Back to top

# Star imports [¶](\#star-imports "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_import_star_error.png)

marimo raises this error you attempt to use `import *`.

In this example, `x` was already defined, and the subsequent cell raised an
error when we tried to run it. In your case, perhaps your variable is a loop
variable ( `i`), a dataframe ( `df`), or a plot ( `fig`, `ax`).

## Why can't I use `*` imports? [¶](\#why-cant-i-use-imports "Permanent link")

Star imports are incompatible with marimo's git-friendly file format and reproducible reactive execution:

- marimo's Python file format stores code in functions, so notebooks can be imported as regular Python modules without executing all their code. But Python disallows `import *` everywhere except at the top-level of a module.
- Star imports would also silently add names to globals, which would be
incompatible with [reactive execution](../../reactivity/).

Even Python's [official style guide](https://peps.python.org/pep-0008/) discourages the use of `import *`, writing:

> Wildcard imports (from import \*) should be avoided, as they make it unclear which names are present in the namespace, confusing both readers and many automated tools.

**What do I get in return?**

By accepting this constraint on imports, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Fixing this error is simple: just import the module, and use `.` notation
to access its members.

instead of

 Back to top

# Star imports [¶](\#star-imports "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_import_star_error.png)

marimo raises this error you attempt to use `import *`.

In this example, `x` was already defined, and the subsequent cell raised an
error when we tried to run it. In your case, perhaps your variable is a loop
variable ( `i`), a dataframe ( `df`), or a plot ( `fig`, `ax`).

## Why can't I use `*` imports? [¶](\#why-cant-i-use-imports "Permanent link")

Star imports are incompatible with marimo's git-friendly file format and reproducible reactive execution:

- marimo's Python file format stores code in functions, so notebooks can be imported as regular Python modules without executing all their code. But Python disallows `import *` everywhere except at the top-level of a module.
- Star imports would also silently add names to globals, which would be
  incompatible with [reactive execution](../../reactivity/).

Even Python's [official style guide](https://peps.python.org/pep-0008/) discourages the use of `import *`, writing:

> Wildcard imports (from import \*) should be avoided, as they make it unclear which names are present in the namespace, confusing both readers and many automated tools.

**What do I get in return?**

By accepting this constraint on imports, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Fixing this error is simple: just import the module, and use `.` notation
to access its members.

instead of


---

###### Multiple definitions - marimoMultiple definitions¶

*Source: [https://docs.marimo.io/guides/understanding_errors/multiple_definitions/](https://docs.marimo.io/guides/understanding_errors/multiple_definitions/)*

<!-- Source: https://docs.marimo.io/guides/understanding_errors/multiple_definitions/ -->
<!-- Title: Multiple definitions - marimoMultiple definitions¶ -->
<!-- Depth: 7 -->

[Skip to content](#multiple-definitions)

# Multiple definitions [¶](\#multiple-definitions "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_redefines_variables_error.png)

marimo raises this error when a variable is defined in multiple cells. In
this example, `x` was already defined, and the subsequent cell raised
an error when we tried to run it. In your case, perhaps your variable
is a loop variable ( `i`), a dataframe ( `df`), or a plot ( `fig`, `ax`).

Watch our YouTube explainer

## Why can't I redefine variables? [¶](\#why-cant-i-redefine-variables "Permanent link")

**Understanding the error message.** The error message tells you which other
cells defined the multiply defined variable. You can click on the cell name,
and marimo will highlight it.

**Why can't I redefine variables?** marimo guarantees that the code on the page
matches the outputs you see by determining a deterministic execution order on
cells; when one cell runs, marimo knows which others should run. But if two
cells defined `x`, and a third showed `x`, the output of the third cell would
be ambiguous, depending on which of the defining cells ran first (should it be
`0` or `1`?). That's a problem because it creates [hidden state and hidden\
bugs](../../coming_from/jupyter/), and it's part of the reason why [over 96% of\
Jupyter notebooks on GitHub aren't reproducible](https://leomurta.github.io/papers/pimentel2019a.pdf).

**What do I get in return?**

By accepting this constraint on variables, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

You have a few options.

### Use local variables [¶](\#use-local-variables "Permanent link")

In marimo, variables prefixed with an underscore ( `_x` or `_i`) are made local
to a cell, and can be redefined across multiple cells.

Use this in a pinch, but prefer encapsulating code in functions.

### Encapsulate code in a function [¶](\#encapsulate-code-in-a-function "Permanent link")

Python provides local scope through functions: if the variable that was
redefined is meant to be a temporary variable, then you can make it local to
the cell by encapsulating the code in a function. If any of the cell's
variables are not meant to be local, or are outputs meant to be displayed, just
return them from the function.

In general, we recommend writing modular code with meaningful functions. But,
in a pinch, just declare an anonymous function like this one to get a "local scope":

That's what clicking on the "Fix: Wrap in a function" button does. Note the function
`_()` is local to the cell.

### Merge cells [¶](\#merge-cells "Permanent link")

Often you can simply merge the cells that define the same variable into a single cell.
To incrementally show outputs in the cell, use [`mo.output.append`](../../../api/outputs/#marimo.output.append "            marimo.output.append")
or `print()`.

### Chain dataframe methods [¶](\#chain-dataframe-methods "Permanent link")

When working with dataframes, instead of splitting up operations across
multiple cells, chain operations in a single cell. This is especially ergonomic
when using [Polars](https://docs.pola.rs/), Daft, or other modern dataframe
libraries that support lazy execution.

Back to top

# Multiple definitions [¶](\#multiple-definitions "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_redefines_variables_error.png)

marimo raises this error when a variable is defined in multiple cells. In
this example, `x` was already defined, and the subsequent cell raised
an error when we tried to run it. In your case, perhaps your variable
is a loop variable ( `i`), a dataframe ( `df`), or a plot ( `fig`, `ax`).

Watch our YouTube explainer

## Why can't I redefine variables? [¶](\#why-cant-i-redefine-variables "Permanent link")

**Understanding the error message.** The error message tells you which other
cells defined the multiply defined variable. You can click on the cell name,
and marimo will highlight it.

**Why can't I redefine variables?** marimo guarantees that the code on the page
matches the outputs you see by determining a deterministic execution order on
cells; when one cell runs, marimo knows which others should run. But if two
cells defined `x`, and a third showed `x`, the output of the third cell would
be ambiguous, depending on which of the defining cells ran first (should it be
`0` or `1`?). That's a problem because it creates [hidden state and hidden\
bugs](../../coming_from/jupyter/), and it's part of the reason why [over 96% of\
Jupyter notebooks on GitHub aren't reproducible](https://leomurta.github.io/papers/pimentel2019a.pdf).

**What do I get in return?**

By accepting this constraint on variables, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

You have a few options.

### Use local variables [¶](\#use-local-variables "Permanent link")

In marimo, variables prefixed with an underscore ( `_x` or `_i`) are made local
to a cell, and can be redefined across multiple cells.

Use this in a pinch, but prefer encapsulating code in functions.

### Encapsulate code in a function [¶](\#encapsulate-code-in-a-function "Permanent link")

Python provides local scope through functions: if the variable that was
redefined is meant to be a temporary variable, then you can make it local to
the cell by encapsulating the code in a function. If any of the cell's
variables are not meant to be local, or are outputs meant to be displayed, just
return them from the function.

In general, we recommend writing modular code with meaningful functions. But,
in a pinch, just declare an anonymous function like this one to get a "local scope":

That's what clicking on the "Fix: Wrap in a function" button does. Note the function
`_()` is local to the cell.

### Merge cells [¶](\#merge-cells "Permanent link")

Often you can simply merge the cells that define the same variable into a single cell.
To incrementally show outputs in the cell, use [`mo.output.append`](../../../api/outputs/#marimo.output.append "            marimo.output.append")
or `print()`.

### Chain dataframe methods [¶](\#chain-dataframe-methods "Permanent link")

When working with dataframes, instead of splitting up operations across
multiple cells, chain operations in a single cell. This is especially ergonomic
when using [Polars](https://docs.pola.rs/), Daft, or other modern dataframe
libraries that support lazy execution.

 Back to top

# Multiple definitions [¶](\#multiple-definitions "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_redefines_variables_error.png)

marimo raises this error when a variable is defined in multiple cells. In
this example, `x` was already defined, and the subsequent cell raised
an error when we tried to run it. In your case, perhaps your variable
is a loop variable ( `i`), a dataframe ( `df`), or a plot ( `fig`, `ax`).

Watch our YouTube explainer

## Why can't I redefine variables? [¶](\#why-cant-i-redefine-variables "Permanent link")

**Understanding the error message.** The error message tells you which other
cells defined the multiply defined variable. You can click on the cell name,
and marimo will highlight it.

**Why can't I redefine variables?** marimo guarantees that the code on the page
matches the outputs you see by determining a deterministic execution order on
cells; when one cell runs, marimo knows which others should run. But if two
cells defined `x`, and a third showed `x`, the output of the third cell would
be ambiguous, depending on which of the defining cells ran first (should it be
`0` or `1`?). That's a problem because it creates [hidden state and hidden\
bugs](../../coming_from/jupyter/), and it's part of the reason why [over 96% of\
Jupyter notebooks on GitHub aren't reproducible](https://leomurta.github.io/papers/pimentel2019a.pdf).

**What do I get in return?**

By accepting this constraint on variables, marimo makes your notebooks:

- **reproducible**, with a well-defined execution order, no hidden state, and no hidden bugs;
- **executable** as a script;
- **interactive** with UI elements that work without callbacks;
- **shareable as a web app**, with far better performance than streamlit.

As a bonus, you'll find that you end up with cleaner, reusable code.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

You have a few options.

### Use local variables [¶](\#use-local-variables "Permanent link")

In marimo, variables prefixed with an underscore ( `_x` or `_i`) are made local
to a cell, and can be redefined across multiple cells.

Use this in a pinch, but prefer encapsulating code in functions.

### Encapsulate code in a function [¶](\#encapsulate-code-in-a-function "Permanent link")

Python provides local scope through functions: if the variable that was
redefined is meant to be a temporary variable, then you can make it local to
the cell by encapsulating the code in a function. If any of the cell's
variables are not meant to be local, or are outputs meant to be displayed, just
return them from the function.

In general, we recommend writing modular code with meaningful functions. But,
in a pinch, just declare an anonymous function like this one to get a "local scope":

That's what clicking on the "Fix: Wrap in a function" button does. Note the function
`_()` is local to the cell.

### Merge cells [¶](\#merge-cells "Permanent link")

Often you can simply merge the cells that define the same variable into a single cell.
To incrementally show outputs in the cell, use [`mo.output.append`](../../../api/outputs/#marimo.output.append "            marimo.output.append")
or `print()`.

### Chain dataframe methods [¶](\#chain-dataframe-methods "Permanent link")

When working with dataframes, instead of splitting up operations across
multiple cells, chain operations in a single cell. This is especially ergonomic
when using [Polars](https://docs.pola.rs/), Daft, or other modern dataframe
libraries that support lazy execution.


---

###### Setup References - marimoSetup references¶

*Source: [https://docs.marimo.io/guides/understanding_errors/setup/](https://docs.marimo.io/guides/understanding_errors/setup/)*

<!-- Source: https://docs.marimo.io/guides/understanding_errors/setup/ -->
<!-- Title: Setup References - marimoSetup references¶ -->
<!-- Depth: 7 -->

[Skip to content](#setup-references)

# Setup references [¶](\#setup-references "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_setup_error.png)

marimo raises this error when the setup cell references variables defined in
other cells. In the example above, `image` is defined elsewhere in the notebook,
and hence cannot be referenced.

## Why can't I refer to variables? [¶](\#why-cant-i-refer-to-variables "Permanent link")

The setup cell special: it runs before all other cells run, in order to provide
symbols that [top-level functions and classes](../../reusing_functions/) can use.
That's why it can't reference variables defined by other cells.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Define all needed variables in the setup cell. Or, if this code does not
need to run before all other cells (if you are not using top-level functions
or classes), simply move your code to a regular cell.

Back to top

# Setup references [¶](\#setup-references "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_setup_error.png)

marimo raises this error when the setup cell references variables defined in
other cells. In the example above, `image` is defined elsewhere in the notebook,
and hence cannot be referenced.

## Why can't I refer to variables? [¶](\#why-cant-i-refer-to-variables "Permanent link")

The setup cell special: it runs before all other cells run, in order to provide
symbols that [top-level functions and classes](../../reusing_functions/) can use.
That's why it can't reference variables defined by other cells.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Define all needed variables in the setup cell. Or, if this code does not
need to run before all other cells (if you are not using top-level functions
or classes), simply move your code to a regular cell.

 Back to top

# Setup references [¶](\#setup-references "Permanent link")

You're probably on this page because you just saw an error like this one:

![](/_static/docs_setup_error.png)

marimo raises this error when the setup cell references variables defined in
other cells. In the example above, `image` is defined elsewhere in the notebook,
and hence cannot be referenced.

## Why can't I refer to variables? [¶](\#why-cant-i-refer-to-variables "Permanent link")

The setup cell special: it runs before all other cells run, in order to provide
symbols that [top-level functions and classes](../../reusing_functions/) can use.
That's why it can't reference variables defined by other cells.

## How do I fix this error? [¶](\#how-do-i-fix-this-error "Permanent link")

Define all needed variables in the setup cell. Or, if this code does not
need to run before all other cells (if you are not using top-level functions
or classes), simply move your code to a regular cell.


---

###### WebAssembly notebooks - marimoWebAssembly Notebooks¶

*Source: [https://docs.marimo.io/guides/wasm/](https://docs.marimo.io/guides/wasm/)*

<!-- Source: https://docs.marimo.io/guides/wasm/ -->
<!-- Title: WebAssembly notebooks - marimoWebAssembly Notebooks¶ -->
<!-- Depth: 7 -->

[Skip to content](#webassembly-notebooks)

# WebAssembly Notebooks [¶](\#webassembly-notebooks "Permanent link")

marimo lets you execute notebooks _entirely in the browser_,
without a backend executing Python. marimo notebooks that
run entirely in the browser are called WebAssembly notebooks, or WASM notebooks
for short.

Try our online playground

To create your first WASM notebook, try our online playground
at [marimo.new](https://marimo.new). Read the [playground\
docs](../publishing/playground/) to learn more.

WASM notebooks have three benefits compared to notebooks hosted using a
traditional client-server model. WASM notebooks:

1. eliminate the need to install Python, making scientific computing accessible;
2. eliminate the cost and complexity of deploying backend infrastructure, making it easy to share notebooks;
3. eliminate network requests to a remote Python runner, making development feel snappy.

When should I use WASM notebooks?

WASM notebooks are excellent for sharing your work, quickly experimenting
with code and models, doing lightweight data exploration, authoring blog
posts, tutorials, and educational materials, and even building tools. For
notebooks that do heavy computation, [use marimo\
locally](../../getting_started/) or on a backend.

**Try it!** Try editing the below notebook (your browser, not a backend server, is executing it!)

_This feature is powered by [Pyodide](https://pyodide.org), a port_
_of Python to WebAssembly that enables browsers to run Python code._

## Creating WASM notebooks [¶](\#creating-wasm-notebooks "Permanent link")

marimo provides three ways to create and share WASM notebooks:

1. [Export to WASM HTML](../exporting/#export-to-wasm-powered-html),
    which you can host on GitHub Pages or self-host. This is great for
    publishing companion notebooks for research papers that are automatically
    updated on Git push, or for embedding interactive notebooks as part of other
    websites.
2. The [online playground](../publishing/playground/), which lets you
    create one-off notebooks and share via links, no login required. The
    playground is also great for embedding editable notebooks in
    documentation.
3. The [Community Cloud](../publishing/community_cloud/), which
    lets you save a collection of notebook to a workspace (for free!) and share
    publicly or privately with sensible URLs.

### From GitHub [¶](\#from-github "Permanent link")

marimo provides three ways to share notebooks stored on GitHub as WASM notebooks:

1. Automatically publish to GitHub Pages on git push with [our GitHub action](../publishing/github_pages/).
2. Load a notebook by URL into the online playground (New > Open from URL ...)
3. Load a notebook from GitHub in the [Community Cloud](../publishing/community_cloud/).

## Packages [¶](\#packages "Permanent link")

Rendering performance

To make sure markdown and other elements render quickly: make sure to put
`import marimo as mo` in its own cell, with no other lines of code.

WASM notebooks come with many packages pre-installed, including
NumPy, SciPy, scikit-learn, pandas, and matplotlib; see [Pyodide's\
documentation](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)
for a full list.

If you attempt to import a package that is not installed, marimo will
attempt to automatically install it for you. To manually install packages, use
[`micropip`](https://micropip.pyodide.org/en/stable/project/usage.html):

In one cell, import micropip:

In the next cell, install packages:

### Supported packages [¶](\#supported-packages "Permanent link")

All packages with pure Python wheels on PyPI are supported, as well as
additional packages like NumPy, SciPy, scikit-learn, duckdb, polars, and more.
For a full list of supported packages, see [Pyodide's\
documentation on supported packages.](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

If you want a package to be supported, consider [filing an issue](https://github.com/pyodide/pyodide/issues/new?assignees=&labels=new+package+request&projects=&template=package_request.md&title=).

## Including data [¶](\#including-data "Permanent link")

**For notebooks exported to WASM HTML.**
To include data files in notebooks [exported to WASM\
HTML](../exporting/#export-to-wasm-powered-html), place them
in a `public/` folder in the same directory as your notebook. When you
export to WASM HTML, the public folder will be copied to the export directory.

In order to access data both locally and when an exported notebook runs via
WebAssembly (e.g., hosted on GitHub Pages), use
[`mo.notebook_location()`](../../api/miscellaneous/#marimo.notebook_location "            marimo.notebook_location") to construct the path to
your data:

**Fetching data files from the web.**
Instead of bundling data files with your notebook, you can host data files on
the web and fetch them in your notebook. Depending on where your files are
hosted, you may need to use a CORS Proxy; see the [Pyodide\
documentation](https://pyodide.org/en/stable/usage/loading-packages.html#installing-wheels-from-arbitrary-urls)
for more details.

**Playground notebooks.** When opening a playground
notebook from GitHub, all the files in the GitHub repo are made available to
your notebook. See the [Playground\
Guide](../publishing/playground/#including-data-files) for more info.

**Community Cloud notebooks.** Our free [Community\
Cloud](../publishing/community_cloud/) lets you upload a limited
amount of data, and also lets you sync notebooks (and their data) from GitHub.

## Limitations [¶](\#limitations "Permanent link")

While WASM notebooks let you share marimo notebooks seamlessly, they have some
limitations.

**Packages.** Many but not all packages are supported. All packages with pure
Python wheels on PyPI are supported, as well as additional packages like NumPy,
SciPy, scikit-learn, duckdb, polars, and more. For a full list of supported
packages, see [Pyodide's documentation on supported\
packages.](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

If you want a package to be supported, consider [filing an\
issue](https://github.com/pyodide/pyodide/issues/new?assignees=&labels=new+package+request&projects=&template=package_request.md&title=).

**PDB.** PDB is not currently supported.

**Threading and multi-processing.** WASM notebooks do not support multithreading
and multiprocessing. [This may be fixed in the future](https://github.com/pyodide/pyodide/issues/237).

**Memory.** WASM notebooks have a memory limit of 2GB; this may be increased
in the future. If memory consumption is an issue, try offloading memory-intensive
computations to hosted APIs or precomputing expensive operations.

## Browser support [¶](\#browser-support "Permanent link")

WASM notebooks are supported in the latest versions of Chrome, Firefox, Edge, and Safari.

Chrome is the recommended browser for WASM notebooks as it seems to have the
best performance and compatibility.

Back to top

# WebAssembly Notebooks [¶](\#webassembly-notebooks "Permanent link")

marimo lets you execute notebooks _entirely in the browser_,
without a backend executing Python. marimo notebooks that
run entirely in the browser are called WebAssembly notebooks, or WASM notebooks
for short.

Try our online playground

To create your first WASM notebook, try our online playground
at [marimo.new](https://marimo.new). Read the [playground\
docs](../publishing/playground/) to learn more.

WASM notebooks have three benefits compared to notebooks hosted using a
traditional client-server model. WASM notebooks:

1. eliminate the need to install Python, making scientific computing accessible;
2. eliminate the cost and complexity of deploying backend infrastructure, making it easy to share notebooks;
3. eliminate network requests to a remote Python runner, making development feel snappy.

When should I use WASM notebooks?

WASM notebooks are excellent for sharing your work, quickly experimenting
with code and models, doing lightweight data exploration, authoring blog
posts, tutorials, and educational materials, and even building tools. For
notebooks that do heavy computation, [use marimo\
locally](../../getting_started/) or on a backend.

**Try it!** Try editing the below notebook (your browser, not a backend server, is executing it!)

_This feature is powered by [Pyodide](https://pyodide.org), a port_
_of Python to WebAssembly that enables browsers to run Python code._

## Creating WASM notebooks [¶](\#creating-wasm-notebooks "Permanent link")

marimo provides three ways to create and share WASM notebooks:

1. [Export to WASM HTML](../exporting/#export-to-wasm-powered-html),
    which you can host on GitHub Pages or self-host. This is great for
    publishing companion notebooks for research papers that are automatically
    updated on Git push, or for embedding interactive notebooks as part of other
    websites.
2. The [online playground](../publishing/playground/), which lets you
    create one-off notebooks and share via links, no login required. The
    playground is also great for embedding editable notebooks in
    documentation.
3. The [Community Cloud](../publishing/community_cloud/), which
    lets you save a collection of notebook to a workspace (for free!) and share
    publicly or privately with sensible URLs.

### From GitHub [¶](\#from-github "Permanent link")

marimo provides three ways to share notebooks stored on GitHub as WASM notebooks:

1. Automatically publish to GitHub Pages on git push with [our GitHub action](../publishing/github_pages/).
2. Load a notebook by URL into the online playground (New > Open from URL ...)
3. Load a notebook from GitHub in the [Community Cloud](../publishing/community_cloud/).

## Packages [¶](\#packages "Permanent link")

Rendering performance

To make sure markdown and other elements render quickly: make sure to put
`import marimo as mo` in its own cell, with no other lines of code.

WASM notebooks come with many packages pre-installed, including
NumPy, SciPy, scikit-learn, pandas, and matplotlib; see [Pyodide's\
documentation](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)
for a full list.

If you attempt to import a package that is not installed, marimo will
attempt to automatically install it for you. To manually install packages, use
[`micropip`](https://micropip.pyodide.org/en/stable/project/usage.html):

In one cell, import micropip:

In the next cell, install packages:

### Supported packages [¶](\#supported-packages "Permanent link")

All packages with pure Python wheels on PyPI are supported, as well as
additional packages like NumPy, SciPy, scikit-learn, duckdb, polars, and more.
For a full list of supported packages, see [Pyodide's\
documentation on supported packages.](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

If you want a package to be supported, consider [filing an issue](https://github.com/pyodide/pyodide/issues/new?assignees=&labels=new+package+request&projects=&template=package_request.md&title=).

## Including data [¶](\#including-data "Permanent link")

**For notebooks exported to WASM HTML.**
To include data files in notebooks [exported to WASM\
HTML](../exporting/#export-to-wasm-powered-html), place them
in a `public/` folder in the same directory as your notebook. When you
export to WASM HTML, the public folder will be copied to the export directory.

In order to access data both locally and when an exported notebook runs via
WebAssembly (e.g., hosted on GitHub Pages), use
[`mo.notebook_location()`](../../api/miscellaneous/#marimo.notebook_location "            marimo.notebook_location") to construct the path to
your data:

**Fetching data files from the web.**
Instead of bundling data files with your notebook, you can host data files on
the web and fetch them in your notebook. Depending on where your files are
hosted, you may need to use a CORS Proxy; see the [Pyodide\
documentation](https://pyodide.org/en/stable/usage/loading-packages.html#installing-wheels-from-arbitrary-urls)
for more details.

**Playground notebooks.** When opening a playground
notebook from GitHub, all the files in the GitHub repo are made available to
your notebook. See the [Playground\
Guide](../publishing/playground/#including-data-files) for more info.

**Community Cloud notebooks.** Our free [Community\
Cloud](../publishing/community_cloud/) lets you upload a limited
amount of data, and also lets you sync notebooks (and their data) from GitHub.

## Limitations [¶](\#limitations "Permanent link")

While WASM notebooks let you share marimo notebooks seamlessly, they have some
limitations.

**Packages.** Many but not all packages are supported. All packages with pure
Python wheels on PyPI are supported, as well as additional packages like NumPy,
SciPy, scikit-learn, duckdb, polars, and more. For a full list of supported
packages, see [Pyodide's documentation on supported\
packages.](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

If you want a package to be supported, consider [filing an\
issue](https://github.com/pyodide/pyodide/issues/new?assignees=&labels=new+package+request&projects=&template=package_request.md&title=).

**PDB.** PDB is not currently supported.

**Threading and multi-processing.** WASM notebooks do not support multithreading
and multiprocessing. [This may be fixed in the future](https://github.com/pyodide/pyodide/issues/237).

**Memory.** WASM notebooks have a memory limit of 2GB; this may be increased
in the future. If memory consumption is an issue, try offloading memory-intensive
computations to hosted APIs or precomputing expensive operations.

## Browser support [¶](\#browser-support "Permanent link")

WASM notebooks are supported in the latest versions of Chrome, Firefox, Edge, and Safari.

Chrome is the recommended browser for WASM notebooks as it seems to have the
best performance and compatibility.

 Back to top

# WebAssembly Notebooks [¶](\#webassembly-notebooks "Permanent link")

marimo lets you execute notebooks _entirely in the browser_,
without a backend executing Python. marimo notebooks that
run entirely in the browser are called WebAssembly notebooks, or WASM notebooks
for short.

Try our online playground

To create your first WASM notebook, try our online playground
at [marimo.new](https://marimo.new). Read the [playground\
docs](../publishing/playground/) to learn more.

WASM notebooks have three benefits compared to notebooks hosted using a
traditional client-server model. WASM notebooks:

1. eliminate the need to install Python, making scientific computing accessible;
2. eliminate the cost and complexity of deploying backend infrastructure, making it easy to share notebooks;
3. eliminate network requests to a remote Python runner, making development feel snappy.

When should I use WASM notebooks?

WASM notebooks are excellent for sharing your work, quickly experimenting
with code and models, doing lightweight data exploration, authoring blog
posts, tutorials, and educational materials, and even building tools. For
notebooks that do heavy computation, [use marimo\
locally](../../getting_started/) or on a backend.

**Try it!** Try editing the below notebook (your browser, not a backend server, is executing it!)

_This feature is powered by [Pyodide](https://pyodide.org), a port_
_of Python to WebAssembly that enables browsers to run Python code._

## Creating WASM notebooks [¶](\#creating-wasm-notebooks "Permanent link")

marimo provides three ways to create and share WASM notebooks:

1. [Export to WASM HTML](../exporting/#export-to-wasm-powered-html),
    which you can host on GitHub Pages or self-host. This is great for
    publishing companion notebooks for research papers that are automatically
    updated on Git push, or for embedding interactive notebooks as part of other
    websites.
2. The [online playground](../publishing/playground/), which lets you
    create one-off notebooks and share via links, no login required. The
    playground is also great for embedding editable notebooks in
    documentation.
3. The [Community Cloud](../publishing/community_cloud/), which
    lets you save a collection of notebook to a workspace (for free!) and share
    publicly or privately with sensible URLs.

### From GitHub [¶](\#from-github "Permanent link")

marimo provides three ways to share notebooks stored on GitHub as WASM notebooks:

1. Automatically publish to GitHub Pages on git push with [our GitHub action](../publishing/github_pages/).
2. Load a notebook by URL into the online playground (New > Open from URL ...)
3. Load a notebook from GitHub in the [Community Cloud](../publishing/community_cloud/).

## Packages [¶](\#packages "Permanent link")

Rendering performance

To make sure markdown and other elements render quickly: make sure to put
`import marimo as mo` in its own cell, with no other lines of code.

WASM notebooks come with many packages pre-installed, including
NumPy, SciPy, scikit-learn, pandas, and matplotlib; see [Pyodide's\
documentation](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)
for a full list.

If you attempt to import a package that is not installed, marimo will
attempt to automatically install it for you. To manually install packages, use
[`micropip`](https://micropip.pyodide.org/en/stable/project/usage.html):

In one cell, import micropip:

In the next cell, install packages:

### Supported packages [¶](\#supported-packages "Permanent link")

All packages with pure Python wheels on PyPI are supported, as well as
additional packages like NumPy, SciPy, scikit-learn, duckdb, polars, and more.
For a full list of supported packages, see [Pyodide's\
documentation on supported packages.](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

If you want a package to be supported, consider [filing an issue](https://github.com/pyodide/pyodide/issues/new?assignees=&labels=new+package+request&projects=&template=package_request.md&title=).

## Including data [¶](\#including-data "Permanent link")

**For notebooks exported to WASM HTML.**
To include data files in notebooks [exported to WASM\
HTML](../exporting/#export-to-wasm-powered-html), place them
in a `public/` folder in the same directory as your notebook. When you
export to WASM HTML, the public folder will be copied to the export directory.

In order to access data both locally and when an exported notebook runs via
WebAssembly (e.g., hosted on GitHub Pages), use
[`mo.notebook_location()`](../../api/miscellaneous/#marimo.notebook_location "            marimo.notebook_location") to construct the path to
your data:

**Fetching data files from the web.**
Instead of bundling data files with your notebook, you can host data files on
the web and fetch them in your notebook. Depending on where your files are
hosted, you may need to use a CORS Proxy; see the [Pyodide\
documentation](https://pyodide.org/en/stable/usage/loading-packages.html#installing-wheels-from-arbitrary-urls)
for more details.

**Playground notebooks.** When opening a playground
notebook from GitHub, all the files in the GitHub repo are made available to
your notebook. See the [Playground\
Guide](../publishing/playground/#including-data-files) for more info.

**Community Cloud notebooks.** Our free [Community\
Cloud](../publishing/community_cloud/) lets you upload a limited
amount of data, and also lets you sync notebooks (and their data) from GitHub.

## Limitations [¶](\#limitations "Permanent link")

While WASM notebooks let you share marimo notebooks seamlessly, they have some
limitations.

**Packages.** Many but not all packages are supported. All packages with pure
Python wheels on PyPI are supported, as well as additional packages like NumPy,
SciPy, scikit-learn, duckdb, polars, and more. For a full list of supported
packages, see [Pyodide's documentation on supported\
packages.](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

If you want a package to be supported, consider [filing an\
issue](https://github.com/pyodide/pyodide/issues/new?assignees=&labels=new+package+request&projects=&template=package_request.md&title=).

**PDB.** PDB is not currently supported.

**Threading and multi-processing.** WASM notebooks do not support multithreading
and multiprocessing. [This may be fixed in the future](https://github.com/pyodide/pyodide/issues/237).

**Memory.** WASM notebooks have a memory limit of 2GB; this may be increased
in the future. If memory consumption is an issue, try offloading memory-intensive
computations to hosted APIs or precomputing expensive operations.

## Browser support [¶](\#browser-support "Permanent link")

WASM notebooks are supported in the latest versions of Chrome, Firefox, Edge, and Safari.

Chrome is the recommended browser for WASM notebooks as it seems to have the
best performance and compatibility.


---

###### Inputs - marimoInputs¶

*Source: [https://docs.marimo.io/api/inputs/](https://docs.marimo.io/api/inputs/)*

<!-- Source: https://docs.marimo.io/api/inputs/ -->
<!-- Title: Inputs - marimoInputs¶ -->
<!-- Depth: 8 -->

[Skip to content](#inputs)

# Inputs [¶](\#inputs "Permanent link")

marimo comes packaged with interactive UI elements that you can use to build
powerful notebooks and apps. These elements are available in `marimo.ui`.

| Element | Description |
| --- | --- |
| [`marimo.ui.array`](array/#marimo.ui.array "            marimo.ui.array") | Create array inputs |
| [`marimo.ui.batch`](batch/#marimo.ui.batch "            marimo.ui.batch") | Batch operations |
| [`marimo.ui.button`](button/#marimo.ui.button "            marimo.ui.button") | Create buttons |
| [`marimo.ui.chat`](chat/#marimo.ui.chat "            marimo.ui.chat") | Create chat interfaces |
| [`marimo.ui.checkbox`](checkbox/#marimo.ui.checkbox "            marimo.ui.checkbox") | Create checkboxes |
| [`marimo.ui.code_editor`](code_editor/#marimo.ui.code_editor "            marimo.ui.code_editor") | Create code editors |
| [`marimo.ui.dataframe`](dataframe/#marimo.ui.dataframe "            marimo.ui.dataframe") | Interactive dataframes |
| [`marimo.ui.data_explorer`](data_explorer/#marimo.ui.data_explorer "            marimo.ui.data_explorer") | Explore data |
| [`marimo.ui.date`](dates/#marimo.ui.date "            marimo.ui.date") | Date picker |
| [`marimo.ui.datetime`](dates/#marimo.ui.datetime "            marimo.ui.datetime") | Date and time picker |
| [`marimo.ui.date_range`](dates/#marimo.ui.date_range "            marimo.ui.date_range") | Date range picker |
| [`marimo.ui.dictionary`](dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") | Dictionary inputs |
| [`marimo.ui.dropdown`](dropdown/#marimo.ui.dropdown "            marimo.ui.dropdown") | Create dropdowns |
| [`marimo.ui.file`](file/#marimo.ui.file "            marimo.ui.file") | File uploads |
| [`marimo.ui.file_browser`](file_browser/#marimo.ui.file_browser "            marimo.ui.file_browser") | Browse files |
| [`marimo.ui.form`](form/#marimo.ui.form "            marimo.ui.form") | Create forms |
| [`marimo.ui.microphone`](microphone/#marimo.ui.microphone "            marimo.ui.microphone") | Record audio |
| [`marimo.ui.multiselect`](multiselect/#marimo.ui.multiselect "            marimo.ui.multiselect") | Multiple selection |
| [`marimo.ui.number`](number/#marimo.ui.number "            marimo.ui.number") | Number inputs |
| [`marimo.ui.radio`](radio/#marimo.ui.radio "            marimo.ui.radio") | Radio buttons |
| [`marimo.ui.range_slider`](range_slider/#marimo.ui.range_slider "            marimo.ui.range_slider") | Range sliders |
| [`marimo.ui.refresh`](refresh/#marimo.ui.refresh "            marimo.ui.refresh") | Refresh buttons |
| [`marimo.ui.run_button`](run_button/#marimo.ui.run_button "            marimo.ui.run_button") | Run buttons |
| [`marimo.ui.slider`](slider/#marimo.ui.slider "            marimo.ui.slider") | Create sliders |
| [`marimo.ui.switch`](switch/#marimo.ui.switch "            marimo.ui.switch") | Toggle switches |
| [`marimo.ui.tabs`](tabs/#marimo.ui.tabs "            marimo.ui.tabs") | Tabbed interfaces |
| [`marimo.ui.table`](table/#marimo.ui.table "            marimo.ui.table") | Interactive tables |
| [`marimo.ui.text`](text/#marimo.ui.text "            marimo.ui.text") | Text inputs |
| [`marimo.ui.text_area`](text_area/#marimo.ui.text_area "            marimo.ui.text_area") | Multiline text inputs |

To use a UI element, assign it to a global variable and output it in a cell.
When you interact with the frontend element, the Python object's `value`
attribute is automatically updated, and all cells referencing that object
automatically run with the element's latest value.

## Integrations [¶](\#integrations "Permanent link")

| Integration | Description |
| --- | --- |
| [`marimo.ui.altair_chart`](../plotting/#marimo.ui.altair_chart "            marimo.ui.altair_chart") | Interactive Altair charts |
| [`marimo.ui.plotly`](../plotting/#marimo.ui.plotly "            marimo.ui.plotly") | Interactive Plotly charts |
| [`marimo.mpl.interactive`](../plotting/#marimo.mpl.interactive "            marimo.mpl.interactive") | Interactive Matplotlib plots |
| [`marimo.ui.anywidget`](anywidget/#marimo.ui.anywidget "            marimo.ui.anywidget") | Custom widgets |

Back to top

# Inputs [¶](\#inputs "Permanent link")

marimo comes packaged with interactive UI elements that you can use to build
powerful notebooks and apps. These elements are available in `marimo.ui`.

| Element | Description |
| --- | --- |
| [`marimo.ui.array`](array/#marimo.ui.array "            marimo.ui.array") | Create array inputs |
| [`marimo.ui.batch`](batch/#marimo.ui.batch "            marimo.ui.batch") | Batch operations |
| [`marimo.ui.button`](button/#marimo.ui.button "            marimo.ui.button") | Create buttons |
| [`marimo.ui.chat`](chat/#marimo.ui.chat "            marimo.ui.chat") | Create chat interfaces |
| [`marimo.ui.checkbox`](checkbox/#marimo.ui.checkbox "            marimo.ui.checkbox") | Create checkboxes |
| [`marimo.ui.code_editor`](code_editor/#marimo.ui.code_editor "            marimo.ui.code_editor") | Create code editors |
| [`marimo.ui.dataframe`](dataframe/#marimo.ui.dataframe "            marimo.ui.dataframe") | Interactive dataframes |
| [`marimo.ui.data_explorer`](data_explorer/#marimo.ui.data_explorer "            marimo.ui.data_explorer") | Explore data |
| [`marimo.ui.date`](dates/#marimo.ui.date "            marimo.ui.date") | Date picker |
| [`marimo.ui.datetime`](dates/#marimo.ui.datetime "            marimo.ui.datetime") | Date and time picker |
| [`marimo.ui.date_range`](dates/#marimo.ui.date_range "            marimo.ui.date_range") | Date range picker |
| [`marimo.ui.dictionary`](dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") | Dictionary inputs |
| [`marimo.ui.dropdown`](dropdown/#marimo.ui.dropdown "            marimo.ui.dropdown") | Create dropdowns |
| [`marimo.ui.file`](file/#marimo.ui.file "            marimo.ui.file") | File uploads |
| [`marimo.ui.file_browser`](file_browser/#marimo.ui.file_browser "            marimo.ui.file_browser") | Browse files |
| [`marimo.ui.form`](form/#marimo.ui.form "            marimo.ui.form") | Create forms |
| [`marimo.ui.microphone`](microphone/#marimo.ui.microphone "            marimo.ui.microphone") | Record audio |
| [`marimo.ui.multiselect`](multiselect/#marimo.ui.multiselect "            marimo.ui.multiselect") | Multiple selection |
| [`marimo.ui.number`](number/#marimo.ui.number "            marimo.ui.number") | Number inputs |
| [`marimo.ui.radio`](radio/#marimo.ui.radio "            marimo.ui.radio") | Radio buttons |
| [`marimo.ui.range_slider`](range_slider/#marimo.ui.range_slider "            marimo.ui.range_slider") | Range sliders |
| [`marimo.ui.refresh`](refresh/#marimo.ui.refresh "            marimo.ui.refresh") | Refresh buttons |
| [`marimo.ui.run_button`](run_button/#marimo.ui.run_button "            marimo.ui.run_button") | Run buttons |
| [`marimo.ui.slider`](slider/#marimo.ui.slider "            marimo.ui.slider") | Create sliders |
| [`marimo.ui.switch`](switch/#marimo.ui.switch "            marimo.ui.switch") | Toggle switches |
| [`marimo.ui.tabs`](tabs/#marimo.ui.tabs "            marimo.ui.tabs") | Tabbed interfaces |
| [`marimo.ui.table`](table/#marimo.ui.table "            marimo.ui.table") | Interactive tables |
| [`marimo.ui.text`](text/#marimo.ui.text "            marimo.ui.text") | Text inputs |
| [`marimo.ui.text_area`](text_area/#marimo.ui.text_area "            marimo.ui.text_area") | Multiline text inputs |

To use a UI element, assign it to a global variable and output it in a cell.
When you interact with the frontend element, the Python object's `value`
attribute is automatically updated, and all cells referencing that object
automatically run with the element's latest value.

## Integrations [¶](\#integrations "Permanent link")

| Integration | Description |
| --- | --- |
| [`marimo.ui.altair_chart`](../plotting/#marimo.ui.altair_chart "            marimo.ui.altair_chart") | Interactive Altair charts |
| [`marimo.ui.plotly`](../plotting/#marimo.ui.plotly "            marimo.ui.plotly") | Interactive Plotly charts |
| [`marimo.mpl.interactive`](../plotting/#marimo.mpl.interactive "            marimo.mpl.interactive") | Interactive Matplotlib plots |
| [`marimo.ui.anywidget`](anywidget/#marimo.ui.anywidget "            marimo.ui.anywidget") | Custom widgets |

 Back to top

# Inputs [¶](\#inputs "Permanent link")

marimo comes packaged with interactive UI elements that you can use to build
powerful notebooks and apps. These elements are available in `marimo.ui`.

| Element | Description |
| --- | --- |
| [`marimo.ui.array`](array/#marimo.ui.array "            marimo.ui.array") | Create array inputs |
| [`marimo.ui.batch`](batch/#marimo.ui.batch "            marimo.ui.batch") | Batch operations |
| [`marimo.ui.button`](button/#marimo.ui.button "            marimo.ui.button") | Create buttons |
| [`marimo.ui.chat`](chat/#marimo.ui.chat "            marimo.ui.chat") | Create chat interfaces |
| [`marimo.ui.checkbox`](checkbox/#marimo.ui.checkbox "            marimo.ui.checkbox") | Create checkboxes |
| [`marimo.ui.code_editor`](code_editor/#marimo.ui.code_editor "            marimo.ui.code_editor") | Create code editors |
| [`marimo.ui.dataframe`](dataframe/#marimo.ui.dataframe "            marimo.ui.dataframe") | Interactive dataframes |
| [`marimo.ui.data_explorer`](data_explorer/#marimo.ui.data_explorer "            marimo.ui.data_explorer") | Explore data |
| [`marimo.ui.date`](dates/#marimo.ui.date "            marimo.ui.date") | Date picker |
| [`marimo.ui.datetime`](dates/#marimo.ui.datetime "            marimo.ui.datetime") | Date and time picker |
| [`marimo.ui.date_range`](dates/#marimo.ui.date_range "            marimo.ui.date_range") | Date range picker |
| [`marimo.ui.dictionary`](dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") | Dictionary inputs |
| [`marimo.ui.dropdown`](dropdown/#marimo.ui.dropdown "            marimo.ui.dropdown") | Create dropdowns |
| [`marimo.ui.file`](file/#marimo.ui.file "            marimo.ui.file") | File uploads |
| [`marimo.ui.file_browser`](file_browser/#marimo.ui.file_browser "            marimo.ui.file_browser") | Browse files |
| [`marimo.ui.form`](form/#marimo.ui.form "            marimo.ui.form") | Create forms |
| [`marimo.ui.microphone`](microphone/#marimo.ui.microphone "            marimo.ui.microphone") | Record audio |
| [`marimo.ui.multiselect`](multiselect/#marimo.ui.multiselect "            marimo.ui.multiselect") | Multiple selection |
| [`marimo.ui.number`](number/#marimo.ui.number "            marimo.ui.number") | Number inputs |
| [`marimo.ui.radio`](radio/#marimo.ui.radio "            marimo.ui.radio") | Radio buttons |
| [`marimo.ui.range_slider`](range_slider/#marimo.ui.range_slider "            marimo.ui.range_slider") | Range sliders |
| [`marimo.ui.refresh`](refresh/#marimo.ui.refresh "            marimo.ui.refresh") | Refresh buttons |
| [`marimo.ui.run_button`](run_button/#marimo.ui.run_button "            marimo.ui.run_button") | Run buttons |
| [`marimo.ui.slider`](slider/#marimo.ui.slider "            marimo.ui.slider") | Create sliders |
| [`marimo.ui.switch`](switch/#marimo.ui.switch "            marimo.ui.switch") | Toggle switches |
| [`marimo.ui.tabs`](tabs/#marimo.ui.tabs "            marimo.ui.tabs") | Tabbed interfaces |
| [`marimo.ui.table`](table/#marimo.ui.table "            marimo.ui.table") | Interactive tables |
| [`marimo.ui.text`](text/#marimo.ui.text "            marimo.ui.text") | Text inputs |
| [`marimo.ui.text_area`](text_area/#marimo.ui.text_area "            marimo.ui.text_area") | Multiline text inputs |

To use a UI element, assign it to a global variable and output it in a cell.
When you interact with the frontend element, the Python object's `value`
attribute is automatically updated, and all cells referencing that object
automatically run with the element's latest value.

## Integrations [¶](\#integrations "Permanent link")

| Integration | Description |
| --- | --- |
| [`marimo.ui.altair_chart`](../plotting/#marimo.ui.altair_chart "            marimo.ui.altair_chart") | Interactive Altair charts |
| [`marimo.ui.plotly`](../plotting/#marimo.ui.plotly "            marimo.ui.plotly") | Interactive Plotly charts |
| [`marimo.mpl.interactive`](../plotting/#marimo.mpl.interactive "            marimo.mpl.interactive") | Interactive Matplotlib plots |
| [`marimo.ui.anywidget`](anywidget/#marimo.ui.anywidget "            marimo.ui.anywidget") | Custom widgets |


---

###### Array - marimoArray¶

*Source: [https://docs.marimo.io/api/inputs/array/](https://docs.marimo.io/api/inputs/array/)*

<!-- Source: https://docs.marimo.io/api/inputs/array/ -->
<!-- Title: Array - marimoArray¶ -->
<!-- Depth: 8 -->

[Skip to content](#array)

# Array [¶](\#array "Permanent link")

## `` marimo.ui.array [¶](\#marimo.ui.array "Permanent link")

Bases: `UIElement[dict[str, JSONType], Sequence[object]]`

An array of UI elements.

Use an array to:
\- create a dynamic number of UI elements at runtime
\- group together logically related UI elements
\- keep the number of global variables in your program small

Access the values of the elements using the `value` attribute of the array
( `array.value`).

The elements in the array can be accessed using square brackets
( `array[index]`) and embedded in other marimo outputs. You can also
iterate over the UI elements using the `in` operator ( `for element in
array`).

Note: The UI elements in the array are clones of the original elements:
interacting with the array will _not_ update the original elements, and
vice versa.

Examples:

A heterogeneous collection of UI elements:

Get the values of the `slider`, `text`, and `date` elements via
`array.value`:

Access and output a UI element in the array:

Some number of UI elements, determined at runtime:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A list containing the values of the array's entries.<br>**TYPE:** `list` |
| `elements` | A list of the wrapped elements (clones of the originals).<br>**TYPE:** `list` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | The UI elements to include.<br>**TYPE:** `Sequence[UIElement]` |
| `label` | A descriptive name for the array. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[Sequence[object]], None]]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.array.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.array.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.array.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.array.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.array.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.array.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.array.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.array.from_args "Permanent link")

### `` hstack [¶](\#marimo.ui.array.hstack "Permanent link")

Stack the elements horizontally.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Additional arguments passed to `marimo.hstack`.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML representation of horizontally stacked elements.<br>**TYPE:** `Html` |

### `` left [¶](\#marimo.ui.array.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.array.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.array.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` vstack [¶](\#marimo.ui.array.vstack "Permanent link")

Stack the elements vertically.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Additional arguments passed to `marimo.vstack`.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML representation of vertically stacked elements.<br>**TYPE:** `Html` |

Back to top

# Array [¶](\#array "Permanent link")

## `` marimo.ui.array [¶](\#marimo.ui.array "Permanent link")

Bases: `UIElement[dict[str, JSONType], Sequence[object]]`

An array of UI elements.

Use an array to:
\- create a dynamic number of UI elements at runtime
\- group together logically related UI elements
\- keep the number of global variables in your program small

Access the values of the elements using the `value` attribute of the array
( `array.value`).

The elements in the array can be accessed using square brackets
( `array[index]`) and embedded in other marimo outputs. You can also
iterate over the UI elements using the `in` operator ( `for element in
array`).

Note: The UI elements in the array are clones of the original elements:
interacting with the array will _not_ update the original elements, and
vice versa.

Examples:

A heterogeneous collection of UI elements:

Get the values of the `slider`, `text`, and `date` elements via
`array.value`:

Access and output a UI element in the array:

Some number of UI elements, determined at runtime:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A list containing the values of the array's entries.<br>**TYPE:** `list` |
| `elements` | A list of the wrapped elements (clones of the originals).<br>**TYPE:** `list` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | The UI elements to include.<br>**TYPE:** `Sequence[UIElement]` |
| `label` | A descriptive name for the array. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[Sequence[object]], None]]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.array.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.array.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.array.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.array.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.array.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.array.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.array.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.array.from_args "Permanent link")

### `` hstack [¶](\#marimo.ui.array.hstack "Permanent link")

Stack the elements horizontally.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Additional arguments passed to `marimo.hstack`.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML representation of horizontally stacked elements.<br>**TYPE:** `Html` |

### `` left [¶](\#marimo.ui.array.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.array.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.array.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` vstack [¶](\#marimo.ui.array.vstack "Permanent link")

Stack the elements vertically.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Additional arguments passed to `marimo.vstack`.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML representation of vertically stacked elements.<br>**TYPE:** `Html` |

 Back to top

# Array [¶](\#array "Permanent link")

## `` marimo.ui.array [¶](\#marimo.ui.array "Permanent link")

Bases: `UIElement[dict[str, JSONType], Sequence[object]]`

An array of UI elements.

Use an array to:
\- create a dynamic number of UI elements at runtime
\- group together logically related UI elements
\- keep the number of global variables in your program small

Access the values of the elements using the `value` attribute of the array
( `array.value`).

The elements in the array can be accessed using square brackets
( `array[index]`) and embedded in other marimo outputs. You can also
iterate over the UI elements using the `in` operator ( `for element in
array`).

Note: The UI elements in the array are clones of the original elements:
interacting with the array will _not_ update the original elements, and
vice versa.

Examples:

A heterogeneous collection of UI elements:

Get the values of the `slider`, `text`, and `date` elements via
`array.value`:

Access and output a UI element in the array:

Some number of UI elements, determined at runtime:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A list containing the values of the array's entries.<br>**TYPE:** `list` |
| `elements` | A list of the wrapped elements (clones of the originals).<br>**TYPE:** `list` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | The UI elements to include.<br>**TYPE:** `Sequence[UIElement]` |
| `label` | A descriptive name for the array. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[Sequence[object]], None]]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.array.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.array.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.array.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.array.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.array.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.array.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.array.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.array.from_args "Permanent link")

### `` hstack [¶](\#marimo.ui.array.hstack "Permanent link")

Stack the elements horizontally.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Additional arguments passed to `marimo.hstack`.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML representation of horizontally stacked elements.<br>**TYPE:** `Html` |

### `` left [¶](\#marimo.ui.array.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.array.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.array.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` vstack [¶](\#marimo.ui.array.vstack "Permanent link")

Stack the elements vertically.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Additional arguments passed to `marimo.vstack`.<br>**TYPE:** `Any`**DEFAULT:** `{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | HTML representation of vertically stacked elements.<br>**TYPE:** `Html` |


---

###### Batch - marimoBatch¶

*Source: [https://docs.marimo.io/api/inputs/batch/](https://docs.marimo.io/api/inputs/batch/)*

<!-- Source: https://docs.marimo.io/api/inputs/batch/ -->
<!-- Title: Batch - marimoBatch¶ -->
<!-- Depth: 8 -->

[Skip to content](#batch)

# Batch [¶](\#batch "Permanent link")

## `` marimo.ui.batch [¶](\#marimo.ui.batch "Permanent link")

Bases: `_batch_base`

Convert an HTML object with templated text into a UI element.

A `batch` is a UI element that wraps other UI elements, and is
represented by custom HTML or markdown. You can create
a `batch` by calling the `batch()` method on `Html` objects.

Get the value of the wrapped UI elements using the `value` attribute
of the batch.

Examples:

In the below example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and `'birthday'`
(and values equal to the values of their corresponding elements).

To get the value of `name` and `birthday`, use:

You can also instantiate this class directly:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict of the batched elements' values.<br>**TYPE:** `dict` |
| `elements` | A dict of the batched elements (clones of the originals).<br>**TYPE:** `dict` |
| `on_change` | Optional callback to run when this element's value changes.<br>**TYPE:** `Optional[Callable]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html` | A templated Html object.<br>**TYPE:** `Html` |
| `elements` | The UI elements to interpolate into the HTML template.<br>**TYPE:** `dict[str, UIElement]` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[Dict[str, object]], None]]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.batch.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.batch.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.batch.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.batch.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.batch.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.batch.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.batch.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.batch.from_args "Permanent link")

### `` get [¶](\#marimo.ui.batch.get "Permanent link")

Get a UI element by key with an optional default value.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `key` | The key to look up in the batch.<br>**TYPE:** `str` |
| `default` | Value to return if key is not found. Defaults to None.<br>**TYPE:** `Any | None`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The UI element if found, otherwise the default value.<br>**TYPE:** `Any` |

### `` items [¶](\#marimo.ui.batch.items "Permanent link")

Return a view of the batch's items (key-value pairs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ItemsView[str, UIElement[JSONType, object]]` | ItemsView\[str, UIElement\]: A view of the batch's (key, element) pairs. |

### `` left [¶](\#marimo.ui.batch.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.batch.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.batch.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` values [¶](\#marimo.ui.batch.values "Permanent link")

Return a view of the batch's values (UI elements).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ValuesView[UIElement[JSONType, object]]` | ValuesView\[UIElement\]: A view of the batch's UI elements. |

Back to top

# Batch [¶](\#batch "Permanent link")

## `` marimo.ui.batch [¶](\#marimo.ui.batch "Permanent link")

Bases: `_batch_base`

Convert an HTML object with templated text into a UI element.

A `batch` is a UI element that wraps other UI elements, and is
represented by custom HTML or markdown. You can create
a `batch` by calling the `batch()` method on `Html` objects.

Get the value of the wrapped UI elements using the `value` attribute
of the batch.

Examples:

In the below example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and `'birthday'`
(and values equal to the values of their corresponding elements).

To get the value of `name` and `birthday`, use:

You can also instantiate this class directly:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict of the batched elements' values.<br>**TYPE:** `dict` |
| `elements` | A dict of the batched elements (clones of the originals).<br>**TYPE:** `dict` |
| `on_change` | Optional callback to run when this element's value changes.<br>**TYPE:** `Optional[Callable]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html` | A templated Html object.<br>**TYPE:** `Html` |
| `elements` | The UI elements to interpolate into the HTML template.<br>**TYPE:** `dict[str, UIElement]` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[Dict[str, object]], None]]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.batch.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.batch.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.batch.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.batch.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.batch.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.batch.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.batch.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.batch.from_args "Permanent link")

### `` get [¶](\#marimo.ui.batch.get "Permanent link")

Get a UI element by key with an optional default value.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `key` | The key to look up in the batch.<br>**TYPE:** `str` |
| `default` | Value to return if key is not found. Defaults to None.<br>**TYPE:** `Any | None`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The UI element if found, otherwise the default value.<br>**TYPE:** `Any` |

### `` items [¶](\#marimo.ui.batch.items "Permanent link")

Return a view of the batch's items (key-value pairs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ItemsView[str, UIElement[JSONType, object]]` | ItemsView\[str, UIElement\]: A view of the batch's (key, element) pairs. |

### `` left [¶](\#marimo.ui.batch.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.batch.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.batch.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` values [¶](\#marimo.ui.batch.values "Permanent link")

Return a view of the batch's values (UI elements).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ValuesView[UIElement[JSONType, object]]` | ValuesView\[UIElement\]: A view of the batch's UI elements. |

 Back to top

# Batch [¶](\#batch "Permanent link")

## `` marimo.ui.batch [¶](\#marimo.ui.batch "Permanent link")

Bases: `_batch_base`

Convert an HTML object with templated text into a UI element.

A `batch` is a UI element that wraps other UI elements, and is
represented by custom HTML or markdown. You can create
a `batch` by calling the `batch()` method on `Html` objects.

Get the value of the wrapped UI elements using the `value` attribute
of the batch.

Examples:

In the below example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and `'birthday'`
(and values equal to the values of their corresponding elements).

To get the value of `name` and `birthday`, use:

You can also instantiate this class directly:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict of the batched elements' values.<br>**TYPE:** `dict` |
| `elements` | A dict of the batched elements (clones of the originals).<br>**TYPE:** `dict` |
| `on_change` | Optional callback to run when this element's value changes.<br>**TYPE:** `Optional[Callable]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html` | A templated Html object.<br>**TYPE:** `Html` |
| `elements` | The UI elements to interpolate into the HTML template.<br>**TYPE:** `dict[str, UIElement]` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[Dict[str, object]], None]]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.batch.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.batch.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.batch.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.batch.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.batch.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.batch.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.batch.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.batch.from_args "Permanent link")

### `` get [¶](\#marimo.ui.batch.get "Permanent link")

Get a UI element by key with an optional default value.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `key` | The key to look up in the batch.<br>**TYPE:** `str` |
| `default` | Value to return if key is not found. Defaults to None.<br>**TYPE:** `Any | None`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The UI element if found, otherwise the default value.<br>**TYPE:** `Any` |

### `` items [¶](\#marimo.ui.batch.items "Permanent link")

Return a view of the batch's items (key-value pairs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ItemsView[str, UIElement[JSONType, object]]` | ItemsView\[str, UIElement\]: A view of the batch's (key, element) pairs. |

### `` left [¶](\#marimo.ui.batch.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.batch.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.batch.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` values [¶](\#marimo.ui.batch.values "Permanent link")

Return a view of the batch's values (UI elements).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ValuesView[UIElement[JSONType, object]]` | ValuesView\[UIElement\]: A view of the batch's UI elements. |


---

###### Dictionary - marimoDictionary¶

*Source: [https://docs.marimo.io/api/inputs/dictionary/](https://docs.marimo.io/api/inputs/dictionary/)*

<!-- Source: https://docs.marimo.io/api/inputs/dictionary/ -->
<!-- Title: Dictionary - marimoDictionary¶ -->
<!-- Depth: 8 -->

[Skip to content](#dictionary)

# Dictionary [¶](\#dictionary "Permanent link")

## `` marimo.ui.dictionary [¶](\#marimo.ui.dictionary "Permanent link")

Bases: `_batch_base`

A dictionary of UI elements.

Use a dictionary to:
\- create a set of UI elements at runtime
\- group together logically related UI elements
\- keep the number of global variables in your program small

Access the values of the elements using the `value` attribute of the
dictionary. The elements in the dictionary can be accessed using square brackets
( `dictionary[key]`) and embedded in other marimo outputs. You can also
iterate over the UI elements using the same syntax used for Python dicts.

Note

The UI elements in the dictionary are clones of the original
elements: interacting with the dictionary will _not_ update the original
elements, and vice versa.

The main reason to use `mo.ui.dictionary` is for reactive execution — when you
interact with an element in a `mo.ui.dictionary`, all cells that reference the
`mo.ui.dictionary` run automatically, just like all other ui elements. When you
use a regular dictionary, you don't get this reactivity.

Examples:

A heterogeneous collection of UI elements:

Get the values of the `slider`, `text`, and `date` elements via
`d.value`:

Access and output a UI element in the array:

Some number of UI elements, determined at runtime:

Quick layouts of UI elements:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict holding the values of the UI elements, keyed by<br>their names.<br>**TYPE:** `dict` |
| `elements` | A dict of the wrapped elements (clones of the originals).<br>**TYPE:** `dict` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[dict[str, object]], None]]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | A dict mapping names to UI<br>elements to include.<br>**TYPE:** `dict[str, UIElement[Any, Any]]` |
| `label` | A descriptive name for the dictionary. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Callable[[dict[str, object]], None]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.dictionary.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.dictionary.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.dictionary.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.dictionary.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.dictionary.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.dictionary.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.dictionary.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.dictionary.from_args "Permanent link")

### `` get [¶](\#marimo.ui.dictionary.get "Permanent link")

Get a UI element by key with an optional default value.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `key` | The key to look up in the batch.<br>**TYPE:** `str` |
| `default` | Value to return if key is not found. Defaults to None.<br>**TYPE:** `Any | None`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The UI element if found, otherwise the default value.<br>**TYPE:** `Any` |

### `` hstack [¶](\#marimo.ui.dictionary.hstack "Permanent link")

Stack the elements horizontally.

For kwargs, see `marimo.hstack`.

### `` items [¶](\#marimo.ui.dictionary.items "Permanent link")

Return a view of the batch's items (key-value pairs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ItemsView[str, UIElement[JSONType, object]]` | ItemsView\[str, UIElement\]: A view of the batch's (key, element) pairs. |

### `` left [¶](\#marimo.ui.dictionary.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.dictionary.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.dictionary.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` values [¶](\#marimo.ui.dictionary.values "Permanent link")

Return a view of the batch's values (UI elements).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ValuesView[UIElement[JSONType, object]]` | ValuesView\[UIElement\]: A view of the batch's UI elements. |

### `` vstack [¶](\#marimo.ui.dictionary.vstack "Permanent link")

Stack the elements vertically.

For kwargs, see `marimo.vstack`.

Back to top

# Dictionary [¶](\#dictionary "Permanent link")

## `` marimo.ui.dictionary [¶](\#marimo.ui.dictionary "Permanent link")

Bases: `_batch_base`

A dictionary of UI elements.

Use a dictionary to:
\- create a set of UI elements at runtime
\- group together logically related UI elements
\- keep the number of global variables in your program small

Access the values of the elements using the `value` attribute of the
dictionary. The elements in the dictionary can be accessed using square brackets
( `dictionary[key]`) and embedded in other marimo outputs. You can also
iterate over the UI elements using the same syntax used for Python dicts.

Note

The UI elements in the dictionary are clones of the original
elements: interacting with the dictionary will _not_ update the original
elements, and vice versa.

The main reason to use `mo.ui.dictionary` is for reactive execution — when you
interact with an element in a `mo.ui.dictionary`, all cells that reference the
`mo.ui.dictionary` run automatically, just like all other ui elements. When you
use a regular dictionary, you don't get this reactivity.

Examples:

A heterogeneous collection of UI elements:

Get the values of the `slider`, `text`, and `date` elements via
`d.value`:

Access and output a UI element in the array:

Some number of UI elements, determined at runtime:

Quick layouts of UI elements:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict holding the values of the UI elements, keyed by<br>their names.<br>**TYPE:** `dict` |
| `elements` | A dict of the wrapped elements (clones of the originals).<br>**TYPE:** `dict` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[dict[str, object]], None]]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | A dict mapping names to UI<br>elements to include.<br>**TYPE:** `dict[str, UIElement[Any, Any]]` |
| `label` | A descriptive name for the dictionary. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Callable[[dict[str, object]], None]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.dictionary.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.dictionary.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.dictionary.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.dictionary.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.dictionary.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.dictionary.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.dictionary.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.dictionary.from_args "Permanent link")

### `` get [¶](\#marimo.ui.dictionary.get "Permanent link")

Get a UI element by key with an optional default value.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `key` | The key to look up in the batch.<br>**TYPE:** `str` |
| `default` | Value to return if key is not found. Defaults to None.<br>**TYPE:** `Any | None`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The UI element if found, otherwise the default value.<br>**TYPE:** `Any` |

### `` hstack [¶](\#marimo.ui.dictionary.hstack "Permanent link")

Stack the elements horizontally.

For kwargs, see `marimo.hstack`.

### `` items [¶](\#marimo.ui.dictionary.items "Permanent link")

Return a view of the batch's items (key-value pairs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ItemsView[str, UIElement[JSONType, object]]` | ItemsView\[str, UIElement\]: A view of the batch's (key, element) pairs. |

### `` left [¶](\#marimo.ui.dictionary.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.dictionary.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.dictionary.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` values [¶](\#marimo.ui.dictionary.values "Permanent link")

Return a view of the batch's values (UI elements).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ValuesView[UIElement[JSONType, object]]` | ValuesView\[UIElement\]: A view of the batch's UI elements. |

### `` vstack [¶](\#marimo.ui.dictionary.vstack "Permanent link")

Stack the elements vertically.

For kwargs, see `marimo.vstack`.

 Back to top

# Dictionary [¶](\#dictionary "Permanent link")

## `` marimo.ui.dictionary [¶](\#marimo.ui.dictionary "Permanent link")

Bases: `_batch_base`

A dictionary of UI elements.

Use a dictionary to:
\- create a set of UI elements at runtime
\- group together logically related UI elements
\- keep the number of global variables in your program small

Access the values of the elements using the `value` attribute of the
dictionary. The elements in the dictionary can be accessed using square brackets
( `dictionary[key]`) and embedded in other marimo outputs. You can also
iterate over the UI elements using the same syntax used for Python dicts.

Note

The UI elements in the dictionary are clones of the original
elements: interacting with the dictionary will _not_ update the original
elements, and vice versa.

The main reason to use `mo.ui.dictionary` is for reactive execution — when you
interact with an element in a `mo.ui.dictionary`, all cells that reference the
`mo.ui.dictionary` run automatically, just like all other ui elements. When you
use a regular dictionary, you don't get this reactivity.

Examples:

A heterogeneous collection of UI elements:

Get the values of the `slider`, `text`, and `date` elements via
`d.value`:

Access and output a UI element in the array:

Some number of UI elements, determined at runtime:

Quick layouts of UI elements:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict holding the values of the UI elements, keyed by<br>their names.<br>**TYPE:** `dict` |
| `elements` | A dict of the wrapped elements (clones of the originals).<br>**TYPE:** `dict` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Optional[Callable[[dict[str, object]], None]]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | A dict mapping names to UI<br>elements to include.<br>**TYPE:** `dict[str, UIElement[Any, Any]]` |
| `label` | A descriptive name for the dictionary. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback<br>to run when this element's value changes.<br>**TYPE:** `Callable[[dict[str, object]], None]`**DEFAULT:** `None` |

### `` elements`property`[¶](\#marimo.ui.dictionary.elements "Permanent link")

### `` text`property`[¶](\#marimo.ui.dictionary.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.dictionary.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.dictionary.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.dictionary.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.dictionary.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.dictionary.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.dictionary.from_args "Permanent link")

### `` get [¶](\#marimo.ui.dictionary.get "Permanent link")

Get a UI element by key with an optional default value.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `key` | The key to look up in the batch.<br>**TYPE:** `str` |
| `default` | Value to return if key is not found. Defaults to None.<br>**TYPE:** `Any | None`**DEFAULT:** `None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The UI element if found, otherwise the default value.<br>**TYPE:** `Any` |

### `` hstack [¶](\#marimo.ui.dictionary.hstack "Permanent link")

Stack the elements horizontally.

For kwargs, see `marimo.hstack`.

### `` items [¶](\#marimo.ui.dictionary.items "Permanent link")

Return a view of the batch's items (key-value pairs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ItemsView[str, UIElement[JSONType, object]]` | ItemsView\[str, UIElement\]: A view of the batch's (key, element) pairs. |

### `` left [¶](\#marimo.ui.dictionary.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.dictionary.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.dictionary.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### `` values [¶](\#marimo.ui.dictionary.values "Permanent link")

Return a view of the batch's values (UI elements).

| RETURNS | DESCRIPTION |
| --- | --- |
| `ValuesView[UIElement[JSONType, object]]` | ValuesView\[UIElement\]: A view of the batch's UI elements. |

### `` vstack [¶](\#marimo.ui.dictionary.vstack "Permanent link")

Stack the elements vertically.

For kwargs, see `marimo.vstack`.


---

###### Form - marimoForm¶

*Source: [https://docs.marimo.io/api/inputs/form/](https://docs.marimo.io/api/inputs/form/)*

<!-- Source: https://docs.marimo.io/api/inputs/form/ -->
<!-- Title: Form - marimoForm¶ -->
<!-- Depth: 8 -->

[Skip to content](#form)

# Form [¶](\#form "Permanent link")

## `` marimo.ui.form [¶](\#marimo.ui.form "Permanent link")

Bases: `UIElement[Optional[S], Optional[T]]`

A submittable form linked to a UIElement.

Use a `form` to prevent sending UI element values to Python until a button
is clicked.

The value of a `form` is the value of the underlying element the last time
the form was submitted.

Examples:

Create a form with chaining:

Create a form with multiple elements:

Instantiate a form directly:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The value of the wrapped element when the form's submit<br>button was last clicked.<br>**TYPE:** `Any` |
| `element` | A copy of the wrapped element.<br>**TYPE:** `UIElement` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `element` | The element to wrap.<br>**TYPE:** `UIElement[S, T]` |
| `bordered` | Whether the form should have a border.<br>Defaults to True.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>Defaults to "Submit".<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should<br>be disabled. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its<br>contents after submitting. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear<br>button. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>Defaults to "Clear".<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `validate` | A<br>function that takes the form's value and returns an error message if<br>the value is invalid, or None if the value is valid. Defaults to None.<br>**TYPE:** `Callable[[Optional[JSONType]], Optional[str]]`**DEFAULT:** `None` |
| `label` | Markdown label for the form. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to<br>run when this element's value changes. Defaults to None.<br>**TYPE:** `Callable[[Optional[T]], None]`**DEFAULT:** `None` |

### `` element`instance-attribute`[¶](\#marimo.ui.form.element "Permanent link")

### `` text`property`[¶](\#marimo.ui.form.text "Permanent link")

A string of HTML representing this element.

### `` validate`instance-attribute`[¶](\#marimo.ui.form.validate "Permanent link")

### `` value`property``writable`[¶](\#marimo.ui.form.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.form.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.form.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.form.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.form.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.form.from_args "Permanent link")

### `` left [¶](\#marimo.ui.form.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.form.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.form.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

Back to top

# Form [¶](\#form "Permanent link")

## `` marimo.ui.form [¶](\#marimo.ui.form "Permanent link")

Bases: `UIElement[Optional[S], Optional[T]]`

A submittable form linked to a UIElement.

Use a `form` to prevent sending UI element values to Python until a button
is clicked.

The value of a `form` is the value of the underlying element the last time
the form was submitted.

Examples:

Create a form with chaining:

Create a form with multiple elements:

Instantiate a form directly:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The value of the wrapped element when the form's submit<br>button was last clicked.<br>**TYPE:** `Any` |
| `element` | A copy of the wrapped element.<br>**TYPE:** `UIElement` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `element` | The element to wrap.<br>**TYPE:** `UIElement[S, T]` |
| `bordered` | Whether the form should have a border.<br>Defaults to True.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>Defaults to "Submit".<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should<br>be disabled. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its<br>contents after submitting. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear<br>button. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>Defaults to "Clear".<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `validate` | A<br>function that takes the form's value and returns an error message if<br>the value is invalid, or None if the value is valid. Defaults to None.<br>**TYPE:** `Callable[[Optional[JSONType]], Optional[str]]`**DEFAULT:** `None` |
| `label` | Markdown label for the form. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to<br>run when this element's value changes. Defaults to None.<br>**TYPE:** `Callable[[Optional[T]], None]`**DEFAULT:** `None` |

### `` element`instance-attribute`[¶](\#marimo.ui.form.element "Permanent link")

### `` text`property`[¶](\#marimo.ui.form.text "Permanent link")

A string of HTML representing this element.

### `` validate`instance-attribute`[¶](\#marimo.ui.form.validate "Permanent link")

### `` value`property``writable`[¶](\#marimo.ui.form.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.form.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.form.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.form.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.form.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.form.from_args "Permanent link")

### `` left [¶](\#marimo.ui.form.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.form.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.form.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

 Back to top

# Form [¶](\#form "Permanent link")

## `` marimo.ui.form [¶](\#marimo.ui.form "Permanent link")

Bases: `UIElement[Optional[S], Optional[T]]`

A submittable form linked to a UIElement.

Use a `form` to prevent sending UI element values to Python until a button
is clicked.

The value of a `form` is the value of the underlying element the last time
the form was submitted.

Examples:

Create a form with chaining:

Create a form with multiple elements:

Instantiate a form directly:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | The value of the wrapped element when the form's submit<br>button was last clicked.<br>**TYPE:** `Any` |
| `element` | A copy of the wrapped element.<br>**TYPE:** `UIElement` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `element` | The element to wrap.<br>**TYPE:** `UIElement[S, T]` |
| `bordered` | Whether the form should have a border.<br>Defaults to True.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>Defaults to "Submit".<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should<br>be disabled. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its<br>contents after submitting. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear<br>button. Defaults to False.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>Defaults to "Clear".<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>Defaults to None.<br>**TYPE:** `str`**DEFAULT:** `None` |
| `validate` | A<br>function that takes the form's value and returns an error message if<br>the value is invalid, or None if the value is valid. Defaults to None.<br>**TYPE:** `Callable[[Optional[JSONType]], Optional[str]]`**DEFAULT:** `None` |
| `label` | Markdown label for the form. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional callback to<br>run when this element's value changes. Defaults to None.<br>**TYPE:** `Callable[[Optional[T]], None]`**DEFAULT:** `None` |

### `` element`instance-attribute`[¶](\#marimo.ui.form.element "Permanent link")

### `` text`property`[¶](\#marimo.ui.form.text "Permanent link")

A string of HTML representing this element.

### `` validate`instance-attribute`[¶](\#marimo.ui.form.validate "Permanent link")

### `` value`property``writable`[¶](\#marimo.ui.form.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.form.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.form.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.form.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.form.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.form.from_args "Permanent link")

### `` left [¶](\#marimo.ui.form.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.form.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.form.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |


---

###### Plotting - marimoPlotting¶

*Source: [https://docs.marimo.io/api/plotting/](https://docs.marimo.io/api/plotting/)*

<!-- Source: https://docs.marimo.io/api/plotting/ -->
<!-- Title: Plotting - marimoPlotting¶ -->
<!-- Depth: 8 -->

[Skip to content](#plotting)

# Plotting [¶](\#plotting "Permanent link")

marimo supports most major plotting libraries, including Matplotlib, Seaborn,
Plotly, and Altair. Just import your plotting library of choice and use it
as you normally would.

For more information about plotting, see the [plotting guide](../../guides/working_with_data/plotting/).

## Reactive charts with Altair [¶](\#reactive-charts-with-altair "Permanent link")

### Disabling automatic selection [¶](\#disabling-automatic-selection "Permanent link")

marimo automatically adds a default selection based on the mark type, however, you may want to customize the selection behavior of your Altair chart. You can do this by setting `chart_selection` and `legend_selection` to `False`, and using `.add_params` directly on your Altair chart.

## `` marimo.ui.altair\_chart [¶](\#marimo.ui.altair_chart "Permanent link")

Bases: `UIElement[ChartSelection, ChartDataType]`

Make reactive charts with Altair.

Use `mo.ui.altair_chart` to make Altair charts reactive: select chart data
with your cursor on the frontend, get them as a dataframe in Python!

Supports polars, pandas, and arrow DataFrames.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dataframe of the plot data filtered by the selections.<br>**TYPE:** `ChartDataType` |
| `dataframe` | A dataframe of the unfiltered chart data.<br>**TYPE:** `ChartDataType` |
| `selections` | The selection of the chart; this may be an interval<br>along the name of an axis or a selection of points.<br>**TYPE:** `ChartSelection` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `chart` | An Altair Chart object.<br>**TYPE:** `Chart` |
| `chart_selection` | Selection type, "point", "interval", or a bool. Defaults to True which will<br>automatically detect the best selection type. This is ignored if the chart<br>already has a point/interval selection param.<br>**TYPE:** `Union[Literal['point'], Literal['interval'], bool]`**DEFAULT:** `True` |
| `legend_selection` | List of legend fields<br>(columns) for which to enable selection, True to enable selection for all<br>fields, or False to disable selection entirely. This is ignored if the chart<br>already has a legend selection param. Defaults to True.<br>**TYPE:** `Union[list[str], bool]`**DEFAULT:** `True` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional<br>callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[ChartDataType], None]]`**DEFAULT:** `None` |

### `` dataframe`instance-attribute`[¶](\#marimo.ui.altair_chart.dataframe "Permanent link")

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.altair_chart.name "Permanent link")

### `` selections`property`[¶](\#marimo.ui.altair_chart.selections "Permanent link")

### `` text`property`[¶](\#marimo.ui.altair_chart.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.altair_chart.value "Permanent link")

### `` apply\_selection [¶](\#marimo.ui.altair_chart.apply_selection "Permanent link")

Apply the selection to a DataFrame.

This method is useful when you have a layered chart and you want to
apply the selection to a DataFrame.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `df` | A DataFrame to apply the selection to.<br>**TYPE:** `ChartDataType` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `ChartDataType` | A DataFrame of the plot data filtered by the selections.<br>**TYPE:** `ChartDataType` |

Examples:

### `` batch [¶](\#marimo.ui.altair_chart.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.altair_chart.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.altair_chart.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.altair_chart.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.altair_chart.from_args "Permanent link")

### `` left [¶](\#marimo.ui.altair_chart.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.altair_chart.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.altair_chart.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### Performance and Data Transformers [¶](\#performance-and-data-transformers "Permanent link")

Altair has a concept of [data](https://altair-viz.github.io/user_guide/data_transformers.html) transformers, which can be used to improve performance.

Such examples are:

- pandas Dataframe has to be sanitized and serialized to JSON.
- The rows of a Dataframe might need to be sampled or limited to a maximum number.
- The Dataframe might be written to a `.csv` or `.json` file for performance reasons.

By default, Altair uses the `default` data transformer, which is the slowest in marimo. It is limited to 5000 rows (although we increase this to `20_000` rows as marimo can handle this). This includes the data inside the HTML that is being sent over the network, which can also be limited by marimo's maximum message size.

It is recommended to use the `marimo_csv` data transformer, which is the most performant and can handle the largest datasets: it converts the data to a CSV file which is smaller and can be sent over the network. This can handle up to +400,000 rows with no issues.

When using `mo.ui.altair_chart`, we automatically set the data transformer to `marimo_csv` for you. If you are using Altair directly, you can set the data transformer using the following code:

## Reactive plots with Plotly [¶](\#reactive-plots-with-plotly "Permanent link")

mo.ui.plotly only supports scatter plots, treemaps charts, and sunbursts charts.

marimo can render any Plotly plot, but [`mo.ui.plotly`](#marimo.ui.plotly "            marimo.ui.plotly") only
supports reactive selections for scatter plots, treemaps charts, and sunbursts charts. If you require other kinds of
selection, consider using [`mo.ui.altair_chart`](#marimo.ui.altair_chart "            marimo.ui.altair_chart").

## `` marimo.ui.plotly [¶](\#marimo.ui.plotly "Permanent link")

Bases: `UIElement[PlotlySelection, list[dict[str, Any]]]`

Make reactive plots with Plotly.

Use `mo.ui.plotly` to make plotly plots reactive: select data with your
cursor on the frontend, get them as a list of dicts in Python!

This function currently only supports scatter plots, treemaps charts,
and sunbursts charts.

Examples:

Or with custom configuration:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict of the plot data.<br>**TYPE:** `Dict[str, Any]` |
| `ranges` | The selection of the plot; this may be an<br>interval along the name of an axis.<br>**TYPE:** `Dict[str, List[float]]` |
| `points` | The selected points data.<br>**TYPE:** `List[Dict[str, Any]]` |
| `indices` | The indices of selected points.<br>**TYPE:** `List[int]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `figure` | A plotly Figure object.<br>**TYPE:** `Figure` |
| `config` | Configuration for the plot.<br>This is a dictionary that is passed directly to plotly.<br>See the plotly documentation for more information:<br>[https://plotly.com/javascript/configuration-options/](https://plotly.com/javascript/configuration-options/)<br>This takes precedence over the default configuration of the renderer.<br>Defaults to None.<br>**TYPE:** `Optional[Dict[str, Any]]`**DEFAULT:** `None` |
| `renderer_name` | Renderer to use for the plot.<br>If this is not provided, the default renderer ( `pio.renderers.default`)<br>is used. Defaults to None.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Callback to run<br>when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[JSONType], None]]`**DEFAULT:** `None` |

### `` indices`property`[¶](\#marimo.ui.plotly.indices "Permanent link")

Get the indices of selected points in the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[int]` | List\[int\]: A list of indices corresponding to the selected points.<br>Returns an empty list if no points are selected. |

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.plotly.name "Permanent link")

### `` points`property`[¶](\#marimo.ui.plotly.points "Permanent link")

Get the selected points data from the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[dict[str, Any]]` | List\[Dict\[str, Any\]\]: A list of dictionaries containing the data for<br>each selected point. Returns an empty list if no points are selected. |

### `` ranges`property`[¶](\#marimo.ui.plotly.ranges "Permanent link")

Get the range selection of the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, list[float]]` | Dict\[str, List\[float\]\]: A dictionary mapping field names to their<br>selected ranges, where each range is a list of \[min, max\] values.<br>Returns an empty dict if no range selection exists. |

### `` text`property`[¶](\#marimo.ui.plotly.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.plotly.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.plotly.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.plotly.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.plotly.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.plotly.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.plotly.from_args "Permanent link")

### `` left [¶](\#marimo.ui.plotly.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.plotly.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.plotly.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

## Interactive matplotlib [¶](\#interactive-matplotlib "Permanent link")

## `` marimo.mpl.interactive [¶](\#marimo.mpl.interactive "Permanent link")

Render a matplotlib figure using an interactive viewer.

The interactive viewer allows you to pan, zoom, and see plot coordinates
on mouse hover.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `figure` | A matplotlib `Figure` or `Axes` object.<br>**TYPE:** `matplotlib Figure or Axes` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An interactive matplotlib figure as an `Html` object.<br>**TYPE:** `Html` |

Source code in `marimo/_plugins/stateless/mpl/_mpl.py`

## Leafmap support [¶](\#leafmap-support "Permanent link")

marimo supports rendering [Leafmap](https://leafmap.org/) maps using the `folium` and `plotly` backends.

## Other plotting libraries [¶](\#other-plotting-libraries "Permanent link")

You can use all the popular plotting libraries with marimo. Such as:

- [Matplotlib](https://matplotlib.org/)
- [Plotly](https://plotly.com/)
- [Seaborn](https://seaborn.pydata.org/)
- [Bokeh](https://bokeh.org/)
- [Altair](https://altair-viz.github.io/)
- [HoloViews](http://holoviews.org/)
- [hvPlot](https://hvplot.holoviz.org/)
- [Leafmap](https://leafmap.org/)
- [Pygwalker](https://kanaries.net/pygwalker)

Back to top

# Plotting [¶](\#plotting "Permanent link")

marimo supports most major plotting libraries, including Matplotlib, Seaborn,
Plotly, and Altair. Just import your plotting library of choice and use it
as you normally would.

For more information about plotting, see the [plotting guide](../../guides/working_with_data/plotting/).

## Reactive charts with Altair [¶](\#reactive-charts-with-altair "Permanent link")

### Disabling automatic selection [¶](\#disabling-automatic-selection "Permanent link")

marimo automatically adds a default selection based on the mark type, however, you may want to customize the selection behavior of your Altair chart. You can do this by setting `chart_selection` and `legend_selection` to `False`, and using `.add_params` directly on your Altair chart.

## `` marimo.ui.altair\_chart [¶](\#marimo.ui.altair_chart "Permanent link")

Bases: `UIElement[ChartSelection, ChartDataType]`

Make reactive charts with Altair.

Use `mo.ui.altair_chart` to make Altair charts reactive: select chart data
with your cursor on the frontend, get them as a dataframe in Python!

Supports polars, pandas, and arrow DataFrames.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dataframe of the plot data filtered by the selections.<br>**TYPE:** `ChartDataType` |
| `dataframe` | A dataframe of the unfiltered chart data.<br>**TYPE:** `ChartDataType` |
| `selections` | The selection of the chart; this may be an interval<br>along the name of an axis or a selection of points.<br>**TYPE:** `ChartSelection` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `chart` | An Altair Chart object.<br>**TYPE:** `Chart` |
| `chart_selection` | Selection type, "point", "interval", or a bool. Defaults to True which will<br>automatically detect the best selection type. This is ignored if the chart<br>already has a point/interval selection param.<br>**TYPE:** `Union[Literal['point'], Literal['interval'], bool]`**DEFAULT:** `True` |
| `legend_selection` | List of legend fields<br>(columns) for which to enable selection, True to enable selection for all<br>fields, or False to disable selection entirely. This is ignored if the chart<br>already has a legend selection param. Defaults to True.<br>**TYPE:** `Union[list[str], bool]`**DEFAULT:** `True` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional<br>callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[ChartDataType], None]]`**DEFAULT:** `None` |

### `` dataframe`instance-attribute`[¶](\#marimo.ui.altair_chart.dataframe "Permanent link")

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.altair_chart.name "Permanent link")

### `` selections`property`[¶](\#marimo.ui.altair_chart.selections "Permanent link")

### `` text`property`[¶](\#marimo.ui.altair_chart.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.altair_chart.value "Permanent link")

### `` apply\_selection [¶](\#marimo.ui.altair_chart.apply_selection "Permanent link")

Apply the selection to a DataFrame.

This method is useful when you have a layered chart and you want to
apply the selection to a DataFrame.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `df` | A DataFrame to apply the selection to.<br>**TYPE:** `ChartDataType` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `ChartDataType` | A DataFrame of the plot data filtered by the selections.<br>**TYPE:** `ChartDataType` |

Examples:

### `` batch [¶](\#marimo.ui.altair_chart.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.altair_chart.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.altair_chart.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.altair_chart.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.altair_chart.from_args "Permanent link")

### `` left [¶](\#marimo.ui.altair_chart.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.altair_chart.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.altair_chart.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### Performance and Data Transformers [¶](\#performance-and-data-transformers "Permanent link")

Altair has a concept of [data](https://altair-viz.github.io/user_guide/data_transformers.html) transformers, which can be used to improve performance.

Such examples are:

- pandas Dataframe has to be sanitized and serialized to JSON.
- The rows of a Dataframe might need to be sampled or limited to a maximum number.
- The Dataframe might be written to a `.csv` or `.json` file for performance reasons.

By default, Altair uses the `default` data transformer, which is the slowest in marimo. It is limited to 5000 rows (although we increase this to `20_000` rows as marimo can handle this). This includes the data inside the HTML that is being sent over the network, which can also be limited by marimo's maximum message size.

It is recommended to use the `marimo_csv` data transformer, which is the most performant and can handle the largest datasets: it converts the data to a CSV file which is smaller and can be sent over the network. This can handle up to +400,000 rows with no issues.

When using `mo.ui.altair_chart`, we automatically set the data transformer to `marimo_csv` for you. If you are using Altair directly, you can set the data transformer using the following code:

## Reactive plots with Plotly [¶](\#reactive-plots-with-plotly "Permanent link")

mo.ui.plotly only supports scatter plots, treemaps charts, and sunbursts charts.

marimo can render any Plotly plot, but [`mo.ui.plotly`](#marimo.ui.plotly "            marimo.ui.plotly") only
supports reactive selections for scatter plots, treemaps charts, and sunbursts charts. If you require other kinds of
selection, consider using [`mo.ui.altair_chart`](#marimo.ui.altair_chart "            marimo.ui.altair_chart").

## `` marimo.ui.plotly [¶](\#marimo.ui.plotly "Permanent link")

Bases: `UIElement[PlotlySelection, list[dict[str, Any]]]`

Make reactive plots with Plotly.

Use `mo.ui.plotly` to make plotly plots reactive: select data with your
cursor on the frontend, get them as a list of dicts in Python!

This function currently only supports scatter plots, treemaps charts,
and sunbursts charts.

Examples:

Or with custom configuration:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict of the plot data.<br>**TYPE:** `Dict[str, Any]` |
| `ranges` | The selection of the plot; this may be an<br>interval along the name of an axis.<br>**TYPE:** `Dict[str, List[float]]` |
| `points` | The selected points data.<br>**TYPE:** `List[Dict[str, Any]]` |
| `indices` | The indices of selected points.<br>**TYPE:** `List[int]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `figure` | A plotly Figure object.<br>**TYPE:** `Figure` |
| `config` | Configuration for the plot.<br>This is a dictionary that is passed directly to plotly.<br>See the plotly documentation for more information:<br>[https://plotly.com/javascript/configuration-options/](https://plotly.com/javascript/configuration-options/)<br>This takes precedence over the default configuration of the renderer.<br>Defaults to None.<br>**TYPE:** `Optional[Dict[str, Any]]`**DEFAULT:** `None` |
| `renderer_name` | Renderer to use for the plot.<br>If this is not provided, the default renderer ( `pio.renderers.default`)<br>is used. Defaults to None.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Callback to run<br>when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[JSONType], None]]`**DEFAULT:** `None` |

### `` indices`property`[¶](\#marimo.ui.plotly.indices "Permanent link")

Get the indices of selected points in the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[int]` | List\[int\]: A list of indices corresponding to the selected points.<br>Returns an empty list if no points are selected. |

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.plotly.name "Permanent link")

### `` points`property`[¶](\#marimo.ui.plotly.points "Permanent link")

Get the selected points data from the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[dict[str, Any]]` | List\[Dict\[str, Any\]\]: A list of dictionaries containing the data for<br>each selected point. Returns an empty list if no points are selected. |

### `` ranges`property`[¶](\#marimo.ui.plotly.ranges "Permanent link")

Get the range selection of the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, list[float]]` | Dict\[str, List\[float\]\]: A dictionary mapping field names to their<br>selected ranges, where each range is a list of \[min, max\] values.<br>Returns an empty dict if no range selection exists. |

### `` text`property`[¶](\#marimo.ui.plotly.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.plotly.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.plotly.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.plotly.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.plotly.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.plotly.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.plotly.from_args "Permanent link")

### `` left [¶](\#marimo.ui.plotly.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.plotly.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.plotly.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

## Interactive matplotlib [¶](\#interactive-matplotlib "Permanent link")

## `` marimo.mpl.interactive [¶](\#marimo.mpl.interactive "Permanent link")

Render a matplotlib figure using an interactive viewer.

The interactive viewer allows you to pan, zoom, and see plot coordinates
on mouse hover.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `figure` | A matplotlib `Figure` or `Axes` object.<br>**TYPE:** `matplotlib Figure or Axes` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An interactive matplotlib figure as an `Html` object.<br>**TYPE:** `Html` |

Source code in `marimo/_plugins/stateless/mpl/_mpl.py`

## Leafmap support [¶](\#leafmap-support "Permanent link")

marimo supports rendering [Leafmap](https://leafmap.org/) maps using the `folium` and `plotly` backends.

## Other plotting libraries [¶](\#other-plotting-libraries "Permanent link")

You can use all the popular plotting libraries with marimo. Such as:

- [Matplotlib](https://matplotlib.org/)
- [Plotly](https://plotly.com/)
- [Seaborn](https://seaborn.pydata.org/)
- [Bokeh](https://bokeh.org/)
- [Altair](https://altair-viz.github.io/)
- [HoloViews](http://holoviews.org/)
- [hvPlot](https://hvplot.holoviz.org/)
- [Leafmap](https://leafmap.org/)
- [Pygwalker](https://kanaries.net/pygwalker)

 Back to top

# Plotting [¶](\#plotting "Permanent link")

marimo supports most major plotting libraries, including Matplotlib, Seaborn,
Plotly, and Altair. Just import your plotting library of choice and use it
as you normally would.

For more information about plotting, see the [plotting guide](../../guides/working_with_data/plotting/).

## Reactive charts with Altair [¶](\#reactive-charts-with-altair "Permanent link")

### Disabling automatic selection [¶](\#disabling-automatic-selection "Permanent link")

marimo automatically adds a default selection based on the mark type, however, you may want to customize the selection behavior of your Altair chart. You can do this by setting `chart_selection` and `legend_selection` to `False`, and using `.add_params` directly on your Altair chart.

## `` marimo.ui.altair\_chart [¶](\#marimo.ui.altair_chart "Permanent link")

Bases: `UIElement[ChartSelection, ChartDataType]`

Make reactive charts with Altair.

Use `mo.ui.altair_chart` to make Altair charts reactive: select chart data
with your cursor on the frontend, get them as a dataframe in Python!

Supports polars, pandas, and arrow DataFrames.

Examples:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dataframe of the plot data filtered by the selections.<br>**TYPE:** `ChartDataType` |
| `dataframe` | A dataframe of the unfiltered chart data.<br>**TYPE:** `ChartDataType` |
| `selections` | The selection of the chart; this may be an interval<br>along the name of an axis or a selection of points.<br>**TYPE:** `ChartSelection` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `chart` | An Altair Chart object.<br>**TYPE:** `Chart` |
| `chart_selection` | Selection type, "point", "interval", or a bool. Defaults to True which will<br>automatically detect the best selection type. This is ignored if the chart<br>already has a point/interval selection param.<br>**TYPE:** `Union[Literal['point'], Literal['interval'], bool]`**DEFAULT:** `True` |
| `legend_selection` | List of legend fields<br>(columns) for which to enable selection, True to enable selection for all<br>fields, or False to disable selection entirely. This is ignored if the chart<br>already has a legend selection param. Defaults to True.<br>**TYPE:** `Union[list[str], bool]`**DEFAULT:** `True` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Optional<br>callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[ChartDataType], None]]`**DEFAULT:** `None` |

### `` dataframe`instance-attribute`[¶](\#marimo.ui.altair_chart.dataframe "Permanent link")

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.altair_chart.name "Permanent link")

### `` selections`property`[¶](\#marimo.ui.altair_chart.selections "Permanent link")

### `` text`property`[¶](\#marimo.ui.altair_chart.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.altair_chart.value "Permanent link")

### `` apply\_selection [¶](\#marimo.ui.altair_chart.apply_selection "Permanent link")

Apply the selection to a DataFrame.

This method is useful when you have a layered chart and you want to
apply the selection to a DataFrame.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `df` | A DataFrame to apply the selection to.<br>**TYPE:** `ChartDataType` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `ChartDataType` | A DataFrame of the plot data filtered by the selections.<br>**TYPE:** `ChartDataType` |

Examples:

### `` batch [¶](\#marimo.ui.altair_chart.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.altair_chart.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.altair_chart.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.altair_chart.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.altair_chart.from_args "Permanent link")

### `` left [¶](\#marimo.ui.altair_chart.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.altair_chart.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.altair_chart.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

### Performance and Data Transformers [¶](\#performance-and-data-transformers "Permanent link")

Altair has a concept of [data](https://altair-viz.github.io/user_guide/data_transformers.html) transformers, which can be used to improve performance.

Such examples are:

- pandas Dataframe has to be sanitized and serialized to JSON.
- The rows of a Dataframe might need to be sampled or limited to a maximum number.
- The Dataframe might be written to a `.csv` or `.json` file for performance reasons.

By default, Altair uses the `default` data transformer, which is the slowest in marimo. It is limited to 5000 rows (although we increase this to `20_000` rows as marimo can handle this). This includes the data inside the HTML that is being sent over the network, which can also be limited by marimo's maximum message size.

It is recommended to use the `marimo_csv` data transformer, which is the most performant and can handle the largest datasets: it converts the data to a CSV file which is smaller and can be sent over the network. This can handle up to +400,000 rows with no issues.

When using `mo.ui.altair_chart`, we automatically set the data transformer to `marimo_csv` for you. If you are using Altair directly, you can set the data transformer using the following code:

## Reactive plots with Plotly [¶](\#reactive-plots-with-plotly "Permanent link")

mo.ui.plotly only supports scatter plots, treemaps charts, and sunbursts charts.

marimo can render any Plotly plot, but [`mo.ui.plotly`](#marimo.ui.plotly "            marimo.ui.plotly") only
supports reactive selections for scatter plots, treemaps charts, and sunbursts charts. If you require other kinds of
selection, consider using [`mo.ui.altair_chart`](#marimo.ui.altair_chart "            marimo.ui.altair_chart").

## `` marimo.ui.plotly [¶](\#marimo.ui.plotly "Permanent link")

Bases: `UIElement[PlotlySelection, list[dict[str, Any]]]`

Make reactive plots with Plotly.

Use `mo.ui.plotly` to make plotly plots reactive: select data with your
cursor on the frontend, get them as a list of dicts in Python!

This function currently only supports scatter plots, treemaps charts,
and sunbursts charts.

Examples:

Or with custom configuration:

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `value` | A dict of the plot data.<br>**TYPE:** `Dict[str, Any]` |
| `ranges` | The selection of the plot; this may be an<br>interval along the name of an axis.<br>**TYPE:** `Dict[str, List[float]]` |
| `points` | The selected points data.<br>**TYPE:** `List[Dict[str, Any]]` |
| `indices` | The indices of selected points.<br>**TYPE:** `List[int]` |

| PARAMETER | DESCRIPTION |
| --- | --- |
| `figure` | A plotly Figure object.<br>**TYPE:** `Figure` |
| `config` | Configuration for the plot.<br>This is a dictionary that is passed directly to plotly.<br>See the plotly documentation for more information:<br>[https://plotly.com/javascript/configuration-options/](https://plotly.com/javascript/configuration-options/)<br>This takes precedence over the default configuration of the renderer.<br>Defaults to None.<br>**TYPE:** `Optional[Dict[str, Any]]`**DEFAULT:** `None` |
| `renderer_name` | Renderer to use for the plot.<br>If this is not provided, the default renderer ( `pio.renderers.default`)<br>is used. Defaults to None.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `label` | Markdown label for the element. Defaults to "".<br>**TYPE:** `str`**DEFAULT:** `''` |
| `on_change` | Callback to run<br>when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[JSONType], None]]`**DEFAULT:** `None` |

### `` indices`property`[¶](\#marimo.ui.plotly.indices "Permanent link")

Get the indices of selected points in the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[int]` | List\[int\]: A list of indices corresponding to the selected points.<br>Returns an empty list if no points are selected. |

### `` name`class-attribute``instance-attribute`[¶](\#marimo.ui.plotly.name "Permanent link")

### `` points`property`[¶](\#marimo.ui.plotly.points "Permanent link")

Get the selected points data from the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[dict[str, Any]]` | List\[Dict\[str, Any\]\]: A list of dictionaries containing the data for<br>each selected point. Returns an empty list if no points are selected. |

### `` ranges`property`[¶](\#marimo.ui.plotly.ranges "Permanent link")

Get the range selection of the plot.

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, list[float]]` | Dict\[str, List\[float\]\]: A dictionary mapping field names to their<br>selected ranges, where each range is a list of \[min, max\] values.<br>Returns an empty dict if no range selection exists. |

### `` text`property`[¶](\#marimo.ui.plotly.text "Permanent link")

A string of HTML representing this element.

### `` value`property``writable`[¶](\#marimo.ui.plotly.value "Permanent link")

The element's current value.

### `` batch [¶](\#marimo.ui.plotly.batch "Permanent link")

Convert an HTML object with templated text into a UI element.

This method lets you create custom UI elements that are represented
by arbitrary HTML.

Example

In this example, `user_info` is a UI Element whose output is markdown
and whose value is a dict with keys `'name'` and ' `birthday`'
(and values equal to the values of their corresponding elements).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `elements` | the UI elements to interpolate into the HTML template.<br>**TYPE:** `UIElement[JSONType, object]`**DEFAULT:** `{}` |

### `` callout [¶](\#marimo.ui.plotly.callout "Permanent link")

Create a callout containing this HTML element.

A callout wraps your HTML element in a raised box, emphasizing its
importance. You can style the callout for different situations with the
`kind` argument.

Examples:

### `` center [¶](\#marimo.ui.plotly.center "Permanent link")

Center an item.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` form [¶](\#marimo.ui.plotly.form "Permanent link")

Create a submittable form out of this `UIElement`.

Creates a form that gates submission of a `UIElement`'s value until a submit button is clicked.
The form's value is the value of the underlying element from the last submission.

Examples:

Convert any `UIElement` into a form:

Combine with `HTML.batch` to create a form made out of multiple `UIElements`:

| PARAMETER | DESCRIPTION |
| --- | --- |
| `label` | A text label for the form.<br>**TYPE:** `str`**DEFAULT:** `''` |
| `bordered` | Whether the form should have a border.<br>**TYPE:** `bool`**DEFAULT:** `True` |
| `loading` | Whether the form should be in a loading state.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `submit_button_label` | The label of the submit button.<br>**TYPE:** `str`**DEFAULT:** `'Submit'` |
| `submit_button_tooltip` | The tooltip of the submit button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `submit_button_disabled` | Whether the submit button should be disabled.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_on_submit` | Whether the form should clear its contents after submitting.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `show_clear_button` | Whether the form should show a clear button.<br>**TYPE:** `bool`**DEFAULT:** `False` |
| `clear_button_label` | The label of the clear button.<br>**TYPE:** `str`**DEFAULT:** `'Clear'` |
| `clear_button_tooltip` | The tooltip of the clear button.<br>**TYPE:** `Optional[str]`**DEFAULT:** `None` |
| `validate` | A function that takes the form's value and returns an error message if invalid,<br>or `None` if valid.<br>**TYPE:** `Optional[Callable[[Optional[JSONType]], Optional[str]]]`**DEFAULT:** `None` |
| `on_change` | Optional callback to run when this element's value changes. Defaults to None.<br>**TYPE:** `Optional[Callable[[Optional[T]], None]]`**DEFAULT:** `None` |

### `` from\_args`classmethod`[¶](\#marimo.ui.plotly.from_args "Permanent link")

### `` left [¶](\#marimo.ui.plotly.left "Permanent link")

Left-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` right [¶](\#marimo.ui.plotly.right "Permanent link")

Right-justify.

Example

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An `Html` object. |

### `` style [¶](\#marimo.ui.plotly.style "Permanent link")

Wrap an object in a styled container.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `style` | an optional dict of CSS styles, keyed by property name<br>**TYPE:** `Optional[dict[str, Any]]`**DEFAULT:** `None` |
| `**kwargs` | CSS styles as keyword arguments<br>**TYPE:** `Any`**DEFAULT:** `{}` |

## Interactive matplotlib [¶](\#interactive-matplotlib "Permanent link")

## `` marimo.mpl.interactive [¶](\#marimo.mpl.interactive "Permanent link")

Render a matplotlib figure using an interactive viewer.

The interactive viewer allows you to pan, zoom, and see plot coordinates
on mouse hover.

Example

| PARAMETER | DESCRIPTION |
| --- | --- |
| `figure` | A matplotlib `Figure` or `Axes` object.<br>**TYPE:** `matplotlib Figure or Axes` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Html` | An interactive matplotlib figure as an `Html` object.<br>**TYPE:** `Html` |

Source code in `marimo/_plugins/stateless/mpl/_mpl.py`

## Leafmap support [¶](\#leafmap-support "Permanent link")

marimo supports rendering [Leafmap](https://leafmap.org/) maps using the `folium` and `plotly` backends.

## Other plotting libraries [¶](\#other-plotting-libraries "Permanent link")

You can use all the popular plotting libraries with marimo. Such as:

- [Matplotlib](https://matplotlib.org/)
- [Plotly](https://plotly.com/)
- [Seaborn](https://seaborn.pydata.org/)
- [Bokeh](https://bokeh.org/)
- [Altair](https://altair-viz.github.io/)
- [HoloViews](http://holoviews.org/)
- [hvPlot](https://hvplot.holoviz.org/)
- [Leafmap](https://leafmap.org/)
- [Pygwalker](https://kanaries.net/pygwalker)


---

###### Getting Started - marimoGetting Started¶

*Source: [https://docs.marimo.io/getting_started/](https://docs.marimo.io/getting_started/)*

<!-- Source: https://docs.marimo.io/getting_started/ -->
<!-- Title: Getting Started - marimoGetting Started¶ -->
<!-- Depth: 8 -->

[Skip to content](#getting-started)

# Getting Started [¶](\#getting-started "Permanent link")

These tutorials will help you get started with marimo

| Guide | Description |
| --- | --- |
| [Installation](installation/) | Installing marimo |
| [Quickstart](quickstart/) | Create notebooks, run apps, and more from the marimo command-line |
| [Key Concepts](key_concepts/) | A tour of key features and concepts |

Back to top

# Getting Started [¶](\#getting-started "Permanent link")

These tutorials will help you get started with marimo

| Guide | Description |
| --- | --- |
| [Installation](installation/) | Installing marimo |
| [Quickstart](quickstart/) | Create notebooks, run apps, and more from the marimo command-line |
| [Key Concepts](key_concepts/) | A tour of key features and concepts |

 Back to top

# Getting Started [¶](\#getting-started "Permanent link")

These tutorials will help you get started with marimo

| Guide | Description |
| --- | --- |
| [Installation](installation/) | Installing marimo |
| [Quickstart](quickstart/) | Create notebooks, run apps, and more from the marimo command-line |
| [Key Concepts](key_concepts/) | A tour of key features and concepts |


---

###### Migrate from Jupyter - marimoComing from Jupyter¶

*Source: [https://docs.marimo.io/guides/coming_from/jupyter/](https://docs.marimo.io/guides/coming_from/jupyter/)*

<!-- Source: https://docs.marimo.io/guides/coming_from/jupyter/ -->
<!-- Title: Migrate from Jupyter - marimoComing from Jupyter¶ -->
<!-- Depth: 8 -->

[Skip to content](#coming-from-jupyter)

# Coming from Jupyter [¶](\#coming-from-jupyter "Permanent link")

If you're coming from Jupyter, here are a few tips to help you adapt to marimo
notebooks.

## How marimo runs cells [¶](\#how-marimo-runs-cells "Permanent link")

The biggest difference between marimo and Jupyter is the [execution model](../../reactivity/).

A **Jupyter** notebook is a **REPL**: you execute blocks of code one at a time,
and Jupyter has no understanding of how different blocks are related to each
other. As a result a Jupyter notebook can easily
accumulate **"hidden state"** (and hidden bugs) --- you might accidentally execute
cells out of order, or you might run (or delete) a cell but forget to re-run
cells that depended on its variables. Because of this, Jupyter notebooks
suffer from a [reproducibility crisis](../../../faq/#faq-problems), with over
a third of Jupyter notebooks on GitHub failing to reproduce.

Unlike Jupyter, **marimo** notebooks understand how different blocks of
code are related to each other, modeling your code as a graph on cells
based on variable declarations and references. This eliminates hidden
state, and it's also what enables marimo notebooks to be reused as
apps and scripts.

**By default, if you run a cell in marimo, all other cells that read its**
**variables run automatically.** While this ensures that your code and outputs are
in sync, it can take some time getting used to. **Here are some tips and tools to**
**help you adapt to marimo's execution model.**

### Configure marimo's runtime [¶](\#configure-marimos-runtime "Permanent link")

[Configure marimo's runtime](../../configuration/runtime_configuration/) to
not autorun on startup or on cell execution.

Even when autorun is disabled, marimo still tracks dependencies across cells,
marking dependents of a cell as stale when you run it. You can click a single
button to run all your stale cells and bring your notebook back up-to-date.

### Stop execution with `mo.stop` [¶](\#stop-execution-with-mostop "Permanent link")

Use [`mo.stop`](../../../api/control_flow/#marimo.stop "            marimo.stop") to stop a cell from executing if a condition
is met:

Use [`mo.stop()`](../../../api/control_flow/#marimo.stop "            marimo.stop") in conjunction with
[`mo.ui.run_button()`](../../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to require a button press for
expensive cells:

### Working with expensive notebooks [¶](\#working-with-expensive-notebooks "Permanent link")

For more tips on adapting to marimo's execution model, see our guide
on [working with expensive notebooks](../../expensive_notebooks/).

## Redefining variables [¶](\#redefining-variables "Permanent link")

marimo "compiles" your notebook cells into a directed graph on cells,
linked by variable declarations and references, reusing this graph to
run your notebook as a script or app. For marimo's compilation to work,
the same variable cannot be defined in multiple cells; otherwise, marimo
wouldn't know what order to run cells in.

To adapt to the restriction, we suggest:

1. Encapsulating code into functions when possible, to minimize the number
    of global variables.
2. Prefixing temporary variables with an underscore ( `_my_temporary`), which
    makes the variable **local** to a cell.
3. Mutating variables in the cell that defines them.

When working with **dataframes**, you might be used to redefining the same `df`
variable in multiple cells. That won't work in marimo. Instead, try merging
the cells into a single cell:

_Don't_ do this:

_Instead_, do this:

If you do need to transform a dataframe across multiple cells, you can
alias the dataframe:

## marimo's file format [¶](\#marimos-file-format "Permanent link")

marimo stores notebooks as Python, not JSON. This lets you version notebooks
with git, [execute them as scripts](../../scripts/), and import named
cells into other Python files. However, it does mean that your notebook outputs
(e.g., plots) are not stored in the file.

If you'd like to keep a visual record of your notebook work, [enable\
the "Auto-download as HTML/IPYNB" setting](../../configuration/), which will
periodically snapshot your notebook as HTML or IPYNB to a `__marimo__` folder in the
notebook directory.

### Converting Jupyter notebooks to marimo notebooks [¶](\#converting-jupyter-notebooks-to-marimo-notebooks "Permanent link")

Convert Jupyter notebooks to marimo notebooks at the command-line:

### Converting Python scripts to marimo notebooks [¶](\#converting-python-scripts-to-marimo-notebooks "Permanent link")

marimo can also convert regular Python scripts to marimo notebooks:

This supports:
\- **py:percent format**: If your script uses `# %%` cell markers, marimo will convert it to a multi-cell notebook (requires jupytext)
\- **Regular Python scripts**: Scripts without cell markers are converted to a single-cell notebook

For py:percent conversion with uv:

### Exporting marimo notebooks to Jupyter notebooks [¶](\#exporting-marimo-notebooks-to-jupyter-notebooks "Permanent link")

Export to an `ipynb` file with

Note that some marimo library functions, including UI elements,
won't work in Jupyter notebooks.

## Magic commands [¶](\#magic-commands "Permanent link")

Because marimo notebooks are just Python (improving maintainability), marimo
doesn't support IPython magic commands or `!`-prefixed console commands. Here
are some alternatives.

### Run console commands with subprocess.run [¶](\#run-console-commands-with-subprocessrun "Permanent link")

To run a console command, use Python's [subprocess.run](https://docs.python.org/3/library/subprocess.html#subprocess.run):

### Common magic commands replacements [¶](\#common-magic-commands-replacements "Permanent link")

| Magic Command | Replacement |
| --- | --- |
| %cd | `os.chdir()`, see also [`mo.notebook_dir()`](../../../api/miscellaneous/#marimo.notebook_dir "            marimo.notebook_dir") |
| %clear | Right-click or toggle the cell actions |
| %debug | Python's built-in debugger: `breakpoint()` |
| %env | `os.environ` |
| %load | N/A - use Python imports |
| %load\_ext | N/A |
| %autoreload | marimo's [module autoreloader](../../editor_features/module_autoreloading/) |
| %matplotlib | marimo auto-displays plots |
| %pwd | `os.getcwd()` |
| %pip | Use marimo's [built-in package management](../../editor_features/package_management/) |
| %who\_ls | `dir()`, `globals()`, [`mo.refs()`](../../../api/miscellaneous/#marimo.refs "            marimo.refs"), [`mo.defs()`](../../../api/miscellaneous/#marimo.defs "            marimo.defs") |
| %system | `subprocess.run()` |
| %%time | `time.perf_counter()` or Python's timeit module |
| %%timeit | Python's timeit module |
| %%writefile | `with open("file.txt", "w") as f: f.write()` |
| %%capture | [`mo.capture_stdout()`](../../../api/outputs/#marimo.capture_stdout "            marimo.capture_stdout"), [`mo.capture_stderr()`](../../../api/outputs/#marimo.capture_stderr "            marimo.capture_stderr") |
| %%html | [`mo.Html()`](../../../api/html/#marimo.Html "            marimo.Html            dataclass   ") or [`mo.md()`](../../../api/markdown/#marimo.md "            marimo.md") |
| %%latex | [`mo.md(r'$$...$$')`](../../../api/markdown/#marimo.md "            marimo.md") |

### Installing packages with marimo's package manager [¶](\#installing-packages-with-marimos-package-manager "Permanent link")

Use marimo's package management sidebar panel to install packages to your current
environment. Learn more in our [package management\
guide](../../editor_features/package_management/).

## Interactive guide [¶](\#interactive-guide "Permanent link")

This guide contains additional tips to help you adapt to marimo. Fun fact: the
guide is itself a marimo notebook!

Back to top

# Coming from Jupyter [¶](\#coming-from-jupyter "Permanent link")

If you're coming from Jupyter, here are a few tips to help you adapt to marimo
notebooks.

## How marimo runs cells [¶](\#how-marimo-runs-cells "Permanent link")

The biggest difference between marimo and Jupyter is the [execution model](../../reactivity/).

A **Jupyter** notebook is a **REPL**: you execute blocks of code one at a time,
and Jupyter has no understanding of how different blocks are related to each
other. As a result a Jupyter notebook can easily
accumulate **"hidden state"** (and hidden bugs) --- you might accidentally execute
cells out of order, or you might run (or delete) a cell but forget to re-run
cells that depended on its variables. Because of this, Jupyter notebooks
suffer from a [reproducibility crisis](../../../faq/#faq-problems), with over
a third of Jupyter notebooks on GitHub failing to reproduce.

Unlike Jupyter, **marimo** notebooks understand how different blocks of
code are related to each other, modeling your code as a graph on cells
based on variable declarations and references. This eliminates hidden
state, and it's also what enables marimo notebooks to be reused as
apps and scripts.

**By default, if you run a cell in marimo, all other cells that read its**
**variables run automatically.** While this ensures that your code and outputs are
in sync, it can take some time getting used to. **Here are some tips and tools to**
**help you adapt to marimo's execution model.**

### Configure marimo's runtime [¶](\#configure-marimos-runtime "Permanent link")

[Configure marimo's runtime](../../configuration/runtime_configuration/) to
not autorun on startup or on cell execution.

Even when autorun is disabled, marimo still tracks dependencies across cells,
marking dependents of a cell as stale when you run it. You can click a single
button to run all your stale cells and bring your notebook back up-to-date.

### Stop execution with `mo.stop` [¶](\#stop-execution-with-mostop "Permanent link")

Use [`mo.stop`](../../../api/control_flow/#marimo.stop "            marimo.stop") to stop a cell from executing if a condition
is met:

Use [`mo.stop()`](../../../api/control_flow/#marimo.stop "            marimo.stop") in conjunction with
[`mo.ui.run_button()`](../../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to require a button press for
expensive cells:

### Working with expensive notebooks [¶](\#working-with-expensive-notebooks "Permanent link")

For more tips on adapting to marimo's execution model, see our guide
on [working with expensive notebooks](../../expensive_notebooks/).

## Redefining variables [¶](\#redefining-variables "Permanent link")

marimo "compiles" your notebook cells into a directed graph on cells,
linked by variable declarations and references, reusing this graph to
run your notebook as a script or app. For marimo's compilation to work,
the same variable cannot be defined in multiple cells; otherwise, marimo
wouldn't know what order to run cells in.

To adapt to the restriction, we suggest:

1. Encapsulating code into functions when possible, to minimize the number
    of global variables.
2. Prefixing temporary variables with an underscore ( `_my_temporary`), which
    makes the variable **local** to a cell.
3. Mutating variables in the cell that defines them.

When working with **dataframes**, you might be used to redefining the same `df`
variable in multiple cells. That won't work in marimo. Instead, try merging
the cells into a single cell:

_Don't_ do this:

_Instead_, do this:

If you do need to transform a dataframe across multiple cells, you can
alias the dataframe:

## marimo's file format [¶](\#marimos-file-format "Permanent link")

marimo stores notebooks as Python, not JSON. This lets you version notebooks
with git, [execute them as scripts](../../scripts/), and import named
cells into other Python files. However, it does mean that your notebook outputs
(e.g., plots) are not stored in the file.

If you'd like to keep a visual record of your notebook work, [enable\
the "Auto-download as HTML/IPYNB" setting](../../configuration/), which will
periodically snapshot your notebook as HTML or IPYNB to a `__marimo__` folder in the
notebook directory.

### Converting Jupyter notebooks to marimo notebooks [¶](\#converting-jupyter-notebooks-to-marimo-notebooks "Permanent link")

Convert Jupyter notebooks to marimo notebooks at the command-line:

### Converting Python scripts to marimo notebooks [¶](\#converting-python-scripts-to-marimo-notebooks "Permanent link")

marimo can also convert regular Python scripts to marimo notebooks:

This supports:
\- **py:percent format**: If your script uses `# %%` cell markers, marimo will convert it to a multi-cell notebook (requires jupytext)
\- **Regular Python scripts**: Scripts without cell markers are converted to a single-cell notebook

For py:percent conversion with uv:

### Exporting marimo notebooks to Jupyter notebooks [¶](\#exporting-marimo-notebooks-to-jupyter-notebooks "Permanent link")

Export to an `ipynb` file with

Note that some marimo library functions, including UI elements,
won't work in Jupyter notebooks.

## Magic commands [¶](\#magic-commands "Permanent link")

Because marimo notebooks are just Python (improving maintainability), marimo
doesn't support IPython magic commands or `!`-prefixed console commands. Here
are some alternatives.

### Run console commands with subprocess.run [¶](\#run-console-commands-with-subprocessrun "Permanent link")

To run a console command, use Python's [subprocess.run](https://docs.python.org/3/library/subprocess.html#subprocess.run):

### Common magic commands replacements [¶](\#common-magic-commands-replacements "Permanent link")

| Magic Command | Replacement |
| --- | --- |
| %cd | `os.chdir()`, see also [`mo.notebook_dir()`](../../../api/miscellaneous/#marimo.notebook_dir "            marimo.notebook_dir") |
| %clear | Right-click or toggle the cell actions |
| %debug | Python's built-in debugger: `breakpoint()` |
| %env | `os.environ` |
| %load | N/A - use Python imports |
| %load\_ext | N/A |
| %autoreload | marimo's [module autoreloader](../../editor_features/module_autoreloading/) |
| %matplotlib | marimo auto-displays plots |
| %pwd | `os.getcwd()` |
| %pip | Use marimo's [built-in package management](../../editor_features/package_management/) |
| %who\_ls | `dir()`, `globals()`, [`mo.refs()`](../../../api/miscellaneous/#marimo.refs "            marimo.refs"), [`mo.defs()`](../../../api/miscellaneous/#marimo.defs "            marimo.defs") |
| %system | `subprocess.run()` |
| %%time | `time.perf_counter()` or Python's timeit module |
| %%timeit | Python's timeit module |
| %%writefile | `with open("file.txt", "w") as f: f.write()` |
| %%capture | [`mo.capture_stdout()`](../../../api/outputs/#marimo.capture_stdout "            marimo.capture_stdout"), [`mo.capture_stderr()`](../../../api/outputs/#marimo.capture_stderr "            marimo.capture_stderr") |
| %%html | [`mo.Html()`](../../../api/html/#marimo.Html "            marimo.Html            dataclass   ") or [`mo.md()`](../../../api/markdown/#marimo.md "            marimo.md") |
| %%latex | [`mo.md(r'$$...$$')`](../../../api/markdown/#marimo.md "            marimo.md") |

### Installing packages with marimo's package manager [¶](\#installing-packages-with-marimos-package-manager "Permanent link")

Use marimo's package management sidebar panel to install packages to your current
environment. Learn more in our [package management\
guide](../../editor_features/package_management/).

## Interactive guide [¶](\#interactive-guide "Permanent link")

This guide contains additional tips to help you adapt to marimo. Fun fact: the
guide is itself a marimo notebook!

 Back to top

# Coming from Jupyter [¶](\#coming-from-jupyter "Permanent link")

If you're coming from Jupyter, here are a few tips to help you adapt to marimo
notebooks.

## How marimo runs cells [¶](\#how-marimo-runs-cells "Permanent link")

The biggest difference between marimo and Jupyter is the [execution model](../../reactivity/).

A **Jupyter** notebook is a **REPL**: you execute blocks of code one at a time,
and Jupyter has no understanding of how different blocks are related to each
other. As a result a Jupyter notebook can easily
accumulate **"hidden state"** (and hidden bugs) --- you might accidentally execute
cells out of order, or you might run (or delete) a cell but forget to re-run
cells that depended on its variables. Because of this, Jupyter notebooks
suffer from a [reproducibility crisis](../../../faq/#faq-problems), with over
a third of Jupyter notebooks on GitHub failing to reproduce.

Unlike Jupyter, **marimo** notebooks understand how different blocks of
code are related to each other, modeling your code as a graph on cells
based on variable declarations and references. This eliminates hidden
state, and it's also what enables marimo notebooks to be reused as
apps and scripts.

**By default, if you run a cell in marimo, all other cells that read its**
**variables run automatically.** While this ensures that your code and outputs are
in sync, it can take some time getting used to. **Here are some tips and tools to**
**help you adapt to marimo's execution model.**

### Configure marimo's runtime [¶](\#configure-marimos-runtime "Permanent link")

[Configure marimo's runtime](../../configuration/runtime_configuration/) to
not autorun on startup or on cell execution.

Even when autorun is disabled, marimo still tracks dependencies across cells,
marking dependents of a cell as stale when you run it. You can click a single
button to run all your stale cells and bring your notebook back up-to-date.

### Stop execution with `mo.stop` [¶](\#stop-execution-with-mostop "Permanent link")

Use [`mo.stop`](../../../api/control_flow/#marimo.stop "            marimo.stop") to stop a cell from executing if a condition
is met:

Use [`mo.stop()`](../../../api/control_flow/#marimo.stop "            marimo.stop") in conjunction with
[`mo.ui.run_button()`](../../../api/inputs/run_button/#marimo.ui.run_button "            marimo.ui.run_button") to require a button press for
expensive cells:

### Working with expensive notebooks [¶](\#working-with-expensive-notebooks "Permanent link")

For more tips on adapting to marimo's execution model, see our guide
on [working with expensive notebooks](../../expensive_notebooks/).

## Redefining variables [¶](\#redefining-variables "Permanent link")

marimo "compiles" your notebook cells into a directed graph on cells,
linked by variable declarations and references, reusing this graph to
run your notebook as a script or app. For marimo's compilation to work,
the same variable cannot be defined in multiple cells; otherwise, marimo
wouldn't know what order to run cells in.

To adapt to the restriction, we suggest:

1. Encapsulating code into functions when possible, to minimize the number
    of global variables.
2. Prefixing temporary variables with an underscore ( `_my_temporary`), which
    makes the variable **local** to a cell.
3. Mutating variables in the cell that defines them.

When working with **dataframes**, you might be used to redefining the same `df`
variable in multiple cells. That won't work in marimo. Instead, try merging
the cells into a single cell:

_Don't_ do this:

_Instead_, do this:

If you do need to transform a dataframe across multiple cells, you can
alias the dataframe:

## marimo's file format [¶](\#marimos-file-format "Permanent link")

marimo stores notebooks as Python, not JSON. This lets you version notebooks
with git, [execute them as scripts](../../scripts/), and import named
cells into other Python files. However, it does mean that your notebook outputs
(e.g., plots) are not stored in the file.

If you'd like to keep a visual record of your notebook work, [enable\
the "Auto-download as HTML/IPYNB" setting](../../configuration/), which will
periodically snapshot your notebook as HTML or IPYNB to a `__marimo__` folder in the
notebook directory.

### Converting Jupyter notebooks to marimo notebooks [¶](\#converting-jupyter-notebooks-to-marimo-notebooks "Permanent link")

Convert Jupyter notebooks to marimo notebooks at the command-line:

### Converting Python scripts to marimo notebooks [¶](\#converting-python-scripts-to-marimo-notebooks "Permanent link")

marimo can also convert regular Python scripts to marimo notebooks:

This supports:
\- **py:percent format**: If your script uses `# %%` cell markers, marimo will convert it to a multi-cell notebook (requires jupytext)
\- **Regular Python scripts**: Scripts without cell markers are converted to a single-cell notebook

For py:percent conversion with uv:

### Exporting marimo notebooks to Jupyter notebooks [¶](\#exporting-marimo-notebooks-to-jupyter-notebooks "Permanent link")

Export to an `ipynb` file with

Note that some marimo library functions, including UI elements,
won't work in Jupyter notebooks.

## Magic commands [¶](\#magic-commands "Permanent link")

Because marimo notebooks are just Python (improving maintainability), marimo
doesn't support IPython magic commands or `!`-prefixed console commands. Here
are some alternatives.

### Run console commands with subprocess.run [¶](\#run-console-commands-with-subprocessrun "Permanent link")

To run a console command, use Python's [subprocess.run](https://docs.python.org/3/library/subprocess.html#subprocess.run):

### Common magic commands replacements [¶](\#common-magic-commands-replacements "Permanent link")

| Magic Command | Replacement |
| --- | --- |
| %cd | `os.chdir()`, see also [`mo.notebook_dir()`](../../../api/miscellaneous/#marimo.notebook_dir "            marimo.notebook_dir") |
| %clear | Right-click or toggle the cell actions |
| %debug | Python's built-in debugger: `breakpoint()` |
| %env | `os.environ` |
| %load | N/A - use Python imports |
| %load\_ext | N/A |
| %autoreload | marimo's [module autoreloader](../../editor_features/module_autoreloading/) |
| %matplotlib | marimo auto-displays plots |
| %pwd | `os.getcwd()` |
| %pip | Use marimo's [built-in package management](../../editor_features/package_management/) |
| %who\_ls | `dir()`, `globals()`, [`mo.refs()`](../../../api/miscellaneous/#marimo.refs "            marimo.refs"), [`mo.defs()`](../../../api/miscellaneous/#marimo.defs "            marimo.defs") |
| %system | `subprocess.run()` |
| %%time | `time.perf_counter()` or Python's timeit module |
| %%timeit | Python's timeit module |
| %%writefile | `with open("file.txt", "w") as f: f.write()` |
| %%capture | [`mo.capture_stdout()`](../../../api/outputs/#marimo.capture_stdout "            marimo.capture_stdout"), [`mo.capture_stderr()`](../../../api/outputs/#marimo.capture_stderr "            marimo.capture_stderr") |
| %%html | [`mo.Html()`](../../../api/html/#marimo.Html "            marimo.Html            dataclass   ") or [`mo.md()`](../../../api/markdown/#marimo.md "            marimo.md") |
| %%latex | [`mo.md(r'$$...$$')`](../../../api/markdown/#marimo.md "            marimo.md") |

### Installing packages with marimo's package manager [¶](\#installing-packages-with-marimos-package-manager "Permanent link")

Use marimo's package management sidebar panel to install packages to your current
environment. Learn more in our [package management\
guide](../../editor_features/package_management/).

## Interactive guide [¶](\#interactive-guide "Permanent link")

This guide contains additional tips to help you adapt to marimo. Fun fact: the
guide is itself a marimo notebook!


---

###### Custom UI plugins - marimoCustom UI plugins

*Source: [https://docs.marimo.io/guides/integrating_with_marimo/custom_ui_plugins/](https://docs.marimo.io/guides/integrating_with_marimo/custom_ui_plugins/)*

<!-- Source: https://docs.marimo.io/guides/integrating_with_marimo/custom_ui_plugins/ -->
<!-- Title: Custom UI plugins - marimoCustom UI plugins -->
<!-- Depth: 8 -->

# Custom UI plugins

Build custom UI plugins that hook into marimo’s reactive execution engine by
using [anywidget](https://anywidget.dev/). [See our AnyWidget API\
docs](../../../api/inputs/anywidget/) for more information.

Back to top

# Custom UI plugins

Build custom UI plugins that hook into marimo’s reactive execution engine by
using [anywidget](https://anywidget.dev/). [See our AnyWidget API\
docs](../../../api/inputs/anywidget/) for more information.

 Back to top

# Custom UI plugins

Build custom UI plugins that hook into marimo’s reactive execution engine by
using [anywidget](https://anywidget.dev/). [See our AnyWidget API\
docs](../../../api/inputs/anywidget/) for more information.


---

###### Package management - marimoPackage management¶

*Source: [https://docs.marimo.io/guides/package_management/](https://docs.marimo.io/guides/package_management/)*

<!-- Source: https://docs.marimo.io/guides/package_management/ -->
<!-- Title: Package management - marimoPackage management¶ -->
<!-- Depth: 8 -->

[Skip to content](#package-management)

# Package management [¶](\#package-management "Permanent link")

The following guides cover how to import, install, and otherwise manage
the Python dependencies of your notebooks.

| Guide | Description |
| --- | --- |
| [Importing packages](importing_packages/) | How marimo finds packages on import |
| [Installing packages](installing_packages/) | Installing packages with marimo's UI |
| [Inlining dependencies](inlining_dependencies/) | Create self-contained notebooks by inlining dependencies in notebook files |
| [Notebooks in existing projects](notebooks_in_projects/) | Working with marimo notebooks in existing Python projects |
| [Using uv](using_uv/) | A guide to using the uv package manager with marimo |

Back to top

# Package management [¶](\#package-management "Permanent link")

The following guides cover how to import, install, and otherwise manage
the Python dependencies of your notebooks.

| Guide | Description |
| --- | --- |
| [Importing packages](importing_packages/) | How marimo finds packages on import |
| [Installing packages](installing_packages/) | Installing packages with marimo's UI |
| [Inlining dependencies](inlining_dependencies/) | Create self-contained notebooks by inlining dependencies in notebook files |
| [Notebooks in existing projects](notebooks_in_projects/) | Working with marimo notebooks in existing Python projects |
| [Using uv](using_uv/) | A guide to using the uv package manager with marimo |

 Back to top

# Package management [¶](\#package-management "Permanent link")

The following guides cover how to import, install, and otherwise manage
the Python dependencies of your notebooks.

| Guide | Description |
| --- | --- |
| [Importing packages](importing_packages/) | How marimo finds packages on import |
| [Installing packages](installing_packages/) | Installing packages with marimo's UI |
| [Inlining dependencies](inlining_dependencies/) | Create self-contained notebooks by inlining dependencies in notebook files |
| [Notebooks in existing projects](notebooks_in_projects/) | Working with marimo notebooks in existing Python projects |
| [Using uv](using_uv/) | A guide to using the uv package manager with marimo |


---

###### Inlining dependencies - marimoInlining dependencies¶

*Source: [https://docs.marimo.io/guides/package_management/inlining_dependencies/](https://docs.marimo.io/guides/package_management/inlining_dependencies/)*

<!-- Source: https://docs.marimo.io/guides/package_management/inlining_dependencies/ -->
<!-- Title: Inlining dependencies - marimoInlining dependencies¶ -->
<!-- Depth: 8 -->

[Skip to content](#inlining-dependencies)

# Inlining dependencies [¶](\#inlining-dependencies "Permanent link")

marimo is the only Python notebook that is reproducible down to the packages,
letting you inline Python dependencies in notebook files and running notebooks
in isolated or "sandboxed" venvs. This lets you share standalone notebooks
without shipping `requirements.txt` files alongside them, and guarantees your
notebooks will work weeks, months, even years into the future.

To opt-in to dependency inlining, use the `sandbox` flag:

editrunnew

When running with `--sandbox`, marimo:

1. tracks the packages and versions used by your notebook, saving
    them in the notebook file;
2. runs in an isolated virtual environment ("sandbox") that only
    contains the notebook dependencies.

marimo's sandbox provides two key benefits. (1) Notebooks that carry their own
dependencies are easy to share — just send the `.py` file. (2) Isolating a
notebook from other installed packages prevents obscure bugs.

You can also run sandboxed notebooks as scripts:

Requires uv

Sandboxed notebooks require the uv package manager
( [installation\
instructions](https://docs.astral.sh/uv/getting-started/installation/)).

Solving the notebook reproducibility crisis

marimo's support for package sandboxing is only possible because marimo
notebooks are stored as pure Python files, letting marimo take advantage
of new Python standards like [PEP\
723](https://peps.python.org/pep-0723/) and tools like uv. In contrast,
traditional notebooks like Jupyter are stored as JSON files, and which suffer
from a [reproducibility\
crisis](https://leomurta.github.io/papers/pimentel2019a.pdf) due to the lack
of package management.

## Inline script metadata [¶](\#auto-tracking-inline-script-metadata "Permanent link")

When running with `--sandbox`, marimo automatically tracks package metadata in
your notebook file using inline script metadata, which per [PEP\
723](https://peps.python.org/pep-0723/) is essentially a pyproject.toml inlined
as the script's header. This metadata is used to manage the
notebook's dependencies and Python version, and looks something like this:

Example notebooks

The [example\
notebooks](https://github.com/marimo-team/marimo/tree/main/examples) in our
GitHub repo were all created using `--sandbox`. Take a look at any of them
for an example of the full script metadata.

### Adding and removing packages [¶](\#adding-and-removing-packages "Permanent link")

**Using the marimo editor.** When you import a module in the marimo editor, if
marimo detects that it is a third-party package, it will automatically be added
to the script metadata. Removing an import does _not_ remove it from the script
metadata (since library code may still use the package).

Adding packages via the package manager panel will also add packages to script
metadata, and removing packages from the panel will in turn remove them from
the script metadata.

**Adding packages manually.** You can manually manage your notebook's
requirements:

- edit the script metadata manually in an editor like VS Code or neovim.
- use `uv` from the command-line:

### Package locations [¶](\#package-locations "Permanent link")

By default, marimo will look for packages on PyPI. You can edit the script
metadata to look for packages elsewhere, such as on GitHub. Consult the [Python\
packaging\
documentation](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#examples)
for more information.

### Local development with editable installs [¶](\#local-development-with-editable-installs "Permanent link")

When developing a local package, you can install it in editable mode using the
`[tool.uv.sources]` section in the script metadata. For example:

This is particularly useful when you want to test changes to your package
without reinstalling it. The package will be installed in "editable" mode,
meaning changes to the source code will be reflected immediately in your
notebook.

### Specifying alternative package indexes [¶](\#specifying-alternative-package-indexes "Permanent link")

When you need to use packages from a custom PyPI server or alternative index,
you can specify these in your script metadata using the `[[tool.uv.index]]`
section. This is useful for private packages or when you want to use packages
from a specific source.

In this example:

- `[[tool.uv.index]]` defines a custom package index
- `name` is an identifier for the index
- `url` points to your custom PyPI server
- `explicit = true` means this index will only be used for packages explicitly associated with it
- `[tool.uv.sources]` specifies which packages should come from which indexes

This approach ensures that specific packages are always fetched from your designated custom index, while other packages continue to be fetched from the default PyPI repository.

## Configuration [¶](\#configuration "Permanent link")

Running marimo in a sandbox environment uses `uv` to create an isolated virtual
environment. You can use any of `uv`'s [supported environment\
variables](https://docs.astral.sh/uv/configuration/environment/).

### Choosing the Python version [¶](\#choosing-the-python-version "Permanent link")

For example, you can specify the Python version using the `UV_PYTHON` environment variable:

### Other common configuration [¶](\#other-common-configuration "Permanent link")

Another common configuration is `uv`'s link mode:

## Sharing on the web [¶](\#sharing-on-the-web "Permanent link")

You can also upload sandboxed notebooks to the web, such as on GitHub, and have
others run them locally with a single command:

**Note:**

1. This command will run code from a URL. Make sure you trust the source before proceeding.
2. Upon execution, you’ll be prompted:

    To proceed securely, ensure you have [Docker](https://www.docker.com/) installed and running, then press `Y`.

## Specifying dependencies in Markdown files [¶](\#specifying-dependencies-in-markdown-files "Permanent link")

Sandboxing support is also provided in [marimo's markdown file\
format](../../editor_features/watching/#as-markdown) under the
`pyproject` entry of your frontmatter.

Back to top

# Inlining dependencies [¶](\#inlining-dependencies "Permanent link")

marimo is the only Python notebook that is reproducible down to the packages,
letting you inline Python dependencies in notebook files and running notebooks
in isolated or "sandboxed" venvs. This lets you share standalone notebooks
without shipping `requirements.txt` files alongside them, and guarantees your
notebooks will work weeks, months, even years into the future.

To opt-in to dependency inlining, use the `sandbox` flag:

editrunnew

When running with `--sandbox`, marimo:

1. tracks the packages and versions used by your notebook, saving
    them in the notebook file;
2. runs in an isolated virtual environment ("sandbox") that only
    contains the notebook dependencies.

marimo's sandbox provides two key benefits. (1) Notebooks that carry their own
dependencies are easy to share — just send the `.py` file. (2) Isolating a
notebook from other installed packages prevents obscure bugs.

You can also run sandboxed notebooks as scripts:

Requires uv

Sandboxed notebooks require the uv package manager
( [installation\
instructions](https://docs.astral.sh/uv/getting-started/installation/)).

Solving the notebook reproducibility crisis

marimo's support for package sandboxing is only possible because marimo
notebooks are stored as pure Python files, letting marimo take advantage
of new Python standards like [PEP\
723](https://peps.python.org/pep-0723/) and tools like uv. In contrast,
traditional notebooks like Jupyter are stored as JSON files, and which suffer
from a [reproducibility\
crisis](https://leomurta.github.io/papers/pimentel2019a.pdf) due to the lack
of package management.

## Inline script metadata [¶](\#auto-tracking-inline-script-metadata "Permanent link")

When running with `--sandbox`, marimo automatically tracks package metadata in
your notebook file using inline script metadata, which per [PEP\
723](https://peps.python.org/pep-0723/) is essentially a pyproject.toml inlined
as the script's header. This metadata is used to manage the
notebook's dependencies and Python version, and looks something like this:

Example notebooks

The [example\
notebooks](https://github.com/marimo-team/marimo/tree/main/examples) in our
GitHub repo were all created using `--sandbox`. Take a look at any of them
for an example of the full script metadata.

### Adding and removing packages [¶](\#adding-and-removing-packages "Permanent link")

**Using the marimo editor.** When you import a module in the marimo editor, if
marimo detects that it is a third-party package, it will automatically be added
to the script metadata. Removing an import does _not_ remove it from the script
metadata (since library code may still use the package).

Adding packages via the package manager panel will also add packages to script
metadata, and removing packages from the panel will in turn remove them from
the script metadata.

**Adding packages manually.** You can manually manage your notebook's
requirements:

- edit the script metadata manually in an editor like VS Code or neovim.
- use `uv` from the command-line:

### Package locations [¶](\#package-locations "Permanent link")

By default, marimo will look for packages on PyPI. You can edit the script
metadata to look for packages elsewhere, such as on GitHub. Consult the [Python\
packaging\
documentation](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#examples)
for more information.

### Local development with editable installs [¶](\#local-development-with-editable-installs "Permanent link")

When developing a local package, you can install it in editable mode using the
`[tool.uv.sources]` section in the script metadata. For example:

This is particularly useful when you want to test changes to your package
without reinstalling it. The package will be installed in "editable" mode,
meaning changes to the source code will be reflected immediately in your
notebook.

### Specifying alternative package indexes [¶](\#specifying-alternative-package-indexes "Permanent link")

When you need to use packages from a custom PyPI server or alternative index,
you can specify these in your script metadata using the `[[tool.uv.index]]`
section. This is useful for private packages or when you want to use packages
from a specific source.

In this example:

- `[[tool.uv.index]]` defines a custom package index
- `name` is an identifier for the index
- `url` points to your custom PyPI server
- `explicit = true` means this index will only be used for packages explicitly associated with it
- `[tool.uv.sources]` specifies which packages should come from which indexes

This approach ensures that specific packages are always fetched from your designated custom index, while other packages continue to be fetched from the default PyPI repository.

## Configuration [¶](\#configuration "Permanent link")

Running marimo in a sandbox environment uses `uv` to create an isolated virtual
environment. You can use any of `uv`'s [supported environment\
variables](https://docs.astral.sh/uv/configuration/environment/).

### Choosing the Python version [¶](\#choosing-the-python-version "Permanent link")

For example, you can specify the Python version using the `UV_PYTHON` environment variable:

### Other common configuration [¶](\#other-common-configuration "Permanent link")

Another common configuration is `uv`'s link mode:

## Sharing on the web [¶](\#sharing-on-the-web "Permanent link")

You can also upload sandboxed notebooks to the web, such as on GitHub, and have
others run them locally with a single command:

**Note:**

1. This command will run code from a URL. Make sure you trust the source before proceeding.
2. Upon execution, you’ll be prompted:

    To proceed securely, ensure you have [Docker](https://www.docker.com/) installed and running, then press `Y`.

## Specifying dependencies in Markdown files [¶](\#specifying-dependencies-in-markdown-files "Permanent link")

Sandboxing support is also provided in [marimo's markdown file\
format](../../editor_features/watching/#as-markdown) under the
`pyproject` entry of your frontmatter.

 Back to top

# Inlining dependencies [¶](\#inlining-dependencies "Permanent link")

marimo is the only Python notebook that is reproducible down to the packages,
letting you inline Python dependencies in notebook files and running notebooks
in isolated or "sandboxed" venvs. This lets you share standalone notebooks
without shipping `requirements.txt` files alongside them, and guarantees your
notebooks will work weeks, months, even years into the future.

To opt-in to dependency inlining, use the `sandbox` flag:

editrunnew

When running with `--sandbox`, marimo:

1. tracks the packages and versions used by your notebook, saving
    them in the notebook file;
2. runs in an isolated virtual environment ("sandbox") that only
    contains the notebook dependencies.

marimo's sandbox provides two key benefits. (1) Notebooks that carry their own
dependencies are easy to share — just send the `.py` file. (2) Isolating a
notebook from other installed packages prevents obscure bugs.

You can also run sandboxed notebooks as scripts:

Requires uv

Sandboxed notebooks require the uv package manager
( [installation\
instructions](https://docs.astral.sh/uv/getting-started/installation/)).

Solving the notebook reproducibility crisis

marimo's support for package sandboxing is only possible because marimo
notebooks are stored as pure Python files, letting marimo take advantage
of new Python standards like [PEP\
723](https://peps.python.org/pep-0723/) and tools like uv. In contrast,
traditional notebooks like Jupyter are stored as JSON files, and which suffer
from a [reproducibility\
crisis](https://leomurta.github.io/papers/pimentel2019a.pdf) due to the lack
of package management.

## Inline script metadata [¶](\#auto-tracking-inline-script-metadata "Permanent link")

When running with `--sandbox`, marimo automatically tracks package metadata in
your notebook file using inline script metadata, which per [PEP\
723](https://peps.python.org/pep-0723/) is essentially a pyproject.toml inlined
as the script's header. This metadata is used to manage the
notebook's dependencies and Python version, and looks something like this:

Example notebooks

The [example\
notebooks](https://github.com/marimo-team/marimo/tree/main/examples) in our
GitHub repo were all created using `--sandbox`. Take a look at any of them
for an example of the full script metadata.

### Adding and removing packages [¶](\#adding-and-removing-packages "Permanent link")

**Using the marimo editor.** When you import a module in the marimo editor, if
marimo detects that it is a third-party package, it will automatically be added
to the script metadata. Removing an import does _not_ remove it from the script
metadata (since library code may still use the package).

Adding packages via the package manager panel will also add packages to script
metadata, and removing packages from the panel will in turn remove them from
the script metadata.

**Adding packages manually.** You can manually manage your notebook's
requirements:

- edit the script metadata manually in an editor like VS Code or neovim.
- use `uv` from the command-line:

### Package locations [¶](\#package-locations "Permanent link")

By default, marimo will look for packages on PyPI. You can edit the script
metadata to look for packages elsewhere, such as on GitHub. Consult the [Python\
packaging\
documentation](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#examples)
for more information.

### Local development with editable installs [¶](\#local-development-with-editable-installs "Permanent link")

When developing a local package, you can install it in editable mode using the
`[tool.uv.sources]` section in the script metadata. For example:

This is particularly useful when you want to test changes to your package
without reinstalling it. The package will be installed in "editable" mode,
meaning changes to the source code will be reflected immediately in your
notebook.

### Specifying alternative package indexes [¶](\#specifying-alternative-package-indexes "Permanent link")

When you need to use packages from a custom PyPI server or alternative index,
you can specify these in your script metadata using the `[[tool.uv.index]]`
section. This is useful for private packages or when you want to use packages
from a specific source.

In this example:

- `[[tool.uv.index]]` defines a custom package index
- `name` is an identifier for the index
- `url` points to your custom PyPI server
- `explicit = true` means this index will only be used for packages explicitly associated with it
- `[tool.uv.sources]` specifies which packages should come from which indexes

This approach ensures that specific packages are always fetched from your designated custom index, while other packages continue to be fetched from the default PyPI repository.

## Configuration [¶](\#configuration "Permanent link")

Running marimo in a sandbox environment uses `uv` to create an isolated virtual
environment. You can use any of `uv`'s [supported environment\
variables](https://docs.astral.sh/uv/configuration/environment/).

### Choosing the Python version [¶](\#choosing-the-python-version "Permanent link")

For example, you can specify the Python version using the `UV_PYTHON` environment variable:

### Other common configuration [¶](\#other-common-configuration "Permanent link")

Another common configuration is `uv`'s link mode:

## Sharing on the web [¶](\#sharing-on-the-web "Permanent link")

You can also upload sandboxed notebooks to the web, such as on GitHub, and have
others run them locally with a single command:

**Note:**

1. This command will run code from a URL. Make sure you trust the source before proceeding.
2. Upon execution, you’ll be prompted:

    To proceed securely, ensure you have [Docker](https://www.docker.com/) installed and running, then press `Y`.

## Specifying dependencies in Markdown files [¶](\#specifying-dependencies-in-markdown-files "Permanent link")

Sandboxing support is also provided in [marimo's markdown file\
format](../../editor_features/watching/#as-markdown) under the
`pyproject` entry of your frontmatter.


---

###### Using uv - marimoUsing uv¶

*Source: [https://docs.marimo.io/guides/package_management/using_uv/](https://docs.marimo.io/guides/package_management/using_uv/)*

<!-- Source: https://docs.marimo.io/guides/package_management/using_uv/ -->
<!-- Title: Using uv - marimoUsing uv¶ -->
<!-- Depth: 8 -->

[Skip to content](#using-uv)

# Using uv [¶](\#using-uv "Permanent link")

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package and
project manager: you can use it to install packages, manage the dependencies
of Python projects, and run scripts. While marimo supports all major package
managers, it integrates especially tightly with uv. In particular, marimo's
package sandbox feature, which lets you [inline\
dependencies](../inlining_dependencies/) in notebook files, requires uv.

No prior knowledge required

This guide teaches you the basics of using `uv` with marimo. It assumes zero
familiarity with `uv`.

You can manage your notebooks' dependencies in three different ways:

1. inline dependencies: [inlining dependencies](../inlining_dependencies/) in notebook files, using `marimo edit --sandbox notebook.py`
2. projects: using a `uv` project , which define dependencies declaratively in a `pyproject.toml`
    file;
3. non-project environment: dependencies are imperatively installed

We'll walk through each of these three ways in this guide.

## Using inline dependencies [¶](\#using-inline-dependencies "Permanent link")

The easiest way to get started is to use marimo's [package sandbox\
feature](../inlining_dependencies/), which manages your dependencies for you.
Create or edit your notebook with [`uvx`\
command](https://docs.astral.sh/uv/concepts/tools/#the-uv-tool-interface),
making sure to include the `--sandbox` flag:

This command installs marimo in a temporary environment, activates it, then
runs your marimo notebook. The `--sandbox` flag is what tells marimo to keep
track of your dependencies and store them in the notebook file. If there are
any dependencies already tracked in the file, this command will download
them and install them in the environment.

Run sandboxed notebooks as scripts with

### From URLs [¶](\#from-urls "Permanent link")

You can also upload sandboxed notebooks to the web, such as on GitHub, and have others run
them locally with a single command:

**Note:**

1. This command will run code from a URL. Make sure you trust the source before proceeding.
2. Upon execution, you’ll be prompted:

    To proceed securely, ensure you have [Docker](https://www.docker.com/) installed and running, then press `Y`.

To learn more, read our full guide on using [inline dependencies](../inlining_dependencies/).

## Using uv projects [¶](\#using-uv-projects "Permanent link")

A [`uv` project](https://docs.astral.sh/uv/guides/projects/) is a directory in which you can store Python code, including
notebooks, alongside a pyproject.toml file that declares the project's
dependencies.

### Creating a project [¶](\#creating-a-project "Permanent link")

Create a project with `uv init`:

Starter template

Get started quickly by cloning our [starter template](https://github.com/marimo-team/marimo-uv-starter-template).

This creates a pyproject.toml and some starter code.

Next, add marimo to your project:

Omitting marimo from your project

Adding marimo to your project is optional. Instead, you can
run marimo in a temporary environment that has access to
your project's dependencies using `uv run --with marimo marimo edit`.

### Running marimo [¶](\#running-marimo "Permanent link")

Once you've added marimo, use the `uv run` command
to run the version of marimo installed in your project:

Starting marimo in this way will let marimo import any of the packages
installed in your project.

**Scripts.** Run marimo notebooks as scripts with

which will run your notebook in an environment containing your project dependencies.

### Adding and removing dependencies [¶](\#adding-and-removing-dependencies "Permanent link")

#### Using the uv command-line [¶](\#using-the-uv-command-line "Permanent link")

Use `uv add` to add dependencies:

You can also specify a version

Remove packages with `uv remove`:

#### Using the marimo editor [¶](\#using-the-marimo-editor "Permanent link")

If you started marimo with `uv run marimo edit`, the marimo editor's [package\
management features](../installing_packages/) will add and remove packages from
your pyproject.toml, so there's no need to use the `uv` command-line if you
don't want to.

## Using marimo in a non-project environment [¶](\#using-marimo-in-a-non-project-environment "Permanent link")

If you are used to a venv and pip based workflow, you can use the `uv venv` and
`uv pip` commands for a similar but more performant experience:

- `uv venv` creates a virtual environment in the current directory, at `.venv`
- `uv pip` lets you install and uninstall packages in the venv

### Example [¶](\#example "Permanent link")

From here, `import numpy` will work within the notebook, and marimo's UI installer will add
packages to the environment with `uv pip install` on your behalf.

Back to top

# Using uv [¶](\#using-uv "Permanent link")

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package and
project manager: you can use it to install packages, manage the dependencies
of Python projects, and run scripts. While marimo supports all major package
managers, it integrates especially tightly with uv. In particular, marimo's
package sandbox feature, which lets you [inline\
dependencies](../inlining_dependencies/) in notebook files, requires uv.

No prior knowledge required

This guide teaches you the basics of using `uv` with marimo. It assumes zero
familiarity with `uv`.

You can manage your notebooks' dependencies in three different ways:

1. inline dependencies: [inlining dependencies](../inlining_dependencies/) in notebook files, using `marimo edit --sandbox notebook.py`
2. projects: using a `uv` project , which define dependencies declaratively in a `pyproject.toml`
    file;
3. non-project environment: dependencies are imperatively installed

We'll walk through each of these three ways in this guide.

## Using inline dependencies [¶](\#using-inline-dependencies "Permanent link")

The easiest way to get started is to use marimo's [package sandbox\
feature](../inlining_dependencies/), which manages your dependencies for you.
Create or edit your notebook with [`uvx`\
command](https://docs.astral.sh/uv/concepts/tools/#the-uv-tool-interface),
making sure to include the `--sandbox` flag:

This command installs marimo in a temporary environment, activates it, then
runs your marimo notebook. The `--sandbox` flag is what tells marimo to keep
track of your dependencies and store them in the notebook file. If there are
any dependencies already tracked in the file, this command will download
them and install them in the environment.

Run sandboxed notebooks as scripts with

### From URLs [¶](\#from-urls "Permanent link")

You can also upload sandboxed notebooks to the web, such as on GitHub, and have others run
them locally with a single command:

**Note:**

1. This command will run code from a URL. Make sure you trust the source before proceeding.
2. Upon execution, you’ll be prompted:

    To proceed securely, ensure you have [Docker](https://www.docker.com/) installed and running, then press `Y`.

To learn more, read our full guide on using [inline dependencies](../inlining_dependencies/).

## Using uv projects [¶](\#using-uv-projects "Permanent link")

A [`uv` project](https://docs.astral.sh/uv/guides/projects/) is a directory in which you can store Python code, including
notebooks, alongside a pyproject.toml file that declares the project's
dependencies.

### Creating a project [¶](\#creating-a-project "Permanent link")

Create a project with `uv init`:

Starter template

Get started quickly by cloning our [starter template](https://github.com/marimo-team/marimo-uv-starter-template).

This creates a pyproject.toml and some starter code.

Next, add marimo to your project:

Omitting marimo from your project

Adding marimo to your project is optional. Instead, you can
run marimo in a temporary environment that has access to
your project's dependencies using `uv run --with marimo marimo edit`.

### Running marimo [¶](\#running-marimo "Permanent link")

Once you've added marimo, use the `uv run` command
to run the version of marimo installed in your project:

Starting marimo in this way will let marimo import any of the packages
installed in your project.

**Scripts.** Run marimo notebooks as scripts with

which will run your notebook in an environment containing your project dependencies.

### Adding and removing dependencies [¶](\#adding-and-removing-dependencies "Permanent link")

#### Using the uv command-line [¶](\#using-the-uv-command-line "Permanent link")

Use `uv add` to add dependencies:

You can also specify a version

Remove packages with `uv remove`:

#### Using the marimo editor [¶](\#using-the-marimo-editor "Permanent link")

If you started marimo with `uv run marimo edit`, the marimo editor's [package\
management features](../installing_packages/) will add and remove packages from
your pyproject.toml, so there's no need to use the `uv` command-line if you
don't want to.

## Using marimo in a non-project environment [¶](\#using-marimo-in-a-non-project-environment "Permanent link")

If you are used to a venv and pip based workflow, you can use the `uv venv` and
`uv pip` commands for a similar but more performant experience:

- `uv venv` creates a virtual environment in the current directory, at `.venv`
- `uv pip` lets you install and uninstall packages in the venv

### Example [¶](\#example "Permanent link")

From here, `import numpy` will work within the notebook, and marimo's UI installer will add
packages to the environment with `uv pip install` on your behalf.

 Back to top

# Using uv [¶](\#using-uv "Permanent link")

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package and
project manager: you can use it to install packages, manage the dependencies
of Python projects, and run scripts. While marimo supports all major package
managers, it integrates especially tightly with uv. In particular, marimo's
package sandbox feature, which lets you [inline\
dependencies](../inlining_dependencies/) in notebook files, requires uv.

No prior knowledge required

This guide teaches you the basics of using `uv` with marimo. It assumes zero
familiarity with `uv`.

You can manage your notebooks' dependencies in three different ways:

1. inline dependencies: [inlining dependencies](../inlining_dependencies/) in notebook files, using `marimo edit --sandbox notebook.py`
2. projects: using a `uv` project , which define dependencies declaratively in a `pyproject.toml`
    file;
3. non-project environment: dependencies are imperatively installed

We'll walk through each of these three ways in this guide.

## Using inline dependencies [¶](\#using-inline-dependencies "Permanent link")

The easiest way to get started is to use marimo's [package sandbox\
feature](../inlining_dependencies/), which manages your dependencies for you.
Create or edit your notebook with [`uvx`\
command](https://docs.astral.sh/uv/concepts/tools/#the-uv-tool-interface),
making sure to include the `--sandbox` flag:

This command installs marimo in a temporary environment, activates it, then
runs your marimo notebook. The `--sandbox` flag is what tells marimo to keep
track of your dependencies and store them in the notebook file. If there are
any dependencies already tracked in the file, this command will download
them and install them in the environment.

Run sandboxed notebooks as scripts with

### From URLs [¶](\#from-urls "Permanent link")

You can also upload sandboxed notebooks to the web, such as on GitHub, and have others run
them locally with a single command:

**Note:**

1. This command will run code from a URL. Make sure you trust the source before proceeding.
2. Upon execution, you’ll be prompted:

    To proceed securely, ensure you have [Docker](https://www.docker.com/) installed and running, then press `Y`.

To learn more, read our full guide on using [inline dependencies](../inlining_dependencies/).

## Using uv projects [¶](\#using-uv-projects "Permanent link")

A [`uv` project](https://docs.astral.sh/uv/guides/projects/) is a directory in which you can store Python code, including
notebooks, alongside a pyproject.toml file that declares the project's
dependencies.

### Creating a project [¶](\#creating-a-project "Permanent link")

Create a project with `uv init`:

Starter template

Get started quickly by cloning our [starter template](https://github.com/marimo-team/marimo-uv-starter-template).

This creates a pyproject.toml and some starter code.

Next, add marimo to your project:

Omitting marimo from your project

Adding marimo to your project is optional. Instead, you can
run marimo in a temporary environment that has access to
your project's dependencies using `uv run --with marimo marimo edit`.

### Running marimo [¶](\#running-marimo "Permanent link")

Once you've added marimo, use the `uv run` command
to run the version of marimo installed in your project:

Starting marimo in this way will let marimo import any of the packages
installed in your project.

**Scripts.** Run marimo notebooks as scripts with

which will run your notebook in an environment containing your project dependencies.

### Adding and removing dependencies [¶](\#adding-and-removing-dependencies "Permanent link")

#### Using the uv command-line [¶](\#using-the-uv-command-line "Permanent link")

Use `uv add` to add dependencies:

You can also specify a version

Remove packages with `uv remove`:

#### Using the marimo editor [¶](\#using-the-marimo-editor "Permanent link")

If you started marimo with `uv run marimo edit`, the marimo editor's [package\
management features](../installing_packages/) will add and remove packages from
your pyproject.toml, so there's no need to use the `uv` command-line if you
don't want to.

## Using marimo in a non-project environment [¶](\#using-marimo-in-a-non-project-environment "Permanent link")

If you are used to a venv and pip based workflow, you can use the `uv venv` and
`uv pip` commands for a similar but more performant experience:

- `uv venv` creates a virtual environment in the current directory, at `.venv`
- `uv pip` lets you install and uninstall packages in the venv

### Example [¶](\#example "Permanent link")

From here, `import numpy` will work within the notebook, and marimo's UI installer will add
packages to the environment with `uv pip install` on your behalf.


---

###### Community Cloud - marimoCommunity Cloud¶

*Source: [https://docs.marimo.io/guides/publishing/community_cloud/](https://docs.marimo.io/guides/publishing/community_cloud/)*

<!-- Source: https://docs.marimo.io/guides/publishing/community_cloud/ -->
<!-- Title: Community Cloud - marimoCommunity Cloud¶ -->
<!-- Depth: 8 -->

[Skip to content](#community-cloud)

# Community Cloud [¶](\#community-cloud "Permanent link")

Our [Community Cloud](https://marimo.io/dashboard) is a free workspace
for creating, saving, and sharing marimo notebooks. Unlike the
[Playground](../playground/), the Community Cloud requires a login. In
return, it lets you save noteoboks, share them using email-based authorization,
and upload a limited amount of data.

WebAssembly notebooks only

Currently, the Community Cloud only allows the creation of [WebAssembly\
notebooks](../../wasm/). These are easy to share and embed in other
web pages, but have some limitations in packages and performance.

Note: unlike our other publishing options, it is not possible to embed
editable Community Cloud notebooks in other web pages.

Back to top

# Community Cloud [¶](\#community-cloud "Permanent link")

Our [Community Cloud](https://marimo.io/dashboard) is a free workspace
for creating, saving, and sharing marimo notebooks. Unlike the
[Playground](../playground/), the Community Cloud requires a login. In
return, it lets you save noteoboks, share them using email-based authorization,
and upload a limited amount of data.

WebAssembly notebooks only

Currently, the Community Cloud only allows the creation of [WebAssembly\
notebooks](../../wasm/). These are easy to share and embed in other
web pages, but have some limitations in packages and performance.

Note: unlike our other publishing options, it is not possible to embed
editable Community Cloud notebooks in other web pages.

 Back to top

# Community Cloud [¶](\#community-cloud "Permanent link")

Our [Community Cloud](https://marimo.io/dashboard) is a free workspace
for creating, saving, and sharing marimo notebooks. Unlike the
[Playground](../playground/), the Community Cloud requires a login. In
return, it lets you save noteoboks, share them using email-based authorization,
and upload a limited amount of data.

WebAssembly notebooks only

Currently, the Community Cloud only allows the creation of [WebAssembly\
notebooks](../../wasm/). These are easy to share and embed in other
web pages, but have some limitations in packages and performance.

Note: unlike our other publishing options, it is not possible to embed
editable Community Cloud notebooks in other web pages.


---

###### GitHub Pages - marimoPublish to GitHub Pages¶

*Source: [https://docs.marimo.io/guides/publishing/github_pages/](https://docs.marimo.io/guides/publishing/github_pages/)*

<!-- Source: https://docs.marimo.io/guides/publishing/github_pages/ -->
<!-- Title: GitHub Pages - marimoPublish to GitHub Pages¶ -->
<!-- Depth: 8 -->

[Skip to content](#publish-to-github-pages)

# Publish to GitHub Pages [¶](\#publish-to-github-pages "Permanent link")

You can publish executable notebooks to [GitHub Pages](https://pages.github.com/)
for free, after exporting your notebook to a WebAssembly notebook.

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../../wasm/):

Export as a readonly appExport as an editable notebook

See our [exporting guide](../../exporting/#export-to-wasm-powered-html) for
the full documentation.

## Publish using GitHub Actions [¶](\#publish-using-github-actions "Permanent link")

Template repository

Fork our [template repository](https://github.com/marimo-team/marimo-gh-pages-template) for deploying multiple notebooks to GitHub Pages. Once you have forked the repository, add your notebooks to the `notebooks` or `apps` directories,
for editable or readonly respectively.

Publish to GitHub Pages using the following GitHub Actions workflow,
which will republish your notebook on git push.

## Publish manually [¶](\#publish-manually "Permanent link")

You can also publish an exported notebook manually through your repository
settings. Read [GitHub's documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) to learn more.

Make sure to [include a `.nojekyll`\
file](https://github.blog/news-insights/bypassing-jekyll-on-github-pages/) in
root folder from which your site is built to prevent GitHub from interfering
with your site.

Back to top

# Publish to GitHub Pages [¶](\#publish-to-github-pages "Permanent link")

You can publish executable notebooks to [GitHub Pages](https://pages.github.com/)
for free, after exporting your notebook to a WebAssembly notebook.

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../../wasm/):

Export as a readonly appExport as an editable notebook

See our [exporting guide](../../exporting/#export-to-wasm-powered-html) for
the full documentation.

## Publish using GitHub Actions [¶](\#publish-using-github-actions "Permanent link")

Template repository

Fork our [template repository](https://github.com/marimo-team/marimo-gh-pages-template) for deploying multiple notebooks to GitHub Pages. Once you have forked the repository, add your notebooks to the `notebooks` or `apps` directories,
for editable or readonly respectively.

Publish to GitHub Pages using the following GitHub Actions workflow,
which will republish your notebook on git push.

## Publish manually [¶](\#publish-manually "Permanent link")

You can also publish an exported notebook manually through your repository
settings. Read [GitHub's documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) to learn more.

Make sure to [include a `.nojekyll`\
file](https://github.blog/news-insights/bypassing-jekyll-on-github-pages/) in
root folder from which your site is built to prevent GitHub from interfering
with your site.

 Back to top

# Publish to GitHub Pages [¶](\#publish-to-github-pages "Permanent link")

You can publish executable notebooks to [GitHub Pages](https://pages.github.com/)
for free, after exporting your notebook to a WebAssembly notebook.

## Export to WASM-powered HTML [¶](\#export-to-wasm-powered-html "Permanent link")

Export your notebook to a self-contained HTML file that runs using [WebAssembly](../../wasm/):

Export as a readonly appExport as an editable notebook

See our [exporting guide](../../exporting/#export-to-wasm-powered-html) for
the full documentation.

## Publish using GitHub Actions [¶](\#publish-using-github-actions "Permanent link")

Template repository

Fork our [template repository](https://github.com/marimo-team/marimo-gh-pages-template) for deploying multiple notebooks to GitHub Pages. Once you have forked the repository, add your notebooks to the `notebooks` or `apps` directories,
for editable or readonly respectively.

Publish to GitHub Pages using the following GitHub Actions workflow,
which will republish your notebook on git push.

## Publish manually [¶](\#publish-manually "Permanent link")

You can also publish an exported notebook manually through your repository
settings. Read [GitHub's documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) to learn more.

Make sure to [include a `.nojekyll`\
file](https://github.blog/news-insights/bypassing-jekyll-on-github-pages/) in
root folder from which your site is built to prevent GitHub from interfering
with your site.


---

###### Publishing with MkDocs - marimoPublishing with MkDocs¶

*Source: [https://docs.marimo.io/guides/publishing/mkdocs/](https://docs.marimo.io/guides/publishing/mkdocs/)*

<!-- Source: https://docs.marimo.io/guides/publishing/mkdocs/ -->
<!-- Title: Publishing with MkDocs - marimoPublishing with MkDocs¶ -->
<!-- Depth: 8 -->

[Skip to content](#publishing-with-mkdocs)

# Publishing with MkDocs [¶](\#publishing-with-mkdocs "Permanent link")

[MkDocs](https://www.mkdocs.org/) is a widely used static site generator that uses plain markdown files as page inputs.

You can use MkDocs to publish your marimo notebooks as a interactive website with marimo's [MkDocs extension](https://github.com/marimo-team/mkdocs-marimo).

Try the markdown file format tutorial

Learn more with `marimo tutorial markdown-format` at the command line.

Alternatively, see our [Quarto guide](../quarto/) for other methods of publishing marimo's markdown format.

Back to top

# Publishing with MkDocs [¶](\#publishing-with-mkdocs "Permanent link")

[MkDocs](https://www.mkdocs.org/) is a widely used static site generator that uses plain markdown files as page inputs.

You can use MkDocs to publish your marimo notebooks as a interactive website with marimo's [MkDocs extension](https://github.com/marimo-team/mkdocs-marimo).

Try the markdown file format tutorial

Learn more with `marimo tutorial markdown-format` at the command line.

Alternatively, see our [Quarto guide](../quarto/) for other methods of publishing marimo's markdown format.

 Back to top

# Publishing with MkDocs [¶](\#publishing-with-mkdocs "Permanent link")

[MkDocs](https://www.mkdocs.org/) is a widely used static site generator that uses plain markdown files as page inputs.

You can use MkDocs to publish your marimo notebooks as a interactive website with marimo's [MkDocs extension](https://github.com/marimo-team/mkdocs-marimo).

Try the markdown file format tutorial

Learn more with `marimo tutorial markdown-format` at the command line.

Alternatively, see our [Quarto guide](../quarto/) for other methods of publishing marimo's markdown format.


---

###### Reuse functions and classes - marimoImporting functions and classes defined in notebooks¶

*Source: [https://docs.marimo.io/guides/reusing_functions/](https://docs.marimo.io/guides/reusing_functions/)*

<!-- Source: https://docs.marimo.io/guides/reusing_functions/ -->
<!-- Title: Reuse functions and classes - marimoImporting functions and classes defined in notebooks¶ -->
<!-- Depth: 8 -->

[Skip to content](#importing-functions-and-classes-defined-in-notebooks)

# Importing functions and classes defined in notebooks [¶](\#importing-functions-and-classes-defined-in-notebooks "Permanent link")

You can import top-level functions and classes defined in a marimo notebook
into other Python scripts or notebooks using normal Python syntax, as long as
your definitions satisfy the simple criteria described on this page. This makes
your notebook code reusable, testable, and easier to edit in text editors
of your choice.

## Overview [¶](\#overview "Permanent link")

For a function or class to be saved at the top level of the notebook file, it must
meet the following **criteria**:

1. The cell must define just a single function or class.
2. The defined function or class can only refer to symbols defined in the
    [setup cell](#1-create-a-setup-cell), or to other top-level symbols.

### Example [¶](\#example "Permanent link")

In another script or notebook

## Creating a top-level function or class [¶](\#creating-a-top-level-function-or-class "Permanent link")

### 1\. Create a setup cell [¶](\#1-create-a-setup-cell "Permanent link")

First, add a **setup cell** to your notebook for imports that your functions or
classes will need:

To add a setup cell in the editor, open the notebook menu and select "Add setup cell".
(The setup cell is guaranteed to run before other cells.)

### 2\. Define your function [¶](\#2-define-your-function "Permanent link")

Define a single function in a cell. If the
[criteria](#overview) for top-level
functions are met, a marker in the bottom right will indicate that it is a
reusable function.

Note

Functions can only reference symbols defined in the setup cell. If a
function cannot be serialized top-level, the marker in the
bottom right will provide a description why.

Under the hood, marimo decorates top-level functions with `@app.function`,
which you can use to define your own top-level functions if you are editing a
notebook file directly. Top-level classes are decorated with `@app.class_definition`.

### 3\. Import into other Python files [¶](\#3-import-into-other-python-files "Permanent link")

Now you can import your function in other notebooks or Python scripts:

## Best practices [¶](\#best-practices "Permanent link")

- Use setup cells for widely used imports
- Keep function dependencies limited to setup-cell references, or other top-level declarations
- Use descriptive names for your functions
- Add docstrings to document your functions' behavior

Tip

Top-level symbols can reference other top-level symbols.

## Limitations [¶](\#limitations "Permanent link")

- Functions cannot depend on variables defined in regular cells
- Like other cells, cyclic dependencies between functions are not allowed
- Functions cannot be exported from notebooks in [marimo's markdown format](../editor_features/watching/#as-markdown).

## Learn more [¶](\#learn-more "Permanent link")

For more on marimo's file format, check out our [documentation on using your\
own editor](../editor_features/watching/) or view
our [file format tutorial](https://links.marimo.app/tutorial-fileformat).

Back to top

# Importing functions and classes defined in notebooks [¶](\#importing-functions-and-classes-defined-in-notebooks "Permanent link")

You can import top-level functions and classes defined in a marimo notebook
into other Python scripts or notebooks using normal Python syntax, as long as
your definitions satisfy the simple criteria described on this page. This makes
your notebook code reusable, testable, and easier to edit in text editors
of your choice.

## Overview [¶](\#overview "Permanent link")

For a function or class to be saved at the top level of the notebook file, it must
meet the following **criteria**:

1. The cell must define just a single function or class.
2. The defined function or class can only refer to symbols defined in the
    [setup cell](#1-create-a-setup-cell), or to other top-level symbols.

### Example [¶](\#example "Permanent link")

In another script or notebook

## Creating a top-level function or class [¶](\#creating-a-top-level-function-or-class "Permanent link")

### 1\. Create a setup cell [¶](\#1-create-a-setup-cell "Permanent link")

First, add a **setup cell** to your notebook for imports that your functions or
classes will need:

To add a setup cell in the editor, open the notebook menu and select "Add setup cell".
(The setup cell is guaranteed to run before other cells.)

### 2\. Define your function [¶](\#2-define-your-function "Permanent link")

Define a single function in a cell. If the
[criteria](#overview) for top-level
functions are met, a marker in the bottom right will indicate that it is a
reusable function.

Note

Functions can only reference symbols defined in the setup cell. If a
function cannot be serialized top-level, the marker in the
bottom right will provide a description why.

Under the hood, marimo decorates top-level functions with `@app.function`,
which you can use to define your own top-level functions if you are editing a
notebook file directly. Top-level classes are decorated with `@app.class_definition`.

### 3\. Import into other Python files [¶](\#3-import-into-other-python-files "Permanent link")

Now you can import your function in other notebooks or Python scripts:

## Best practices [¶](\#best-practices "Permanent link")

- Use setup cells for widely used imports
- Keep function dependencies limited to setup-cell references, or other top-level declarations
- Use descriptive names for your functions
- Add docstrings to document your functions' behavior

Tip

Top-level symbols can reference other top-level symbols.

## Limitations [¶](\#limitations "Permanent link")

- Functions cannot depend on variables defined in regular cells
- Like other cells, cyclic dependencies between functions are not allowed
- Functions cannot be exported from notebooks in [marimo's markdown format](../editor_features/watching/#as-markdown).

## Learn more [¶](\#learn-more "Permanent link")

For more on marimo's file format, check out our [documentation on using your\
own editor](../editor_features/watching/) or view
our [file format tutorial](https://links.marimo.app/tutorial-fileformat).

 Back to top

# Importing functions and classes defined in notebooks [¶](\#importing-functions-and-classes-defined-in-notebooks "Permanent link")

You can import top-level functions and classes defined in a marimo notebook
into other Python scripts or notebooks using normal Python syntax, as long as
your definitions satisfy the simple criteria described on this page. This makes
your notebook code reusable, testable, and easier to edit in text editors
of your choice.

## Overview [¶](\#overview "Permanent link")

For a function or class to be saved at the top level of the notebook file, it must
meet the following **criteria**:

1. The cell must define just a single function or class.
2. The defined function or class can only refer to symbols defined in the
    [setup cell](#1-create-a-setup-cell), or to other top-level symbols.

### Example [¶](\#example "Permanent link")

In another script or notebook

## Creating a top-level function or class [¶](\#creating-a-top-level-function-or-class "Permanent link")

### 1\. Create a setup cell [¶](\#1-create-a-setup-cell "Permanent link")

First, add a **setup cell** to your notebook for imports that your functions or
classes will need:

To add a setup cell in the editor, open the notebook menu and select "Add setup cell".
(The setup cell is guaranteed to run before other cells.)

### 2\. Define your function [¶](\#2-define-your-function "Permanent link")

Define a single function in a cell. If the
[criteria](#overview) for top-level
functions are met, a marker in the bottom right will indicate that it is a
reusable function.

Note

Functions can only reference symbols defined in the setup cell. If a
function cannot be serialized top-level, the marker in the
bottom right will provide a description why.

Under the hood, marimo decorates top-level functions with `@app.function`,
which you can use to define your own top-level functions if you are editing a
notebook file directly. Top-level classes are decorated with `@app.class_definition`.

### 3\. Import into other Python files [¶](\#3-import-into-other-python-files "Permanent link")

Now you can import your function in other notebooks or Python scripts:

## Best practices [¶](\#best-practices "Permanent link")

- Use setup cells for widely used imports
- Keep function dependencies limited to setup-cell references, or other top-level declarations
- Use descriptive names for your functions
- Add docstrings to document your functions' behavior

Tip

Top-level symbols can reference other top-level symbols.

## Limitations [¶](\#limitations "Permanent link")

- Functions cannot depend on variables defined in regular cells
- Like other cells, cyclic dependencies between functions are not allowed
- Functions cannot be exported from notebooks in [marimo's markdown format](../editor_features/watching/#as-markdown).

## Learn more [¶](\#learn-more "Permanent link")

For more on marimo's file format, check out our [documentation on using your\
own editor](../editor_features/watching/) or view
our [file format tutorial](https://links.marimo.app/tutorial-fileformat).


---

###### Recipes - marimoRecipes¶

*Source: [https://docs.marimo.io/recipes/](https://docs.marimo.io/recipes/)*

<!-- Source: https://docs.marimo.io/recipes/ -->
<!-- Title: Recipes - marimoRecipes¶ -->
<!-- Depth: 8 -->

[Skip to content](#recipes)

# Recipes [¶](\#recipes "Permanent link")

This page includes code snippets or " **recipes**" for a variety of common tasks.
Use them as building blocks or examples when making your own notebooks.

In these recipes, **each code block represents a cell**.

## Control Flow [¶](\#control-flow "Permanent link")

### Show an output conditionally [¶](\#show-an-output-conditionally "Permanent link")

**Use cases.** Hide an output until a condition is met ( _e.g._, until algorithm
parameters are valid), or show different outputs depending on the value of a UI
element or some other Python object

**Recipe.**

1. Use an `if` expression to choose which output to show.

### Run a cell on a timer [¶](\#run-a-cell-on-a-timer "Permanent link")

**Use cases.**

- Load new data periodically, and show updated plots or other outputs. For
example, in a dashboard monitoring a training run, experiment trial,
real-time weather data, ...
- Run a job periodically

**Recipe.**

1. Import packages

1. Create a [`mo.ui.refresh`](../api/inputs/refresh/#marimo.ui.refresh "            marimo.ui.refresh") timer that fires once a second:

1. Reference the timer by name to make this cell run once a second

Requires 'on cell change' autorun

For this to work, the [runtime\
configuration's](../guides/configuration/runtime_configuration/) `on cell
change` should be set to `autorun`

### Require form submission before sending UI value [¶](\#require-form-submission-before-sending-ui-value "Permanent link")

**Use cases.** UI elements automatically send their values to the Python when
they are interacted with, and run all cells referencing the elements. This
makes marimo notebooks responsive, but it can be an issue when the
downstream cells are expensive, or when the input (such as a text box)
needs to be filled out completely before it is considered valid. Forms
let you gate submission of UI element values on manual confirmation, via a
button press.

**Recipe.**

1. Import packages

1. Create a submittable form.

1. Get the value of the form.

### Stop execution of a cell and its descendants [¶](\#stop-execution-of-a-cell-and-its-descendants "Permanent link")

**Use cases.** For example, don't run a cell or its descendants if a form is
unsubmitted.

**Recipe.**

1. Import packages

1. Create a submittable form.

1. Use [`mo.stop`](../api/control_flow/#marimo.stop "            marimo.stop") to stop execution when
the form is unsubmitted.

## Grouping UI elements together [¶](\#grouping-ui-elements-together "Permanent link")

### Create an array of UI elements [¶](\#create-an-array-of-ui-elements "Permanent link")

**Use cases.** In order to synchronize UI elements between the frontend and
backend (Python), marimo requires you to
[assign UI elements to global variables](../guides/interactivity/).
But sometimes you don't know the number of elements to make until runtime:
for example, maybe you want to make a list of sliders, and the number of sliders
to make depends on the value of some other UI element.

You might be tempted to create a Python list of UI elements,
such as `l = [mo.ui.slider(1, 10) for i in range(number.value)]`: _however,_
_this won't work, because the sliders are not bound to global variables_.

For such cases, marimo provides the "higher-order" UI element
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"), which lets you make
a new UI element out of a list of UI elements:
`l = mo.ui.array([mo.ui.slider(1, 10) for i in range(number.value)])`.
The value of an `array` element is a list of the values of the elements
it wraps (in this case, a list of the slider values). Any time you interact
with any of the UI elements in the array, all cells referencing the array
by name (in this case, " `l`") will run automatically.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array") to group together
    many UI elements into a list.

1. Get the value of the UI elements using `array.value`

### Create a dictionary of UI elements [¶](\#create-a-dictionary-of-ui-elements "Permanent link")

**Use cases.** Same as for creating an array of UI elements, but lets you
name each of the wrapped elements with a string key.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") to
    group together many UI elements into a list.

1. Get the value of the UI elements using `dictionary.value`

### Embed a dynamic number of UI elements in another output [¶](\#embed-a-dynamic-number-of-ui-elements-in-another-output "Permanent link")

**Use cases.** When you want to embed a dynamic number of UI elements
in other outputs (like tables or markdown).

**Recipe.**

1. Import packages

1. Group the elements with

[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") or
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"), then retrieve them from the container
and display them elsewhere.

1. Get the value of the elements

### Create a hstack (or vstack) of UI elements with `on_change` handlers [¶](\#create-a-hstack-or-vstack-of-ui-elements-with-on_change-handlers "Permanent link")

**Use cases.** Arrange a dynamic number of UI elements in a hstack or vstack,
for example some number of buttons, and execute some side-effect when an
element is interacted with, e.g. when a button is clicked.

**Recipe.**

1. Import packages

1. Create buttons in `mo.ui.array` and pass them to hstack -- a regular
Python list won't work. Make sure to assign the array to a global variable.

1. Get the state value

### Create a table column of buttons with `on_change` handlers [¶](\#create-a-table-column-of-buttons-with-on_change-handlers "Permanent link")

**Use cases.** Arrange a dynamic number of UI elements in a column of
a table, and execute some side-effect when an element is interacted with, e.g.
when a button is clicked.

**Recipe.**

1. Import packages

1. Create buttons in `mo.ui.array` and pass them to `mo.ui.table`.
Make sure to assign the table and array to global variables

1. Get the state value

### Create a form with multiple UI elements [¶](\#create-a-form-with-multiple-ui-elements "Permanent link")

**Use cases.** Combine multiple UI elements into a form so that submission
of the form sends all its elements to Python.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.form`](../api/inputs/form/#marimo.ui.form "            marimo.ui.form") and [`Html.batch`](../api/html/#marimo.Html.batch "            batch") to
    create a form with multiple elements.

1. Get the submitted form value.

### Populating form with pre-defined examples [¶](\#populating-form-with-pre-defined-examples "Permanent link")

**Use cases.** To give examples of how a filled form looks like. Useful for illustrating complex API requests or database queries.
The form can also be populated from [URL query parameters](../api/query_params/) ( [notebook example](https://marimo.app/l/w21a3x?t1=query&t2=params)).

**Recipe.**

1. Import packages

1. Create dropdown of examples

1. Create form from examples.

1. Run pre-populated from or recompute with new input.

## Working with buttons [¶](\#working-with-buttons "Permanent link")

### Create a button that triggers computation when clicked [¶](\#create-a-button-that-triggers-computation-when-clicked "Permanent link")

**Use cases.** To trigger a computation on button click and only on button
click, use [`mo.ui.run_button()`](../api/inputs/run_button/).

**Recipe.**

1. Import packages

1. Create a run button

1. Run something only if the button has been clicked.

### Create a counter button [¶](\#create-a-counter-button "Permanent link")

**Use cases.** A counter button, i.e. a button that counts the number of times
it has been clicked, is a helpful building block for reacting to button clicks
(see other recipes in this section).

**Recipe.**

1. Import packages

1. Use [`mo.ui.button`](../api/inputs/button/#marimo.ui.button "            marimo.ui.button") and its
    `on_click` argument to create a counter button.

1. Get the button value

### Create a toggle button [¶](\#create-a-toggle-button "Permanent link")

**Use cases.** Toggle between two states using a button with a button
that toggles between `True` and `False`. (Tip: you can also just use
[`mo.ui.switch`](../api/inputs/switch/#marimo.ui.switch "            marimo.ui.switch").)

**Recipe.**

1. Import packages

1. Use [`mo.ui.button`](../api/inputs/button/#marimo.ui.button "            marimo.ui.button") and its
    `on_click` argument to create a toggle button.

1. Toggle between two outputs using the button value.

### Re-run a cell when a button is pressed [¶](\#re-run-a-cell-when-a-button-is-pressed "Permanent link")

**Use cases.** For example, you have a cell showing a random sample of data,
and you want to resample on button press.

**Recipe.**

1. Import packages

1. Create a button without a value, to function as a _trigger_.

1. Reference the button in another cell.

### Run a cell when a button is pressed, but not before [¶](\#run-a-cell-when-a-button-is-pressed-but-not-before "Permanent link")

**Use cases.** Wait for confirmation before executing downstream cells
(similar to a form).

**Recipe.**

1. Import packages

1. Create a counter button.

1. Only execute when the count is greater than 0.

### Reveal an output when a button is pressed [¶](\#reveal-an-output-when-a-button-is-pressed "Permanent link")

**Use cases.** Incrementally reveal a user interface.

**Recipe.**

1. Import packages

1. Create a counter button.

1. Show an output after the button is clicked.

## Caching [¶](\#caching "Permanent link")

### Cache function outputs in memory [¶](\#cache-function-outputs-in-memory "Permanent link")

**Use case.** Because marimo runs cells automatically as code and
UI elements change, it can be helpful to cache expensive intermediate
computations. For example, perhaps your notebook computes t-SNE, UMAP, or PyMDE
embeddings, and exposes their parameters as UI elements. Caching the embeddings
for different configurations of the elements would greatly speed up your notebook.

**Recipe.**

1. Use [`mo.cache`](../api/caching/#marimo.cache "            marimo.cache") to cache function outputs given inputs.

Whenever `compute_predictions` is called with a value of `problem_parameters`
it has not seen, it will compute the predictions and store them in a cache. The
next time it is called with the same parameters, instead of recomputing the
predictions, it will return the previously computed value from the cache.

### Persistent caching for very expensive computations [¶](\#persistent-caching-for-very-expensive-computations "Permanent link")

**Use case.** If you are using marimo to capture very compute intensive
results, you may want to save the state of your computations to disk. Ideally,
if you update your code, then this save should be invalidated. It may also be
advantageous to add UI elements to explore your results, without having to
recompute expensive computations. You can achieve this with
[`mo.persistent_cache`](../api/caching/#marimo.persistent_cache "            marimo.persistent_cache").

**Recipe.**

1. Use `mo.persistent_cache` to cache blocks of code to disk.

If the execution conditions are the same, then cache will load results from
disk, and populate variable definitions.

Back to top

# Recipes [¶](\#recipes "Permanent link")

This page includes code snippets or " **recipes**" for a variety of common tasks.
Use them as building blocks or examples when making your own notebooks.

In these recipes, **each code block represents a cell**.

## Control Flow [¶](\#control-flow "Permanent link")

### Show an output conditionally [¶](\#show-an-output-conditionally "Permanent link")

**Use cases.** Hide an output until a condition is met ( _e.g._, until algorithm
parameters are valid), or show different outputs depending on the value of a UI
element or some other Python object

**Recipe.**

1. Use an `if` expression to choose which output to show.

### Run a cell on a timer [¶](\#run-a-cell-on-a-timer "Permanent link")

**Use cases.**

- Load new data periodically, and show updated plots or other outputs. For
example, in a dashboard monitoring a training run, experiment trial,
real-time weather data, ...
- Run a job periodically

**Recipe.**

1. Import packages

1. Create a [`mo.ui.refresh`](../api/inputs/refresh/#marimo.ui.refresh "            marimo.ui.refresh") timer that fires once a second:

1. Reference the timer by name to make this cell run once a second

Requires 'on cell change' autorun

For this to work, the [runtime\
configuration's](../guides/configuration/runtime_configuration/) `on cell
change` should be set to `autorun`

### Require form submission before sending UI value [¶](\#require-form-submission-before-sending-ui-value "Permanent link")

**Use cases.** UI elements automatically send their values to the Python when
they are interacted with, and run all cells referencing the elements. This
makes marimo notebooks responsive, but it can be an issue when the
downstream cells are expensive, or when the input (such as a text box)
needs to be filled out completely before it is considered valid. Forms
let you gate submission of UI element values on manual confirmation, via a
button press.

**Recipe.**

1. Import packages

1. Create a submittable form.

1. Get the value of the form.

### Stop execution of a cell and its descendants [¶](\#stop-execution-of-a-cell-and-its-descendants "Permanent link")

**Use cases.** For example, don't run a cell or its descendants if a form is
unsubmitted.

**Recipe.**

1. Import packages

1. Create a submittable form.

1. Use [`mo.stop`](../api/control_flow/#marimo.stop "            marimo.stop") to stop execution when
the form is unsubmitted.

## Grouping UI elements together [¶](\#grouping-ui-elements-together "Permanent link")

### Create an array of UI elements [¶](\#create-an-array-of-ui-elements "Permanent link")

**Use cases.** In order to synchronize UI elements between the frontend and
backend (Python), marimo requires you to
[assign UI elements to global variables](../guides/interactivity/).
But sometimes you don't know the number of elements to make until runtime:
for example, maybe you want to make a list of sliders, and the number of sliders
to make depends on the value of some other UI element.

You might be tempted to create a Python list of UI elements,
such as `l = [mo.ui.slider(1, 10) for i in range(number.value)]`: _however,_
_this won't work, because the sliders are not bound to global variables_.

For such cases, marimo provides the "higher-order" UI element
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"), which lets you make
a new UI element out of a list of UI elements:
`l = mo.ui.array([mo.ui.slider(1, 10) for i in range(number.value)])`.
The value of an `array` element is a list of the values of the elements
it wraps (in this case, a list of the slider values). Any time you interact
with any of the UI elements in the array, all cells referencing the array
by name (in this case, " `l`") will run automatically.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array") to group together
    many UI elements into a list.

1. Get the value of the UI elements using `array.value`

### Create a dictionary of UI elements [¶](\#create-a-dictionary-of-ui-elements "Permanent link")

**Use cases.** Same as for creating an array of UI elements, but lets you
name each of the wrapped elements with a string key.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") to
    group together many UI elements into a list.

1. Get the value of the UI elements using `dictionary.value`

### Embed a dynamic number of UI elements in another output [¶](\#embed-a-dynamic-number-of-ui-elements-in-another-output "Permanent link")

**Use cases.** When you want to embed a dynamic number of UI elements
in other outputs (like tables or markdown).

**Recipe.**

1. Import packages

1. Group the elements with

[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") or
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"), then retrieve them from the container
and display them elsewhere.

1. Get the value of the elements

### Create a hstack (or vstack) of UI elements with `on_change` handlers [¶](\#create-a-hstack-or-vstack-of-ui-elements-with-on_change-handlers "Permanent link")

**Use cases.** Arrange a dynamic number of UI elements in a hstack or vstack,
for example some number of buttons, and execute some side-effect when an
element is interacted with, e.g. when a button is clicked.

**Recipe.**

1. Import packages

1. Create buttons in `mo.ui.array` and pass them to hstack -- a regular
Python list won't work. Make sure to assign the array to a global variable.

1. Get the state value

### Create a table column of buttons with `on_change` handlers [¶](\#create-a-table-column-of-buttons-with-on_change-handlers "Permanent link")

**Use cases.** Arrange a dynamic number of UI elements in a column of
a table, and execute some side-effect when an element is interacted with, e.g.
when a button is clicked.

**Recipe.**

1. Import packages

1. Create buttons in `mo.ui.array` and pass them to `mo.ui.table`.
Make sure to assign the table and array to global variables

1. Get the state value

### Create a form with multiple UI elements [¶](\#create-a-form-with-multiple-ui-elements "Permanent link")

**Use cases.** Combine multiple UI elements into a form so that submission
of the form sends all its elements to Python.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.form`](../api/inputs/form/#marimo.ui.form "            marimo.ui.form") and [`Html.batch`](../api/html/#marimo.Html.batch "            batch") to
    create a form with multiple elements.

1. Get the submitted form value.

### Populating form with pre-defined examples [¶](\#populating-form-with-pre-defined-examples "Permanent link")

**Use cases.** To give examples of how a filled form looks like. Useful for illustrating complex API requests or database queries.
The form can also be populated from [URL query parameters](../api/query_params/) ( [notebook example](https://marimo.app/l/w21a3x?t1=query&t2=params)).

**Recipe.**

1. Import packages

1. Create dropdown of examples

1. Create form from examples.

1. Run pre-populated from or recompute with new input.

## Working with buttons [¶](\#working-with-buttons "Permanent link")

### Create a button that triggers computation when clicked [¶](\#create-a-button-that-triggers-computation-when-clicked "Permanent link")

**Use cases.** To trigger a computation on button click and only on button
click, use [`mo.ui.run_button()`](../api/inputs/run_button/).

**Recipe.**

1. Import packages

1. Create a run button

1. Run something only if the button has been clicked.

### Create a counter button [¶](\#create-a-counter-button "Permanent link")

**Use cases.** A counter button, i.e. a button that counts the number of times
it has been clicked, is a helpful building block for reacting to button clicks
(see other recipes in this section).

**Recipe.**

1. Import packages

1. Use [`mo.ui.button`](../api/inputs/button/#marimo.ui.button "            marimo.ui.button") and its
    `on_click` argument to create a counter button.

1. Get the button value

### Create a toggle button [¶](\#create-a-toggle-button "Permanent link")

**Use cases.** Toggle between two states using a button with a button
that toggles between `True` and `False`. (Tip: you can also just use
[`mo.ui.switch`](../api/inputs/switch/#marimo.ui.switch "            marimo.ui.switch").)

**Recipe.**

1. Import packages

1. Use [`mo.ui.button`](../api/inputs/button/#marimo.ui.button "            marimo.ui.button") and its
    `on_click` argument to create a toggle button.

1. Toggle between two outputs using the button value.

### Re-run a cell when a button is pressed [¶](\#re-run-a-cell-when-a-button-is-pressed "Permanent link")

**Use cases.** For example, you have a cell showing a random sample of data,
and you want to resample on button press.

**Recipe.**

1. Import packages

1. Create a button without a value, to function as a _trigger_.

1. Reference the button in another cell.

### Run a cell when a button is pressed, but not before [¶](\#run-a-cell-when-a-button-is-pressed-but-not-before "Permanent link")

**Use cases.** Wait for confirmation before executing downstream cells
(similar to a form).

**Recipe.**

1. Import packages

1. Create a counter button.

1. Only execute when the count is greater than 0.

### Reveal an output when a button is pressed [¶](\#reveal-an-output-when-a-button-is-pressed "Permanent link")

**Use cases.** Incrementally reveal a user interface.

**Recipe.**

1. Import packages

1. Create a counter button.

1. Show an output after the button is clicked.

## Caching [¶](\#caching "Permanent link")

### Cache function outputs in memory [¶](\#cache-function-outputs-in-memory "Permanent link")

**Use case.** Because marimo runs cells automatically as code and
UI elements change, it can be helpful to cache expensive intermediate
computations. For example, perhaps your notebook computes t-SNE, UMAP, or PyMDE
embeddings, and exposes their parameters as UI elements. Caching the embeddings
for different configurations of the elements would greatly speed up your notebook.

**Recipe.**

1. Use [`mo.cache`](../api/caching/#marimo.cache "            marimo.cache") to cache function outputs given inputs.

Whenever `compute_predictions` is called with a value of `problem_parameters`
it has not seen, it will compute the predictions and store them in a cache. The
next time it is called with the same parameters, instead of recomputing the
predictions, it will return the previously computed value from the cache.

### Persistent caching for very expensive computations [¶](\#persistent-caching-for-very-expensive-computations "Permanent link")

**Use case.** If you are using marimo to capture very compute intensive
results, you may want to save the state of your computations to disk. Ideally,
if you update your code, then this save should be invalidated. It may also be
advantageous to add UI elements to explore your results, without having to
recompute expensive computations. You can achieve this with
[`mo.persistent_cache`](../api/caching/#marimo.persistent_cache "            marimo.persistent_cache").

**Recipe.**

1. Use `mo.persistent_cache` to cache blocks of code to disk.

If the execution conditions are the same, then cache will load results from
disk, and populate variable definitions.

 Back to top

# Recipes [¶](\#recipes "Permanent link")

This page includes code snippets or " **recipes**" for a variety of common tasks.
Use them as building blocks or examples when making your own notebooks.

In these recipes, **each code block represents a cell**.

## Control Flow [¶](\#control-flow "Permanent link")

### Show an output conditionally [¶](\#show-an-output-conditionally "Permanent link")

**Use cases.** Hide an output until a condition is met ( _e.g._, until algorithm
parameters are valid), or show different outputs depending on the value of a UI
element or some other Python object

**Recipe.**

1. Use an `if` expression to choose which output to show.

### Run a cell on a timer [¶](\#run-a-cell-on-a-timer "Permanent link")

**Use cases.**

- Load new data periodically, and show updated plots or other outputs. For
   example, in a dashboard monitoring a training run, experiment trial,
   real-time weather data, ...
- Run a job periodically

**Recipe.**

1. Import packages

1. Create a [`mo.ui.refresh`](../api/inputs/refresh/#marimo.ui.refresh "            marimo.ui.refresh") timer that fires once a second:

1. Reference the timer by name to make this cell run once a second

Requires 'on cell change' autorun

For this to work, the [runtime\
configuration's](../guides/configuration/runtime_configuration/) `on cell
change` should be set to `autorun`

### Require form submission before sending UI value [¶](\#require-form-submission-before-sending-ui-value "Permanent link")

**Use cases.** UI elements automatically send their values to the Python when
they are interacted with, and run all cells referencing the elements. This
makes marimo notebooks responsive, but it can be an issue when the
downstream cells are expensive, or when the input (such as a text box)
needs to be filled out completely before it is considered valid. Forms
let you gate submission of UI element values on manual confirmation, via a
button press.

**Recipe.**

1. Import packages

1. Create a submittable form.

1. Get the value of the form.

### Stop execution of a cell and its descendants [¶](\#stop-execution-of-a-cell-and-its-descendants "Permanent link")

**Use cases.** For example, don't run a cell or its descendants if a form is
unsubmitted.

**Recipe.**

1. Import packages

1. Create a submittable form.

1. Use [`mo.stop`](../api/control_flow/#marimo.stop "            marimo.stop") to stop execution when
   the form is unsubmitted.

## Grouping UI elements together [¶](\#grouping-ui-elements-together "Permanent link")

### Create an array of UI elements [¶](\#create-an-array-of-ui-elements "Permanent link")

**Use cases.** In order to synchronize UI elements between the frontend and
backend (Python), marimo requires you to
[assign UI elements to global variables](../guides/interactivity/).
But sometimes you don't know the number of elements to make until runtime:
for example, maybe you want to make a list of sliders, and the number of sliders
to make depends on the value of some other UI element.

You might be tempted to create a Python list of UI elements,
such as `l = [mo.ui.slider(1, 10) for i in range(number.value)]`: _however,_
_this won't work, because the sliders are not bound to global variables_.

For such cases, marimo provides the "higher-order" UI element
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"), which lets you make
a new UI element out of a list of UI elements:
`l = mo.ui.array([mo.ui.slider(1, 10) for i in range(number.value)])`.
The value of an `array` element is a list of the values of the elements
it wraps (in this case, a list of the slider values). Any time you interact
with any of the UI elements in the array, all cells referencing the array
by name (in this case, " `l`") will run automatically.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array") to group together
    many UI elements into a list.

1. Get the value of the UI elements using `array.value`

### Create a dictionary of UI elements [¶](\#create-a-dictionary-of-ui-elements "Permanent link")

**Use cases.** Same as for creating an array of UI elements, but lets you
name each of the wrapped elements with a string key.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") to
    group together many UI elements into a list.

1. Get the value of the UI elements using `dictionary.value`

### Embed a dynamic number of UI elements in another output [¶](\#embed-a-dynamic-number-of-ui-elements-in-another-output "Permanent link")

**Use cases.** When you want to embed a dynamic number of UI elements
in other outputs (like tables or markdown).

**Recipe.**

1. Import packages

1. Group the elements with

[`mo.ui.dictionary`](../api/inputs/dictionary/#marimo.ui.dictionary "            marimo.ui.dictionary") or
[`mo.ui.array`](../api/inputs/array/#marimo.ui.array "            marimo.ui.array"), then retrieve them from the container
and display them elsewhere.

1. Get the value of the elements

### Create a hstack (or vstack) of UI elements with `on_change` handlers [¶](\#create-a-hstack-or-vstack-of-ui-elements-with-on_change-handlers "Permanent link")

**Use cases.** Arrange a dynamic number of UI elements in a hstack or vstack,
for example some number of buttons, and execute some side-effect when an
element is interacted with, e.g. when a button is clicked.

**Recipe.**

1. Import packages

1. Create buttons in `mo.ui.array` and pass them to hstack -- a regular
   Python list won't work. Make sure to assign the array to a global variable.

1. Get the state value

### Create a table column of buttons with `on_change` handlers [¶](\#create-a-table-column-of-buttons-with-on_change-handlers "Permanent link")

**Use cases.** Arrange a dynamic number of UI elements in a column of
a table, and execute some side-effect when an element is interacted with, e.g.
when a button is clicked.

**Recipe.**

1. Import packages

1. Create buttons in `mo.ui.array` and pass them to `mo.ui.table`.
   Make sure to assign the table and array to global variables

1. Get the state value

### Create a form with multiple UI elements [¶](\#create-a-form-with-multiple-ui-elements "Permanent link")

**Use cases.** Combine multiple UI elements into a form so that submission
of the form sends all its elements to Python.

**Recipe.**

1. Import packages.

1. Use [`mo.ui.form`](../api/inputs/form/#marimo.ui.form "            marimo.ui.form") and [`Html.batch`](../api/html/#marimo.Html.batch "            batch") to
    create a form with multiple elements.

1. Get the submitted form value.

### Populating form with pre-defined examples [¶](\#populating-form-with-pre-defined-examples "Permanent link")

**Use cases.** To give examples of how a filled form looks like. Useful for illustrating complex API requests or database queries.
The form can also be populated from [URL query parameters](../api/query_params/) ( [notebook example](https://marimo.app/l/w21a3x?t1=query&t2=params)).

**Recipe.**

1. Import packages

1. Create dropdown of examples

1. Create form from examples.

1. Run pre-populated from or recompute with new input.

## Working with buttons [¶](\#working-with-buttons "Permanent link")

### Create a button that triggers computation when clicked [¶](\#create-a-button-that-triggers-computation-when-clicked "Permanent link")

**Use cases.** To trigger a computation on button click and only on button
click, use [`mo.ui.run_button()`](../api/inputs/run_button/).

**Recipe.**

1. Import packages

1. Create a run button

1. Run something only if the button has been clicked.

### Create a counter button [¶](\#create-a-counter-button "Permanent link")

**Use cases.** A counter button, i.e. a button that counts the number of times
it has been clicked, is a helpful building block for reacting to button clicks
(see other recipes in this section).

**Recipe.**

1. Import packages

1. Use [`mo.ui.button`](../api/inputs/button/#marimo.ui.button "            marimo.ui.button") and its
    `on_click` argument to create a counter button.

1. Get the button value

### Create a toggle button [¶](\#create-a-toggle-button "Permanent link")

**Use cases.** Toggle between two states using a button with a button
that toggles between `True` and `False`. (Tip: you can also just use
[`mo.ui.switch`](../api/inputs/switch/#marimo.ui.switch "            marimo.ui.switch").)

**Recipe.**

1. Import packages

1. Use [`mo.ui.button`](../api/inputs/button/#marimo.ui.button "            marimo.ui.button") and its
    `on_click` argument to create a toggle button.

1. Toggle between two outputs using the button value.

### Re-run a cell when a button is pressed [¶](\#re-run-a-cell-when-a-button-is-pressed "Permanent link")

**Use cases.** For example, you have a cell showing a random sample of data,
and you want to resample on button press.

**Recipe.**

1. Import packages

1. Create a button without a value, to function as a _trigger_.

1. Reference the button in another cell.

### Run a cell when a button is pressed, but not before [¶](\#run-a-cell-when-a-button-is-pressed-but-not-before "Permanent link")

**Use cases.** Wait for confirmation before executing downstream cells
(similar to a form).

**Recipe.**

1. Import packages

1. Create a counter button.

1. Only execute when the count is greater than 0.

### Reveal an output when a button is pressed [¶](\#reveal-an-output-when-a-button-is-pressed "Permanent link")

**Use cases.** Incrementally reveal a user interface.

**Recipe.**

1. Import packages

1. Create a counter button.

1. Show an output after the button is clicked.

## Caching [¶](\#caching "Permanent link")

### Cache function outputs in memory [¶](\#cache-function-outputs-in-memory "Permanent link")

**Use case.** Because marimo runs cells automatically as code and
UI elements change, it can be helpful to cache expensive intermediate
computations. For example, perhaps your notebook computes t-SNE, UMAP, or PyMDE
embeddings, and exposes their parameters as UI elements. Caching the embeddings
for different configurations of the elements would greatly speed up your notebook.

**Recipe.**

1. Use [`mo.cache`](../api/caching/#marimo.cache "            marimo.cache") to cache function outputs given inputs.

Whenever `compute_predictions` is called with a value of `problem_parameters`
it has not seen, it will compute the predictions and store them in a cache. The
next time it is called with the same parameters, instead of recomputing the
predictions, it will return the previously computed value from the cache.

### Persistent caching for very expensive computations [¶](\#persistent-caching-for-very-expensive-computations "Permanent link")

**Use case.** If you are using marimo to capture very compute intensive
results, you may want to save the state of your computations to disk. Ideally,
if you update your code, then this save should be invalidated. It may also be
advantageous to add UI elements to explore your results, without having to
recompute expensive computations. You can achieve this with
[`mo.persistent_cache`](../api/caching/#marimo.persistent_cache "            marimo.persistent_cache").

**Recipe.**

1. Use `mo.persistent_cache` to cache blocks of code to disk.

If the execution conditions are the same, then cache will load results from
disk, and populate variable definitions.
