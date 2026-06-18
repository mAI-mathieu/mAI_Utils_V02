# AGENTS.md

## Purpose

This repository is a ComfyUI custom node pack.

The goal is to build clean, maintainable, reusable mAI nodes that are easy to extend, test, and keep stable across saved ComfyUI workflows.

Follow this document for all coding tasks in this repository.

## Core principles

* Do not modify ComfyUI core files.
* Work only inside this custom node repository.
* Keep changes small, focused, and easy to review.
* Prefer readable, explicit code over clever abstractions.
* Keep Python backend logic and frontend JavaScript behavior separated.
* Do not break existing nodes when adding new ones.
* Do not rename or remove existing nodes unless explicitly requested.
* Do not change released socket names or output order unless explicitly requested.
* Avoid heavy dependencies.
* Do not add API keys, tokens, local machine paths, or personal config files.
* Add tests for reusable pure logic whenever possible.

## Repository structure

Use this structure unless the existing repo already has a clear equivalent structure:

```text
.
├── __init__.py
├── nodes/
├── utils/
├── web/
│   └── js/
├── tests/
├── docs/
├── README.md
└── AGENTS.md
```

Recommended responsibilities:

```text
nodes/      ComfyUI node classes
utils/      Pure reusable logic
web/js/     ComfyUI frontend extensions
tests/      Pure Python tests
docs/       Optional documentation
```

## Naming conventions

### Node display names

ComfyUI display names must be human-readable.

Format:

```text
mAI [feature name]
```

Examples:

```text
mAI main input
mAI image resize fit
mAI RGBA to RGB
mAI prompt hub
mAI sequential folder loader
```

Rules:

* Always start with `mAI`.
* Use plain readable words.
* Do not use underscores in display names.
* Do not add version numbers to display names unless explicitly requested.
* Do not remove older nodes unless explicitly requested.

### Python class names

Python classes must use PascalCase.

Format:

```text
MAI[FeatureName]
```

Examples:

```python
MAIMainInput
MAIImageResizeFit
MAIRgbaToRgb
MAISequentialFolderLoader
```

Rules:

* Start with `MAI`.
* Do not add version numbers unless explicitly requested.
* Keep the class name stable after release.

### ComfyUI mapping keys

Use the Python class name as the internal mapping key.

Example:

```python
NODE_CLASS_MAPPINGS = {
    "MAIMainInput": MAIMainInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIMainInput": "mAI main input",
}
```

Rules:

* Mapping keys should match class names.
* Display names should follow the `mAI [feature name]` format.
* Keep mappings easy to scan.

### Python file names

Python files must use snake_case.

Examples:

```text
main_input.py
image_resize_fit.py
rgba_to_rgb.py
sequential_folder_loader.py
```

Rules:

* One main node per file when practical.
* Small related helper nodes may share a file if they are tightly connected.
* Avoid generic new files like `nodes.py`, `helpers.py`, or `misc.py`.

### Utility file names

Utility files should be named by responsibility.

Examples:

```text
aspect_ratios.py
image_defaults.py
path_validation.py
tensor_conversion.py
```

Rules:

* Utility code should be reusable.
* Utility code should avoid ComfyUI dependencies unless necessary.
* Pure utility logic should be testable outside ComfyUI.

### JavaScript file names

Frontend JS files should match the node feature name.

Examples:

```text
main_input.js
image_resize_fit.js
prompt_hub.js
```

Rules:

* Put ComfyUI frontend extensions in `web/js/`.
* JS should handle UI behavior only.
* Python remains the source of truth for backend execution.
* Do not duplicate complex backend logic in JS unless needed for live UI behavior.

### Test file names

Tests must use this format:

```text
test_[module_name].py
```

Examples:

```text
test_aspect_ratios.py
test_path_validation.py
test_image_defaults.py
```

## Stability rules

Saved ComfyUI workflows rely on node names, sockets, and output order.

Create a separate new node only when explicitly requested or when a change would clearly break existing workflows.

Breaking changes include:

* input names change
* output names change
* output order changes
* socket types change
* default behavior changes significantly
* UI structure changes significantly

Small internal fixes can update the existing node if they do not break saved workflows.

## Socket rules

### Output sockets

Output names and order are part of the workflow contract.

Rules:

* Do not change released output names.
* Do not change released output order.
* Do not change released output types.
* Keep outputs stable even if some UI widgets are hidden.
* Avoid dynamically removing sockets unless explicitly requested.
* If sockets must change, ask first or create a clearly separate node if requested.

Good:

```python
RETURN_TYPES = ("STRING", "INT", "INT")
RETURN_NAMES = ("Prompt", "Width", "Height")
```

Bad:

```python
# Changing output order in an existing released node
RETURN_TYPES = ("INT", "STRING", "INT")
RETURN_NAMES = ("Width", "Prompt", "Height")
```

### Input and widget names

Backend input names should use snake_case.

Examples:

```text
main_prompt
aspect_ratio
custom_width
custom_height
use_llm_prompt
image
mask
```

Rules:

* Keep backend input names stable.
* Use clear names.
* Avoid ambiguous names like `value1` unless the node is intentionally generic.
* Prefer `custom_width` over `width` when the field only applies in custom mode.

## Node implementation style

Keep node classes small.

Preferred structure:

```python
from ..utils.some_logic import resolve_value

class MAISomeNode:
    CATEGORY = "mAI / Category"
    FUNCTION = "run"
    RETURN_TYPES = (...)
    RETURN_NAMES = (...)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                ...
            },
            "optional": {
                ...
            },
        }

    def run(self, ...):
        result = resolve_value(...)
        return (...)
```

Avoid:

```python
# Huge node class with all parsing, validation, UI logic,
# file handling, and business logic mixed together.
```

## Categories

Use clear ComfyUI categories.

Examples:

```python
CATEGORY = "mAI / Input"
CATEGORY = "mAI / Image"
CATEGORY = "mAI / Text"
CATEGORY = "mAI / Utils"
CATEGORY = "mAI / Mask"
CATEGORY = "mAI / IO"
```

Rules:

* Keep categories consistent.
* Do not create too many category names.
* Prefer simple names users can understand.

## Frontend JavaScript rules

Use frontend JS only when Python cannot provide a good UI experience alone.

Good uses:

* show or hide widgets
* resize nodes after UI changes
* add UI buttons
* improve widget layout
* react to dropdown changes

Rules:

* Put JS in `web/js/`.
* Register `WEB_DIRECTORY = "./web"` in `__init__.py` when needed.
* Do not remove sockets dynamically unless explicitly requested.
* Do not rely on JS for important backend validation.
* Python backend must still work if JS fails to load.

## Registration rules

The root `__init__.py` must clearly register node classes.

Example:

```python
from .nodes.main_input import MAIMainInput
from .nodes.rgba_to_rgb import MAIRgbaToRgb

NODE_CLASS_MAPPINGS = {
    "MAIMainInput": MAIMainInput,
    "MAIRgbaToRgb": MAIRgbaToRgb,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIMainInput": "mAI main input",
    "MAIRgbaToRgb": "mAI RGBA to RGB",
}

WEB_DIRECTORY = "./web"
```

Rules:

* Keep imports explicit.
* Keep mappings easy to scan.
* Do not silently remove existing registered nodes.
* Only add `WEB_DIRECTORY` if frontend files are used.

## Optional input rules

When a node has optional inputs:

* The node must not crash when an optional input is missing.
* Missing optional values should have clear fallback behavior.
* Return `None` only when the output type and downstream ComfyUI behavior can handle it.
* Document optional behavior in the README.

## Error handling

Errors should be clear and useful.

Good:

```python
raise ValueError(f"Unknown aspect ratio preset: {aspect_ratio}")
```

Bad:

```python
raise Exception("error")
```

Rules:

* Validate user-facing values.
* Do not hide important errors silently.
* Avoid broad `except Exception` unless re-raising with useful context.
* Do not print noisy logs during every execution unless debugging is explicitly requested.

## Dependency rules

Avoid dependencies unless the node truly needs them.

Do not add:

```text
torch
numpy
opencv-python
Pillow
requests
```

unless the feature actually requires them.

For ComfyUI image tensor work, use the dependencies already available in the ComfyUI environment whenever possible.

Do not add heavy dependencies for simple utility nodes.

## Testing rules

Add tests for pure logic.

Good test targets:

* aspect ratio parsing
* path validation
* naming helpers
* config serialization
* pure image dimension calculations
* prompt/text transformation helpers

Avoid tests that require a full ComfyUI runtime unless explicitly requested.

Tests should be runnable with:

```bash
python -m pytest
```

If pytest is not installed, the code should still be understandable and testable manually.

## README rules

When adding or changing a node, update `README.md`.

Include:

* node name
* what it does
* where it appears in ComfyUI
* inputs
* outputs
* default behavior
* known limitations
* short testing instructions

Keep README practical and short.

## Documentation rules

Use `docs/` only for longer notes.

Examples:

```text
docs/main_input.md
docs/development_notes.md
docs/frontend_extension_notes.md
```

Do not duplicate the README unless more detail is needed.

## Forbidden changes

Do not:

* modify ComfyUI core files
* hardcode local machine paths
* store API keys or secrets
* add unrelated features
* add large dependencies without need
* rename released nodes without request
* change released socket order without creating a new separate node
* hide errors silently
* add shell commands inside node execution
* run arbitrary system commands from nodes
* make network calls from nodes unless explicitly requested

## Code quality checklist

Before finishing a task, verify:

* ComfyUI can import the node pack without errors.
* Existing nodes still register.
* New nodes appear under the expected `mAI / ...` category.
* Output names and order match the task.
* Optional inputs do not crash when missing.
* Pure utility tests pass, if tests were added.
* README is updated if behavior changed.
* No ComfyUI core files were modified.
* No unrelated files were changed.

## Final report expected

After implementation, report:

* files changed
* what was added or changed
* how the node is registered
* how to test it in ComfyUI
* any limitation or risk
* whether tests were run
