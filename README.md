# mAI Utils V02

ComfyUI custom node pack for small, reusable mAI utility nodes.

Install this folder under ComfyUI's `custom_nodes` directory, then restart ComfyUI.
The currently registered nodes are listed below.

## mAI Composite Layer

Location:

```text
mAI / Image
```

Purpose:
Composite one image layer over a base image.
Use multiple copies of the node chained together to build multi-layer compositions.

Inputs:

* `base_image`
* optional `base_mask`
* `layer_image`
* optional `layer_mask`

Widgets:

* `x`
* `y`
* `scale`
* `opacity`
* `anchor`
* `fit_mode`

Outputs:

* `image`
* `mask`

Default behavior:

* Outputs RGB image data.
* Uses `layer_mask` when connected.
* Otherwise uses embedded alpha when `layer_image` has 4 channels.
* Otherwise treats the layer as opaque.
* Supports negative `x` and `y` values.
* Allows layers to be partially outside the base image.

Mask behavior:

* The `mask` output is the placed layer alpha.
* If `base_mask` is connected, the output mask combines `base_mask` and the current layer alpha.
* This makes it possible to chain Composite Layer nodes while preserving mask coverage.

Known limitations:

* The old `mAI Image Layer Stack` implementation remains in the codebase but is not registered by default.

## mAI Save Text File

Location:

```text
mAI / IO
```

Purpose:
Save text to a UTF-8 file.

Inputs:

* `text`
* `file_name`
* `folder`
* `overwrite`

Outputs:

* `file_path`
* `saved`

Default behavior:

* Writes text using UTF-8 encoding.
* Saves into the current working directory when `folder` is empty.
* Creates `folder` if it does not exist.
* Uses the exact provided `file_name`.
* Does not add timestamps or change the file name.
* Does not auto-add `.txt`; include `.txt` in `file_name` when wanted.
* Raises `FileExistsError` when `overwrite` is false and the target file already exists.

Known limitations:

* `file_name` must be a simple file name, not a nested or absolute path.
* Put folder paths in `folder`, not in `file_name`.

## mAI Type Converter

Location:

```text
mAI / Utils
```

Purpose:
Convert one selected input type into boolean, string, int, and float outputs.

Inputs:

* `source_type`
* `boolean`
* `string`
* `int`
* `float`
* `strict`

Outputs:

* `boolean`
* `string`
* `int`
* `float`

Default behavior:

* `source_type` decides which input value is used for conversion.
* `strict = false` falls back to safe defaults for invalid conversions.
* `strict = true` raises errors for invalid conversions.

Safe defaults:

* Invalid boolean conversions return `false`.
* Invalid int conversions return `0`.
* Invalid float conversions return `0.0`.

## mAI Random Line

Location:

```text
mAI / Utils
```

Purpose:
Randomly select one non-empty line from a multiline text input.

Input:

* `text`

Output:

* `line`

Default behavior:

* Uses Windows and Unix line endings.
* Ignores empty and whitespace-only lines.
* Preserves the selected line text except for the removed line break.
* Returns an empty string when there are no non-empty lines.
* Produces a fresh random choice on each queued execution.

Example:

Input:

```text
Line 1
Line 2
```

Possible output:

```text
Line 2
```

Known limitations:

* There is no visible seed input, so results are intentionally not reproducible.

## mAI Example Text Node

Location:

```text
mAI / Template
```

Purpose:
Small registered example node that returns a text string, optionally stripping leading and trailing whitespace.

Inputs:

* `text`
* `strip_whitespace`

Output:

* `text`

Default behavior:

* `strip_whitespace = true` removes leading and trailing whitespace.
* `strip_whitespace = false` returns the input text unchanged.

## Testing

Run the Python tests from this folder:

```powershell
python -m pytest
```

The tests cover pure node and utility behavior where possible.
