# mAI Utils V02

ComfyUI custom node pack for small, reusable mAI utility nodes.

Install this folder under ComfyUI's `custom_nodes` directory, then restart ComfyUI.
The currently registered nodes are listed below.

## mAI background lighting match

Location:

```text
mAI / Image
```

Purpose:
Correct a brightness and contrast shift after inpainting. The node measures the
untouched background in both the original and edited images, then applies the
resulting lighting correction to the complete edited image.

Inputs:

* `original_image` — the image before inpainting
* `edited_image` — the complete image after inpainting or object removal
* `edit_mask` — white marks the edited area; black marks the untouched background

Output:

* `image` — the corrected edited image

Default behavior:

* Uses only the inverse of `edit_mask` for analysis.
* Treats soft mask values as proportional background weights.
* Matches the mean (brightness) and standard deviation (contrast) independently
  for the red, green, and blue channels.
* Applies one affine correction per RGB channel to every edited pixel, including
  pixels inside the mask.
* Supports image batches and broadcasts a batch of one where possible.

Known limitations:

* The original image, edited image, and mask must have matching dimensions.
* The mask must leave some untouched background visible.
* Contrast cannot be reconstructed if an edited background channel is completely
  flat while the corresponding original channel has contrast; the node reports a
  clear error in that case.
* Output values are limited to `[0, 1]`. Clipping at the range boundaries can make
  the final measured statistics differ slightly from the mathematical target.

## mAI video loader

Location:

```text
mAI / IO
```

Purpose:
Upload or select a video from ComfyUI's input directory and decode it into a
ComfyUI image batch, along with its source metadata and audio stream.

Input:

* `video`

Outputs, in order:

* `frames` (`IMAGE`) — all decoded video frames as one image batch
* `fps` (`FLOAT`) — the video's average source frame rate
* `audio` (`AUDIO`) — the decoded audio stream, or no value when the video has no audio
* `frame_count` (`INT`) — the number of decoded frames
* `width` (`INT`) — decoded frame width
* `height` (`INT`) — decoded frame height

Default behavior:

* Lists video files in ComfyUI's input directory and provides a video upload button.
* Uses ComfyUI's native video decoder and returns frames in
  `[frame, height, width, channel]` format.
* Applies video rotation metadata through ComfyUI's decoder before reporting width
  and height.

Known limitations:

* The complete decoded frame sequence is held in memory; long or high-resolution
  videos can require substantial RAM.
* Variable-frame-rate sources report their average frame rate.
* A video without an audio stream returns no `AUDIO` value. Nodes that require
  audio should only be connected when the source contains audio.

## mAI Inpaint Crop - Separate Stitch Mask

Location:

```text
mAI / Image
```

Purpose:
Crop an image using the current `ComfyUI-Inpaint-CropAndStitch` workflow while
generating a separate full-size stitch/blend mask inside the node.

```text
render mask = what the model edits
stitch mask = where the generated result is blended back
```

The generated stitch mask has two modes:

* `rectangle (full crop)` blends the complete cropped rectangle back into the
  source image.
* `extended mask` follows the render-mask shape, expands it with
  `stitch_mask_expand_pixels`, and ensures the crop is large enough to contain
  that expanded region.

Only `cropped_mask` is sent to the inpainting model.
`stitch_mask_blend_pixels` feathers the generated stitch boundary in either
mode.

Inputs:

* All current inputs from the installed upstream `Inpaint Crop` node.
* `stitch_mask_mode` (`rectangle (full crop)` or `extended mask`)
* `stitch_mask_expand_pixels` (used by `extended mask`, default `32`)
* `stitch_mask_blend_pixels` (default `32`)

Outputs:

* `stitcher`
* `cropped_image`
* `cropped_mask` (the render/inpainting mask)
* `cropped_stitch_mask` (the exact mask stored in the stitcher for compositing)

Example workflow:

```text
IMAGE ─────────────────────────────┐
                                  │
RENDER MASK ──────────────────────┤
                                  ▼
                     mAI Inpaint Crop
                                  │
                    ┌─────────────┼──────────────────────┐
                    │             │                      │
              cropped image   render mask            STITCHER
                    │             │            (internal stitch mask)
                    └──────► IMAGE MODEL                 │
                              │                          │
                              ▼                          │
                       rendered crop                     │
                              │                          │
                              └────────► Inpaint Stitch ◄┘
                                               │
                                               ▼
                                          FINAL IMAGE
```

Connect `stitcher` and the model's rendered crop to the standard upstream
`Inpaint Stitch` node. `cropped_stitch_mask` previews the exact internally
generated mask used by that stitch operation.

Known limitations:

* This integration follows the locally installed upstream v3 stitcher format and
  requires the `cropped_mask_for_blend` field exposed by that format.
* The optional dependency must be available when ComfyUI loads this node pack.

### CropAndStitch dependency installation

`ComfyUI-Inpaint-CropAndStitch` must also be installed. It is intentionally not
listed in `requirements.txt` because it is a separate ComfyUI custom-node pack.

Recommended installation:

```text
ComfyUI Manager
→ search for ComfyUI-Inpaint-CropAndStitch
→ install
→ restart ComfyUI
```

Manual installation, following the current upstream README:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git
```

Restart ComfyUI after installation. If the dependency is absent, this node is
skipped and the rest of mAI Utils continues to load.

Acknowledgment: this node integrates with
[`ComfyUI-Inpaint-CropAndStitch`](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch),
which is licensed under GNU GPL v3. No upstream source or license text is
vendored here.

## mAI prepare image for Minimax H3

Location:

```text
mAI / Image
```

Purpose:
Resize an image for Minimax H3 while keeping its proportions as close as possible
and ensuring that both output dimensions are multiples of 32.

Inputs:

* `image`
* `target_megapixels` (`0.2 MP` through `2.0 MP` in `0.1 MP` steps)
* `resize_mode` (`Target megapixels` or `Short side 768 px`)

Output:

* `image`

Default behavior:

* Targets approximately `1.0 MP` by default (`1024 × 1024` total pixels in ComfyUI's convention).
* Calculates a proportional target size, then rounds width and height to the nearest multiples of 32.
* Uses Lanczos resampling without cropping.
* Supports image batches.

With `Short side 768 px` selected, the megapixel value is ignored. The shorter
dimension is set to exactly 768 pixels and the longer dimension is rounded to
the nearest multiple of 32.

Known limitations:

* Rounding both dimensions to a 32-pixel grid can introduce a small aspect-ratio difference.
* The selected megapixel value is approximate because valid output dimensions are discrete.

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

## mAI mask bounding box

Location:

```text
mAI / Mask
```

Purpose:
Create a pure white, filled rectangular mask covering the bounding box of the
original mask's nonzero pixels.

Input:

* `mask`

Output:

* `mask`

Default behavior:

* Treats every input value greater than zero as part of the mask.
* Fills the smallest axis-aligned rectangle containing those pixels with white (`1.0`).
* Keeps pixels outside the rectangle black (`0.0`).
* Processes each mask in a batch independently.
* Returns an empty mask when the input mask is empty.

Known limitations:

* Very small positive mask values count toward the bounding box.

## mAI mask outline

Location:

```text
mAI / Mask
```

Purpose:
Draw a colored outline that follows the mask's actual contour on the original
image.

Inputs:

* `image` — the original image to annotate
* `mask` — the shape to outline
* `color` — outline color in `#RRGGBB` format (default `#00FF00`)
* `thickness` — outline width in pixels (default `10`)
* `padding` — pixels used to expand the mask shape before outlining (default `0`)
* `threshold` — minimum included mask value on a `0` to `255` scale (default `0`)

Output:

* `image` — a copy of the input image with the outline drawn over it

Default behavior:

* Treats every mask value greater than the selected threshold as part of the shape.
* Expands the mask by `padding` pixels before tracing its contour.
* Centers even stroke widths across the contour; odd widths place the extra pixel inside.
* Processes each batch item independently and broadcasts a batch of one where possible.
* Leaves the image unchanged when the mask contains no included pixels.
* Preserves additional image channels, including alpha; the outline changes RGB only.

Known limitations:

* The mask height and width must match the image.
* `threshold` uses `0` to `255` while ComfyUI stores mask values internally as `0.0` to `1.0`.
* The morphology uses square pixel neighborhoods, so heavily padded diagonal corners can look slightly squared.
* Outlines and padding are clipped at image edges.

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

## mAI text sequence randomizer

Location:

```text
mAI / Text
```

Purpose:
Randomize the order of items in a text sequence with a reproducible seed.

Inputs:

* `text`
* `separator` (`comma` or `line break`)
* `seed`

Output:

* `text`

Default behavior:

* Uses comma-separated items by default.
* Trims whitespace around each item and ignores empty items.
* Joins comma-separated output with a comma and one space.
* A fixed seed always produces the same order for the same text and separator.
* The seed widget supports ComfyUI's Fixed, Increment, Decrement, and Randomize modes.
* Returns an empty string when there are no non-empty items.

Example:

Input:

```text
text 1, text 2, text 3
```

Possible output:

```text
text 2, text 1, text 3
```

Known limitations:

* Commas inside an item are treated as separators in `comma` mode.
* Random shuffling can occasionally leave the sequence in its original order.

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
