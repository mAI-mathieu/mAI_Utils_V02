"""Render repeatable native-resolution Auto comparisons for supplied image pairs.

Usage: python scripts/evaluate_source_preservation.py source.png edit1.png edit2.png
       --output-dir output/preserve_review --tag before
Run again with --tag after to produce before/after comparison sheets.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.image_edit_detection import detect_image_edits
from utils.image_preserve_source import preserve_source_image


def load_image(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).float()[None] / 255


def image_from_tensor(tensor):
    array = (tensor[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("edits", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/preserve_review"))
    parser.add_argument("--tag", default="after")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = load_image(args.source)
    source_image = image_from_tensor(source)
    reports = []
    sheets = []
    for index, path in enumerate(args.edits, 1):
        edit = load_image(path)
        image, mask = preserve_source_image(source, edit, mode="auto")
        _, settings = detect_image_edits(source[..., :3], edit[..., :3], torch.ones(source.shape[:3]))
        report = dict(case=index, filename=path.name, **asdict(settings),
                      original_percent=round(100 * float((mask == 0).float().mean()), 2),
                      transition_percent=round(100 * float(((mask > 0) & (mask < 1)).float().mean()), 2))
        reports.append(report)
        print(json.dumps(report), flush=True)
        result_image = image_from_tensor(image)
        mask_image = image_from_tensor(mask)
        result_image.save(args.output_dir / f"{index:02d}_{args.tag}.png")
        mask_image.save(args.output_dir / f"{index:02d}_{args.tag}_mask.png")
        before_path = args.output_dir / f"{index:02d}_before.png"
        tiles = [("Source", source_image), ("Model edit", image_from_tensor(edit))]
        if args.tag != "before" and before_path.exists():
            tiles.append(("Previous Auto", Image.open(before_path).convert("RGB")))
        tiles += [(f"Auto: {args.tag}", result_image), ("Mask", mask_image.convert("RGB"))]
        tile_width, tile_height = 228, 400
        sheet = Image.new("RGB", (len(tiles) * tile_width, tile_height + 32), "#202020")
        draw = ImageDraw.Draw(sheet)
        for column, (title, tile) in enumerate(tiles):
            draw.text((column * tile_width + 8, 8), f"{index}: {title}", fill="white")
            sheet.paste(tile.resize((tile_width, tile_height)), (column * tile_width, 32))
        sheet.save(args.output_dir / f"{index:02d}_{args.tag}_comparison.jpg", quality=94)
        sheets.append(sheet)
    overview = Image.new("RGB", (sheets[0].width, sum(sheet.height for sheet in sheets)))
    offset = 0
    for sheet in sheets:
        overview.paste(sheet, (0, offset))
        offset += sheet.height
    overview.save(args.output_dir / f"{args.tag}_overview.jpg", quality=94)
    (args.output_dir / f"{args.tag}_report.json").write_text(json.dumps(reports, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
