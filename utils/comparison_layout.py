"""
Utilities for saving comparison layouts while preserving image resolution.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_uint8(image_array):
    array = np.asarray(image_array)

    if array.dtype == np.uint8:
        return array

    return np.clip(array, 0, 255).astype(np.uint8)


def _to_pil_image(image_array):
    array = _to_uint8(image_array)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L")

    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")

    raise ValueError("Comparison layouts support only grayscale or RGB images.")


def _load_font(font_size):
    for font_name in ["DejaVuSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resolve_title_height(images, requested_title_height):
    if requested_title_height is not None:
        return requested_title_height

    tallest_image = max(image.height for image in images)
    return max(72, min(120, int(round(tallest_image * 0.06))))


def _resolve_output_mode(images):
    return "RGB" if any(image.mode == "RGB" for image in images) else "L"


def _resolve_background(output_mode, background):
    if output_mode == "RGB":
        if isinstance(background, tuple):
            return background
        return (background, background, background)
    return background


def _make_labeled_panel(image_array, title, title_height, label_padding=18, background=255, output_mode=None):
    image = _to_pil_image(image_array)
    if output_mode is None:
        output_mode = image.mode
    image = image.convert(output_mode)
    background_value = _resolve_background(output_mode, background)
    panel = Image.new(output_mode, (image.width, image.height + title_height), color=background_value)
    panel.paste(image, (0, title_height))

    draw = ImageDraw.Draw(panel)
    font = _load_font(max(28, int(title_height * 0.50)))
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = max(label_padding, (panel.width - text_width) // 2)
    text_y = max(0, (title_height - text_height) // 2 - 4)
    text_fill = (0, 0, 0) if output_mode == "RGB" else 0
    draw.text((text_x, text_y), title, fill=text_fill, font=font)

    return panel


def save_comparison_row(image_arrays, titles, output_path, gap=24, title_height=None, background=255):
    pil_images = [_to_pil_image(image_array) for image_array in image_arrays]
    output_mode = _resolve_output_mode(pil_images)
    resolved_title_height = _resolve_title_height(pil_images, title_height)
    background_value = _resolve_background(output_mode, background)

    panels = [
        _make_labeled_panel(
            image_array,
            title,
            title_height=resolved_title_height,
            background=background_value,
            output_mode=output_mode,
        )
        for image_array, title in zip(image_arrays, titles)
    ]

    total_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    max_height = max(panel.height for panel in panels)
    canvas = Image.new(output_mode, (total_width, max_height), color=background_value)

    current_x = 0
    for panel in panels:
        canvas.paste(panel, (current_x, 0))
        current_x += panel.width + gap

    canvas.save(output_path)


def save_comparison_grid(rows, output_path, gap_x=24, gap_y=28, title_height=None, background=255):
    pil_images = [_to_pil_image(image_array) for row in rows for image_array, _ in row]
    output_mode = _resolve_output_mode(pil_images)
    resolved_title_height = _resolve_title_height(pil_images, title_height)
    background_value = _resolve_background(output_mode, background)
    row_canvases = []
    row_widths = []
    row_heights = []

    for row in rows:
        panels = [
            _make_labeled_panel(
                image_array,
                title,
                title_height=resolved_title_height,
                background=background_value,
                output_mode=output_mode,
            )
            for image_array, title in row
        ]

        row_width = sum(panel.width for panel in panels) + gap_x * (len(panels) - 1)
        row_height = max(panel.height for panel in panels)
        row_canvas = Image.new(output_mode, (row_width, row_height), color=background_value)

        current_x = 0
        for panel in panels:
            row_canvas.paste(panel, (current_x, 0))
            current_x += panel.width + gap_x

        row_canvases.append(row_canvas)
        row_widths.append(row_width)
        row_heights.append(row_height)

    total_width = max(row_widths)
    total_height = sum(row_heights) + gap_y * (len(row_canvases) - 1)
    canvas = Image.new(output_mode, (total_width, total_height), color=background_value)

    current_y = 0
    for row_canvas in row_canvases:
        offset_x = (total_width - row_canvas.width) // 2
        canvas.paste(row_canvas, (offset_x, current_y))
        current_y += row_canvas.height + gap_y

    canvas.save(output_path)
