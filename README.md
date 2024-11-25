# Image Gridder

A Python tool for processing grayscale images by:

- Automatically cropping to content
- Adding grid overlay
- Autorotating along the vertical axis
- Saving with customizable output names

# Background

claude.ai is surprisingly excellent at image segmentation and analysis but it can't transform images or superimpose its findings onto a modified image in the way that ChatGPT can do. By adding a grid (todo: with axis markings), claude can express its findings to a user-provided input file in terms of specific grid coordinates. This allows further interrogation and visual validation of the outputs.

## Requirements

```bash 
brew install uv #uv simplifies package management
uv init
uv venv
uv add -r requirements.txt
uv pip install -r requirements.txt

```

## Usage

### As a Command Line Tool

Basic usage: 

```bash 

(python | uv run) image_gridder_w_autorotate.py input_image.png 

```

With options: 

```bash 

(python | uv run) image_gridder.py input_image.png \
--grid-interval 30 \
--grid-color 0.75 \
--alpha_grid 0.15 \
--threshold 0.2 \
--margin 10 \
--[auto_rotate]
--[show]

```

### As a Python Module

```python 

from image_gridder_w_autorotate import GridParameters, ImageGridder

#Basic usage# 
(python | uv run) image_gridder_basic).py input_image.png 

params = GridParameters(image_path=`input_image.png`) gridder = ImageGridder(params) output_file = gridder.process()

#Custom parameters# 
params = GridParameters( image_path=`input_image.png`, grid_interval=30, grid_color=0.75, alpha_grid=0.15, threshold=0.2, margin=10, show_plt=True ) 

```

## Parameters

- `image_path`: Path to input grayscale image
- `grid_interval`: Spacing between grid lines in pixels (default: 25)
- `grid_color`: Grid line color (0=black, 1=white, default: 0.55)
- `alpha_grid`: Grid transparency (0=transparent, 1=opaque, default: 0.15)
- `threshold`: Threshold for content detection (default: 0.2)
- `margin`: Extra margin around detected content in pixels (default: 10)
- `output_prefix`: Prefix for output filename (default: `cropped_`)
- `output_suffix`: Suffix for output filename (default: `_focused`)
- `auto_rotate`: Provide if the image should be autorotated along the vertical axis
- `show`: Display result using matplotlib interactively (default: False)

## TODO

- <input disabled="" type="checkbox"> Add example images
- <input disabled="" type="checkbox"> Add batch processing capability
- <input disabled="" type="checkbox"> Add batch processing capability
- <input disabled="" type="checkbox"> Add grid markings
- <input disabled="" type="checkbox"> Resizing of images based on "acceptable" image dimensions (useful for providing images to claude/chatgpt)
- <input disabled="" type="checkbox"> Add tests 
