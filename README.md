# Image Gridder

A Python tool for processing grayscale images by:

- Automatically cropping to content
- Adding grid overlay
- Autorotating along the vertical axis
- Saving with customizable output names

### Table of Contents
**[Installation Instructions](#installation)**<br>
**[Usage Instructions](#usage)**<br>
**[Parameter Reference](#parameters)**<br>

# Background

claude.ai is surprisingly excellent at image segmentation and analysis but it can't transform images or superimpose its findings onto a modified image in the way that ChatGPT can do. By adding a grid (todo: with axis markings), claude can express its findings to a user-provided input file in terms of specific grid coordinates. This allows further interrogation and visual validation of the outputs.

## Installation

```bash 

git clone #this repository
brew install uv ;  uv init; uv venv; uv add -r requirements.txt  #setup uv (or use pip natively)
uv pip install -r requirements.txt

```

## Usage

```bash 

export IMAGE_FILE="example_image.jpg"
uv run ./image_gridder_w_autorotate.py --auto-rotate  --margin 100  $IMAGE_FILE   --show --grid-interval 100 --grid-color 0.2 --grid-alpha 0.5
...
INFO:__main__:Using device: mps
INFO:auto_rotater:Best alignment angle: -2.80°
INFO:__main__:Gridlines added with interval: 100px
```

![image](https://github.com/user-attachments/assets/1bff50cf-d320-4f12-aab7-c25948edf507)

This gets much more interesting when combined with Claude, here we go:

![image](https://github.com/user-attachments/assets/50f74ed4-7897-43e6-8367-729759ae72ea)

## As Python Module

Alternatively, just import the module within your existing code

```python 

from image_gridder_w_autorotate import GridParameters, ImageGridder

params = GridParameters(image_path=`input_image.png`) gridder = ImageGridder(params) output_file = gridder.process()

#Custom parameters# 
params = GridParameters( image_path=`input_image.png`, grid_interval=30, grid_color=0.75, alpha_grid=0.15, threshold=0.2, margin=10, show_plt=True ) 

```

## Parameters

- `image_path`: Path to input grayscale image
- `grid_interval`: Spacing between grid lines in pixels (default: 25)
- `grid_color`: Grid line color (0=black, 1=white, default: 0.55)
- `grid_alpha`: Grid transparency (0=transparent, 1=opaque, default: 0.15)
- `threshold`: Threshold for content detection (default: 0.2)
- `margin`: Extra margin around detected content in pixels (default: 10)
- `output_prefix`: Prefix for output filename (default: `cropped_`)
- `output_suffix`: Suffix for output filename (default: `_focused`)
- `auto_rotate`: Provide if the image should be autorotated along the vertical axis
- `show`: Display result using matplotlib interactively (default: False)

## TODO

- <input disabled="" type="checkbox"> Add batch processing capability
- <input disabled="" type="checkbox"> Add grid axis markings
- <input disabled="" type="checkbox"> Resizing of images based on "acceptable" image dimensions (useful for providing images to claude/chatgpt)
- <input disabled="" type="checkbox"> Add tests 
