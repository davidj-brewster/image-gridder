# Image Gridder

A Python tool for processing grayscale images by:

Automatically cropping to content
Adding configurable grid overlay
Saving with customizable output names

## Requirements

```bash 
brew install uv #so good
uv init
uv venv
uv add -r requirements.txt
uv pip install -r requirements.txt

```

## Usage

### As a Command Line Tool

Basic usage: 

```bash 

python image_gridder.py input_image.png 

```

With options: 

```bash 

python image_gridder.py input_image.png \
--grid-interval 30 \
--grid-color 0.75 \
--alpha_grid 0.15 \
--threshold 0.2 \
--margin 10 \
--show 

```

### As a Python Module

```python 

from image_gridder import GridParameters, ImageGridder

#Basic usage# 

params = GridParameters(image_path=`input_image.png`) gridder = ImageGridder(params) output_file = gridder.process()

#Custom parameters# 
params = GridParameters( image_path=`input_image.png`, grid_interval=30, grid_color=0.75, alpha_grid=0.15, threshold=0.2, margin=10, show_plt=True ) 

```

## Parameters

`image_path`: Path to input grayscale image
`grid_interval`: Spacing between grid lines in pixels (default: 25)
`grid_color`: Grid line color (0=black, 1=white, default: 0.55)
`alpha_grid`: Grid transparency (0=transparent, 1=opaque, default: 0.15)
`threshold`: Threshold for content detection (default: 0.2)
`margin`: Extra margin around detected content in pixels (default: 10)
`output_prefix`: Prefix for output filename (default: `cropped_`)
`output_suffix`: Suffix for output filename (default: `_focused`)
`show`: Display result using matplotlib (default: False)

## TODO

<input disabled="" type="checkbox"> Add example images
<input disabled="" type="checkbox"> Add support for color images
<input disabled="" type="checkbox"> Add batch processing capability
<input disabled="" type="checkbox"> Add more output format options
<input disabled="" type="checkbox"> Add tests ```
