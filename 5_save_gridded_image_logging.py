import logging.config
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import logging
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class GridParameters:
    image_path: str
    grid_interval: int = 25
    grid_color: float = 0.75
    alpha_grid: float = 0.15
    threshold: float = 0
    margin: int = 10
    output_prefix: str = "cropped_"
    output_suffix: str = "_focused"
    show_plt: bool = False

class ImageGridder:
    def __init__(self, params: GridParameters):
        self.params : GridParameters = params
        self.logger = logging.getLogger(__name__)
        self.image_tensor : torch.Tensor = None
        self.source_prefix :str = None
        self.source_ext :str= None

    def process(self) -> str:
        """Main processing function that returns output filename"""
        self._load_image()
        cropped_tensor = self._crop_image()
        grid_tensor = self._add_grid(cropped_tensor)
        
        if self.params.show_plt:
            self._display_grid(grid_tensor)
            
        return self._save_image(grid_tensor)

    def _load_image(self):
        source_filename = os.path.basename(self.params.image_path)
        self.source_prefix, self.source_ext = os.path.splitext(source_filename)
        
        self.logger.info(f"Loading image: {source_filename}")
        try:
            image = Image.open(self.params.image_path).convert("L")
            self.image_tensor = torch.from_numpy(np.array(image)/ 255.0 )
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {self.params.image_path}")
            raise e
        except Exception as e:
            self.logger.error(f"Error loading image: {self.params.image_path}")
            raise e

    def _crop_image(self) -> torch.Tensor:
        cropped_tensor = self.image_tensor.clone()
        binary_mask = self.image_tensor > self.params.threshold
        non_black_indices = torch.nonzero(binary_mask)
        y_min, x_min = non_black_indices.min(dim=0).values
        y_max, x_max = non_black_indices.max(dim=0).values

        y_min = max(y_min - self.params.margin, 0)
        x_min = max(x_min - self.params.margin, 0)
        y_max = min(y_max + self.params.margin, self.image_tensor.shape[0])
        x_max = min(x_max + self.params.margin, self.image_tensor.shape[1])

        self.logger.info(f"Cropping to bounding box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        return cropped_tensor[y_min:y_max, x_min:x_max]

    def _add_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        grid_tensor = tensor.clone()
        height, width = grid_tensor.shape
        self.logger.info(f"New image dimensions: x=[{width}], y=[{height}]")

        for y in range(0, height, self.params.grid_interval):
            grid_tensor[y, :] = (1.0-self.params.alpha_grid) * grid_tensor[y, :] + self.params.alpha_grid * self.params.grid_color

        for x in range(0, width, self.params.grid_interval):
            grid_tensor[:, x] = (1.0-self.params.alpha_grid) * grid_tensor[:, x] + self.params.alpha_grid * self.params.grid_color

        self.logger.info(f"Gridlines added with interval: {self.params.grid_interval}px")
        
        return grid_tensor

    def _display_grid(self, grid_tensor: torch.Tensor):
        try:
            plt.imshow(grid_tensor.numpy(), cmap="gray")
            plt.title("Image with Faint Grid Overlay (PyTorch)")
            plt.axis("off")
            plt.show()
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            raise e

    def _save_image(self, grid_tensor: torch.Tensor) -> str:
        output_filename = f"{self.params.output_prefix}{self.source_prefix}{self.params.output_suffix}{self.source_ext}"
        self.logger.info(f"Saving cropped image as: {output_filename}")
        
        try:
            image = Image.fromarray((grid_tensor.numpy() * 255).astype(np.uint8))
            image.save(output_filename, compress_level=9)
        except Exception as e:
            self.logger.error(f"Error saving image: {output_filename}: {e}")
            raise e

        height, width = grid_tensor.shape
        self.logger.info(f"Image saved as: {output_filename} - dimensions x=[{width}], y=[{height}]")
        return output_filename

def parse_args() -> GridParameters:
    parser = argparse.ArgumentParser(description='Process image with grid overlay.')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--grid-interval', type=int, default=25, help='Grid spacing in pixels')
    parser.add_argument('--grid-color', type=float, default=0.55, help='Grid color (0=black, 1=white)')
    parser.add_argument('--alpha_grid', type=float, default=0.15, help='Grid transparency (0=transparent, 1=opaque)')
    parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for non-black areas')
    parser.add_argument('--margin', type=int, default=10, help='Margin around detected object')
    parser.add_argument('--output-prefix', default='cropped_', help='Output filename prefix')
    parser.add_argument('--output-suffix', default='_focused', help='Output filename suffix')
    parser.add_argument('--show', action='store_true', help='Show the plot interactively')
    
    args = parser.parse_args()
    return GridParameters(
        image_path=args.image_path,
        grid_interval=args.grid_interval,
        grid_color=args.grid_color,
        alpha_grid=args.alpha_grid,
        threshold=args.threshold,
        margin=args.margin,
        output_prefix=args.output_prefix,
        output_suffix=args.output_suffix,
        show_plt=args.show
    )

def main():
    logging.basicConfig(level=logging.INFO)
    params = parse_args()
    gridder = ImageGridder(params)
    gridder.process()

if __name__ == '__main__':
    main()

