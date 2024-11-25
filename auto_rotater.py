from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt
import sys
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

@dataclass
class RotationParameters:
    auto_rotate: bool = True
    rotation_angle: float = 0.0

class RotationWindow(QMainWindow):
    def __init__(self, tensor: torch.Tensor, rotate_fn, parent=None):
        super().__init__(parent)
        self.tensor = tensor
        self.rotate_fn = rotate_fn
        self.result = {"angle": 0.0, "confirmed": False}
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Adjust Rotation")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(self.canvas)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        layout.addWidget(controls)

        self.angle_label = QLabel("Angle: 0.0°")
        controls_layout.addWidget(self.angle_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(-300, 300)
        self.slider.valueChanged.connect(self.update_rotation)
        controls_layout.addWidget(self.slider)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.confirm)
        controls_layout.addWidget(confirm_btn)

        self.update_preview(0.0)

    def update_preview(self, angle):
        rotated = self.rotate_fn(self.tensor, angle)
        self.ax.clear()
        self.ax.imshow(rotated.numpy(), cmap='gray')
        self.ax.axvline(x=rotated.shape[1]//2, color='r', alpha=0.3)
        self.canvas.draw()

    def update_rotation(self):
        angle = self.slider.value() / 10.0
        self.result["angle"] = angle
        self.angle_label.setText(f"Angle: {angle:.1f}°")
        self.update_preview(angle)

    def confirm(self):
        self.result["confirmed"] = True
        self.close()

class AutoRotator:
    def __init__(self, params: RotationParameters):
        self.params = params
        self.logger = logging.getLogger(__name__)

    def detect_and_adjust_midline(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.params.auto_rotate:
            return self._auto_rotate(tensor)
        else:
            return self._interactive_rotation(tensor, self.params.rotation_angle)

    def _auto_rotate(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.unsqueeze(0).unsqueeze(0)
        angles = torch.linspace(-10, 10, 41)
        scores = torch.tensor([
            self._get_symmetry_score(x, angle.item()) 
            for angle in angles
        ])
        best_angle = angles[torch.argmax(scores)].item()
        self.logger.info(f"Best alignment angle: {best_angle:.2f}°")
        return self._rotate_tensor(tensor, best_angle)

    def detect_and_adjust_midline(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Aligns image using PyTorch operations to maximize symmetry along vertical midline.
        
        Args:
            tensor (torch.Tensor): Input image tensor [H,W]
            
        Returns:
            torch.Tensor: Aligned image tensor [H,W]
        """
        if self.params.auto_rotate:
            # Add batch and channel dims for torchvision
            x = tensor.unsqueeze(0).unsqueeze(0)
            
            # Compute edges using diff
            padding_y = torch.nn.functional.pad(x, (0, 0, 0, 1), mode='replicate')
            padding_x = torch.nn.functional.pad(x, (0, 1, 0, 0), mode='replicate')
            
            dy = torch.diff(padding_y, dim=2)
            dx = torch.diff(padding_x, dim=3)
            edge_magnitude = torch.sqrt(dx**2 + dy**2)

            angles = torch.linspace(-10, 10, 201)
            scores = torch.tensor([
                self.get_symmetry_score(x, angle.item()) 
                for angle in angles
            ])
            self.logger.debug(f"Alignmnet scores {scores}")
            best_angle = angles[torch.argmax(scores)].item()
            self.logger.info(f"Best alignment angle: {best_angle:.2f}°")
            
            return self._rotate_tensor(tensor, best_angle)
        else:
            return tensor #self._interactive_rotation(tensor, best_angle)

    def get_symmetry_score(self, img: torch.Tensor, angle: float) -> float:
        """
        Compute symmetry score for given angle.
        """
        # Rotate image first
        rotated = TF.rotate(
            img, 
            angle=-angle,  # Negative for clockwise rotation
            interpolation=TF.InterpolationMode.BILINEAR
        )
        
        # Compute edges on rotated image
        padding_y = torch.nn.functional.pad(rotated, (0, 0, 0, 1), mode='replicate')
        padding_x = torch.nn.functional.pad(rotated, (0, 1, 0, 0), mode='replicate')
        
        dy = torch.diff(padding_y, dim=2)
        dx = torch.diff(padding_x, dim=3)
        edge_magnitude = torch.sqrt(dx**2 + dy**2)
        
        _, _, h, w = edge_magnitude.shape
        mid = w // 2
        
        # Create distance weights
        x_coords = torch.arange(w, device=img.device, dtype=torch.float32)
        dist_from_mid = torch.abs(x_coords - mid) / (w/4)
        dist_weights = torch.exp(-dist_from_mid**2)
        dist_weights = dist_weights.view(1, 1, 1, -1).expand(-1, -1, h, -1)
        
        # Weight edges
        weighted = edge_magnitude * dist_weights
        
        # Compare sides
        left = weighted[:, :, :, :mid]
        right = torch.flip(weighted[:, :, :, mid:], [3])
        
        min_width = min(left.size(3), right.size(3))
        left = left[:, :, :, -min_width:]
        right = right[:, :, :, :min_width]
        
        # Focus on strongest edges
        threshold = torch.quantile(weighted, 0.9)
        edge_mask = (left + right) > threshold
        
        if edge_mask.sum() > 0:
            left_strong = left[edge_mask]
            right_strong = right[edge_mask]
            corr = torch.corrcoef(torch.stack([
                left_strong.flatten(),
                right_strong.flatten()
            ]))[0,1]
        else:
            corr = torch.corrcoef(torch.stack([
                left.flatten(),
                right.flatten()
            ]))[0,1]
            
        return corr.item()
            
    def _rotate_tensor(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotate image tensor by given angle using torchvision.
        
        Args:
            tensor (torch.Tensor): Input image tensor
            angle (float): Rotation angle in degrees
            
        Returns:
            torch.Tensor: Rotated image tensor
        """
        # Add batch and channel dimensions required by torchvision
        batched = tensor.unsqueeze(0).unsqueeze(0)
    
        # Perform rotation - torchvision uses counter-clockwise rotation
        rotated = TF.rotate(
            batched,
            angle=-angle,  # Negative for clockwise rotation
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0
        )
        
        # Remove batch and channel dimensions
        return rotated.squeeze(0).squeeze(0)

    def _interactive_rotation(self, tensor: torch.Tensor, estimated_angle: float) -> torch.Tensor:
        """
        Interactive rotation adjustment using PyQt6.
        """
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            window = RotationWindow(tensor, self._rotate_tensor)
            window.slider.setValue(int(estimated_angle * 10))
            window.show()
            app.exec()

            if window.result["confirmed"]:
                return self._rotate_tensor(tensor, window.result["angle"])
            return self._rotate_tensor(tensor, estimated_angle)

        except Exception as e:
            self.logger.error(f"Interactive rotation failed: {e}")
            return self._rotate_tensor(tensor, estimated_angle)

