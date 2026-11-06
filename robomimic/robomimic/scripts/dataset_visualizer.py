from pathlib import Path
import argparse
import os

import h5py
import numpy as np
# /home/sydney1/Repos/robomimic/datasets/transport/ph/image_v15.hdf5
# "/home/sydney1/Repos/robomimic/rollouts/temporal/waitatgoal/tc_diffusion_policy_mod_nocrop/20260810170345_model_epoch_100_waitatgoal_image_success_0.98.hdf5"

DEFAULT_PATH = "/home/sydney1/Repos/robomimic/datasets/temporal/liftqa_image.hdf5"

parser = argparse.ArgumentParser(description="Visualize episodes and timesteps from a robomimic HDF5 dataset")
parser.add_argument("--source", default=DEFAULT_PATH, help="Path to HDF5 dataset file")
# parser.add_argument("--source", default="rollouts/can/ph/tc_diffusion_policy/20260501172718_model_epoch_240_image_v15_success_1.0.pth.hdf5", help="Path to HDF5 dataset file")
parser.add_argument("--image-key", default=None, help="Primary image key to display (if not set, will detect first image key)")
parser.add_argument("--window-title", default="Robomimic Dataset Visualizer", help="Window title")
parser.add_argument("--image-size", type=int, default=512, help="Displayed image size")
parser.add_argument("--max-attribute-preview-elements", type=int, default=32, help="Max flattened elements to preview per attribute")


try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    PYQT_IMPORT_ERROR = None
except ImportError as error:
    PYQT_IMPORT_ERROR = error


def run_visualizer(
    source: str,
    image_key: str = None,
    window_title: str = "Robomimic Dataset Visualizer",
    image_size: int = 512,
    max_attribute_preview_elements: int = 32,
):
    if PYQT_IMPORT_ERROR is not None:
        raise ImportError(
            "PyQt6 is required for dataset_visualizer.py. Install it with: pip install PyQt6"
        ) from PYQT_IMPORT_ERROR

    app = QApplication([])
    window = DatasetVisualizer(
        source=source,
        image_key=image_key,
        window_title=window_title,
        image_size=image_size,
        max_attribute_preview_elements=max_attribute_preview_elements,
    )
    window.resize(1400, 1200)
    window.show()
    app.exec()


class DatasetVisualizer(QMainWindow):
    def __init__(self, source: str, image_key: str, window_title: str, image_size: int, max_attribute_preview_elements: int):
        super().__init__()
        
        # Expand and validate path
        self.source = os.path.expanduser(source)
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Dataset file not found at {self.source}")
        
        # Open HDF5 file
        self.h5_file = h5py.File(self.source, 'r')
        
        # Get list of demonstrations
        if "data" not in self.h5_file:
            raise ValueError("Invalid robomimic HDF5 file: missing 'data' group")
        
        # Sort demos numerically (demo_1, demo_2, ..., demo_10, etc.)
        try:
            self.demos = sorted(list(self.h5_file["data"].keys()), 
                              key=lambda x: int(x.split('_')[-1]) if '_' in x else int(x))
        except (ValueError, IndexError):
            # Fallback to lexicographic sort if numeric parsing fails
            self.demos = sorted(list(self.h5_file["data"].keys()))
        if not self.demos:
            raise ValueError("No demonstrations found in dataset")
        
        self.window_title = window_title
        self.image_size = image_size
        self.max_attribute_preview_elements = max_attribute_preview_elements
        
        # Determine available image keys from first demo
        first_demo = self.demos[0]
        demo_obs_keys = list(self.h5_file[f"data/{first_demo}/obs"].keys())
        image_keys = [k for k in demo_obs_keys if "image" in k.lower()]
        print(image_keys)
        if not image_keys:
            raise ValueError("No image observations found in dataset")
        
        # Use provided image_key or default to first found
        if image_key is None:
            self.image_key = image_keys[0]
        else:
            if image_key not in image_keys:
                raise ValueError(f"Image key '{image_key}' not found. Available: {image_keys}")
            self.image_key = image_key
        self.image_keys = [self.image_key] + [key for key in image_keys if key != self.image_key]
        
        self.current_episode_index = 0
        self.current_demo_key = None
        self.current_episode_length = 0
        
        self.setWindowTitle(window_title)
        self._build_ui()
        self._populate_episodes()
        self._load_episode(0)

    def closeEvent(self, event):
        """Properly close the HDF5 file when the window closes."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
        event.accept()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        layout = QGridLayout(central)

        control_row = QHBoxLayout()
        self.episode_label = QLabel("Episode:")
        self.episode_dropdown = QComboBox()
        self.episode_dropdown.currentIndexChanged.connect(self._on_episode_changed)

        self.step_label = QLabel("Step: 0/0")
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(0)
        self.step_slider.valueChanged.connect(self._on_step_changed)

        control_row.addWidget(self.episode_label)
        control_row.addWidget(self.episode_dropdown)
        control_row.addSpacing(16)
        control_row.addWidget(self.step_label)
        control_row.addWidget(self.step_slider)

        # Image panel with space for all available image observations side-by-side
        image_panel = QWidget()
        image_layout = QHBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)

        self.image_widgets = {}
        for key in self.image_keys:
            image_column = QWidget()
            image_column_layout = QVBoxLayout(image_column)
            image_column_layout.setContentsMargins(0, 0, 0, 0)
            image_column_layout.setSpacing(4)

            title = QLabel(key)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            image_widget = QLabel("No image")
            image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_widget.setMinimumSize(self.image_size, self.image_size)

            image_column_layout.addWidget(title)
            image_column_layout.addWidget(image_widget)
            image_layout.addWidget(image_column)
            self.image_widgets[key] = image_widget

        self.attribute_table = QTableWidget()
        self.attribute_table.setColumnCount(2)
        self.attribute_table.setHorizontalHeaderLabels(["Attribute", "Value (current step)"])
        self.attribute_table.horizontalHeader().setStretchLastSection(True)
        self.attribute_table.setWordWrap(True)

        layout.addLayout(control_row, 0, 0)
        layout.addWidget(image_panel, 1, 0)
        layout.addWidget(self.attribute_table, 2, 0)

    def _populate_episodes(self):
        self.episode_dropdown.clear()
        for demo_idx, demo_key in enumerate(self.demos):
            demo_group = self.h5_file[f"data/{demo_key}"]
            demo_length = demo_group.attrs.get("num_samples", 0)
            self.episode_dropdown.addItem(f"{demo_key} (len={demo_length})", userData=demo_key)

    def _load_episode(self, episode_idx: int):
        if not self.demos:
            self.current_demo_key = None
            self.current_episode_length = 0
            self.step_slider.setMaximum(0)
            self.step_label.setText("Step: 0/0")
            for image_widget in self.image_widgets.values():
                image_widget.setText("No episodes found")
            self.attribute_table.setRowCount(0)
            return

        self.current_episode_index = max(0, min(episode_idx, len(self.demos) - 1))
        self.current_demo_key = self.demos[self.current_episode_index]
        
        demo_group = self.h5_file[f"data/{self.current_demo_key}"]
        self.current_episode_length = demo_group.attrs.get("num_samples", 0)

        if self.current_episode_length == 0:
            self.step_slider.setMaximum(0)
            self.step_slider.setValue(0)
            self.step_label.setText("Step: 0/0")
            for image_widget in self.image_widgets.values():
                image_widget.setText("Empty episode")
            self.attribute_table.setRowCount(0)
            return

        max_step = max(0, self.current_episode_length - 1)

        self.step_slider.blockSignals(True)
        self.step_slider.setMaximum(max_step)
        self.step_slider.setValue(0)
        self.step_slider.blockSignals(False)

        self._render_step(0)

    def _on_episode_changed(self, index: int):
        if index < 0:
            return
        self._load_episode(index)

    def _on_step_changed(self, step_idx: int):
        self._render_step(step_idx)

    def _render_step(self, step_idx: int):
        if self.current_demo_key is None or self.current_episode_length == 0:
            return

        step_idx = max(0, min(step_idx, self.current_episode_length - 1))
        self.step_label.setText(f"Step: {step_idx}/{max(0, self.current_episode_length - 1)}")

        self._render_image(step_idx)
        self._render_attributes(step_idx)

    def _render_image(self, step_idx: int):
        demo_group = self.h5_file[f"data/{self.current_demo_key}"]

        for image_key in self.image_keys:
            image_widget = self.image_widgets[image_key]
            if f"obs/{image_key}" in demo_group:
                frame = demo_group[f"obs/{image_key}"][step_idx]
                frame_hwc = self._to_hwc_uint8(frame)

                if frame_hwc is not None:
                    self._set_image_pixmap(image_widget, frame_hwc)
                else:
                    image_widget.setText(f"Unsupported shape: {frame.shape}")
            else:
                image_widget.setText(f"Key 'obs/{image_key}' not found")

    def _set_image_pixmap(self, label: QLabel, frame_hwc: np.ndarray):
        height, width, channels = frame_hwc.shape
        bytes_per_line = channels * width
        image = QImage(
            frame_hwc.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.image_size,
            self.image_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def _render_attributes(self, step_idx: int):
        if self.current_demo_key is None:
            return
        
        demo_group = self.h5_file[f"data/{self.current_demo_key}"]
        
        # Collect all keys from the demo (obs, next_obs, actions, etc.)
        keys_to_display = []
        
        # Add obs keys
        if "obs" in demo_group:
            for key in sorted(demo_group["obs"].keys()):
                keys_to_display.append(f"obs/{key}")
        
        # Add next_obs keys
        if "next_obs" in demo_group:
            for key in sorted(demo_group["next_obs"].keys()):
                keys_to_display.append(f"next_obs/{key}")
        
        # Add other keys (actions, rewards, etc.)
        for key in sorted(demo_group.keys()):
            if key not in ["obs", "next_obs"]:
                keys_to_display.append(key)
        
        self.attribute_table.setRowCount(len(keys_to_display))

        for row_idx, key_path in enumerate(keys_to_display):
            try:
                if "/" in key_path:
                    group_name, key_name = key_path.split("/", 1)
                    value = demo_group[group_name][key_name][step_idx]
                else:
                    value = demo_group[key_path][step_idx]
                
                formatted = self._format_value(value, max_elements=self.max_attribute_preview_elements)

                self.attribute_table.setItem(row_idx, 0, QTableWidgetItem(key_path))
                self.attribute_table.setItem(row_idx, 1, QTableWidgetItem(formatted))
            except Exception as e:
                self.attribute_table.setItem(row_idx, 0, QTableWidgetItem(key_path))
                self.attribute_table.setItem(row_idx, 1, QTableWidgetItem(f"Error: {str(e)}"))

        self.attribute_table.resizeRowsToContents()

    @staticmethod
    def _to_hwc_uint8(image: np.ndarray):
        if not isinstance(image, np.ndarray):
            return None

        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] != 3:
                return None
        else:
            return None

        if np.issubdtype(arr.dtype, np.floating):
            if float(np.nanmax(arr)) <= 1.0 + 1e-4:
                arr = (arr * 255.0).clip(0, 255)
            else:
                arr = arr.clip(0, 255)
        else:
            arr = arr.clip(0, 255)

        return np.ascontiguousarray(arr.astype(np.uint8))

    @staticmethod
    def _format_value(value, max_elements: int = 8) -> str:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                scalar = value.item()
                if isinstance(scalar, (float, np.floating)):
                    return f"{float(scalar):.4f}"
                return str(scalar)

            flat = value.reshape(-1)
            formatter = {'float_kind': lambda x: f"{float(x):.4f}"}
            if max_elements <= 0 or flat.size <= max_elements:
                preview = np.array2string(
                    flat,
                    separator=", ",
                    max_line_width=10_000_000,
                    formatter=formatter,
                )
            else:
                preview = np.array2string(
                    flat[:max_elements],
                    separator=", ",
                    max_line_width=10_000_000,
                    formatter=formatter,
                )
                preview = f"{preview} ..."
            preview = preview.replace("\n", " ")
            return f"shape={value.shape}, dtype={value.dtype}, data={preview}"

        if np.isscalar(value):
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4f}"
            return str(value)

        return str(value)


def main():
    args = parser.parse_args()
    run_visualizer(
        source=args.source,
        image_key=args.image_key,
        window_title=args.window_title,
        image_size=args.image_size,
        max_attribute_preview_elements=args.max_attribute_preview_elements,
    )


if __name__ == "__main__":
    main()
