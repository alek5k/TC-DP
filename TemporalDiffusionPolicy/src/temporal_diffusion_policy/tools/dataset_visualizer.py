from pathlib import Path
import argparse

import numpy as np
from temporal_diffusion_policy.replay_buffer import ReplayBuffer

# /mnt/shared/TemporalDiffusionPolicy/data/inference/LiftQA-Real_b08d56_ckpt20260324_201551_.zarr
# /home/sydney1/Repos/ldp/data/inference/liftqa_h32_o16_s1_a8_e500_epoch250_eval/rollouts.zarr
# /mnt/shared/TemporalDiffusionPolicy/data/inference/LiftQA_b08d56_ckpt20260122_000453_.zarr
# /home/sydney1/Repos/ldp/data/inference/liftqa_h32_o16_s1_a8_e500_epoch50_eval/rollouts.zarr

parser = argparse.ArgumentParser(description="Visualize episodes and timesteps from a zarr dataset")
parser.add_argument("--source", default="/mnt/shared/TemporalDiffusionPolicy/data/demonstration/piano_20eps.zarr", help="Dataset path")
parser.add_argument("--image-key", default="full_image", help="Primary image key to display")
parser.add_argument("--window-title", default="Dataset Visualizer", help="Window title")
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
    image_key: str = "full_image",
    window_title: str = "Dataset Visualizer",
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
        self.source = source
        self.image_key = image_key
        self.window_title = window_title
        self.image_size = image_size
        self.max_attribute_preview_elements = max_attribute_preview_elements
        self.buffer = ReplayBuffer.create_from_path(source, "r")
        self.current_episode_index = 0
        self.current_episode = None

        self.setWindowTitle(window_title)
        self._build_ui()
        self._populate_episodes()
        self._load_episode(0)

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

        # Image panel with space for full_image and hand_image side-by-side
        image_panel = QWidget()
        image_layout = QHBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)

        self.full_image_widget = QLabel("No image")
        self.full_image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.full_image_widget.setMinimumSize(self.image_size, self.image_size)

        self.hand_image_widget = QLabel("No image")
        self.hand_image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hand_image_widget.setMinimumSize(self.image_size, self.image_size)

        image_layout.addWidget(self.full_image_widget)
        image_layout.addWidget(self.hand_image_widget)

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
        episode_lengths = self.buffer.episode_lengths
        for episode_idx, episode_length in enumerate(episode_lengths):
            self.episode_dropdown.addItem(f"Episode {episode_idx} (len={episode_length})")

    def _load_episode(self, episode_idx: int):
        if self.buffer.n_episodes == 0:
            self.current_episode = None
            self.step_slider.setMaximum(0)
            self.step_label.setText("Step: 0/0")
            self.full_image_widget.setText("No episodes found")
            self.hand_image_widget.setText("")
            self.attribute_table.setRowCount(0)
            return

        self.current_episode_index = max(0, min(episode_idx, self.buffer.n_episodes - 1))
        self.current_episode = self.buffer.get_episode(self.current_episode_index)

        if len(self.current_episode) == 0:
            self.step_slider.setMaximum(0)
            self.step_slider.setValue(0)
            self.step_label.setText("Step: 0/0")
            self.full_image_widget.setText("Empty episode")
            self.hand_image_widget.setText("")
            self.attribute_table.setRowCount(0)
            return

        episode_length = len(next(iter(self.current_episode.values())))
        max_step = max(0, episode_length - 1)

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
        if self.current_episode is None or len(self.current_episode) == 0:
            return

        episode_length = len(next(iter(self.current_episode.values())))
        step_idx = max(0, min(step_idx, episode_length - 1))
        self.step_label.setText(f"Step: {step_idx}/{max(0, episode_length - 1)}")

        self._render_image(step_idx)
        self._render_attributes(step_idx)

    def _render_image(self, step_idx: int):
        # Render full_image
        if self.image_key in self.current_episode:
            frame = self.current_episode[self.image_key][step_idx]
            frame_hwc = self._to_hwc_uint8(frame)

            if frame_hwc is not None:
                self._set_image_pixmap(self.full_image_widget, frame_hwc)
            else:
                self.full_image_widget.setText(f"Unsupported shape: {frame.shape}")
        else:
            self.full_image_widget.setText(f"Key '{self.image_key}' not found")

        # Render hand_image if it exists
        if "hand_image" in self.current_episode:
            frame = self.current_episode["hand_image"][step_idx]
            frame_hwc = self._to_hwc_uint8(frame)

            if frame_hwc is not None:
                self._set_image_pixmap(self.hand_image_widget, frame_hwc)
            else:
                self.hand_image_widget.setText(f"Unsupported shape: {frame.shape}")
        else:
            self.hand_image_widget.setText("hand_image\nnot found")

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
        keys = list(self.current_episode.keys())
        self.attribute_table.setRowCount(len(keys))

        for row_idx, key in enumerate(keys):
            value = self.current_episode[key][step_idx]
            formatted = self._format_value(
                value,
                max_elements=self.max_attribute_preview_elements,
            )

            self.attribute_table.setItem(row_idx, 0, QTableWidgetItem(key))
            self.attribute_table.setItem(row_idx, 1, QTableWidgetItem(formatted))

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
