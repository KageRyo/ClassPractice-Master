import io
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.schemas.enhancement_results_schema import EnhancementResultsSchema
from src.utils.image_utils import ImageHistogramCalculator

ProcessedItem = Tuple[str, np.ndarray, EnhancementResultsSchema]


class ImageReviewApp:
    """Tkinter GUI for browsing processed enhancement results."""

    def __init__(self, processed_items: Optional[List[ProcessedItem]], gamma_value: float):
        self.processed_items = list(processed_items or [])
        self.gamma_value = gamma_value
        self.current_index = 0
        self.display_titles = [
            "Original",
            "Power-law",
            "Histogram Equalization",
            "Laplacian",
        ]
        self.photo_cache: dict[str, list[ImageTk.PhotoImage]] = {"images": [], "hist": []}
        self.histogram_calculator = ImageHistogramCalculator()

        self.root = tk.Tk()
        self.root.title("Image Enhancement Review")
        self.root.geometry("1280x900")

        self._build_layout()
        if self.processed_items:
            self._set_controls_enabled(True)
            self.update_processed_items(self.processed_items)
            self.set_processing_message("Processing complete")
        else:
            self._set_controls_enabled(False)
            self._clear_slots()
            self.set_processing_message("Processing images...")

    def _build_layout(self):
        self.processing_status_var = tk.StringVar(value="Processing images...")
        self.image_status_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")
        self.selection_var = tk.StringVar(value="")

        header_frame = ttk.Frame(self.root, padding=(12, 12, 12, 6))
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="614410073 張健勳", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(header_frame, textvariable=self.processing_status_var, font=("Segoe UI", 12)).pack(side=tk.RIGHT)

        control_frame = ttk.Frame(self.root, padding=(12, 6))
        control_frame.pack(fill=tk.X)

        self.prev_button = ttk.Button(control_frame, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(control_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(control_frame, text="Jump to:").pack(side=tk.LEFT, padx=(15, 0))
        self.selection_combo = ttk.Combobox(
            control_frame,
            textvariable=self.selection_var,
            state="disabled",
            width=30,
        )
        self.selection_combo.bind("<<ComboboxSelected>>", self._on_combo_selected)
        self.selection_combo.pack(side=tk.LEFT)

        ttk.Label(control_frame, textvariable=self.image_status_var).pack(side=tk.RIGHT)

        content_frame = ttk.Frame(self.root, padding=12)
        content_frame.pack(fill=tk.BOTH, expand=True)
        for column_index in range(4):
            content_frame.columnconfigure(column_index, weight=1)
        content_frame.rowconfigure(0, weight=3)
        content_frame.rowconfigure(1, weight=2)

        self.image_slots: list[tk.Label] = []
        self.hist_slots: list[tk.Label] = []

        for index, title in enumerate(self.display_titles):
            image_container = ttk.Frame(content_frame, padding=6)
            image_container.grid(row=0, column=index, sticky="nsew")
            ttk.Label(image_container, text=title, font=("Segoe UI", 12, "bold")).pack(anchor=tk.N)
            image_label = tk.Label(image_container, borderwidth=1, relief=tk.SOLID, background="#1f1f1f")
            image_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            self.image_slots.append(image_label)

            hist_container = ttk.Frame(content_frame, padding=6)
            hist_container.grid(row=1, column=index, sticky="nsew")
            ttk.Label(hist_container, text=f"{title} Histogram", font=("Segoe UI", 11)).pack(anchor=tk.N)
            hist_label = tk.Label(hist_container, borderwidth=1, relief=tk.SOLID, background="white")
            hist_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            self.hist_slots.append(hist_label)

        details_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        details_frame.pack(fill=tk.X)
        ttk.Label(details_frame, textvariable=self.detail_var).pack(anchor=tk.W)

    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.prev_button.configure(state=state)
        self.next_button.configure(state=state)
        combo_state = "readonly" if enabled else "disabled"
        self.selection_combo.configure(state=combo_state)

    def _clear_slots(self):
        for slot in self.image_slots + self.hist_slots:
            slot.configure(image="", text="Processing...", compound=tk.CENTER)
            slot.image = None  # type: ignore[attr-defined]
        self.photo_cache = {"images": [], "hist": []}
        self.image_status_var.set("")
        self.detail_var.set("Processing images... Please wait.")

    def _on_combo_selected(self, event):  # pragma: no cover - GUI callback
        if not self.processed_items:
            return
        selected_name = self.selection_var.get()
        for idx, (name, _, _) in enumerate(self.processed_items):
            if name == selected_name:
                self.current_index = idx
                self._update_display()
                break

    def _array_to_photo(self, array: np.ndarray) -> ImageTk.PhotoImage:
        arr = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
        if image.mode != "L":
            image = image.convert("L")

        max_dimension = 360
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

        return ImageTk.PhotoImage(image)

    def _hist_to_photo(self, array: np.ndarray, title: str) -> ImageTk.PhotoImage:
        histogram = self.histogram_calculator.calculate_image_pixel_histogram(array)
        figure = Figure(figsize=(3.0, 2.2), dpi=100)
        axis = figure.add_subplot(111)
        axis.bar(range(256), histogram, color="#4c72b0", alpha=0.85)
        axis.set_xlim(0, 255)
        axis.set_title(f"{title} Histogram", fontsize=9)
        axis.set_xlabel("Intensity", fontsize=8)
        axis.set_ylabel("Frequency", fontsize=8)
        axis.tick_params(labelsize=8)
        figure.tight_layout()

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(figure)
        buffer.seek(0)
        image = Image.open(buffer)
        rgb_image = image.convert("RGB")
        photo = ImageTk.PhotoImage(rgb_image)
        rgb_image.close()
        image.close()
        buffer.close()
        return photo

    def _update_display(self):  # pragma: no cover - GUI side effect
        if not self.processed_items:
            return

        self.photo_cache = {"images": [], "hist": []}
        name, original_image, results = self.processed_items[self.current_index]
        arrays = [
            original_image,
            results.power_law,
            results.hist_eq,
            results.laplacian,
        ]

        for slot, array in zip(self.image_slots, arrays):
            photo = self._array_to_photo(array)
            slot.configure(image=photo, text="", compound=tk.NONE)
            slot.image = photo  # type: ignore[attr-defined]
            self.photo_cache["images"].append(photo)

        for slot, array, title in zip(self.hist_slots, arrays, self.display_titles):
            photo = self._hist_to_photo(array, title)
            slot.configure(image=photo, text="", compound=tk.NONE)
            slot.image = photo  # type: ignore[attr-defined]
            self.photo_cache["hist"].append(photo)

        self.selection_var.set(name)
        self.image_status_var.set(f"Image {self.current_index + 1} of {len(self.processed_items)}")
        self.detail_var.set(f"Gamma: {self.gamma_value}    File: {name}")

    def update_processed_items(self, processed_items: List[ProcessedItem]):
        self.processed_items = list(processed_items)
        if not self.processed_items:
            self._set_controls_enabled(False)
            self.selection_combo.configure(values=[])
            self._clear_slots()
            return

        self.selection_combo.configure(values=[item[0] for item in self.processed_items])
        self.current_index = 0
        self._set_controls_enabled(True)
        self._update_display()

    def set_processing_message(self, message: str):
        self.processing_status_var.set(message)

    def schedule_processed_items(self, processed_items: List[ProcessedItem]):
        self.root.after(0, lambda: self.update_processed_items(processed_items))

    def schedule_processing_message(self, message: str):
        self.root.after(0, lambda: self.set_processing_message(message))

    def schedule_error(self, message: str):
        self.root.after(0, lambda: self.show_error(message))

    def show_next(self):  # pragma: no cover - GUI callback
        if not self.processed_items:
            return
        self.current_index = (self.current_index + 1) % len(self.processed_items)
        self._update_display()

    def show_previous(self):  # pragma: no cover - GUI callback
        if not self.processed_items:
            return
        self.current_index = (self.current_index - 1) % len(self.processed_items)
        self._update_display()

    def show_error(self, message: str):  # pragma: no cover - GUI side effect
        messagebox.showerror("Processing Error", message)
        self.root.quit()

    def run(self):  # pragma: no cover - GUI loop
        self.root.mainloop()