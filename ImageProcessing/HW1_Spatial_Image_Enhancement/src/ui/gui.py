import os
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageTk

from src.schemas.enhancement_results_schema import EnhancementResultsSchema


@dataclass
class ProcessedItem:
    filename: str
    original_image: np.ndarray
    results: EnhancementResultsSchema
    histogram_paths: Dict[str, str]


class ImageReviewApp:
    """Tkinter GUI for browsing processed enhancement results."""

    def __init__(self, processed_items: Optional[List[ProcessedItem]], gamma_value: float):
        self.processed_items = list(processed_items or [])
        self.gamma_value = gamma_value
        self.current_index = 0
        self.variant_keys = ["original", "power_law", "hist_eq", "laplacian"]
        self.display_titles = [
            "Original",
            "Power-law",
            "Histogram Equalization",
            "Laplacian",
        ]
        self.current_image_photos: List[ImageTk.PhotoImage] = []
        self.hist_photo_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.current_hist_photos: List[ImageTk.PhotoImage] = []

        self.root = tk.Tk()
        self.root.title("Image Enhancement Review")
        self.root.geometry("1280x940")

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

        self.image_slots: List[tk.Label] = []
        self.hist_slots: List[tk.Label] = []

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
            hist_label = tk.Label(hist_container, borderwidth=1, relief=tk.SOLID, background="#ffffff")
            hist_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            self.hist_slots.append(hist_label)

        details_frame = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        details_frame.pack(fill=tk.X)
        ttk.Label(details_frame, textvariable=self.detail_var).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(self.root, text="Logs", padding=(12, 8))
        log_frame.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 12))
        self.log_text = tk.Text(
            log_frame,
            height=8,
            state=tk.DISABLED,
            wrap=tk.NONE,
            font=("Consolas", 10),
            background="#101010",
            foreground="#eaeaea",
            insertbackground="#eaeaea",
        )
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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
        self.current_image_photos = []
        self.current_hist_photos = []
        self.image_status_var.set("")
        self.detail_var.set("Processing images... Please wait.")

    def _on_combo_selected(self, event):  # pragma: no cover - GUI callback
        if not self.processed_items:
            return
        selected_name = self.selection_var.get()
        for idx, item in enumerate(self.processed_items):
            if item.filename == selected_name:
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

    def _load_hist_photo(self, path: str) -> ImageTk.PhotoImage:
        if path in self.hist_photo_cache:
            return self.hist_photo_cache[path]
        image = Image.open(path)
        photo = ImageTk.PhotoImage(image)
        image.close()
        self.hist_photo_cache[path] = photo
        return photo

    def _update_display(self):  # pragma: no cover - GUI side effect
        if not self.processed_items:
            return

        item = self.processed_items[self.current_index]
        arrays = [
            item.original_image,
            item.results.power_law,
            item.results.hist_eq,
            item.results.laplacian,
        ]

        self.current_image_photos = []
        for slot, array in zip(self.image_slots, arrays):
            photo = self._array_to_photo(array)
            slot.configure(image=photo, text="", compound=tk.NONE)
            slot.image = photo  # type: ignore[attr-defined]
            self.current_image_photos.append(photo)

        self.current_hist_photos = []
        for slot, key in zip(self.hist_slots, self.variant_keys):
            path = item.histogram_paths.get(key)
            if path and os.path.exists(path):
                try:
                    photo = self._load_hist_photo(path)
                    slot.configure(image=photo, text="", compound=tk.NONE)
                    slot.image = photo  # type: ignore[attr-defined]
                    self.current_hist_photos.append(photo)
                except Exception:
                    slot.configure(image="", text="Histogram unavailable", compound=tk.CENTER)
                    slot.image = None  # type: ignore[attr-defined]
            else:
                slot.configure(image="", text="Histogram unavailable", compound=tk.CENTER)
                slot.image = None  # type: ignore[attr-defined]

        self.selection_var.set(item.filename)
        self.image_status_var.set(f"Image {self.current_index + 1} of {len(self.processed_items)}")
        self.detail_var.set(f"Gamma: {self.gamma_value}    File: {item.filename}")

    def append_log_message(self, message: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def update_processed_items(self, processed_items: List[ProcessedItem]):
        self.processed_items = list(processed_items)
        if not self.processed_items:
            self._set_controls_enabled(False)
            self.selection_combo.configure(values=[])
            self._clear_slots()
            return

        self.selection_combo.configure(values=[item.filename for item in self.processed_items])
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

    def schedule_log_message(self, message: str):
        try:
            self.root.after(0, lambda: self.append_log_message(message))
        except tk.TclError:
            pass

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