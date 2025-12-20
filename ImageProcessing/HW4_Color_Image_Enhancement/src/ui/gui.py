"""
Color Image Review GUI Module

Tkinter-based GUI for browsing and reviewing color image enhancement results.
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image, ImageTk


@dataclass
class ProcessedItem:
    """Data class for storing processed image information."""
    filename: str
    comparison_figure_path: Optional[str]
    gamma_value: Optional[float] = None
    technique_description: str = ""


class ColorImageReviewApp:
    """Tkinter GUI for browsing processed color enhancement comparison figures."""

    def __init__(self, processed_items: Optional[List[ProcessedItem]], gamma_value: Optional[float]):
        self.processed_items = list(processed_items or [])
        self.default_gamma_value = gamma_value
        self.current_index = 0

        self.comparison_cache: Dict[str, Image.Image] = {}
        self.comparison_photo: Optional[ImageTk.PhotoImage] = None
        self.comparison_source_path: Optional[str] = None
        self.comparison_after_id: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("614410073 - Color Image Enhancement HW4")
        self.root.geometry("1200x900")
        self.root.minsize(900, 650)

        self.build_layout()
        if self.processed_items:
            self.set_controls_enabled(True)
            self.update_processed_items(self.processed_items)
            self.set_processing_message("Processing complete")
        else:
            self.set_controls_enabled(False)
            self.clear_display()
            self.set_processing_message("Processing images...")

    def build_layout(self):
        self.processing_status_var = tk.StringVar(value="Processing images...")
        self.image_status_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")
        self.selection_var = tk.StringVar(value="")

        # Header frame
        header_frame = ttk.Frame(self.root, padding=(12, 12, 12, 6))
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="614410073 張健勳 - Color Image Enhancement", 
                  font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(header_frame, textvariable=self.processing_status_var, 
                  font=("Segoe UI", 12)).pack(side=tk.RIGHT)

        # Control frame
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
            width=35,
        )
        self.selection_combo.bind("<<ComboboxSelected>>", self.handle_combo_selected)
        self.selection_combo.pack(side=tk.LEFT)

        ttk.Label(control_frame, textvariable=self.image_status_var).pack(side=tk.RIGHT)

        # Image display container
        comparison_container = ttk.Frame(self.root, padding=12)
        comparison_container.pack(fill=tk.BOTH, expand=True)
        comparison_container.rowconfigure(0, weight=1)
        comparison_container.columnconfigure(0, weight=1)

        self.comparison_label = tk.Label(comparison_container, borderwidth=0, background="#ffffff")
        self.comparison_label.grid(row=0, column=0, sticky="nsew")
        self.comparison_label.bind("<Configure>", self.handle_comparison_resize)

        # Details frame
        details_frame = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        details_frame.pack(fill=tk.X)
        ttk.Label(details_frame, textvariable=self.detail_var, wraplength=1100).pack(anchor=tk.W)

        # Log frame
        log_frame = ttk.LabelFrame(self.root, text="Processing Logs", padding=(12, 8))
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

    def set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.prev_button.configure(state=state)
        self.next_button.configure(state=state)
        combo_state = "readonly" if enabled else "disabled"
        self.selection_combo.configure(state=combo_state)

    def clear_display(self):
        self.comparison_label.configure(image="", text="Processing...", compound=tk.CENTER)
        self.comparison_label.image = None
        self.comparison_photo = None
        self.comparison_source_path = None
        if self.comparison_after_id:
            self.root.after_cancel(self.comparison_after_id)
            self.comparison_after_id = None
        self.image_status_var.set("")
        self.detail_var.set("Processing images... Please wait.")

    def handle_combo_selected(self, event):
        if not self.processed_items:
            return
        selected_name = self.selection_var.get()
        for index, item in enumerate(self.processed_items):
            if item.filename == selected_name:
                self.current_index = index
                self.update_display()
                break

    def load_comparison_base_image(self, path: Optional[str]) -> Optional[Image.Image]:
        if not path or not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        cache_key = f"{path}:{mtime}" if mtime is not None else path

        cached_image = self.comparison_cache.get(cache_key)
        if cached_image is not None:
            return cached_image

        with Image.open(path) as source_image:
            base_image = source_image.convert("RGB")

        stale_keys = [key for key in self.comparison_cache if key.startswith(path)]
        for key in stale_keys:
            del self.comparison_cache[key]
        
        self.comparison_cache[cache_key] = base_image
        return base_image

    def update_display(self):
        if not self.processed_items:
            self.clear_display()
            return

        item = self.processed_items[self.current_index]
        self.selection_var.set(item.filename)
        self.image_status_var.set(f"Image {self.current_index + 1} of {len(self.processed_items)}")

        gamma_str = f"γ={item.gamma_value:.3f}" if item.gamma_value else "adaptive gamma"
        technique_str = item.technique_description if item.technique_description else \
            "RGB Histogram Eq., HSI Intensity Hist. Eq., HSI Gamma Correction, HSI Saturation Enhancement"
        self.detail_var.set(f"File: {item.filename} | {gamma_str} | Techniques: {technique_str}")

        if item.comparison_figure_path and os.path.exists(item.comparison_figure_path):
            self.comparison_source_path = item.comparison_figure_path
            self.schedule_comparison_resize()
        else:
            self.comparison_label.configure(image="", text="No comparison figure available")
            self.comparison_photo = None

    def schedule_comparison_resize(self):
        if self.comparison_after_id:
            self.root.after_cancel(self.comparison_after_id)
        self.comparison_after_id = self.root.after(50, self.do_comparison_resize)

    def handle_comparison_resize(self, event=None):
        if self.comparison_source_path:
            self.schedule_comparison_resize()

    def do_comparison_resize(self):
        self.comparison_after_id = None
        if not self.comparison_source_path:
            return

        base_image = self.load_comparison_base_image(self.comparison_source_path)
        if base_image is None:
            self.comparison_label.configure(image="", text="Failed to load image")
            return

        label_width = self.comparison_label.winfo_width()
        label_height = self.comparison_label.winfo_height()
        if label_width < 10 or label_height < 10:
            return

        img_width, img_height = base_image.size
        scale = min(label_width / img_width, label_height / img_height, 1.0)
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))

        resized = base_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.comparison_photo = ImageTk.PhotoImage(resized)
        self.comparison_label.configure(image=self.comparison_photo, text="")

    def show_previous(self):
        if self.processed_items and self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def show_next(self):
        if self.processed_items and self.current_index < len(self.processed_items) - 1:
            self.current_index += 1
            self.update_display()

    def update_processed_items(self, items: List[ProcessedItem]):
        self.processed_items = list(items)
        self.current_index = 0
        if self.processed_items:
            filenames = [item.filename for item in self.processed_items]
            self.selection_combo['values'] = filenames
            self.set_controls_enabled(True)
            self.update_display()
        else:
            self.selection_combo['values'] = []
            self.set_controls_enabled(False)
            self.clear_display()

    def set_processing_message(self, message: str):
        self.processing_status_var.set(message)

    def append_log(self, message: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def run(self):
        self.root.mainloop()
