import os
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image, ImageTk


@dataclass
class ProcessedItem:
    filename: str
    comparison_figure_path: Optional[str]


class ImageReviewApp:
    """Tkinter GUI for browsing processed enhancement comparison figures."""

    def __init__(self, processed_items: Optional[List[ProcessedItem]], gamma_value: float):
        self.processed_items = list(processed_items or [])
        self.gamma_value = gamma_value
        self.current_index = 0

        self.comparison_cache: Dict[str, Image.Image] = {}
        self.comparison_photo: Optional[ImageTk.PhotoImage] = None
        self.comparison_source_path: Optional[str] = None
        self._comparison_after_id: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("614410073 - Image Processing HW1")
        self.root.geometry("1100x820")
        self.root.minsize(860, 600)

        self._build_layout()
        if self.processed_items:
            self._set_controls_enabled(True)
            self.update_processed_items(self.processed_items)
            self.set_processing_message("Processing complete")
        else:
            self._set_controls_enabled(False)
            self._clear_display()
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

        comparison_container = ttk.Frame(self.root, padding=12)
        comparison_container.pack(fill=tk.BOTH, expand=True)
        comparison_container.rowconfigure(0, weight=1)
        comparison_container.columnconfigure(0, weight=1)

        self.comparison_label = tk.Label(comparison_container, borderwidth=0, background="#ffffff")
        self.comparison_label.grid(row=0, column=0, sticky="nsew")
        self.comparison_label.bind("<Configure>", self._on_comparison_resize)

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

    def _clear_display(self):
        self.comparison_label.configure(image="", text="Processing...", compound=tk.CENTER)
        self.comparison_label.image = None  # type: ignore[attr-defined]
        self.comparison_photo = None
        self.comparison_source_path = None
        if self._comparison_after_id:
            self.root.after_cancel(self._comparison_after_id)
            self._comparison_after_id = None
        self.image_status_var.set("")
        self.detail_var.set("Processing images... Please wait.")

    def _on_combo_selected(self, _event):  # pragma: no cover - GUI callback
        if not self.processed_items:
            return
        selected_name = self.selection_var.get()
        for idx, item in enumerate(self.processed_items):
            if item.filename == selected_name:
                self.current_index = idx
                self._update_display()
                break

    def _get_comparison_base_image(self, path: Optional[str]) -> Optional[Image.Image]:
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
            self.comparison_cache.pop(key, None)
        self.comparison_cache[cache_key] = base_image
        return base_image

    def _render_comparison_image(self):
        if self._comparison_after_id:
            try:
                self.root.after_cancel(self._comparison_after_id)
            except tk.TclError:
                pass
            self._comparison_after_id = None

        if not self.comparison_source_path:
            return

        label = self.comparison_label
        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1 or height <= 1:
            self._comparison_after_id = self.root.after(60, self._render_comparison_image)
            return

        base_image = self._get_comparison_base_image(self.comparison_source_path)
        if base_image is None or base_image.width == 0 or base_image.height == 0:
            label.configure(image="", text="Comparison figure unavailable", compound=tk.CENTER)
            label.image = None  # type: ignore[attr-defined]
            self.comparison_photo = None
            self.comparison_source_path = None
            return

        available_width = max(width - 24, 1)
        available_height = max(height - 24, 1)
        scale = min(available_width / base_image.width, available_height / base_image.height, 1.0)
        target_width = max(1, int(base_image.width * scale))
        target_height = max(1, int(base_image.height * scale))

        if scale < 1.0:
            rendered = base_image.resize((target_width, target_height), Image.LANCZOS)
        else:
            rendered = base_image

        photo = ImageTk.PhotoImage(rendered)
        label.configure(image=photo, text="", compound=tk.NONE)
        label.image = photo  # type: ignore[attr-defined]
        self.comparison_photo = photo

    def _on_comparison_resize(self, _event):  # pragma: no cover - GUI callback
        if not self.comparison_source_path:
            return
        if self._comparison_after_id:
            try:
                self.root.after_cancel(self._comparison_after_id)
            except tk.TclError:
                pass
        self._comparison_after_id = self.root.after(40, self._render_comparison_image)

    def _update_display(self):  # pragma: no cover - GUI side effect
        if not self.processed_items:
            return

        item = self.processed_items[self.current_index]
        if self._comparison_after_id:
            try:
                self.root.after_cancel(self._comparison_after_id)
            except tk.TclError:
                pass
            self._comparison_after_id = None

        if item.comparison_figure_path and os.path.exists(item.comparison_figure_path):
            self.comparison_source_path = item.comparison_figure_path
            self.comparison_photo = None
            self.comparison_label.configure(image="", text="Loading...", compound=tk.CENTER)
            self.comparison_label.image = None  # type: ignore[attr-defined]
            self._render_comparison_image()
        else:
            self.comparison_source_path = None
            self.comparison_photo = None
            self.comparison_label.configure(image="", text="Comparison figure unavailable", compound=tk.CENTER)
            self.comparison_label.image = None  # type: ignore[attr-defined]

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
            self._clear_display()
            return

        self.comparison_cache.clear()
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