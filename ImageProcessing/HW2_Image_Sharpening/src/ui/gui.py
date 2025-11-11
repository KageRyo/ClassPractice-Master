import logging
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, List, Optional

from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


@dataclass
class ProcessedItem:
    filename: str
    comparison_figure_path: Optional[str]
    detail_text: str = ''


class ImageReviewApp:
    """Tkinter UI for browsing sharpening comparison figures and logs."""

    def __init__(self, processed_items: Optional[List[ProcessedItem]], parameter_summary: str):
        self.processed_items = list(processed_items or [])
        self.parameter_summary = parameter_summary
        self.current_index = 0

        self.comparison_cache: Dict[str, Image.Image] = {}
        self.comparison_photo: Optional[ImageTk.PhotoImage] = None
        self.current_base_image: Optional[Image.Image] = None

        self.root = tk.Tk()
        self.root.title('614410073 - Image Processing HW2')
        self.root.geometry('1180x860')
        self.root.minsize(880, 600)

        self.build_layout()
        if self.processed_items:
            self.update_processed_items(self.processed_items)
        else:
            self.set_controls_enabled(False)
            self.processing_status_var.set('Waiting for processing to start...')
            self.image_status_var.set('No processed images yet.')

    def build_layout(self) -> None:
        self.processing_status_var = tk.StringVar(value='Processing images...')
        self.image_status_var = tk.StringVar(value='')
        self.detail_var = tk.StringVar(value=self.parameter_summary)
        self.selection_var = tk.StringVar(value='')

        header_frame = ttk.Frame(self.root, padding=(12, 12, 12, 6))
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text='614410073 張健勳', font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT)
        ttk.Label(header_frame, textvariable=self.processing_status_var, font=('Segoe UI', 12)).pack(side=tk.RIGHT)

        control_frame = ttk.Frame(self.root, padding=(12, 6))
        control_frame.pack(fill=tk.X)

        self.prev_button = ttk.Button(control_frame, text='Previous', command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(control_frame, text='Next', command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(control_frame, text='Jump to:').pack(side=tk.LEFT, padx=(15, 0))
        self.selection_combo = ttk.Combobox(control_frame, textvariable=self.selection_var, state='disabled', width=32)
        self.selection_combo.bind('<<ComboboxSelected>>', self.handle_combo_selected)
        self.selection_combo.pack(side=tk.LEFT)

        status_frame = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, textvariable=self.image_status_var, font=('Segoe UI', 11)).pack(side=tk.LEFT)

        detail_frame = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        detail_frame.pack(fill=tk.X)
        ttk.Label(detail_frame, textvariable=self.detail_var, font=('Segoe UI', 10)).pack(side=tk.LEFT)

        canvas_frame = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.image_canvas = tk.Canvas(canvas_frame, background='#101010', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind('<Configure>', self.handle_canvas_resize)

        log_frame = ttk.LabelFrame(self.root, text='Processing Log', padding=(12, 6))
        log_frame.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 12))
        self.log_text = tk.Text(log_frame, height=10, wrap='word', state='disabled', font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def set_controls_enabled(self, enabled: bool) -> None:
        state = 'normal' if enabled else 'disabled'
        self.prev_button.config(state=state)
        self.next_button.config(state=state)
        self.selection_combo.config(state=state if self.processed_items else 'disabled')

    def clear_display(self) -> None:
        self.image_canvas.delete('all')
        self.comparison_photo = None
        self.current_base_image = None

    def handle_combo_selected(self, event) -> None:  # pragma: no cover - UI callback
        selection = self.selection_var.get()
        for index, item in enumerate(self.processed_items):
            if item.filename == selection:
                self.current_index = index
                self.update_display()
                break

    def load_comparison_base_image(self, path: Optional[str]) -> Optional[Image.Image]:
        if not path:
            return None
        if path in self.comparison_cache:
            return self.comparison_cache[path]
        if not os.path.exists(path):
            logger.error('Comparison figure not found: %s', path)
            return None
        image = Image.open(path)
        copied_image = image.copy()
        image.close()
        self.comparison_cache[path] = copied_image
        return copied_image

    def render_comparison_image(self) -> None:
        if self.current_base_image is None:
            self.clear_display()
            return
        canvas_width = max(self.image_canvas.winfo_width(), 1)
        canvas_height = max(self.image_canvas.winfo_height(), 1)
        image_width, image_height = self.current_base_image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = max(scale, 0.01)
        resized = self.current_base_image.resize(
            (max(1, int(image_width * scale)), max(1, int(image_height * scale))),
            Image.LANCZOS,
        )
        self.comparison_photo = ImageTk.PhotoImage(resized)
        self.image_canvas.delete('all')
        self.image_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=self.comparison_photo,
        )

    def handle_canvas_resize(self, event) -> None:  # pragma: no cover - UI callback
        if self.current_base_image is not None:
            self.render_comparison_image()

    def update_display(self) -> None:
        if not self.processed_items:
            self.clear_display()
            self.image_status_var.set('No processed images available.')
            return
        self.current_index = max(0, min(self.current_index, len(self.processed_items) - 1))
        current_item = self.processed_items[self.current_index]
        self.selection_var.set(current_item.filename)
        self.image_status_var.set(f"Showing {self.current_index + 1} of {len(self.processed_items)}: {current_item.filename}")
        detail_text = current_item.detail_text or self.parameter_summary
        self.detail_var.set(detail_text)
        self.current_base_image = self.load_comparison_base_image(current_item.comparison_figure_path)
        self.render_comparison_image()

    def append_log_message(self, message: str) -> None:
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def update_processed_items(self, processed_items: List[ProcessedItem]) -> None:
        self.processed_items = list(processed_items)
        filenames = [item.filename for item in self.processed_items]
        self.selection_combo['values'] = filenames
        if filenames:
            self.selection_combo.current(0)
            self.current_index = 0
            self.set_controls_enabled(True)
            self.update_display()
        else:
            self.clear_display()
            self.set_controls_enabled(False)

    def set_processing_message(self, message: str) -> None:
        self.processing_status_var.set(message)

    def schedule_processed_items(self, processed_items: List[ProcessedItem]) -> None:
        self.root.after(0, lambda: self.update_processed_items(processed_items))

    def schedule_processing_message(self, message: str) -> None:
        self.root.after(0, lambda: self.set_processing_message(message))

    def schedule_error(self, message: str) -> None:
        self.root.after(0, lambda: self.show_error(message))

    def schedule_log_message(self, message: str) -> None:
        self.root.after(0, lambda: self.append_log_message(message))

    def show_next(self) -> None:
        if not self.processed_items:
            return
        self.current_index = (self.current_index + 1) % len(self.processed_items)
        self.update_display()

    def show_previous(self) -> None:
        if not self.processed_items:
            return
        self.current_index = (self.current_index - 1) % len(self.processed_items)
        self.update_display()

    def show_error(self, message: str) -> None:
        messagebox.showerror('Processing Error', message)

    def run(self) -> None:
        self.root.mainloop()
