import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np
import os

class PaletteChanger:
    def __init__(self, root):
        self.root = root
        self.img = None
        self.palette_path = None
        self.file_path = None
        self.setup_ui()

    def setup_ui(self):
        self.root.title("PaletteChanger")
        self.root.geometry("250x470")
        self.root.resizable(False, False)
        self.image_frame = Frame(self.root, width=256, height=256)
        self.image_frame.pack_propagate(False)
        self.image_frame.pack(pady=5)
        self.img_label = Label(self.image_frame)
        self.img_label.pack(expand=True)
        Button(self.root, text="Select Image", command=self.get_image).pack(pady=5)
        Button(self.root, text="Select Color Palette", command=self.select_palette).pack(pady=5)
        self.palette_label = Label(self.root, text="")
        self.palette_label.pack()
        Label(self.root, text="Colors to Use:").pack(pady=(5, 0))
        self.scale = Scale(self.root, from_=1, to=256, orient=HORIZONTAL, highlightthickness=0)
        self.scale.pack()
        self.scale.focus_set()
        self.scale.bind("<Left>", self.change_scale(-1))
        self.scale.bind("<Right>", self.change_scale(1))
        self.scale.bind("<Down>", self.change_scale(-1))
        self.scale.bind("<Up>", self.change_scale(1))
        Button(self.root, text="Convert Image", command=self.convert_image).pack(pady=5)

    def change_scale(self, delta):
        def handler(event):
            current_value = self.scale.get()
            new_value = current_value + delta
            if self.scale.cget("from") <= new_value <= self.scale.cget("to"):
                self.scale.set(new_value)
            return "break"
        return handler

    def select_palette(self):
        self.palette_path = filedialog.askopenfilename(title='Select Color Palette Image')
        if self.palette_path:
            palette_image = Image.open(self.palette_path)
            colors = np.array(palette_image).reshape(-1, 3)
            max_colors = min(len(np.unique(colors, axis=0)), 256)
            self.scale.config(to=max_colors)
            self.scale.set(max_colors)
            # Truncate filename if too long
            filename = os.path.basename(self.palette_path)
            max_length = 40
            if len(filename) > max_length:
                filename = filename[:max_length-3] + '...'
            self.palette_label.config(text=filename)

    def get_image(self):
        self.file_path = filedialog.askopenfilename()
        if not self.file_path:
            return
        self.img = Image.open(self.file_path)
        self.img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        self.img_preview = ImageTk.PhotoImage(self.img)
        self.img_label.config(image=self.img_preview)  # type: ignore

    def convert_image(self):
        if not self.palette_path or not self.file_path:
            print("Please select both an image and a color palette first.")
            return
        palette_image = Image.open(self.palette_path)
        self.img = Image.open(self.file_path)
        if self.img.mode != 'RGBA':
            self.img = self.img.convert('RGB')
        colors = self._quantize_palette(palette_image)
        new_image = self._map_pixel_to_palette(self.img, colors)
        self._save_converted_image(new_image)

    def _quantize_palette(self, palette_image):
        resized_palette_image = palette_image.resize((256, 256))
        num_colors = int(self.scale.get())
        result = resized_palette_image.quantize(colors=num_colors)
        result = result.convert('RGB')
        color_counts = result.getcolors()
        return np.array([color for count, color in color_counts])

    def _map_pixel_to_palette(self, image, colors):
        new_image = Image.new(image.mode, image.size)
        img_array = np.array(image)
        memo = {}
        for x in range(image.width):
            for y in range(image.height):
                pixel_color = tuple(img_array[y, x])
                # Handle transparent pixels in RGBA images
                if image.mode == 'RGBA' and pixel_color[:3] == (255, 255, 255):
                    new_image.putpixel((x, y), (0, 0, 0, 0))
                    continue
                # Use only RGB values for distance calculation
                color_key = pixel_color[:3] if image.mode == 'RGBA' else pixel_color
                if color_key in memo:
                    closest_color = memo[color_key]
                else:
                    distances = np.sum((colors - color_key) ** 2, axis=1)
                    closest_color_index = np.argmin(distances)
                    closest_color = tuple(colors[closest_color_index])
                    memo[color_key] = closest_color
                if image.mode == 'RGBA':
                    new_image.putpixel((x, y), closest_color + (pixel_color[3],))
                else:
                    new_image.putpixel((x, y), closest_color)
        return new_image

    def _save_converted_image(self, new_image):
        if self.file_path is None:
            print("Please select an image first.")
            return
        file_name = os.path.splitext(os.path.basename(self.file_path))[0] + "_converted_palette.png"
        save_file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=file_name)
        if save_file_path:
            new_image.save(save_file_path)

def main():
    root = tk.Tk()
    icon_data = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAATklEQVQ4jWNk5hb+z4AGmFi50IUg4myY4kxYVZIAhoEBLGysmhiCrL/FsCoW+69NfReMGsDAwGLMkoMhKMaBGdoMDAwMorw61HfBMDAAAKMZBStpAogmAAAAAElFTkSuQmCC"
    icon = tk.PhotoImage(data=icon_data)
    root.iconphoto(True, icon)
    app = PaletteChanger(root)
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    root.mainloop()

if __name__ == "__main__":
    main()