from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os

class PaletteChanger:
    def __init__(self, root):
        self.root = root
        self.palette_path = None
        self.file_path = None
        self.img = None
        self.setup_ui()

    def setup_ui(self):
        self.root.title("PaletteChanger")
        self.root.geometry("250x420")
        self.root.resizable(False, False)

        self.image_frame = Frame(self.root, height=250)
        self.image_frame.pack()

        self.img_label = Label(self.root)
        self.img_label.pack()
        image_button = tk.Button(self.root, text='Select Image', command=self.get_image)
        image_button.pack(pady=5)

        palette_button = tk.Button(self.root, text='Select Color Palette', command=self.select_palette)
        palette_button.pack(pady=5)
        self.palette_label = Label(self.root, text='')
        self.palette_label.pack()

        convert_button = tk.Button(self.root, text='Convert Image', command=self.convert_image)
        convert_button.pack(pady=5)

    def select_palette(self):
        self.palette_path = filedialog.askopenfilename(title='Select Color Palette Image')
        self.palette_label.config(text=os.path.basename(self.palette_path))

    def get_image(self):
        self.file_path = filedialog.askopenfilename()
        self.img = Image.open(self.file_path)
        if self.img.mode != 'RGBA':
            self.img = self.img.convert('RGB')
        img_preview = self.img.copy()
        max_size = (250, 250)
        img_preview.thumbnail(max_size, Image.LANCZOS)
        self.image_frame.pack_forget()
        img_preview = ImageTk.PhotoImage(img_preview)
        self.img_label.config(image=img_preview)
        self.img_label.image = img_preview

    def convert_image(self):
        palette_image = Image.open(self.palette_path)
        small_img = palette_image.resize((256, 256))
        result = small_img.quantize(colors=256)
        result = result.convert('RGB')
        color_counts = result.getcolors()
        colors = np.array([color for count, color in color_counts])

        if self.img.mode == 'RGBA':
            new_image = Image.new('RGBA', self.img.size)
            img_array = np.array(self.img)
            memo = {}
            for x in range(self.img.width):
                for y in range(self.img.height):
                    pixel_color = tuple(img_array[y, x])
                    if pixel_color[:3] == (255, 255, 255):
                        new_image.putpixel((x, y), (0, 0, 0, 0))
                    else:
                        if pixel_color[:3] in memo:
                            closest_color = memo[pixel_color[:3]]
                        else:
                            distances = np.sum((colors - pixel_color[:3]) ** 2, axis=1)
                            closest_color_index = np.argmin(distances)
                            closest_color = tuple(colors[closest_color_index])
                            memo[pixel_color[:3]] = closest_color
                        new_image.putpixel((x, y), closest_color + (pixel_color[3],))
        else:
            new_image = Image.new('RGB', self.img.size)
            img_array = np.array(self.img)
            memo = {}
            for x in range(self.img.width):
                for y in range(self.img.height):
                    pixel_color = tuple(img_array[y, x])
                    if pixel_color in memo:
                        closest_color = memo[pixel_color]
                    else:
                        distances = np.sum((colors - pixel_color) ** 2, axis=1)
                        closest_color_index = np.argmin(distances)
                        closest_color = tuple(colors[closest_color_index])
                        memo[pixel_color] = closest_color
                    new_image.putpixel((x, y), closest_color)

        file_name = os.path.basename(self.file_path)
        file_name = os.path.splitext(file_name)[0] + "_converted_palette.png"
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