import os
import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, Scale, HORIZONTAL, Toplevel, messagebox

import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans


class PaletteChanger:
    """
    Main application class for changing images' color palettes.

    Provides functionality to select an image and a color palette image,
    and then convert the source image to use only the colors present in the palette.

    Attributes:
        root (tk.Tk): The root Tkinter window.
        img (PIL.Image.Image): The source image being processed.
        palette_path (str): Path to the selected color palette image.
        file_path (str): Path to the selected source image.
    """

    def __init__(self, root):
        """
        Initialize the PaletteChanger application.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.img = None
        self.palette_path = None
        self.file_path = None
        self.last_preview_window = None
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the user interface components.

        Creates and arranges all GUI elements including frames, labels, buttons, and scales.
        """
        self.root.title("PaletteChanger")
        self.root.geometry("256x475")
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
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        Button(button_frame, text="Preview Image", command=self.preview_image).pack(side=tk.LEFT, padx=5)
        Button(button_frame, text="Save Image", command=self.convert_image).pack(side=tk.LEFT, padx=5)

    def change_scale(self, delta):
        """
        Create a handler for keyboard events to change the color scale value.

        Args:
            delta (int): The amount to change the scale value by (positive or negative).

        Returns:
            function: An event handler function that adjusts the scale value.
        """

        def handler(event):
            current_value = self.scale.get()
            new_value = current_value + delta
            if self.scale.cget("from") <= new_value <= self.scale.cget("to"):
                self.scale.set(new_value)
            return "break"

        return handler

    def select_palette(self):
        """
        Open a file dialog to select a color palette image.

        Analyzes the selected image to determine the maximum number of unique colors,
        then updates the scale widget accordingly.
        """
        # Ask for a new palette path, store in temporary variable
        new_palette_path = filedialog.askopenfilename(title="Select Color Palette Image")
        if new_palette_path:
            try:
                palette_image = Image.open(new_palette_path)
                # Ensure the image is in RGB format before converting to numpy array
                if palette_image.mode != "RGB":
                    palette_image = palette_image.convert("RGB")
                colors = np.array(palette_image).reshape(-1, 3)
                max_colors = min(len(np.unique(colors, axis=0)), 256)
                self.palette_path = new_palette_path
                self.scale.config(to=max_colors)
                self.scale.set(max_colors)
                # Truncate filename if too long
                filename = os.path.basename(self.palette_path)
                max_length = 40
                if len(filename) > max_length:
                    filename = filename[: max_length - 3] + "..."
                self.palette_label.config(text=filename)
            except Exception as e:
                messagebox.showerror(
                    "Error Loading Palette",
                    f"Could not process the selected palette image: {new_palette_path}\nError: {e}",
                )

    def get_image(self):
        """
        Open a file dialog to select the source image for conversion.

        Loads the selected image and displays a thumbnail preview in the GUI.
        """
        self.file_path = filedialog.askopenfilename()
        if not self.file_path:
            return
        self.img = Image.open(self.file_path)
        self.img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        self.img_preview = ImageTk.PhotoImage(self.img)
        self.img_label.config(image=self.img_preview)  # type: ignore

    def convert_image(self):
        """
        Convert the source image to use the selected color palette.

        Processes the image by quantizing the palette to the specified number of colors,
        then mapping each pixel in the source image to the closest color in the palette.
        """
        if not self.palette_path or not self.file_path:
            print("Please select both an image and a color palette first.")
            return
        palette_image = Image.open(self.palette_path)
        self.img = Image.open(self.file_path)
        if self.img.mode != "RGBA":
            self.img = self.img.convert("RGB")
        colors = self._quantize_palette(palette_image)
        new_image = self._map_pixel_to_palette(self.img, colors)
        self._save_converted_image(new_image)

    def _quantize_palette(self, palette_image):
        """
        Extract the dominant colors from the palette image using k-Means clustering.

        Args:
            palette_image (PIL.Image.Image): The palette image to process.

        Returns:
            numpy.ndarray: Array of dominant colors (cluster centers) from the palette.
        """
        # Suppress console warning
        cpu_count = int(os.cpu_count() / 2)
        os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_count)
        # Convert to RGB if needed
        if palette_image.mode != "RGB":
            palette_image = palette_image.convert("RGB")
        img_array = np.array(palette_image)
        pixels = img_array.reshape(-1, 3)
        num_colors = int(self.scale.get())
        kmeans = KMeans(n_clusters=num_colors, init="k-means++", n_init="auto", random_state=0)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return colors

    def _map_pixel_to_palette(self, image, colors):
        """
        Map each pixel in the source image to the closest color in the palette.

        Args:
            image (PIL.Image.Image): The source image to process.
            colors (numpy.ndarray): Array of colors from the quantized palette.

        Returns:
            PIL.Image.Image: New image with colors mapped to the palette.
        """
        new_image = Image.new(image.mode, image.size)
        img_array = np.array(image)
        memo = {}
        for x in range(image.width):
            for y in range(image.height):
                pixel_color = tuple(img_array[y, x])
                color_key = pixel_color[:3] if image.mode == "RGBA" else pixel_color
                if color_key in memo:
                    closest_color = memo[color_key]
                else:
                    distances = np.sum((colors - color_key) ** 2, axis=1)
                    closest_color_index = np.argmin(distances)
                    closest_color = tuple(colors[closest_color_index])
                    memo[color_key] = closest_color
                if image.mode == "RGBA":
                    new_image.putpixel((x, y), closest_color + (pixel_color[3],))
                else:
                    new_image.putpixel((x, y), closest_color)
        return new_image

    def preview_image(self):
        """
        Generate and display a preview of the converted image in a new window.
        """
        if not self.file_path or not self.palette_path:
            print("Please select both an image and a color palette first.")
            return
        try:
            palette_image = Image.open(self.palette_path)
            source_image = Image.open(self.file_path)
            if source_image.mode not in ["RGB", "RGBA"]:
                source_image = source_image.convert("RGB")
            colors = self._quantize_palette(palette_image)
            converted_image = self._map_pixel_to_palette(source_image, colors)
            if self.last_preview_window and self.last_preview_window.winfo_exists():
                preview_window = self.last_preview_window
                for widget in preview_window.winfo_children():
                    widget.destroy()
            else:
                preview_window = Toplevel(self.root)
                self.last_preview_window = preview_window
            preview_window.title("Image Preview")
            max_preview_dim = 512  # Resize preview image
            preview_img_display = converted_image.copy()
            preview_img_display.thumbnail((max_preview_dim, max_preview_dim), Image.Resampling.LANCZOS)
            img_w, img_h = preview_img_display.size
            img_tk = ImageTk.PhotoImage(preview_img_display)
            preview_label = Label(preview_window, image=img_tk, borderwidth=0, highlightthickness=0)
            preview_label.image = img_tk
            preview_label.pack()
            preview_window.lift()
            preview_window.focus_set()
            preview_window.lift()
            preview_window.focus_set()
            try:
                if (
                    not preview_window.winfo_geometry()
                    or "1x1+0+0" in preview_window.winfo_geometry()
                    or preview_window is self.last_preview_window
                ):
                    preview_window.update_idletasks()
                    main_x = self.root.winfo_x()
                    main_y = self.root.winfo_y()
                    main_w = self.root.winfo_width()
                    win_w = img_w
                    win_h = img_h
                    new_x = main_x + main_w + 5  # add small gap
                    new_y = main_y
                    # Adjust if off-screen (basic check)
                    screen_w = self.root.winfo_screenwidth()
                    screen_h = self.root.winfo_screenheight()
                    if new_x + win_w > screen_w:
                        new_x = max(0, main_x - win_w - 10)
                    if new_y + win_h > screen_h:
                        new_y = max(0, screen_h - win_h - 10)
                    preview_window.geometry(f"{win_w}x{win_h}+{new_x}+{new_y}")
                    preview_window.resizable(False, False)
                    preview_window.transient(self.root)
            except tk.TclError:
                pass

        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find the image or palette file. Please select them again.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate preview.\n{e}")

    def _save_converted_image(self, new_image):
        """
        Save the converted image to disk.

        Opens a file dialog to select the save location and filename.

        Args:
            new_image (PIL.Image.Image): The converted image to save.
        """
        if self.file_path is None:
            print("Please select an image first.")
            return
        base_name, original_ext = os.path.splitext(os.path.basename(self.file_path))
        suggested_filename = f"{base_name}_converted_palette{original_ext}"
        # Basic list of file types to use for dialog
        file_types = [
            ("Original format", f"*{original_ext}"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("WEBP files", "*.webp;*.webp"),
            ("All files", "*.*"),
        ]
        save_file_path = filedialog.asksaveasfilename(
            initialfile=suggested_filename, defaultextension=original_ext, filetypes=file_types
        )
        if save_file_path:
            save_ext = os.path.splitext(save_file_path)[1].lower()
            save_format = save_ext[1:]
            if save_format == "jpg":
                save_format = "jpeg"
            try:
                if save_format == "jpeg" and new_image.mode == "RGBA":
                    rgb_image = new_image.convert("RGB")
                    rgb_image.save(save_file_path, format=save_format.upper())
                else:
                    new_image.save(save_file_path, format=save_format.upper())
            except ValueError:
                try:
                    # Fallback to PNG
                    fallback_path = os.path.splitext(save_file_path)[0] + ".png"
                    new_image.save(fallback_path, format="PNG")
                except Exception:
                    pass
            except Exception:
                pass


def main():
    """
    Initialize and run the PaletteChanger application.

    Creates the root Tkinter window, sets up the application icon,
    centers the window on the screen, and starts the main event loop.
    """
    root = tk.Tk()
    icon_data = (
        "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAATklEQVQ4jWNk5hb+z4AGmFi5"
        "0IUg4myY4kxYVZIAhoEBLGysmhiCrL/FsCoW+69NfReMGsDAwGLMkoMhKMaBGdoMDAwMorw6"
        "1HfBMDAAAKMZBStpAogmAAAAAElFTkSuQmCC"
    )
    icon = tk.PhotoImage(data=icon_data)
    root.iconphoto(True, icon)
    PaletteChanger(root)
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.mainloop()


if __name__ == "__main__":
    main()
