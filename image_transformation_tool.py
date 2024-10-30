import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageTransformationTool:
    def __init__(self, master):
        self.master = master
        master.title("Image Transformation Tool")

        self.canvas_width = 800
        self.canvas_height = 600

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Frame for buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.transform_button = tk.Button(self.button_frame, text="Correct Image", command=self.correct_image, state=tk.DISABLED)
        self.transform_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset Points", command=self.reset_points, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(self.button_frame, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.image = None
        self.transformed_image = None
        self.photo = None
        self.points = []

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image",
                                               filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return
            self.transformed_image = None
            self.display_image(self.image)
            self.points = []
            self.transform_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
            self.canvas.delete("point")

    def display_image(self, img):
        self.current_img = img  # Keep reference to the image displayed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize image to fit canvas while maintaining aspect ratio
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((self.canvas_width, self.canvas_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas_image_id = self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.canvas.config(width=img_pil.width, height=img_pil.height)

        # Calculate the ratio of the original image to the displayed image
        self.ratio_x = img.shape[1] / img_pil.width
        self.ratio_y = img.shape[0] / img_pil.height

    def on_canvas_click(self, event):
        if self.image is None:
            return

        x = event.x
        y = event.y

        # Ensure click is within the displayed image area
        if x > self.tk_img.width() or y > self.tk_img.height():
            return

        # Map the clicked point to the original image coordinates
        original_x = int(x * self.ratio_x)
        original_y = int(y * self.ratio_y)

        # Add point to list
        self.points.append([original_x, original_y])

        # Draw circle on the canvas
        r = 3  # radius of point marker
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='red', outline='red', tags="point")

        if len(self.points) == 64:
            messagebox.showinfo("Info", "64 points selected. You can now correct the image.")
            self.transform_button.config(state=tk.NORMAL)

    def reset_points(self):
        self.points = []
        self.canvas.delete("point")
        self.transform_button.config(state=tk.DISABLED)
        self.save_button.config(state=self.transformed_image is not None)

    def correct_image(self):
        if self.image is not None and len(self.points) == 64:
            corrected_img = self.process_image(self.image, self.points)
            if corrected_img is not None:
                self.transformed_image = corrected_img
                self.display_image(self.transformed_image)
                self.image = self.transformed_image  # Update the image
                self.points = []
                self.canvas.delete("point")
                self.transform_button.config(state=tk.DISABLED)
                self.reset_button.config(state=tk.DISABLED)
                self.save_button.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Warning", "Please select 64 points first.")

    def save_image(self):
        if self.transformed_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("BMP Image", "*.bmp")]
            )
            if file_path:
                # Convert image from RGB to BGR before saving
                img_to_save = cv2.cvtColor(self.transformed_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(file_path, img_to_save)
                if success:
                    messagebox.showinfo("Success", "Image saved successfully!")
                else:
                    messagebox.showerror("Error", "Failed to save image.")
        else:
            messagebox.showwarning("Warning", "No transformed image to save.")

    def process_image(self, img, src_points):
        # Source points from user clicks
        src_points = np.array(src_points, dtype=np.float32)

        # Destination points: create an 8x8 grid
        grid_size = 100  # Adjust as needed
        dst_points = []
        for i in range(8):
            for j in range(8):
                dst_points.append([j * grid_size, i * grid_size])
        dst_points = np.array(dst_points, dtype=np.float32)

        # Compute homography using all points with a perspective transform
        try:
            # Since we have more than 4 points, use a more robust method like RANSAC
            H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            if H is not None:
                # Determine the size of the output image
                height = 8 * grid_size
                width = 8 * grid_size
                # Warp the image using the homography
                corrected_img = cv2.warpPerspective(img, H, (width, height))
                # Convert BGR to RGB for consistent display
                corrected_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
                return corrected_img
            else:
                messagebox.showerror("Error", "Homography computation failed.")
                return None
        except cv2.error as e:
            messagebox.showerror("Error", f"OpenCV error: {e}")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTransformationTool(root)
    root.mainloop()
