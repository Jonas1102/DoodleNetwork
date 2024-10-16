import tkinter as tk
from PIL import Image, ImageDraw


class DoodleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Doodle App")

        # Define canvas size and pixel size
        self.canvas_size = 28
        self.pixel_size = 20  # Size of each pixel on the screen
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_size * self.pixel_size,
                                height=self.canvas_size * self.pixel_size)
        self.canvas.pack()

        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events for painting
        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<B1-Motion>", self.paint_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.canvas.bind("<KeyPress-c>", self.save_doodle)
        self.canvas.focus_set()

        self.drawing = False  # To track whether we are drawing

    def start_paint(self, event):
        """Start painting when mouse button is pressed."""
        self.drawing = True
        self.paint_pixel(event)  # Paint the initial pixel

    def paint_pixel(self, event):
        """Paint pixels at the current mouse position."""
        if self.drawing:
            # Calculate which pixel is clicked
            x = event.x // self.pixel_size
            y = event.y // self.pixel_size

            # Draw the pixel on the canvas
            self.canvas.create_rectangle(x * self.pixel_size, y * self.pixel_size,
                                         (x + 1) * self.pixel_size, (y + 1) * self.pixel_size,
                                         fill="black", outline="black")

            # Update the image in memory
            self.draw.rectangle([x, y, x + 1, y + 1], fill="black")

    def stop_paint(self, event):
        """Stop painting when mouse button is released."""
        self.drawing = False

    def save_doodle(self, event):
        """Save the doodle as an image file."""
        filename = "doodle_{}.png".format(self.get_doodle_count())
        self.image.save(filename)
        print(f"Doodle saved as {filename}")

        self.reset_canvas()

    def reset_canvas(self):
        """Reset the canvas for a new doodle."""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

    def get_doodle_count(self):
        """Count the number of doodles saved and return the next count."""
        try:
            with open("doodle_count.txt", "r") as f:
                count = int(f.read().strip()) + 1
        except FileNotFoundError:
            count = 1

        with open("doodle_count.txt", "w") as f:
            f.write(str(count))

        return count


if __name__ == "__main__":
    root = tk.Tk()
    app = DoodleApp(root)
    root.mainloop()
