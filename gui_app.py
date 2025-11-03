'''
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from model_pix2pix import build_generator

# ----------------------------
# Paths
# ----------------------------
CHECKPOINT_PATH = "checkpoints/generator.weights.h5"
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Generator
# ----------------------------
generator = build_generator()
generator.load_weights(CHECKPOINT_PATH)
print("✅ Generator loaded successfully!")

# ----------------------------
# GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Sketch to Real Image Generator")
root.geometry("900x500")
root.resizable(False, False)

input_image_path = None
generated_image = None

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_sketch(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = (tf.cast(img, tf.float32) / 127.5) - 1
    img = tf.expand_dims(img, axis=0)
    return img

def generate_real_image():
    global generated_image
    if not input_image_path:
        messagebox.showwarning("Warning", "Please upload a sketch first!")
        return

    # Preprocess and generate
    sketch_input = preprocess_sketch(input_image_path)
    gen_output = generator(sketch_input, training=False)
    gen_output = (gen_output[0] + 1) * 127.5
    gen_output = tf.cast(gen_output, tf.uint8).numpy()
    generated_image = Image.fromarray(gen_output)

    # Display generated image
    display_generated = generated_image.resize((400, 400))
    gen_photo = ImageTk.PhotoImage(display_generated)
    output_label.config(image=gen_photo)
    output_label.image = gen_photo

def upload_sketch():
    global input_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return
    input_image_path = file_path

    # Display sketch
    sketch_img = Image.open(file_path).resize((400, 400))
    sketch_photo = ImageTk.PhotoImage(sketch_img)
    input_label.config(image=sketch_photo)
    input_label.image = sketch_photo

def save_images():
    if not input_image_path or generated_image is None:
        messagebox.showwarning("Warning", "No images to save!")
        return

    # Save input sketch
    sketch_save_path = os.path.join(DATA_DIR, os.path.basename(input_image_path))
    Image.open(input_image_path).save(sketch_save_path)

    # Save generated image
    gen_save_path = os.path.join(OUTPUT_DIR, os.path.basename(input_image_path))
    generated_image.save(gen_save_path)

    messagebox.showinfo("Saved", f"Saved:\nSketch: {sketch_save_path}\nGenerated: {gen_save_path}")

def redo():
    global input_image_path, generated_image
    input_image_path = None
    generated_image = None
    input_label.config(image='')
    output_label.config(image='')

def exit_app():
    root.destroy()

# ----------------------------
# GUI Layout
# ----------------------------
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

upload_btn = tk.Button(top_frame, text="Upload Sketch", command=upload_sketch, width=15, height=2)
upload_btn.grid(row=0, column=0, padx=10)

generate_btn = tk.Button(top_frame, text="Generate Image", command=generate_real_image, width=15, height=2)
generate_btn.grid(row=0, column=1, padx=10)

save_btn = tk.Button(top_frame, text="Save", command=save_images, width=15, height=2)
save_btn.grid(row=0, column=2, padx=10)

redo_btn = tk.Button(top_frame, text="Redo", command=redo, width=15, height=2)
redo_btn.grid(row=0, column=3, padx=10)

exit_btn = tk.Button(top_frame, text="Exit", command=exit_app, width=15, height=2)
exit_btn.grid(row=0, column=4, padx=10)

# Image Display
image_frame = tk.Frame(root)
image_frame.pack(pady=20)

input_label = tk.Label(image_frame)
input_label.grid(row=0, column=0, padx=20)

output_label = tk.Label(image_frame)
output_label.grid(row=0, column=1, padx=20)

root.mainloop()
'''

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from model_pix2pix import build_generator

# ----------------------------
# Configuration
# ----------------------------
ctk.set_appearance_mode("dark")   # Options: "light", "dark", "system"
ctk.set_default_color_theme("blue")

CHECKPOINT_PATH = "checkpoints/generator.weights.h5"
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Generator
# ----------------------------
generator = build_generator()
generator.load_weights(CHECKPOINT_PATH)
print("✅ Generator loaded successfully!")

# ----------------------------
# GUI Setup
# ----------------------------
app = ctk.CTk()
app.title("🎨 Sketch to Real Image Generator")
app.geometry("1000x600")
app.resizable(False, False)

input_image_path = None
generated_image = None

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_sketch(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = (tf.cast(img, tf.float32) / 127.5) - 1
    img = tf.expand_dims(img, axis=0)
    return img

def generate_real_image():
    global generated_image
    if not input_image_path:
        messagebox.showwarning("Warning", "Please upload a sketch first!")
        return

    sketch_input = preprocess_sketch(input_image_path)
    gen_output = generator(sketch_input, training=False)
    gen_output = (gen_output[0] + 1) * 127.5
    gen_output = tf.cast(gen_output, tf.uint8).numpy()
    generated_image = Image.fromarray(gen_output)

    display_generated = generated_image.resize((400, 400))
    gen_photo = ImageTk.PhotoImage(display_generated)
    output_label.configure(image=gen_photo)
    output_label.image = gen_photo

    status_label.configure(text="✅ Image generated successfully!", text_color="lightgreen")

def upload_sketch():
    global input_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return
    input_image_path = file_path

    sketch_img = Image.open(file_path).resize((400, 400))
    sketch_photo = ImageTk.PhotoImage(sketch_img)
    input_label.configure(image=sketch_photo)
    input_label.image = sketch_photo

    status_label.configure(text="📤 Sketch uploaded", text_color="white")

def save_images():
    if not input_image_path or generated_image is None:
        messagebox.showwarning("Warning", "No images to save!")
        return

    sketch_save_path = os.path.join(DATA_DIR, os.path.basename(input_image_path))
    Image.open(input_image_path).save(sketch_save_path)

    gen_save_path = os.path.join(OUTPUT_DIR, os.path.basename(input_image_path))
    generated_image.save(gen_save_path)

    messagebox.showinfo("Saved", f"✅ Saved:\nSketch: {sketch_save_path}\nGenerated: {gen_save_path}")
    status_label.configure(text="💾 Images saved successfully", text_color="lightblue")

def redo():
    global input_image_path, generated_image
    input_image_path = None
    generated_image = None
    input_label.configure(image=None)
    output_label.configure(image=None)
    status_label.configure(text="🔁 Ready for new input", text_color="gray")

def exit_app():
    app.destroy()

# ----------------------------
# GUI Layout
# ----------------------------
# Header
header = ctk.CTkLabel(app, text="Sketch → Real Image Generator",
                      font=("Segoe UI Semibold", 26), text_color="skyblue")
header.pack(pady=20)

# Button Frame
button_frame = ctk.CTkFrame(app, corner_radius=15)
button_frame.pack(pady=10)

upload_btn = ctk.CTkButton(button_frame, text="📤 Upload Sketch", command=upload_sketch, width=150)
upload_btn.grid(row=0, column=0, padx=10, pady=10)

generate_btn = ctk.CTkButton(button_frame, text="⚡ Generate Image", command=generate_real_image, width=150)
generate_btn.grid(row=0, column=1, padx=10, pady=10)

save_btn = ctk.CTkButton(button_frame, text="💾 Save", command=save_images, width=150)
save_btn.grid(row=0, column=2, padx=10, pady=10)

redo_btn = ctk.CTkButton(button_frame, text="🔁 Redo", command=redo, width=150)
redo_btn.grid(row=0, column=3, padx=10, pady=10)

exit_btn = ctk.CTkButton(button_frame, text="❌ Exit", command=exit_app, width=150, fg_color="red", hover_color="#b30000")
exit_btn.grid(row=0, column=4, padx=10, pady=10)

# Image Display Frame
image_frame = ctk.CTkFrame(app, corner_radius=15)
image_frame.pack(pady=25)

input_label = ctk.CTkLabel(image_frame, text="Input Sketch", width=400, height=400, fg_color="#1e1e1e", corner_radius=15)
input_label.grid(row=0, column=0, padx=25, pady=10)

output_label = ctk.CTkLabel(image_frame, text="Generated Image", width=400, height=400, fg_color="#1e1e1e", corner_radius=15)
output_label.grid(row=0, column=1, padx=25, pady=10)

# Status Bar
status_label = ctk.CTkLabel(app, text="Ready", text_color="gray", font=("Segoe UI", 14))
status_label.pack(pady=10)

app.mainloop()
