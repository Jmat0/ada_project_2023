import tkinter as tk
from PIL import Image, ImageTk
import csv
import os
import random

# Create the main window
window = tk.Tk()
window.title("Watch Selector")
window.geometry('800x800')

# Calculate the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate the x and y coordinates for centering the window
x = (screen_width // 2) - (600 // 2)
y = (screen_height // 2) - (600 // 2)

# Set the window position
window.geometry(f'800x800+{x}+{y}')

# Make the window static and non-resizable
window.resizable(False, False)

# Define the placeholder image
placeholder_image_path = "placeholder.png"
placeholder_original = Image.open(placeholder_image_path)
placeholder_resized = placeholder_original.resize((130, 130))
placeholder_image = ImageTk.PhotoImage(placeholder_resized)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Create the left frame
left_frame = tk.Frame(window, bg='#3A3845')
left_frame.grid(row=0, column=0, sticky='nsew')

left_frame.grid_rowconfigure(1, minsize=100)

# Open and resize the image
image_path_title = "image.jpg"
original_image = Image.open(image_path_title)
resized_image = original_image.resize((125, 125))
logo_image = ImageTk.PhotoImage(resized_image)

# Add image logo and text to the left frame
logo_label = tk.Label(left_frame, image=logo_image, bg='#3A3845', borderwidth=0, highlightthickness=0)
logo_label.grid(row=0, column=0, sticky='w')

text_label = tk.Label(left_frame, text="Find your perfect watch!", font=('Helvetica 20 underline'), bg=('#3A3845'),
                      fg='white')
text_label.grid(row=0, column=0, columnspan=2, sticky='e', padx=140)

# Add instruction text for user on the left frame
text_instruction = tk.Label(left_frame, text="Use the filters below to adjust your preferences", font=('Helvetica', 15),
                            bg=('#3A3845'), fg='white')
text_instruction.grid(row=1, column=0, sticky='w', columnspan=2)

# Read the CSV file
csv_file_path = "data_with_images.csv"

# Read the CSV file and extract the image file names
image_file_names = []
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image_file_names.append(str(row['image_file']))

# Construct the image paths
image_folder_path = "Watches_Images"
image_paths = [os.path.join(image_folder_path, image_file) for image_file in image_file_names]


# Function to extract unique values from a column in the CSV file
def extract_unique_values(column_name):
    unique_values = set()
    with open(csv_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            unique_values.add(row[column_name])
    return unique_values


# Extract unique brand values
brand_values = extract_unique_values("Brand")
brand_values = sorted(list(brand_values))
brand_values.insert(0, "Any")

# Extract unique color values
color_values = extract_unique_values("Color")
color_values = sorted(list(color_values))
color_values.insert(0, "Any")

# Extract unique shape values
shape_values = extract_unique_values("Shape")
shape_values = sorted(list(shape_values))
shape_values.insert(0, "Any")

# Diameter
diameter_values = extract_unique_values("Diameter")
sorted_values_diameter = sorted(diameter_values, key=lambda x: float(x.split()[0]) if x.split() else 0)
sorted_values_diameter.insert(0, "Any")

# Water Resistance
wr_values = extract_unique_values("W/R")
sorted_values_wr = sorted(diameter_values, key=lambda x: float(x.split()[0]) if x.split() else 0)
sorted_values_wr.insert(0, "Any")

# Create the dropdown buttons
dropdown_frame1 = tk.Frame(left_frame, relief='flat')
dropdown_frame1.grid(row=3, column=0, padx=10, pady=10, sticky='w')

title_label1 = tk.Label(dropdown_frame1, text="Brand")
title_label1.configure(width=18)
title_label1.pack(anchor='w')

dropdown_var1 = tk.StringVar()
dropdown_var1.set(brand_values[0])  # Set the default value to "Any"
dropdown_menu1 = tk.OptionMenu(dropdown_frame1, dropdown_var1, *brand_values)
dropdown_menu1.configure(width=14)
dropdown_menu1.pack(anchor='w')

dropdown_frame2 = tk.Frame(left_frame, relief='flat')
dropdown_frame2.grid(row=4, column=0, padx=10, pady=10, sticky='w')

title_label2 = tk.Label(dropdown_frame2, text="Color")
title_label2.configure(width=18)
title_label2.pack(anchor='w')

dropdown_var2 = tk.StringVar()
dropdown_var2.set(color_values[0])  # Set the default value to "Any"

# Update the dropdown menu options with the sorted values
dropdown_menu2 = tk.OptionMenu(dropdown_frame2, dropdown_var2, *color_values)

dropdown_menu2.configure(width=14)
dropdown_menu2.pack(anchor='w')

dropdown_frame3 = tk.Frame(left_frame, relief='flat')
dropdown_frame3.grid(row=5, column=0, padx=10, pady=10, sticky='w')

title_label3 = tk.Label(dropdown_frame3, text="Shape")
title_label3.configure(width=18)
title_label3.pack(anchor='w')

dropdown_var3 = tk.StringVar()
dropdown_var3.set(shape_values[0])  # Set the default value to "Any"
dropdown_menu3 = tk.OptionMenu(dropdown_frame3, dropdown_var3, *shape_values)
dropdown_menu3.configure(width=14)
dropdown_menu3.pack(anchor='w')

dropdown_frame4 = tk.Frame(left_frame, relief='flat')
dropdown_frame4.grid(row=6, column=0, padx=10, pady=10, sticky='w')

title_label4 = tk.Label(dropdown_frame4, text="Diameter")
title_label4.configure(width=18)
title_label4.pack(anchor='w')

dropdown_var4 = tk.StringVar()
dropdown_var4.set(sorted_values_diameter[0])
dropdown_menu4 = tk.OptionMenu(dropdown_frame4, dropdown_var4, *sorted_values_diameter)
dropdown_menu4.configure(width=14)
dropdown_menu4.pack(anchor='w')

dropdown_frame5 = tk.Frame(left_frame, relief='flat')
dropdown_frame5.grid(row=7, column=0, padx=10, pady=10, sticky='w')

title_label5 = tk.Label(dropdown_frame5, text="Water Resistance")
title_label5.configure(width=18)
title_label5.pack(anchor='w')

dropdown_var5 = tk.StringVar()
dropdown_var5.set(sorted_values_wr[0])
dropdown_menu5 = tk.OptionMenu(dropdown_frame5, dropdown_var5, *sorted_values_wr)
dropdown_menu5.configure(width=14)
dropdown_menu5.pack(anchor='w')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Create the right frame
right_frame = tk.Frame(window, bg='#526D82')
right_frame.grid(row=0, column=1, sticky='nsew')

# Configure grid weights to make frames expand equally
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)

# Add text label to the right frame
suggestions_label = tk.Label(right_frame, text="Here are some suggestions:", font=('Helvetica', 20), bg=('#526D82'),
                             fg=('white'))
suggestions_label.grid(row=0, column=1, sticky='w', padx=50, pady=48)

# Create a list to store all the brand labels
brand_labels = []

# Create a list to store all the reference labels
reference_labels = []

# Create a list to store all the placeholder labels
placeholder_labels = []

# Create placeholder labels (White images where output images will be shown)
for i in range(3):
    placeholder_label = tk.Label(right_frame, image=placeholder_image, bg='#526D82', borderwidth=0,
                                 highlightthickness=0)
    placeholder_label.grid(row=i + 1, column=1, sticky='w', padx=105, pady=36)
    placeholder_labels.append(placeholder_label)

    brand_label = tk.Label(right_frame, text="", font=('Helvetica', 13), bg='#526D82', fg='white')
    brand_label.grid(row=i + 2, column=1, sticky='n', padx=10, pady=0)
    brand_labels.append(brand_label)

    reference_label = tk.Label(right_frame, text="", font=('Helvetica', 13), bg='#526D82', fg='white')
    reference_label.grid(row=i + 2, column=1, sticky='n', padx=0, pady=18)
    reference_labels.append(reference_label)


# Create class
class Watch:
    def __init__(self, brand, color, shape, diameter, wr, reference, image_path):
        self.brand = brand
        self.color = color
        self.shape = shape
        self.diameter = diameter
        self.wr = wr
        self.reference = reference
        self.image_path = image_path


# We add a "no results" image when the filter doesn't match specific watches
no_results_path = "no_results.png"
no_results_original = Image.open(no_results_path)
no_results_resized = no_results_original.resize((130, 130))
no_results_image = ImageTk.PhotoImage(no_results_resized)

# Function to handle image filtering output
def search_button_click():
    selected_brand = dropdown_var1.get()
    selected_color = dropdown_var2.get()
    selected_shape = dropdown_var3.get()
    selected_diameter = dropdown_var4.get()
    selected_wr = dropdown_var5.get()

    # Filter watches based on selected options
    filtered_watches = []
    for watch in watches_list:
        if (selected_brand == "Any" or watch.brand == selected_brand) and \
                (selected_color == "Any" or watch.color == selected_color) and \
                (selected_shape == "Any" or watch.shape == selected_shape) and \
                (selected_diameter == "Any" or watch.diameter == selected_diameter) and \
                (selected_wr == "Any" or watch.wr == selected_wr):
            filtered_watches.append(watch)

    # Calculate the start and end indices for the current page
    start_index = current_page * 3
    end_index = start_index + 3

    # Take the watches for the current page
    selected_watches = filtered_watches[start_index:end_index]

    # Clear the image labels and display placeholder images
    for label in placeholder_labels:
        label.configure(image=placeholder_image)
        label.image = placeholder_image  # Store a reference image

    # Update the placeholder labels with the images and text of the watches
    for i, watch in enumerate(selected_watches):
        image_path = watch.image_path
        try:
            watch_image = Image.open(image_path)
            resized_image = watch_image.resize((130, 130))
            watch_photo = ImageTk.PhotoImage(resized_image)
            placeholder_labels[i].configure(image=watch_photo)
            placeholder_labels[i].image = watch_photo  # Store a reference to prevent image garbage collection
        except FileNotFoundError as e:
            # Handle the exception if the image file is not found
            placeholder_labels[i].configure(image=no_results_image)
            placeholder_labels[i].image = no_results_image  # Store a reference to prevent image garbage collection

        # Update brand label
        brand_labels[i].configure(text=watch.brand)

        # Update reference label
        reference_labels[i].configure(text=watch.reference)

    # Display "no results" image if no watches match the filter
    if not selected_watches:
        for label in placeholder_labels:
            label.configure(image=no_results_image)
            label.image = no_results_image  # Store a reference image

        suggestions_label.configure(text="No results found")

# Read the CSV file and create watch objects
watches_list = []
with open(csv_file_path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        brand = row["Brand"]
        color = row["Color"]
        shape = row["Shape"]
        diameter = row["Diameter"]
        wr = row["W/R"]
        reference = row["Reference"]
        image_file = row["image_file"]

        image_path = os.path.join(image_folder_path, image_file)

        watch = Watch(brand, color, shape, diameter, wr, reference, image_path)
        watches_list.append(watch)

# Shuffle the watches
random.shuffle(watches_list)

# Function to handle the next button click

# Define page start and finish for navigation
current_page = 0
max_pages = 3
def next_button_click():
    global current_page, max_pages
    if current_page < max_pages - 1:
        current_page += 1
        search_button_click()

# Function to handle the previous button click
def previous_button_click():
    global current_page
    if current_page > 0:
        current_page -= 1
        search_button_click()

# Create the search button
search_button = tk.Button(left_frame, text="Search", font=('Arial', 12), bg='#526D82', command=search_button_click)
search_button.grid(row=8, column=0, sticky='w', padx=10, pady=20)

# Create the next and previous buttons
next_button = tk.Button(right_frame, text="Next", font=('Arial', 12), bg='#526D82', command=next_button_click, highlightthickness=0)
next_button.grid(row=4, column=1, sticky='e', padx=45, pady=20)

previous_button = tk.Button(right_frame, text="Previous", font=('Arial', 12), bg='#526D82', command=previous_button_click, highlightthickness=0)
previous_button.grid(row=4, column=1, sticky='w', padx=15, pady=15)

# Create the reinitialize button
def reinitialize_button_click():
    for label in placeholder_labels:
        label.configure(image=placeholder_image)
        label.image = placeholder_image  # Store a reference to prevent image garbage collection

    for brand_label in brand_labels:
        brand_label.configure(text="")

    for reference_label in reference_labels:
        reference_label.configure(text="")

    suggestions_label.configure(text="Here are some suggestions:")

reinitialize_button = tk.Button(left_frame, text="Reinitialize", font=('Arial', 12), bg='#526D82', command=reinitialize_button_click)
reinitialize_button.grid(row=9, column=0, sticky='w', padx=10, pady=0)

# Run the main window loop
window.mainloop()