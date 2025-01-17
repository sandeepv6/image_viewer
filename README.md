# Image Viewer

The **Image Viewer** is a versatile Python application designed to provide users with advanced image processing and editing capabilities through a graphical user interface (GUI). This project utilizes libraries such as OpenCV, NumPy, PySimpleGUI, and Matplotlib to create a rich and interactive experience for users.

## Features

- **Image Display and Viewing**
  - Load and display images in a user-friendly interface.
  - Resize images while maintaining or ignoring the aspect ratio.

- **Filters and Effects**
  - Apply **Average Filter** and **Gaussian Filter** to enhance or blur images.
  - Real-time adjustment of filter properties using sliders.

- **Image Enhancements**
  - **Adjust Saturation:** Modify the vibrancy of colors.
  - **Adjust Contrast:** Use an S-shaped curve for dynamic contrast control.
  - **Adjust Temperature:** Warm up or cool down the color tones.

- **Advanced Image Processing**
  - Add painted strokes with customizable properties (width, length, and edge threshold).
  - Perform **Histogram Equalization** for improved brightness and contrast.

- **Customization and Settings**
  - Save and load settings to/from YAML files for reproducibility.
  - Reset images and settings to default values.

- **Resize Options**
  - Resize images using **Nearest Neighbor** or **Bilinear Interpolation** methods.
  - Customizable dimensions with an option to maintain aspect ratio.

- **Histogram Visualization**
  - View histograms of the original and processed images side by side.

- **File Operations**
  - Save the processed images in various formats, including PNG and JPEG.

## Requirements

Ensure you have the following Python libraries installed:

- `opencv-python`
- `numpy`
- `PySimpleGUI`
- `Pillow`
- `matplotlib`
- `yaml`

To install the required libraries, run:

```bash
pip install opencv-python numpy PySimpleGUI Pillow matplotlib pyyaml
```

## Usage
- Run the application:
```bash
- python image_viewer.py <path_to_image_file>
```
- Use the GUI sliders and buttons to apply filters, adjust settings, and process your image.
- Save your work using the Save button.

## Key Functionality
- Filters
- Adjust the intensity of Gaussian and Average filters using the sliders provided.
- Color Adjustments
- Fine-tune saturation, contrast, and temperature in real time.
- Painted Look
- Add artistic strokes to your image with customizable properties for a unique effect.
- Histogram
- Visualize the distribution of pixel intensities before and after processing.
- Resizing
- Easily resize images with options to maintain or ignore aspect ratio.
- Save and Load Settings
- Save your current settings to a YAML file and reload them later for consistent results.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Your feedback and suggestions are always welcome!
