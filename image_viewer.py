import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
from scipy.ndimage import convolve
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

modified_image = None

def draw_painted_strokes(image, stroke_width_range, stroke_length_range, edge_threshold):
    # Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients along x and y
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and angle
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    # Threshold the edges to get binary edge map
    edges = (magnitude > edge_threshold).astype(np.uint8)
    
    # Create output image initialized as a copy of the original
    painted_image = image.copy()
    
    # Get random points across the image
    points = np.argwhere(edges == 0)  # Non-edge points for strokes
    random_points = points[np.random.choice(points.shape[0], size=int(points.shape[0] * 0.05), replace=False)]
    
    for point in random_points:
        y, x = point
        color = image[y, x].tolist()  # Stroke color matches pixel color
        
        # Randomize stroke length and width
        stroke_length = np.random.randint(stroke_length_range[0], stroke_length_range[1])
        stroke_width = np.random.randint(stroke_width_range[0], stroke_width_range[1])
        
        # Set stroke direction at 45 degrees
        dx = int(stroke_length * np.cos(np.pi / 4))
        dy = int(stroke_length * np.sin(np.pi / 4))
        
        # Determine the start and end points for the stroke
        start_point = (x - dx // 2, y - dy // 2)
        end_point = (x + dx // 2, y + dy // 2)
        
        # Clip strokes at edges
        if edges[y, x] == 1:
            continue  # Skip drawing stroke to avoid spillover
        
        # Draw the stroke
        cv2.line(painted_image, start_point, end_point, color, stroke_width)
    
    return painted_image

# Popup window for setting stroke properties
def open_stroke_properties_window(current_width, current_length, current_threshold):
    layout = [
        [sg.Text('Stroke Width'), sg.Slider(range=(1, 10), orientation='h', default_value=current_width, key='-POPUP_STROKE_WIDTH-')],
        [sg.Text('Stroke Length'), sg.Slider(range=(5, 20), orientation='h', default_value=current_length, key='-POPUP_STROKE_LENGTH-')],
        [sg.Text('Edge Threshold'), sg.Slider(range=(0, 255), orientation='h', default_value=current_threshold, key='-POPUP_EDGE_THRESH-')],
        [sg.Button('Apply'), sg.Button('Cancel')]
    ]

    window = sg.Window('Set Stroke Properties', layout, modal=True)

    stroke_values = None
    while True:
        event, values = window.read()
        if event == 'Apply':
            # Retrieve values from the sliders
            stroke_values = {
                'stroke_width': int(values['-POPUP_STROKE_WIDTH-']),
                'stroke_length': int(values['-POPUP_STROKE_LENGTH-']),
                'edge_threshold': int(values['-POPUP_EDGE_THRESH-'])
            }
            break
        elif event in (sg.WINDOW_CLOSED, 'Cancel'):
            break

    window.close()
    return stroke_values

def adjust_saturation(image, saturation_scale):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image = hsv_image.astype(np.float32)
    hsv_image[..., 1] *= saturation_scale  # Adjust the saturation channel
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)  # Ensure values are valid
    hsv_image = hsv_image.astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# Function to adjust contrast using an S-shaped curve
def adjust_contrast(image, contrast_factor):
    # Adjust the contrast using an S-shaped function
    normalized_image = image / 255.0  # Normalize image to [0, 1]
    s_curve = 1 / (1 + np.exp(-contrast_factor * (normalized_image - 0.5)))
    adjusted_image = np.clip(s_curve * 255, 0, 255)  # Scale back to [0, 255]
    return adjusted_image.astype(np.uint8)

# Function to adjust color temperature
def adjust_temperature(image, temperature_scale):
    image = image.astype(np.float32)
    image[..., 0] = np.clip(image[..., 0] + temperature_scale, 0, 255)  # Adjust red channel
    image[..., 2] = np.clip(image[..., 2] - temperature_scale, 0, 255)  # Adjust blue channel
    return image.astype(np.uint8)

# Function to save settings to a YAML file
def save_settings(settings_dict, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(settings_dict, file)

# Function to load settings from a YAML file
def load_settings(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def nearest_neighbor_resize(image, new_width, new_height):
    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    row_ratio = original_height / new_height
    col_ratio = original_width / new_width

    for row in range(new_height):
        for col in range(new_width):
            src_row = int(row * row_ratio)
            src_col = int(col * col_ratio)
            resized_image[row, col] = image[src_row, src_col]

    return resized_image

def bilinear_resize(image, new_width, new_height):
    original_height, original_width = image.shape[:2]
    
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    x_ratio = (original_width - 1) / new_width
    y_ratio = (original_height - 1) / new_height

    for i in range(new_height):
        for j in range(new_width):
            x_l, y_l = int(x_ratio * j), int(y_ratio * i)
            x_h, y_h = min(x_l + 1, original_width - 1), min(y_l + 1, original_height - 1)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            pixel = (a * (1 - x_weight) * (1 - y_weight) +
                     b * x_weight * (1 - y_weight) +
                     c * y_weight * (1 - x_weight) +
                     d * x_weight * y_weight)

            resized_image[i, j] = pixel

    return resized_image


def open_window():
    layout = [
        [sg.Text("Enter Width:"), sg.InputText(key='Width', size=(20, 1))],
        [sg.Text("Enter Height:"), sg.InputText(key='Height', size=(20, 1))],
        [sg.Checkbox("Constrained", key='Constrained', default=True)],
        [sg.Radio("Nearest Neighbor", "RESIZE", key='Nearest', default=True), sg.Radio("Bilinear", "RESIZE", key='Bilinear')],
        [sg.Button("Submit"), sg.Button("Cancel")]
    ]
    window = sg.Window("Set Dimensions", layout, modal=True)

    while True:
        event, values = window.read()
        if event == "Submit":
            try:
                width = int(values['Width'])
                height = int(values['Height'])
                constrained = values['Constrained']
                method = 'nearest' if values['Nearest'] else 'bilinear'
                window.close()
                return width, height, constrained, method
            except ValueError:
                sg.popup_error("Please enter valid integers for width and height.")
        elif event == "Cancel" or event == sg.WIN_CLOSED:
            window.close()
            return None

        window.close()
        
def get_aspect_ratio_dimensions(image, target_width=None, target_height=None):
    original_height, original_width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    
    if target_width and not target_height:  # Calculate height based on width
        target_height = int(target_width / aspect_ratio)
    elif target_height and not target_width:  # Calculate width based on height
        target_width = int(target_height * aspect_ratio)
    
    return target_width, target_height

def create_average_kernel(size):
    return np.ones((size, size), np.float32) / (size * size)

def create_gaussian_kernel(size):
    sigma = size / 3
    gauss_kernel_1d = cv2.getGaussianKernel(size, sigma)
    gauss_kernel_2d = np.outer(gauss_kernel_1d, gauss_kernel_1d)
    return gauss_kernel_2d

def apply_filter_to_patch(patch, filter):
    fw,fh = filter.shape
    total_sum  = 0
    for i in range(fw):
        for j in range(fh):
            total_sum+= patch[i,j] * filter[i,j]
    return total_sum

def apply_filter_to_image_channel(image_channel, filter):
    ih, iw = image_channel.shape
    fh, fw = filter.shape
    
    output_h = ih - fh + 1
    output_w = iw - fw + 1
    output_image = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            patch = image_channel[i:i+fh, j:j+fw]
            output_image[i, j] = np.sum(patch * filter)
    
    return output_image

def apply_filter_to_image(image, filter):
    if len(image.shape) == 3:  # Color image
        ih, iw, _ = image.shape
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(apply_filter_to_image_channel, image[:, :, channel], filter) for channel in range(image.shape[2])]
            output_image = np.dstack([f.result() for f in futures])
        return output_image
    else:
        return apply_filter_to_image_channel(image, filter)
    
def numpy_convolution(image, filter):
    if len(image.shape) == 3:  # Color image
        return np.stack([convolve(image[:, :, i], filter) for i in range(image.shape[2])], axis=2)
    else:
        return convolve(image, filter)

def resize_image_with_aspect_ratio(image, target_width=None, target_height=None):
    original_height, original_width = image.shape[:2]
    
    
    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    print(aspect_ratio)
    if target_width and not target_height: #Calculate height based on width
        target_height = int(target_width / aspect_ratio)
    elif target_height and not target_width: #Vice versa
        target_width = int(target_height * aspect_ratio)
    
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR) #Use same resize method but with new dimension

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def construct_image_histogram(np_image):
    L = 256
    bins = np.arange(L+1)
    hist, _ = np.histogram(np_image, bins)
    return hist


def draw_hist(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    #tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return tkcanvas


def histogram_equalization_numpy(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Mask the CDF to avoid dividing by zero
    cdf_m = np.ma.masked_equal(cdf, 0)

    # Normalize the CDF to be in the range [0, 255]
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # Fill masked values and cast to uint8
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Map original image pixel values to equalized values
    equalized_image = cdf[image]

    return equalized_image

def time_filter_application(image, filter, use_numpy=False):
    start_time = time.time()
    if use_numpy:
        numpy_convolution(image, filter)
    else:
        apply_filter_to_image(image, filter)
    end_time = time.time()
    return end_time - start_time



def display_image(np_image):
    


    # Convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)

    # Get original dimensions of the image
    height, width = np_image.shape[:2]

    hist = construct_image_histogram(np_image)
    fig, ax = plt.subplots(1)
    ax.bar(np.arange(len(hist)), hist, alpha=0.5, label = "Original")

    # Define the layout
    layout = [
        [sg.Graph(
            canvas_size=(width, height),
            graph_bottom_left=(0, 0),
            graph_top_right=(width, height),
            key='-IMAGE-',
            background_color='white',
            change_submits=True,
            drag_submits=True),
         sg.Graph(
             canvas_size=(width, height),
             graph_bottom_left=(0, 0),
             graph_top_right=(width, height),
             key='-IMAGE2-',
             background_color='white',
             change_submits=True,
             drag_submits=True),
         sg.Canvas(key='-HIST-', size=(400, 200))],
        [sg.Slider((0, 15), orientation='horizontal', key="-avg-", enable_events=True), sg.Text("Average Filter")],
        [sg.Slider((0, 15), orientation='horizontal', key="-gauss-", enable_events=True), sg.Text("Gaussian Filter")],
        [sg.Text('Saturation'), sg.Slider(range=(0.0, 3.0), resolution=0.1, orientation='h', key='-SAT-', default_value=1.0,enable_events=True)],
        [sg.Text('Contrast'), sg.Slider(range=(0.5, 10), resolution=0.5, orientation='h', key='-CON-', default_value=5,enable_events=True)],
        [sg.Text('Temperature'), sg.Slider(range=(-50, 50), orientation='h', key='-TEMP-', default_value=0,enable_events=True)],
        [sg.Button('Set Stroke Properties'), sg.Button('Apply Painted Look')],
        [sg.Button("Open Window", key="open"), sg.Button('Convert'), sg.Button('Histogram Equalization'), sg.Button("Save"),
         sg.Button('Save Settings'), sg.Button('Load Settings'),sg.Button('Reset Image'), sg.Button('Exit')]
    ]
    stroke_properties = {
        'stroke_width': 3,
        'stroke_length': 10,
        'edge_threshold': 100
    }

    # Create the window
    window = sg.Window('Display Image', layout, finalize=True) 
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))
    window['-IMAGE2-'].draw_image(data=image_data, location=(0, height))
    draw_call = draw_hist(window['-HIST-'].TKCanvas, fig)
    #draw_call.draw()

    
    # Event loop
    while True:
        event, values = window.read()
        
        modified_image = np_image.copy()  # Start with the original for each update
        
        # Apply saturation, contrast, and temperature
        saturation_value = values['-SAT-']
        contrast_value = values['-CON-']
        temperature_value = values['-TEMP-']

        modified_image = adjust_saturation(modified_image, saturation_value)
        modified_image = adjust_contrast(modified_image, contrast_value)
        modified_image = adjust_temperature(modified_image, temperature_value)
        if event == 'Set Stroke Properties':
            # Open a popup window for stroke properties
            new_stroke_properties = open_stroke_properties_window(
                stroke_properties['stroke_width'],
                stroke_properties['stroke_length'],
                stroke_properties['edge_threshold']
            )
            if new_stroke_properties:
                # Update stroke properties with values from the popup
                stroke_properties.update(new_stroke_properties)

        # Apply the "Painted Look" effect
        if event == 'Apply Painted Look':
            # Apply the painted look effect with the current stroke properties
            modified_image = draw_painted_strokes(
                np_image,
                (1, stroke_properties['stroke_width']),
                (5, stroke_properties['stroke_length']),
                stroke_properties['edge_threshold']
            )

            # Update the displayed image
            window['-IMAGE2-'].draw_image(data=np_im_to_data(modified_image), location=(0, height))

        if event == 'Reset Image':
            modified_image = np_image.copy()  # Reset to the original image
            window['-IMAGE2-'].draw_image(data=np_im_to_data(modified_image), location=(0, height))
            # Reset all sliders to their initial values
            window['-SAT-'].update(value=1.0)
            window['-CON-'].update(value=5)
            window['-TEMP-'].update(value=0)
            window['-avg-'].update(value=0)
            window['-gauss-'].update(value=0)
            continue  # Skip the rest of the loop for this event
        
        if event == 'Convert':
            #Setup Grayscaled Image
            image_grayscaled = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            image_data2 = np_im_to_data(image_grayscaled)
            window['-IMAGE2-'].draw_image(data=image_data2, location=(0, height))
            modified_image = image_grayscaled
            print("Image Converted")
        
        if event == 'Histogram Equalization':
            # Perform histogram equalization on the color image
            hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = histogram_equalization_numpy(hsv[:, :, 2]) #All rows, all columns, but take their third element which is the V section, This is all we apply the equalization to
            equalized_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            equalized_image_data = np_im_to_data(equalized_img)
            window['-IMAGE2-'].draw_image(data=equalized_image_data, location=(0, height))
            modified_image = equalized_img
            
            # Construct the equalized histogram
            equalized_hist = construct_image_histogram(equalized_img)
            print("Histogram Equalization applied to color image")
            ax.bar(np.arange(len(equalized_hist)), equalized_hist, alpha=0.5, label = "Comparison")
            fig.canvas.flush_events()
            # Update the histogram plot with both original and equalized histograms
            #draw_call = draw_hist(window['-HIST-'].TKCanvas, fig)
            draw_call.draw()

        if event == '-avg-':
            kernel_size = int(values['-avg-'])
            print(kernel_size)
            if kernel_size > 0:
                avg_kernel = create_average_kernel(kernel_size)
                filtered_image = apply_filter_to_image(modified_image, avg_kernel)

                # Ensure the image data type and range are correct | Causes problems when saving
                filtered_image = np.clip(filtered_image, 0, 255)  # Clip values to ensure they are within [0, 255]  
                filtered_image = filtered_image.astype('uint8')   # Convert to uint8

                image_data_filtered = np_im_to_data(filtered_image)
                window['-IMAGE2-'].draw_image(data=image_data_filtered, location=(0, height))
                modified_image = filtered_image  # Update current image dat

        if event == '-gauss-':
        # Apply the Gaussian filter based on the slider value
            kernel_size = int(values['-gauss-'])
            print(kernel_size)
            if kernel_size > 0:
                gauss_kernel = create_gaussian_kernel(kernel_size)
                filtered_image = apply_filter_to_image(modified_image, gauss_kernel)

                # Ensure the image data type and range are correct | Causes problems when  saving otherwise
                filtered_image = np.clip(filtered_image, 0, 255)  # Clip values to ensure they are within [0, 255] 
                filtered_image = filtered_image.astype('uint8')   # Convert to uint8

                image_data_filtered = np_im_to_data(filtered_image)
                window['-IMAGE2-'].draw_image(data=image_data_filtered, location=(0, height))
                modified_image = filtered_image  # Update current image data
            
        if event == 'Save':
            if modified_image is not None:
                file_path = sg.popup_get_file('Save', save_as=True, no_window=True, file_types=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")))
                if file_path:
                    im = Image.fromarray(modified_image)  # Convert numpy array to PIL Image
                    im.save(file_path)
                    print(f"Image saved to {file_path}")
            else:
                sg.popup("No image to save", title="Error")

        
        if event == 'open':
            result = open_window()
            if result:
                width, height, constrained, method_flag = result
                if constrained:
                    # Resize maintaining aspect ratio
                    new_width, new_height = get_aspect_ratio_dimensions(np_image, target_width=width, target_height=None) if width < height else get_aspect_ratio_dimensions(np_image, target_width=None, target_height=height)
                else:
                    # Resize without preserving aspect ratio
                    new_width, new_height = width, height
                    
                if method_flag == 'nearest':
                    resized_image = nearest_neighbor_resize(np_image, new_width, new_height)
                elif method_flag == 'bilinear':
                    resized_image = bilinear_resize(np_image, new_width, new_height)
                    
                resized_image_data = np_im_to_data(resized_image)
                
                window['-IMAGE2-'].set_size((new_width, new_height))
                #window['-IMAGE-'].set_size((new_width, new_height))
                window['-HIST-'].set_size((new_width, 200))  # Assuming you want to adjust this as well
                window.size = (new_width + 20, new_height + 250)
                window['-IMAGE2-'].erase()
                print(new_height)
                window['-IMAGE2-'].change_coordinates((0,0),(new_width,new_height))
                window['-IMAGE2-'].draw_image(data=resized_image_data, location=(0, new_height))
                modified_image = resized_image
                
            else:
                print("Resize operation cancelled.")
                if event == sg.WINDOW_CLOSED or event == 'Exit':
                    break
                
        if event in ('-SAT-', '-CON-', '-TEMP-'):
            saturation_value = values['-SAT-']
            contrast_value = values['-CON-']
            temperature_value = values['-TEMP-']

            modified_image = adjust_saturation(np_image, saturation_value)
            modified_image = adjust_contrast(modified_image, contrast_value)
            modified_image = adjust_temperature(modified_image, temperature_value)

            window['-IMAGE2-'].draw_image(data=np_im_to_data(modified_image), location=(0, height))

        if event == 'Save Settings':
            save_path = sg.popup_get_file('Save Settings As', save_as=True, file_types=(("YAML Files", "*.yaml"),))
            if save_path:
                settings = {
                    'saturation': values['-SAT-'],
                    'contrast': values['-CON-'],
                    'temperature': values['-TEMP-'],
                    'avg_filter': values['-avg-'],
                    'gauss_filter': values['-gauss-']
                }
                save_settings(settings, save_path)
                sg.popup(f'Settings saved to {save_path}')

        if event == 'Load Settings':
            load_path = sg.popup_get_file('Load Settings', file_types=(("YAML Files", "*.yaml"),))
            if load_path:
                settings = load_settings(load_path)
                window['-SAT-'].update(value=settings['saturation'])
                window['-CON-'].update(value=settings['contrast'])
                window['-TEMP-'].update(value=settings['temperature'])
                window['-avg-'].update(value=settings['avg_filter'])
                window['-gauss-'].update(value=settings['gauss_filter'])

                # Apply loaded settings to the image
                saturation_value = settings['saturation']
                contrast_value = settings['contrast']
                temperature_value = settings['temperature']

                modified_image = adjust_saturation(np_image, saturation_value)
                modified_image = adjust_contrast(modified_image, contrast_value)
                modified_image = adjust_temperature(modified_image, temperature_value)

                window['-IMAGE2-'].draw_image(data=np_im_to_data(modified_image), location=(0, height))
                sg.popup('Settings loaded and applied')
                
                
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        

    window.close()

def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser = argparse.ArgumentParser(description="Compare custom convolution vs numpy's built-in convolution")
    parser.add_argument('file', help='Image file to process')
    parser.add_argument('--use-numpy', action='store_true', help='Use numpy convolution instead of custom')
    args = parser.parse_args()

    # Load the image and convert to grayscale
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #target_width = 1200
    #image = resize_image_with_aspect_ratio(image, target_width=target_width)
    """ print(f'Resized to {image.shape} while preserving aspect ratio')
    print("only smaller filter sizes used due to how long the program will take for other sizes")
    
    filter_sizes = [3, 5]  # Various filter half-lengths
    speed_results = []

    for size in filter_sizes:
        avg_filter = create_average_kernel(size)
        gauss_filter = create_gaussian_kernel(size)
        
        # Time the custom method (apply_filter_to_image)
        avg_time_custom = time_filter_application(image, avg_filter, use_numpy=False)
        gauss_time_custom = time_filter_application(image, gauss_filter, use_numpy=False)

        # Time the NumPy method
        avg_time_numpy = time_filter_application(image, avg_filter, use_numpy=True)
        gauss_time_numpy = time_filter_application(image, gauss_filter, use_numpy=True)
        
        speed_results.append({
            'Filter Size': size,
            'Custom Avg Time': avg_time_custom,
            'Custom Gauss Time': gauss_time_custom,
            'NumPy Avg Time': avg_time_numpy,
            'NumPy Gauss Time': gauss_time_numpy
        }) """

    # Convert results to a pandas DataFrame
    #df = pd.DataFrame(speed_results)
    #print(df)
    
    # Save to CSV file for later analysis
    #df.to_csv('convolution_speed_comparison.csv', index=False)

    target_width = 400  # Define a target width for the window
    #Call Resize image with our target width
    image = resize_image_with_aspect_ratio(image, target_width=target_width)


    print(f'Resized to {image.shape} while preserving aspect ratio')

    display_image(image)

    

if __name__ == '__main__':
    main()

