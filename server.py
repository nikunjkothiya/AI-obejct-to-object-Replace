import base64
import io
import uuid
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import imageio.v2 as imageio
from trainer import Trainer
from utils.tools import get_config
import torch.nn.functional as F
from iopaint.single_processing import batch_inpaint
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def resize_image(input_image_base64, width=640, height=640):
    """Resizes an image from base64 data and returns the resized image as bytes."""
    try:
        # Decode base64 string to bytes
        input_image_data = base64.b64decode(input_image_base64)
        # Convert bytes to NumPy array
        img = np.frombuffer(input_image_data, dtype=np.uint8)
        # Decode NumPy array as an image
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Resize while maintaining the aspect ratio
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (width, height)  # the shape to resize to

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # Resize the image
        im = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Pad the image
        color = (114, 114, 114)  # color used for padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # divide padding into 2 sides
        dw /= 2
        dh /= 2
        # compute padding on all corners
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # Convert the resized and padded image to bytes
        resized_image_bytes = cv2.imencode('.png', im)[1].tobytes()
        return resized_image_bytes

    except Exception as e:
        print(f"Error resizing image: {e}")
        return None  # Or handle differently as needed


def load_weights(path, device):
    model_weights = torch.load(path)
    return {
        k: v.to(device)
        for k, v in model_weights.items()
    }


# Function to convert image to base64
def convert_image_to_base64(image):
    # Convert image to bytes
    _, buffer = cv2.imencode('.png', image)
    # Convert bytes to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def convert_to_base64(image):
    # Read the image file as binary data
    image_data = image.read()
    # Encode the binary data as base64
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded


@app.route('/process_images', methods=['POST'])
def process_images():
    # Static paths
    config_path = Path('configs/config.yaml')
    model_path = Path('pretrained-model/torch_model.p')

    # Check if the request contains files
    if 'input_image' not in request.files or 'append_image' not in request.files:
        return jsonify({'error': 'No files found'}), 419

    # Get the objectName from the request or use default "chair" if not provided
    default_class = request.form.get('objectName', 'chair')

    # Convert the images to base64
    try:
        input_base64 = convert_to_base64(request.files['input_image'])
        append_base64 = convert_to_base64(request.files['append_image'])
    except Exception as e:
        return jsonify({'error': 'Failed to read files'}), 419

    # Resize input image and get base64 data of resized image
    input_resized_image_bytes = resize_image(input_base64)

    # Convert resized image bytes to base64
    input_resized_base64 = base64.b64encode(input_resized_image_bytes).decode('utf-8')

    # Decode the resized image from base64 data directly
    img = cv2.imdecode(np.frombuffer(input_resized_image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Failed to decode resized image'}), 419

    H, W, _ = img.shape
    x_point = 0
    y_point = 0
    width = 1
    height = 1

    # Load a model
    model = YOLO('pretrained-model/yolov8m-seg.pt')  # pretrained YOLOv8m-seg model

    # Run batched inference on a list of images
    results = model(img, imgsz=(W,H), conf=0.5)  # chair class 56 with confidence >= 0.5
    names = model.names
    print(names)

    class_found = False
    for result in results:
        for i, label in enumerate(result.boxes.cls):
            # Check if the label matches the chair label
            if names[int(label)] == default_class:
                class_found = True
                # Convert the tensor to a numpy array
                chair_mask_np = result.masks.data[i].numpy()

                kernel = np.ones((5, 5), np.uint8)  # Create a 5x5 kernel for dilation
                chair_mask_np = cv2.dilate(chair_mask_np, kernel, iterations=2)  # Apply dilation

                # Find contours to get bounding box
                contours, _ = cv2.findContours((chair_mask_np == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Iterate over contours to find the bounding box of each object
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_point = x
                    y_point = y
                    width = w
                    height = h

                # Get the corresponding mask
                mask = result.masks.data[i].numpy() * 255
                dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # Apply dilation
                # Resize the mask to match the dimensions of the original image
                resized_mask = cv2.resize(dilated_mask, (img.shape[1], img.shape[0]))
                # Convert mask to base64
                mask_base64 = convert_image_to_base64(resized_mask)

                # call repainting and merge function
                output_base64 = repaitingAndMerge(append_base64,str(model_path), str(config_path),width, height, x_point, y_point, input_resized_base64, mask_base64)
                # Return the output base64 image in the API response
                return jsonify({'output_base64': output_base64}), 200

    # return class not found in prediction
    if not class_found:
        return jsonify({'message': f'{default_class} object not found in the image'}), 200

def repaitingAndMerge(append_image_base64_image, model_path, config_path, width, height, xposition, yposition, input_base64, mask_base64):
    config = get_config(config_path)
    device = torch.device("cpu")
    trainer = Trainer(config)
    trainer.load_state_dict(load_weights(model_path, device), strict=False)
    trainer.eval()

    # lama inpainting start
    print("lama inpainting start")
    inpaint_result_base64 = batch_inpaint('lama', 'cpu', input_base64, mask_base64)
    print("lama inpainting end")

    # Decode base64 to bytes
    inpaint_result_bytes = base64.b64decode(inpaint_result_base64)

    # Convert bytes to NumPy array
    inpaint_result_np = np.array(Image.open(io.BytesIO(inpaint_result_bytes)))

    # Create PIL Image from NumPy array
    final_image = Image.fromarray(inpaint_result_np)

    print("merge start")
    # Decode base64 to binary data
    decoded_image_data = base64.b64decode(append_image_base64_image)
    # Convert binary data to a NumPy array
    append_image = cv2.imdecode(np.frombuffer(decoded_image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    # Resize the append image while preserving transparency
    resized_image = cv2.resize(append_image, (width, height), interpolation=cv2.INTER_AREA)
    # Convert the resized image to RGBA format (assuming it's in BGRA format)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGBA)
    # Create a PIL Image from the resized image with transparent background
    append_image_pil = Image.fromarray(resized_image)
    # Paste the append image onto the final image
    final_image.paste(append_image_pil, (xposition, yposition), append_image_pil)
    # Save the resulting image
    print("merge end")
    # Convert the final image to base64
    with io.BytesIO() as output_buffer:
        final_image.save(output_buffer, format='PNG')
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

    return output_base64


if __name__ == '__main__':
    app.run(debug=True)