import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained CNN models
cnn_model1 = load_model('fruit_veg_cnn_model.keras')
cnn_model2 = load_model('brand_pred_3.keras')

# Function to preprocess image for CNN input
def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Function for CNN Model 1 prediction
def process_with_cnn1(image_path):
    cnn1_input_shape = (224, 224, 3)  # Example input shape for CNN1
    preprocessed_image = preprocess_image(image_path, cnn1_input_shape)
    
    # Run the image through CNN Model 1
    cnn1_output = cnn_model1.predict(preprocessed_image)
    
    return cnn1_output

# Function for CNN Model 2 prediction
def process_with_cnn2(image_path):
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # Load your trained .keras model
    model = load_model('fruit_veg_cnn_model.keras')


    # Path to the input image
    image_path = input("Enter img path:")

    # Function to preprocess the image
    def preprocess_image(image_path, target_size):
        # Load the image with target size
        img = load_img(image_path, target_size=target_size)
        # Convert the image to an array
        img_array = img_to_array(img)
        # Normalize the image (assuming your model was trained with normalized images)
        img_array = img_array / 255.0
        # Add a batch dimension (model expects batches of images)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    # Preprocess the image to match the model's input format
    img_size = (150, 150, 3)  # Change to the input size of your model
    preprocessed_image = preprocess_image(image_path, target_size=img_size)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Decode the predictions (assuming it's a classification model)
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest score
    predicted_prob = np.max(predictions)  # Get the highest probability

    class_labels = [
        'Apple_Bad', 'Apple_Good', 'Apple_mix',
        'Banana_Bad', 'Banana_Good', 'Banana_mix',
        'Bell Pepper_Bad', 'Bell Pepper_Good', 'Bell Pepper_Mixed',
        'Chilli Pepper_Bad', 'Chilli Pepper_Good', 'Chilli Pepper_Mixed',
        'Green Chilli_Bad', 'Green Chilli_Good', 'Green Chilli_Mixed',
        'Lemon_mix', 'Lime_Bad', 'Lime_Good',
        'Orange_Bad', 'Orange_Good', 'Orange_mix',
        'Tomato_Bad', 'Tomato_Good', 'Tomato_Mixed'
    ]


    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class[0]]

    # Output the predicted class and confidence
    print(f"Predicted class: {predicted_class_label}, Confidence: {predicted_prob:.2f}")

# Function to extract text using OCR
def extract_text(image_path):
    from ocr_test import process_image

    return process_image(image_path)
# Main function to combine results from both CNNs and OCR
def combine_cnn_ocr_results(image_path):
    # Get outputs from both CNN models
    cnn1_output = process_with_cnn1(image_path)
    cnn2_output = process_with_cnn2(image_path)
    
    # Get text extracted via OCR
    extracted_text = extract_text(image_path)
    
    # Combine results into a dictionary
    combined_result = {
        "CNN1_Output": cnn1_output,
        "CNN2_Output": cnn2_output,
        "Extracted_Text": extracted_text
    }
    
    return combined_result

# Test the functions
if __name__ == "__main__":
    image_path = 'test_image.jpg'
    
    # Combine the CNN and OCR results
    result = combine_cnn_ocr_results(image_path)
    
    # Print the result
    print("CNN1 Output:", result['CNN1_Output'])
    print("CNN2 Output:", result['CNN2_Output'])
    print("Extracted Text:", result['Extracted_Text'])
