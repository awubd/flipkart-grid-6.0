# # # # import cv2
# # # # import numpy as np
# # # # from tensorflow.keras.models import load_model
# # # # import time
# # # # import os
# # # # import re
# # # # import glob
# # # # import pytesseract
# # # # from ocr import ImageOCR
# # # # from postprocessing import TextPostprocessing
# # # # from fuzzywuzzy import process

# # # # # Set the path for Tesseract OCR executable
# # # # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # # # Load pre-trained CNN models
# # # # cnn_model1 = load_model('fruit_veg_cnn_model.keras')
# # # # cnn_model2 = load_model('fruit_veg_cnn_model.keras')

# # # # # Sample labels of biscuit brands
# # # # biscuit_labels = [
# # # #     "Sunfeast Dark Fantasy",
# # # #     "Bourbon",
# # # #     "Good Day",
# # # #     "Parle-G",
# # # #     "Hide & Seek",
# # # #     "Milk Bikis",
# # # #     "Treat",
# # # #     "Nice Time",
# # # #     "Marie Gold"
# # # # ]

# # # # # Function to preprocess image for CNN input
# # # # def preprocess_image(image_path, input_shape):
# # # #     image = cv2.imread(image_path)
# # # #     if image is None:
# # # #         raise FileNotFoundError(f"Image file '{image_path}' not found.")
# # # #     image = cv2.resize(image, (input_shape[1], input_shape[0]))
# # # #     image = image / 255.0  # Normalize
# # # #     return np.expand_dims(image, axis=0)

# # # # # Function for CNN Model 1 prediction
# # # # def process_with_cnn1(image_path):
# # # #     cnn1_input_shape = (150, 150, 3)   # Input shape for CNN1
# # # #     preprocessed_image = preprocess_image(image_path, cnn1_input_shape)
    
# # # #     # Run the image through CNN Model 1
# # # #     cnn1_output = cnn_model1.predict(preprocessed_image)
    
# # # #     return cnn1_output

# # # # # Function for CNN Model 2 prediction
# # # # def process_with_cnn2(image_path):
# # # #     cnn2_input_shape = (150, 150, 3)  # Input shape for CNN2
# # # #     preprocessed_image = preprocess_image(image_path, cnn2_input_shape)
    
# # # #     # Run the image through CNN Model 2
# # # #     cnn2_output = cnn_model2.predict(preprocessed_image)
    
# # # #     return cnn2_output

# # # # # Function to extract text using OCR
# # # # def extract_text(image_path):
# # # #     def filename_encoder(dir):
# # # #         '''Reads the file number and language(alpha-3 format) of the image from the file name'''
# # # #         filename = os.path.basename(dir)
# # # #         name = os.path.splitext(filename)[0]
# # # #         lang = filename[0:3]
# # # #         number = re.findall(r'[0-9]{1,5}', name)
        
# # # #         if number:  # Check if any number was found
# # # #             return (lang, number[0])
# # # #         else:
# # # #             return (lang, None)

# # # #     def extract_text_from_image(img):
# # # #         # Perform OCR to extract text
# # # #         extracted_text = pytesseract.image_to_string(img, lang='eng')
# # # #         return extracted_text

# # # #     def find_closest_label(extracted_text, labels):
# # # #         # Use fuzzy matching to find the closest label
# # # #         closest_match, score = process.extractOne(extracted_text, labels)
# # # #         return closest_match, score

# # # #     img = cv2.imread(image_path)
# # # #     if img is None:
# # # #         raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
# # # #     start = time.time()
# # # #     language, number = filename_encoder(image_path)
# # # #     print(f'Image № {number}, Filename language {language}')

# # # #     ocr = ImageOCR(img)

# # # #     # Extract text from image using OCR
# # # #     recognized_text = ocr.get_text(text_lang='eng', crop=1)
# # # #     print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
# # # #     # Extract and clean the text from the recognized output
# # # #     extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
# # # #     extracted_text_cleaned = extracted_text.strip()
# # # #     print(f"Extracted Text:\n{extracted_text_cleaned}")

# # # #     # Find the closest biscuit label
# # # #     closest_label, match_score = find_closest_label(extracted_text_cleaned, biscuit_labels)
# # # #     print(f"Closest Label: {closest_label}, Match Score: {match_score}")

# # # #     # We iterate over the found text blocks and remove unnecessary characters
# # # #     postprocessing = TextPostprocessing()
# # # #     for dict in recognized_text:
# # # #         lang = dict['lang']
# # # #         cleared_text = postprocessing.stringFilter(input_string=dict['text'])
# # # #         print(f'Clear text:\n{cleared_text}\nRecognized language: {lang}')

# # # #     print('-' * 30)

# # # #     return closest_label, match_score

# # # # # Main function to combine results from both CNNs and OCR
# # # # def combine_cnn_ocr_results(image_path):
# # # #     # Get outputs from both CNN models
# # # #     cnn1_output = process_with_cnn1(image_path)
# # # #     cnn2_output = process_with_cnn2(image_path)
    
# # # #     # Get text extracted via OCR
# # # #     extracted_label, match_score = extract_text(image_path)
    
# # # #     # Combine results into a dictionary
# # # #     combined_result = {
# # # #         "CNN1_Output": cnn1_output,
# # # #         "CNN2_Output": cnn2_output,
# # # #         "Extracted_Label": extracted_label,
# # # #         "Match_Score": match_score
# # # #     }
    
# # # #     return combined_result

# # # # # Test the functions
# # # # if __name__ == "__main__":
# # # #     image_path = 'image_test/eng-1.jpg'  # Modify the path as needed
    
# # # #     # Combine the CNN and OCR results
# # # #     result = combine_cnn_ocr_results(image_path)
    
# # # #     # Print the result
# # # #     print("CNN1 Output:", result['CNN1_Output'])
# # # #     print("CNN2 Output:", result['CNN2_Output'])
# # # #     print("Extracted Label:", result['Extracted_Label'])
# # # #     print("Match Score:", result['Match_Score'])


# # # import cv2
# # # import numpy as np
# # # from tensorflow.keras.models import load_model
# # # import time
# # # import os
# # # import re
# # # import glob
# # # import pytesseract
# # # from ocr import ImageOCR
# # # from postprocessing import TextPostprocessing
# # # from fuzzywuzzy import process
# # # from keras.preprocessing.image import load_img, img_to_array

# # # # Set the path for Tesseract OCR executable
# # # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # # Load pre-trained CNN models
# # # # cnn_model1 = load_model('brand_pred_4.keras')
# # # cnn_model2 = load_model('fruit_veg_cnn_model.keras')

# # # # Sample labels of biscuit brands
# # # biscuit_labels = [
# # #     "Anik_Ghee",
# # #     "Biscuit_Parle_G",
# # #     "Biscuit_Parle_Karckjack",
# # #     "Biscuit_Parle_Monaco",
# # #     "Biscuit_Patanjali_ButterCookies",
# # #     "Colddrink_pepsi",
# # #     "Detergent_Aerial",
# # #     "Detergent_Aerial_Matic_Topload",
# # #     "Detergent_Ghari",
# # #     "Detergent_Patanjali_Herbo_Wash",
# # #     "Detergent_Surfexcel_Matic_Topload",
# # #     "Detergent_Surfexcel_Powder",
# # #     "Detergent_Tide",
# # #     "Detol_Liquid",
# # #     "Ezee",
# # #     "Fabric_Softner_comfort",
# # #     "Ghadi_Matic",
# # #     "Halonix_Prime",
# # #     "Headphones_Zebronics",
# # #     "Maggi",
# # #     "Shampoo_ClinicPlus",
# # #     "Shampoo_Head_and_Shoulder",
# # #     "Shampoo_Himalaya",
# # #     "Shampoo_Himalaya_Gentle_Body",
# # #     "Shampoo_Pantene",
# # #     "Soap_Detol",
# # #     "Soap_Dove",
# # #     "Soap_Lux",
# # #     "Soap_Mysore_Sandal",
# # #     "Soap_Rin_bar",
# # #     "Talc_Ponds",
# # #     "Toothpaste_Closeup",
# # #     "Toothpaste_Colgate",
# # #     "Toothpaste_Pepsudent"
# # # ]

# # # # Function to preprocess image for CNN input
# # # def preprocess_image(image_path, input_shape):
# # #     image = cv2.imread(image_path)
# # #     if image is None:
# # #         raise FileNotFoundError(f"Image file '{image_path}' not found.")
# # #     image = cv2.resize(image, (input_shape[1], input_shape[0]))
# # #     image = image / 255.0  # Normalize
# # #     return np.expand_dims(image, axis=0)

# # # # Function for CNN Model 1 prediction
# # # # def process_with_cnn1(image_path):
# # # #     cnn1_input_shape = (150, 150, 3)   # Input shape for CNN1
# # # #     preprocessed_image = preprocess_image(image_path, cnn1_input_shape)
    
# # # #     # Run the image through CNN Model 1
# # # #     cnn1_output = cnn_model1.predict(preprocessed_image)
    
# # # #     return cnn1_output
# # # def preprocess_image_2(image_path, target_size):
# # #     # Load the image with target size
# # #     img = load_img(image_path, target_size=target_size)
# # #     # Convert the image to an array
# # #     img_array = img_to_array(img)
# # #     # Normalize the image (assuming your model was trained with normalized images)
# # #     img_array = img_array / 255.0
# # #     # Add a batch dimension (model expects batches of images)
# # #     img_array = np.expand_dims(img_array, axis=0)
# # #     print("12122222",img_array)

# # #     return img_array

# # # def process_with_cnn1(image_path):
# # #     img_size = (150, 150, 3)  # Change to the input size of your model
# # #     preprocessed_image = preprocess_image_2(image_path, target_size=img_size)

# # #     # Make predictions
# # #     predictions = cnn_model1.predict(preprocessed_image)

# # #     # Decode the predictions (assuming it's a classification model)
# # #     predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest score
# # #     predicted_prob = np.max(predictions)  # Get the highest probability

# # #     class_labels = [
# # #         'Anik_Ghee',                          # 0
# # #         'Biscuit_Parle_G',                    # 1
# # #         'Biscuit_Parle_Karckjack',            # 2
# # #         'Biscuit_Parle_Monaco',               # 3
# # #         'Biscuit_Patanjali_ButterCookies',    # 4
# # #         'Colddrink_Pepsi',                    # 5
# # #         'Detergent_Aerial',                   # 6
# # #         'Detergent_Aerial_Matic_Topload',     # 7
# # #         'Detergent_Ghari',                    # 8
# # #         'Detergent_Patanjali_Herbo_Wash',     # 9
# # #         'Detergent_Surfexcel_Matic_Topload',  # 10
# # #         'Detergent_Surfexcel_Powder',         # 11
# # #         'Detergent_Tide',                     # 12
# # #         'Detol_Liquid',                       # 13
# # #         'Ezee',                               # 14
# # #         'Fabric_Softner_Comfort',             # 15
# # #         'Ghadi_Matic',                        # 16
# # #         'Halonix_Prime',                      # 17
# # #         'Headphones_Zebronics',               # 18
# # #         'Maggi',                              # 19
# # #         'Shampoo_ClinicPlus',                 # 20
# # #         'Shampoo_Head_&_Shoulder',            # 21
# # #         'Shampoo_Himalaya',                   # 22
# # #         'Shampoo_Himalaya_Gentle_Body',       # 23
# # #         'Shampoo_Pantene',                    # 24
# # #         'Soap_Detol',                         # 25
# # #         'Soap_Dove',                          # 26
# # #         'Soap_Lux',                           # 27
# # #         'Soap_Mysore_Sandal',                 # 28
# # #         'Soap_Rin_Bar',                       # 29
# # #         'Talc_Ponds',                         # 30
# # #         'Toothpaste_Closeup',                 # 31
# # #         'Toothpaste_Colgate',                 # 32
# # #         'Toothpaste_Pepsudent'                # 33
# # #     ]

# # #     # Get the predicted class label
# # #     predicted_class_label = class_labels[predicted_class[0]]
    
# # #     # Output the predicted class and confidence
# # #     print(f"Predicted class: {predicted_class_label}, Confidence: {predicted_prob:.2f}")
    
# # #     return predicted_class_label, predicted_prob

# # # # Function for CNN Model 2 prediction
# # # def process_with_cnn2(image_path):
# # #     cnn2_input_shape = (150, 150, 3)  # Input shape for CNN2
# # #     preprocessed_image = preprocess_image(image_path, cnn2_input_shape)
    
# # #     # Run the image through CNN Model 2
# # #     cnn2_output = cnn_model2.predict(preprocessed_image)
    
# # #     return cnn2_output

# # # # Function to extract text using OCR
# # # def extract_text(image_path):
# # #     def filename_encoder(dir):
# # #         '''Reads the file number and language(alpha-3 format) of the image from the file name'''
# # #         filename = os.path.basename(dir)
# # #         name = os.path.splitext(filename)[0]
# # #         lang = filename[0:3]
# # #         number = re.findall(r'[0-9]{1,5}', name)
        
# # #         if number:  # Check if any number was found
# # #             return (lang, number[0])
# # #         else:
# # #             return (lang, None)

# # #     def extract_text_from_image(img):
# # #         # Perform OCR to extract text
# # #         extracted_text = pytesseract.image_to_string(img, lang='eng')
# # #         return extracted_text

# # #     def find_closest_label(extracted_text, labels):
# # #         # Use fuzzy matching to find the closest label
# # #         closest_match, score = process.extractOne(extracted_text, labels)
# # #         return closest_match, score

# # #     img = cv2.imread(image_path)
# # #     if img is None:
# # #         raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
# # #     start = time.time()
# # #     language, number = filename_encoder(image_path)
# # #     print(f'Image № {number}, Filename language {language}')

# # #     ocr = ImageOCR(img)

# # #     # Extract text from image using OCR
# # #     recognized_text = ocr.get_text(text_lang='eng', crop=1)
# # #     print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
# # #     # Extract and clean the text from the recognized output
# # #     extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
# # #     extracted_text_cleaned = extracted_text.strip()
# # #     print(f"Extracted Text:\n{extracted_text_cleaned}")

# # #     # Find the closest biscuit label
# # #     closest_label, match_score = find_closest_label(extracted_text_cleaned, biscuit_labels)
# # #     print(f"Closest Label: {closest_label}, Match Score: {match_score}")

# # #     # We iterate over the found text blocks and remove unnecessary characters
# # #     postprocessing = TextPostprocessing()
# # #     for dict in recognized_text:
# # #         lang = dict['lang']
# # #         cleared_text = postprocessing.stringFilter(input_string=dict['text'])
# # #         print(f'Clear text:\n{cleared_text}\nRecognized language: {lang}')

# # #     print('-' * 30)

# # #     return closest_label, match_score

# # # # Main function to combine results from both CNNs and OCR
# # # def combine_cnn_ocr_results(image_path):
# # #     # Get outputs from both CNN models
# # #     # cnn1_output = process_with_cnn1(image_path)
# # #     cnn2_output = process_with_cnn2(image_path)
    
# # #     # Get predicted classes from CNN outputs
# # #     # cnn1_predicted_class = np.argmax(cnn1_output, axis=1)[0]
# # #     cnn2_predicted_class = np.argmax(cnn2_output, axis=1)[0]

# # #     # Define the label mapping
# # #     label_mapping = {
# # #         0: "Apple_Bad",
# # #         1: "Apple_Good",
# # #         2: "Apple_mix",
# # #         3: "Banana_Bad",
# # #         4: "Banana_Good",
# # #         5: "Banana_mix",
# # #         6: "Bell Pepper_Bad",
# # #         7: "Bell Pepper_Good",
# # #         8: "Bell Pepper_Mixed",
# # #         9: "Chile Pepper_Bad",
# # #         10: "Chile Pepper_Good",
# # #         11: "Chile Pepper_Mixed",
# # #         12: "Green Chile_Bad",
# # #         13: "Green Chile_Good",
# # #         14: "Green Chile_Mixed",
# # #         15: "Lemon_mix",
# # #         16: "Lime_Bad",
# # #         17: "Lime_Good",
# # #         18: "Orange_Bad",
# # #         19: "Orange_Good",
# # #         20: "Orange_mix",
# # #         21: "Tomato_Bad",
# # #         22: "Tomato_Good",
# # #         23: "Tomato_Mixed"
# # #     }

# # #     # Get text extracted via OCR
# # #     extracted_label, match_score = extract_text(image_path)
    
# # #     # Map the predicted class to its label
# # #     cnn2_predicted_label = label_mapping.get(cnn2_predicted_class, "Unknown Class")
    
# # #     # Combine results into a dictionary
# # #     combined_result = {
# # #         # "CNN1_Predicted_Class": cnn1_predicted_class,
# # #         "CNN2_Predicted_Class": cnn2_predicted_class,
# # #         "CNN2_Predicted_Label": cnn2_predicted_label,
# # #         "Extracted_Label": extracted_label,
# # #         "Match_Score": match_score
# # #     }
    
# # #     # Print the predicted class and label
# # #     print(f"CNN2 Predicted Class: {cnn2_predicted_class} - Label: {cnn2_predicted_label}")
    
# # #     return combined_result



# # # # Test the functions
# # # if __name__ == "__main__":
# # #     image_path = r"C:\Users\HP\OneDrive\Desktop\WhatsApp Image 2024-10-20 at 23.48.18_ffc23208.jpg"  # Modify the path as needed
    
# # #     # Combine the CNN and OCR results
# # #     result = combine_cnn_ocr_results(image_path)
    
# # #     # Print the result
# # #     # print("CNN1 Predicted Class:", result['CNN1_Predicted_Class'])
# # #     print("CNN2 Predicted Class:", result['CNN2_Predicted_Class'])
# # #     print("Extracted Label:", result['Extracted_Label'])
# # #     print("Match Score:", result['Match_Score'])






# # import cv2
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # import tensorflow as tf
# # import time
# # import os
# # import re
# # import pytesseract
# # from ocr import ImageOCR
# # from postprocessing import TextPostprocessing
# # from fuzzywuzzy import process

# # # Set the path for Tesseract OCR executable
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # Load pre-trained CNN models
# # cnn_model1 = tf.keras.models.load_model('fruit_veg_cnn_model_4.keras')
# # cnn_model2 = tf.keras.models.load_model('brand_pred_3.keras')

# # # Sample labels of biscuit brands
# # biscuit_labels = [
# #     "Sunfeast Dark Fantasy",
# #     "Bourbon",
# #     "Good Day",
# #     "Parle-G",
# #     "Hide & Seek",
# #     "Milk Bikis",
# #     "Treat",
# #     "Nice Time",
# #     "MarieGold",
# # ]

# # # Function to preprocess image for CNN input
# # def preprocess_image(image, input_shape):
# #     image = cv2.resize(image, (input_shape[1], input_shape[0]))
# #     image = image / 255.0  # Normalize
# #     return np.expand_dims(image, axis=0)

# # # Function for CNN Model 1 prediction
# # def process_with_cnn1(image):
# #     cnn1_input_shape = (150, 150, 3)   # Input shape for CNN1
# #     preprocessed_image = preprocess_image(image, cnn1_input_shape)
    
# #     # Run the image through CNN Model 1
# #     cnn1_output = cnn_model1.predict(preprocessed_image)
    
# #     return cnn1_output

# # # Function for CNN Model 2 prediction
# # def process_with_cnn2(image):
# #     cnn2_input_shape = (150, 150, 3)  # Input shape for CNN2
# #     preprocessed_image = preprocess_image(image, cnn2_input_shape)
    
# #     # Run the image through CNN Model 2
# #     cnn2_output = cnn_model2.predict(preprocessed_image)
    
# #     return cnn2_output

# # # Function to extract text using OCR
# # def extract_text(image):
# #     def find_closest_label(extracted_text, labels):
# #         # Use fuzzy matching to find the closest label
# #         closest_match, score = process.extractOne(extracted_text, labels)
# #         return closest_match, score

# #     start = time.time()
# #     ocr = ImageOCR(image)

# #     # Extract text from image using OCR
# #     recognized_text = ocr.get_text(text_lang='eng', crop=1)
# #     print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
# #     # Extract and clean the text from the recognized output
# #     extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
# #     extracted_text_cleaned = extracted_text.strip()
# #     print(f"Extracted Text:\n{extracted_text_cleaned}")

# #     # Find the closest biscuit label
# #     closest_label, match_score = find_closest_label(extracted_text_cleaned, biscuit_labels)
# #     print(f"Closest Label: {closest_label}, Match Score: {match_score}")

# #     # We iterate over the found text blocks and remove unnecessary characters
# #     postprocessing = TextPostprocessing()
# #     for dict in recognized_text:
# #         lang = dict['lang']
# #         cleared_text = postprocessing.stringFilter(input_string=dict['text'])
# #         print(f'Clear text:\n{cleared_text}\nRecognized language: {lang}')

# #     print('-' * 30)

# #     return closest_label, match_score

# # # Function to capture image from webcam
# # def capture_from_webcam():
# #     print("Capturing image from webcam...")
# #     cap = cv2.VideoCapture(0)

# #     if not cap.isOpened():
# #         print("Error: Could not open webcam.")
# #         return None

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("Error: Failed to capture image.")
# #             break

# #         # Show the captured frame
# #         cv2.imshow("Press 's' to save or 'q' to quit", frame)

# #         # Wait for user input: 's' to save the frame, 'q' to quit
# #         key = cv2.waitKey(1) & 0xFF
# #         if key == ord('s'):
# #             # Save the captured frame and return it
# #             cap.release()
# #             cv2.destroyAllWindows()
# #             return frame
# #         elif key == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()
# #     return None

# # # Main function to combine results from both CNNs and OCR
# # def combine_cnn_ocr_results(image):
# #     # Get outputs from both CNN models
# #     cnn1_output = process_with_cnn1(image)
# #     cnn2_output = process_with_cnn2(image)
    
# #     # Get predicted classes from CNN outputs
# #     cnn1_predicted_class = np.argmax(cnn1_output, axis=1)[0]
# #     cnn2_predicted_class = np.argmax(cnn2_output, axis=1)[0]

# #     # Get text extracted via OCR
# #     extracted_label, match_score = extract_text(image)
    
# #     # Combine results into a dictionary
# #     combined_result = {
# #         "CNN1_Predicted_Class": cnn1_predicted_class,
# #         "CNN2_Predicted_Class": cnn2_predicted_class,
# #         "Extracted_Label": extracted_label,
# #         "Match_Score": match_score
# #     }
    
# #     return combined_result

# # # Main script to choose image source
# # if __name__ == "__main__":
# #     while True:
# #         # Ask user to choose between webcam capture or loading an image
# #         choice = input("Choose input method: [1] Webcam, [2] Insert Image, or type 'quit' to exit: ")

# #         if choice.lower() == 'quit':
# #             print("Exiting the program.")
# #             break
# #         elif choice == '1':
# #             # Capture image from webcam
# #             image = capture_from_webcam()
# #             if image is None:
# #                 print("Error: No image captured.")
# #                 continue
# #         elif choice == '2':
# #             # Load image from inserted file
# #             image_path = input("Enter the image file path: ")
# #             image = cv2.imread(image_path)
# #             if image is None:
# #                 print(f"Error: Image file '{image_path}' not found.")
# #                 continue
# #         else:
# #             print("Invalid choice. Please try again.")
# #             continue

# #         # Combine the CNN and OCR results
# #         result = combine_cnn_ocr_results(image)
        
# #         # Print the result
# #         print("CNN1 Predicted Class:", result['CNN1_Predicted_Class'])
# #         print("CNN2 Predicted Class:", result['CNN2_Predicted_Class'])
# #         print("Extracted Label:", result['Extracted_Label'])
# #         print("Match Score:", result['Match_Score'])






# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import time
# import pytesseract
# from ocr import ImageOCR
# from postprocessing import TextPostprocessing
# from fuzzywuzzy import process

# # Set the path for Tesseract OCR executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load pre-trained CNN models
# cnn_model1 = tf.keras.models.load_model('fruit_veg_cnn_model_4.keras')
# cnn_model2 = tf.keras.models.load_model('brand_pred_3.keras')


# # Label mappings for CNN1
# cnn1_labels = [
#     "Apple_Bad", "Apple_Good", "Apple_mix", "Banana_Bad", "Banana_Good", "Banana_mix",
#     "Bell Pepper_Bad", "Bell Pepper_Good", "Bell Pepper_Mixed", "Chile Pepper_Bad",
#     "Chile Pepper_Good", "Chile Pepper_Mixed", "Green Chile_Bad", "Green Chile_Good",
#     "Green Chile_Mixed", "Lemon_mix", "Lime_Bad", "Lime_Good", "Orange_Bad", "Orange_Good",
#     "Orange_mix", "Tomato_Bad", "Tomato_Good", "Tomato_Mixed"
# ]

# # Label mappings for CNN2
# cnn2_labels = [
#     "Anik_Ghee", "Biscuit_Parle_G", "Biscuit_Parle_Karckjack", "Biscuit_Parle_Monaco",
#     "Biscuit_Patanjali_ButterCookies", "Colddrink_pepsi", "Detergent_Aerial",
#     "Detergent_Aerial_Matic_Topload", "Detergent_Ghari", "Detergent_Patanjali Herbo Wash",
#     "Detergent_Surfexcel_Matic_Topload", "Detergent_Surfexcel_Powder", "Detergent_Tide",
#     "Detol_Liquid", "Ezee", "Fabric_Softner_comfort", "Ghadi Matic", "Halonix_Prime",
#     "Headphones_Zebronics", "Maggi", "Shampoo_ClinicPlus", "Shampoo_Head_&_Shoulder",
#     "Shampoo_Himalaya", "Shampoo_Himalaya_Gentle_Body", "Shampoo_Pantene", "Soap_Detol",
#     "Soap_Dove", "Soap_Lux", "Soap_Mysore_Sandal", "Soap_Rin_bar", "Talc_Ponds",
#     "Toothpaste_Closeup", "Toothpaste_Colgate", "Toothpaste_Pepsudent"
# ]

# # Function to preprocess image for CNN input
# def preprocess_image(image, input_shape):
#     image = cv2.resize(image, (input_shape[1], input_shape[0]))
#     image = image / 255.0  # Normalize
#     return np.expand_dims(image, axis=0)

# # Function for CNN Model 1 prediction
# def process_with_cnn1(image):
#     cnn1_input_shape = (150, 150, 3)  # Input shape for CNN1
#     preprocessed_image = preprocess_image(image, cnn1_input_shape)
    
#     # Run the image through CNN Model 1
#     cnn1_output = cnn_model1.predict(preprocessed_image)
    
#     # Get predicted class and map to label
#     cnn1_predicted_class = np.argmax(cnn1_output, axis=1)[0]
#     return cnn1_predicted_class, cnn1_labels[cnn1_predicted_class]

# # Function for CNN Model 2 prediction
# def process_with_cnn2(image):
#     cnn2_input_shape = (150, 150, 3)  # Input shape for CNN2
#     preprocessed_image = preprocess_image(image, cnn2_input_shape)
    
#     # Run the image through CNN Model 2
#     cnn2_output = cnn_model2.predict(preprocessed_image)
    
#     # Get predicted class and map to label
#     cnn2_predicted_class = np.argmax(cnn2_output, axis=1)[0]
#     return cnn2_predicted_class, cnn2_labels[cnn2_predicted_class]

# # Function to extract text using OCR
# def extract_text(image):
#     def find_closest_label(extracted_text, labels):
#         # Use fuzzy matching to find the closest label
#         closest_match, score = process.extractOne(extracted_text, labels)
#         return closest_match, score

#     start = time.time()
#     ocr = ImageOCR(image)

#     # Extract text from image using OCR
#     recognized_text = ocr.get_text(text_lang='eng', crop=1)
#     print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
#     # Extract and clean the text from the recognized output
#     extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
#     extracted_text_cleaned = extracted_text.strip()
#     print(f"Extracted Text:\n{extracted_text_cleaned}")

#     return extracted_text_cleaned

# # Function to capture image from webcam
# def capture_from_webcam():
#     print("Capturing image from webcam...")
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break

#         # Show the captured frame
#         cv2.imshow("Press 's' to save or 'q' to quit", frame)

#         # Wait for user input: 's' to save the frame, 'q' to quit
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('s'):
#             # Save the captured frame and return it
#             cap.release()
#             cv2.destroyAllWindows()
#             return frame
#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return None

# # Main function to run user-specified functionality
# def main():
#     # Ask user to choose between webcam capture or loading an image
#     choice = input("Choose input method: [1] Webcam, [2] Insert Image: ")

#     if choice == '1':
#         # Capture image from webcam
#         image = capture_from_webcam()
#         if image is None:
#             print("Error: No image captured.")
#             return
#     elif choice == '2':
#         # Load image from inserted file
#         image_path = input("Enter the image file path: ")
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Error: Image file '{image_path}' not found.")
#             return
#     else:
#         print("Invalid choice.")
#         return

#     # Ask user to choose which function to run: CNN1, CNN2, or Text Extraction
#     model_choice = input("Choose function to run: [1] CNN1, [2] CNN2, [3] Extract Text: ")

#     if model_choice == '1':
#         predicted_class, label = process_with_cnn1(image)
#         print(f"CNN1 Predicted Class: {predicted_class} ({label})")
#     elif model_choice == '2':
#         predicted_class, label = process_with_cnn2(image)
#         print(f"CNN2 Predicted Class: {predicted_class} ({label})")
#     elif model_choice == '3':
#         extracted_text = extract_text(image)
#         print(f"Extracted Text: {extracted_text}")
#     else:
#         print("Invalid choice.")

# if __name__ == "__main__":
#     main()




import cv2
import numpy as np
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import os
import re
import pytesseract
from ocr import ImageOCR
from postprocessing import TextPostprocessing
from fuzzywuzzy import process

weights_path = "OCR-pipeline-for-product-labels-main\yolov3.weights"  # Absolute path to the weights file
config_path = "OCR-pipeline-for-product-labels-main\yolov3.cfg"       # Absolute path to the configuration file


# Set the path for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load pre-trained CNN models
cnn_model1 = tf.keras.models.load_model('fruit_veg_cnn_model_4.keras')
cnn_model2 = tf.keras.models.load_model('brand_pred_3.keras')

# Class labels for CNN1 (fruits and vegetables)
cnn1_labels = [
    "Apple_Bad", "Apple_Good", "Apple_mix",
    "Banana_Bad", "Banana_Good", "Banana_mix",
    "Bell Pepper_Bad", "Bell Pepper_Good", "Bell Pepper_Mixed",
    "Chile Pepper_Bad", "Chile Pepper_Good", "Chile Pepper_Mixed",
    "Green Chile_Bad", "Green Chile_Good", "Green Chile_Mixed",
    "Lemon_mix", "Lime_Bad", "Lime_Good",
    "Orange_Bad", "Orange_Good", "Orange_mix",
    "Tomato_Bad", "Tomato_Good", "Tomato_Mixed"
]

# Class labels for CNN2 (biscuit brands)
cnn2_labels = [
    "Anik_Ghee", "Biscuit_Parle_G", "Biscuit_Parle_Karckjack",
    "Biscuit_Parle_Monaco", "Detergent_Surfexcel_Powder",  "Colddrink_pepsi",
    "Detergent_Aerial", "Detergent_Aerial_Matic_Topload", "Detergent_Ghari",
    "Detergent_Patanjali Herbo Wash", "Detergent_Surfexcel_Matic_Topload", 
    "Biscuit_Patanjali_ButterCookies", "Detergent_Tide", "Detol_Liquid", "Ezee", 
    "Fabric_Softner_comfort", "Ghadi Matic", "Halonix_Prime", "Headphones_Zebronics", 
    "Maggi", "Shampoo_ClinicPlus", "Shampoo_Head_&_Shoulder", "Shampoo_Himalaya", 
    "Shampoo_Himalaya_Gentle_Body", "Shampoo_Pantene", "Soap_Detol", "Soap_Dove", 
    "Soap_Lux", "Soap_Mysore_Sandal", "Soap_Rin_bar", "Talc_Ponds", "Toothpaste_Closeup", 
    "Toothpaste_Colgate", "Toothpaste_Pepsudent"
]

biscuits_labels = [
    "Sunfeast Dark Fantasy",
    "Bourbon",
    "Good Day",
    "Parle-G",
    "Hide & Seek",
    "Milk Bikis",
    "Treat",
    "Nice Time",
    "MarieGold",
]

# Function to preprocess image for CNN input
def preprocess_image(image, input_shape):
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Function for CNN Model 1 prediction with shelf life logic
def process_with_cnn1(image):
    cnn1_input_shape = (150, 150, 3)  # Input shape for CNN1
    preprocessed_image = preprocess_image(image, cnn1_input_shape)
    
    # Run the image through CNN Model 1
    cnn1_output = cnn_model1.predict(preprocessed_image)
    
    # Get predicted class and map to label
    cnn1_predicted_class = np.argmax(cnn1_output, axis=1)[0]
    predicted_label = cnn1_labels[cnn1_predicted_class]
    
    # Determine shelf life based on the class name
    if "Good" in predicted_label:
        shelf_life = "Shelf life = 1 Week"
    elif "mix" in predicted_label:
        shelf_life = "Shelf life = 3 Days"
    elif "Bad" in predicted_label:
        shelf_life = "Shelf life = 0 Days"
    else:
        shelf_life = "Shelf life not available"
    
    return cnn1_predicted_class, predicted_label, shelf_life

# Function for CNN Model 2 prediction
def process_with_cnn2(image):
    cnn2_input_shape = (150, 150, 3)  # Input shape for CNN2
    preprocessed_image = preprocess_image(image, cnn2_input_shape)
    
    # Run the image through CNN Model 2
    cnn2_output = cnn_model2.predict(preprocessed_image)
    
    # Get predicted class and map to label
    cnn2_predicted_class = np.argmax(cnn2_output, axis=1)[0]
    predicted_label = cnn2_labels[cnn2_predicted_class]
    
    # Object detection part
    
    # Detect objects in the image
#     bbox, label, conf = cv.detect_common_objects(image, model='yolov3', confidence=0.5, weights_path=weights_path, config_path=config_path)

# # Draw bounding boxes
#     output_image = draw_bbox(image, bbox, label, conf)

# # Show the output
#     cv2.imshow("Detected Objects", output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
        
#     print("Number of objects detected:", len(label))
    
    return cnn2_predicted_class, predicted_label
# Function to extract text using OCR
def extract_text(image):
    def find_closest_label(extracted_text, labels):
        # Use fuzzy matching to find the closest label
        closest_match, score = process.extractOne(extracted_text, labels)
        return closest_match, score

    start = time.time()
    ocr = ImageOCR(image)

    # Extract text from image using OCR
    recognized_text = ocr.get_text(text_lang='eng', crop=1)
    print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
    # Extract and clean the text from the recognized output
    extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
    extracted_text_cleaned = extracted_text.strip()
    print(f"Extracted Text:\n{extracted_text_cleaned}")

    # Find the closest biscuit label
    closest_label, match_score = find_closest_label(extracted_text_cleaned, biscuits_labels)
    print(f"Closest Label: {closest_label}, Match Score: {match_score}")

    return closest_label, match_score

# Function to capture image from webcam
def capture_from_webcam():
    print("Capturing image from webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Show the captured frame
        cv2.imshow("Press 's' to save or 'q' to quit", frame)

        # Wait for user input: 's' to save the frame, 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the captured frame and return it
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# Main function to run user-specified functionality
def main():
    # Ask user to choose between webcam capture or loading an image
    choice = input("Choose input method: [1] Webcam, [2] Insert Image: ")

    if choice == '1':
        # Capture image from webcam
        image = capture_from_webcam()
        if image is None:
            print("Error: No image captured.")
            return
    elif choice == '2':
        # Load image from inserted file
        image_path = input("Enter the image file path: ")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image file '{image_path}' not found.")
            return
    else:
        print("Invalid choice.")
        return

    # Ask user to choose which function to run: CNN1, CNN2, or Text Extraction
    model_choice = input("Choose function to run: [1] Freshness, [2] Brand_Prediction, [3] Extract Text: ")

    if model_choice == '1':
        predicted_class, label, shelf_life = process_with_cnn1(image)
        print(f"Freshness Predicted Class: {predicted_class} ({label})")
        print(shelf_life)
    elif model_choice == '2':
        predicted_class, label = process_with_cnn2(image)
        print(f"Brand_Prediction Predicted Class: {predicted_class} ({label})")
    elif model_choice == '3':
        extracted_label, match_score = extract_text(image)
        print(f"Extracted Label: {extracted_label}, Match Score: {match_score}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()