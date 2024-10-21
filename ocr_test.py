# # import time
# # import os
# # import re
# # import glob
# # import cv2
# # from ocr import ImageOCR
# # from postprocessing import TextPostprocessing
# # import pytesseract

# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # def filename_encoder(dir):
# #     '''Reads the file number and language(alpha-3 format) of the image from the file name'''
# #     filename = os.path.basename(dir)
# #     name = os.path.splitext(filename)[0]
# #     lang = filename[0:3]
# #     number = re.findall(r'[0-9]{1,5}', name)[0]
# #     return (lang, number)

# # # Get the path to each image in the specified folder
# # filenames = glob.glob(r"C:\Users\admin\Downloads\test_dataset\Biscuits\Biscuit_Parle-G Original Gluco Biscuits\WhatsApp Image 2024-10-08 at 7.20.28 PM.jpeg")
# # filenames.sort()
# # for filename in filenames:
# #     start = time.time()

# #     language, number = filename_encoder(filename)
# #     print(f'Image № {number}, Filename language {language}')

# #     img = cv2.imread(filename)
# #     ocr = ImageOCR(img)

# #     # If the image with text contains a lot of unnecessary and needs cropping, set the parameter crop = 1
# #     # The recognition quality can be improved by setting the desired font size(set_font parameter) to which the
# #     # text will be scaled. A font that is too large can slow down the recognition speed.
# #     # The language is set in the alpha-3 format or False, if the image language needs
# #     # to be determined automatically (this will take more time).
# #     recognized_text = ocr.get_text(text_lang='eng', crop=1)
# #     print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
# #     print(f"Recognized text:\n{recognized_text}")

# #     image_with_boxes = ocr.draw_boxes(max_resolution=700)
# #     cv2.imshow("Selected text", image_with_boxes)
# #     cv2.waitKey(1000)

# #     # We iterate over the found text blocks and remove unnecessary characters
# #     for dict in recognized_text:
# #         lang = dict['lang']
# #         postprocessing = TextPostprocessing()
# #         cleared_text = postprocessing.stringFilter(input_string=dict['text'])
# #         print(f'Сlear text:\n{cleared_text}\nRecognized language: {lang}')

# #     print('-' * 30)



# import time
# import os
# import re
# import glob
# import cv2
# from ocr import ImageOCR
# from postprocessing import TextPostprocessing
# import pytesseract
# from fuzzywuzzy import process

# # Set the path for Tesseract OCR executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Sample labels of biscuit brands
# biscuit_labels = biscuit_labels = [
#     "Sunfeast Dark Fantasy",
#     "Bourbon",
#     "Good Day",
#     "Parle-G",
#     "Hide & Seek",
#     "Milk Bikis",
#     "Treat",
#     "Nice Time",
#     "MarieGold",
# ]

# def filename_encoder(dir):
#     '''Reads the file number and language(alpha-3 format) of the image from the file name'''
#     filename = os.path.basename(dir)
#     name = os.path.splitext(filename)[0]
#     lang = filename[0:3]
#     number = re.findall(r'[0-9]{1,5}', name)
    
#     if number:  # Check if any number was found
#         return (lang, number[0])
#     else:
#         return (lang, None)

# def extract_text_from_image(img):
#     # Perform OCR to extract text
#     extracted_text = pytesseract.image_to_string(img, lang='eng')
#     return extracted_text

# def find_closest_label(extracted_text, labels):
#     # Use fuzzy matching to find the closest label
#     closest_match, score = process.extractOne(extracted_text, labels)
#     return closest_match, score

# # Get the path to each image in the specified folder
# filenames = glob.glob(r"C:\Users\HP\OneDrive\Desktop\WhatsApp Image 2024-10-20 at 23.56.40_55dc1ec8.jpg")
# # filenames = img_path
# filenames.sort()

# for filename in filenames:
#     start = time.time()

#     language, number = filename_encoder(filename)
#     print(f'Image № {number}, Filename language {language}')

#     img = cv2.imread(filename)
#     ocr = ImageOCR(img)

#     # Extract text from image using OCR
#     recognized_text = ocr.get_text(text_lang='eng', crop=1)
#     print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
#     # Extract and clean the text from the recognized output
#     extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
#     extracted_text_cleaned = extracted_text.strip()
#     print(f"Extracted Text:\n{extracted_text_cleaned}")

#     # Find the closest biscuit label
#     closest_label, match_score = find_closest_label(extracted_text_cleaned, biscuit_labels)
#     print(f"Closest Label: {closest_label}, Match Score: {match_score}")

#     # Show image with boxes (if required)
#     image_with_boxes = ocr.draw_boxes(max_resolution=700)
#     cv2.imshow("Selected text", image_with_boxes)
#     cv2.waitKey(10000)

#     # We iterate over the found text blocks and remove unnecessary characters
#     postprocessing = TextPostprocessing()
#     for dict in recognized_text:
#         lang = dict['lang']
#         cleared_text = postprocessing.stringFilter(input_string=dict['text'])
#         print(f'Clear text:\n{cleared_text}\nRecognized language: {lang}')

#     print('-' * 30)

# cv2.destroyAllWindows()  # Close all OpenCV windows at the end

# def process_image(filename):
#     img = cv2.imread(filename)
    
#     # Extract text from image using OCR
#     extracted_text, _ = extract_text_from_image(img)
#     # global img_path
#     # img_path= filename
#     # Find the closest biscuit label
#     closest_label, match_score = find_closest_label(extracted_text)

#     return closest_label, match_score

import time
import os
import re
import glob
import cv2
from ocr import ImageOCR
from postprocessing import TextPostprocessing
import pytesseract
from fuzzywuzzy import process

# Set the path for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Sample labels of biscuit brands
biscuit_labels = [
    "Sunfeast Dark Fantasy",
    "Bourbon",
    "Good Day",
    "Parle-G",
    "Hide & Seek",
    "Milk Bikis",
    "Treat",
    "Nice Time",
    "Marie Gold"
]

def filename_encoder(dir):
    '''Reads the file number and language(alpha-3 format) of the image from the file name'''
    filename = os.path.basename(dir)
    name = os.path.splitext(filename)[0]
    lang = filename[0:3]
    number = re.findall(r'[0-9]{1,5}', name)
    
    if number:  # Check if any number was found
        return (lang, number[0])
    else:
        return (lang, None)

def extract_text_from_image(img):
    # Perform OCR to extract text
    extracted_text = pytesseract.image_to_string(img, lang='eng')
    return extracted_text

def find_closest_label(extracted_text, labels):
    # Use fuzzy matching to find the closest label
    closest_match, score = process.extractOne(extracted_text, labels)
    return closest_match, score

# Get the path to each image in the specified folder
filenames = glob.glob(r"C:\Users\HP\Downloads\images.jpg")
# filenames = img_path
filenames.sort()

for filename in filenames:
    start = time.time()

    language, number = filename_encoder(filename)
    print(f'Image № {number}, Filename language {language}')

    img = cv2.imread(filename)
    ocr = ImageOCR(img)

    # Extract text from image using OCR
    recognized_text = ocr.get_text(text_lang='eng', crop=1)
    print("[TEXT EXTRACTOR] Time [{:.6f}] sec".format(time.time() - start))
    
    # Extract and clean the text from the recognized output
    extracted_text = "\n".join([dict['text'] for dict in recognized_text if 'text' in dict])
    extracted_text_cleaned = extracted_text.strip()
    print(f"Extracted Text:\n{extracted_text_cleaned}")

    # Find the closest biscuit label
    closest_label, match_score = find_closest_label(extracted_text_cleaned, biscuit_labels)
    print(f"Closest Label: {closest_label}, Match Score: {match_score}")

    # Show image with boxes (if required)
    image_with_boxes = ocr.draw_boxes(max_resolution=700)
    cv2.imshow("Selected text", image_with_boxes)
    cv2.waitKey(10000)

    # We iterate over the found text blocks and remove unnecessary characters
    postprocessing = TextPostprocessing()
    for dict in recognized_text:
        lang = dict['lang']
        cleared_text = postprocessing.stringFilter(input_string=dict['text'])
        print(f'Clear text:\n{cleared_text}\nRecognized language: {lang}')

    print('-' * 30)

cv2.destroyAllWindows()  # Close all OpenCV windows at the end

def process_image(filename):
    img = cv2.imread(filename)
    
    # Extract text from image using OCR
    extracted_text, _ = extract_text_from_image(img)
    # global img_path
    # img_path= filename
    # Find the closest biscuit label
    closest_label, match_score = find_closest_label(extracted_text)

    return closest_label,match_score