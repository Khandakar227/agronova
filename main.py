import pickle
import cv2
from ultralytics import YOLOv10 as YOLO
import pandas as pd

# Paths for models
insect_detection_model_path = 'weights/insect.50.pt'
weed_crop_model_path = 'weights/weedCrop.140.pt'
crop_yield_model_path = 'weights/BRCropYield.pkl'
crop_prediction_model_path = 'weights/cropPredictionRF.pkl'

# Mapping for area and item
area_mapping = {'Albania': 0, 'Algeria': 1, 'Angola': 2, 'Argentina': 3, 'Armenia': 4, 'Australia': 5, 'Austria': 6, 'Azerbaijan': 7, 'Bahamas': 8, 'Bahrain': 9, 'Bangladesh': 10, 'Belarus': 11, 'Belgium': 12, 'Botswana': 13, 'Brazil': 14, 'Bulgaria': 15, 'Burkina Faso': 16, 'Burundi': 17, 'Cameroon': 18, 'Canada': 19, 'Central African Republic': 20, 'Chile': 21, 'Colombia': 22, 'Croatia': 23, 'Denmark': 24, 'Dominican Republic': 25, 'Ecuador': 26, 'Egypt': 27, 'El Salvador': 28, 'Eritrea': 29, 'Estonia': 30, 'Finland': 31, 'France': 32, 'Germany': 33, 'Ghana': 34, 'Greece': 35, 'Guatemala': 36, 'Guinea': 37, 'Guyana': 38, 'Haiti': 39, 'Honduras': 40, 'Hungary': 41, 'India': 42, 'Indonesia': 43, 'Iraq': 44, 'Ireland': 45, 'Italy': 46, 'Jamaica': 47, 'Japan': 48, 'Kazakhstan': 49, 'Kenya': 50, 'Latvia': 51, 'Lebanon': 52, 'Lesotho': 53, 'Libya': 54, 'Lithuania': 55, 'Madagascar': 56, 'Malawi': 57, 'Malaysia': 58, 'Mali': 59, 'Mauritania': 60, 'Mauritius': 61, 'Mexico': 62, 'Montenegro': 63, 'Morocco': 64, 'Mozambique': 65, 'Namibia': 66, 'Nepal': 67, 'Netherlands': 68, 'New Zealand': 69, 'Nicaragua': 70, 'Niger': 71, 'Norway': 72, 'Pakistan': 73, 'Papua New Guinea': 74, 'Peru': 75, 'Poland': 76, 'Portugal': 77, 'Qatar': 78, 'Romania': 79, 'Rwanda': 80, 'Saudi Arabia': 81, 'Senegal': 82, 'Slovenia': 83, 'South Africa': 84, 'Spain': 85, 'Sri Lanka': 86, 'Sudan': 87, 'Suriname': 88, 'Sweden': 89, 'Switzerland': 90, 'Tajikistan': 91, 'Thailand': 92, 'Tunisia': 93, 'Turkey': 94, 'Uganda': 95, 'Ukraine': 96, 'United Kingdom': 97, 'Uruguay': 98, 'Zambia': 99, 'Zimbabwe': 100}
item_mapping = {'Cassava': 0, 'Maize': 1, 'Plantains and others': 2, 'Potatoes': 3, 'Rice, paddy': 4, 'Sorghum': 5, 'Soybeans': 6, 'Sweet potatoes': 7, 'Wheat': 8, 'Yams': 9}

# Load models
insect_detection_model = YOLO(insect_detection_model_path)
weed_crop_model = YOLO(weed_crop_model_path)
crop_yield_model = pickle.load(open(crop_yield_model_path, 'rb'))
crop_prediction_model = pickle.load(open(crop_prediction_model_path, 'rb'))

def detect_insect(image_path):
    results = insect_detection_model.predict(image_path)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (640, 480))
    return annotated_frame

def detect_weed_crop(image_path):
    results = weed_crop_model.predict(image_path)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (640, 480))
    return annotated_frame

def recommend_crop(N,P,K,temp, hum, ph, rainfall):
    x = pd.DataFrame([[N,P,K,temp, hum, ph, rainfall]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = crop_prediction_model.predict(x)
    return prediction

def predict_yield(country, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp_celsius):
    x = pd.DataFrame([[area_mapping[country], item_mapping[item], year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp_celsius]], columns=['Area', 'Item', 'Year','average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
    prediction = crop_yield_model.predict(x)
    return prediction


# Test on the images
insect_image_path = ['images/colorado-beetle-eats-a-potato-leaves-young-pests-destroy-a-crop-in-the-field-parasites.jpg', 'images/1073dbb7e33a2bca70ce4286c2ac6c1d.jpg', 'images/insect.jpg',]
weed_crop_image_path = ['images/weed-1.jpg','images/weed-2.jpg', 'images/weed-3.jpg']
    
weed_crop_annotated_frame = detect_weed_crop(weed_crop_image_path[0])
cv2.imshow('Weed Crop Detection', weed_crop_annotated_frame)

insect_annotated_frame = detect_insect(insect_image_path[0])
cv2.imshow('Insect Detection', insect_annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

recommended_crops = recommend_crop(N=50, P=50, K=30, temp=32, hum=80, ph=6.5, rainfall=100)
print('\n\nRecommended Crops:', ", ".join(recommended_crops))

predicted_yield = predict_yield('Bangladesh', 'Plantains and others', 2024, 1083.0, 75000.0, 32)
print('\nPredicted Yield:', ", ".join(map( "{:.2f}".format, predicted_yield)), 'Hectogram per hectare\n')
