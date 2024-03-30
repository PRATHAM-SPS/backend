from __future__ import print_function
import json
from flask_cors import CORS
from fpdf import FPDF
from flask import Flask, request, render_template , jsonify
from markupsafe import Markup
import numpy as np
import pickle
import pandas as pd
from disease import disease_dic
from fertilizer import fertilizer_dic
import requests
import io
from crop_predict import Crop_Predict
import os
import openai
import datetime 
import google.generativeai as genai
from datetime import date

import pymongo

from dotenv import load_dotenv
import os

load_dotenv()


# fruit disease prediction

import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import speech_recognition as sr

openai.api_key = "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" #os.getenv("OPENAI_API_KEY_IMAGE")
geminikey = "AIzaSyAwYtdZDx-tP7EAjyM6hlXCWOo2nRQ5dTQ"
genai.configure(api_key=geminikey)  

# client = pymongo.MongoClient("mongodb+srv://pratham:kheti1234@cluster0.saiervy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# test_db = client["test"]

app = Flask(__name__)
CORS(app)



# prediction function
def CropPredictor(to_predict_list):
    to_predict = np.array([to_predict_list])
    loaded_model = pickle.load(open("models/RandomForest.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


def FertilizerPredictor(to_predict_list):
    to_predict = np.array([to_predict_list])
    loaded_model = pickle.load(open("models/classifier.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    print(result)
    return result[0]


def WeatherPredictor(to_predict_list):
    to_predict = np.array([to_predict_list])
    loaded_model = pickle.load(open("models/weather.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]




# routing
@app.route("/", methods=["GET"])
def home():
    return "server started..."

@app.route('/hello')
def hello_world():
	return 'Hello World!'


# @app.route("/crop-predict2", methods=["POST"])
# def result():
#     if request.method == "POST":
#         print(request.json)
#         to_predict_list = request.json
#         to_predict_list = list(to_predict_list.values())
#         to_predict_list = list(map(int, to_predict_list))
#         result = CropPredictor(to_predict_list)
#         return result


@app.route("/crop-predict", methods=["POST"])
def crop():
    model = Crop_Predict()
    if request.method == "POST":
        crop_name = model.crop()
        if crop_name == "noData":
            return -1

        return {
            "crop_name": crop_name,
            "no_of_crops": len(crop_name),
        }
@app.route("/crop-predict",methods=["GET"])
def get_crops():
    results = []
    # for dat in data:
    #     i = test_db["crop_recommendation"].count_documents({"output":{"$in":[dat]}})
    #     results.append(i if i is not None else 0)
    # dat = data.to_dict()
    return jsonify({
        "labels":data.to_list(),
        "datasets":results,
        "xlabel":"Crops",
        "ylabel":"counts"
    })


@app.route("/fertilizer-predict", methods=["POST"])
def result2():
    if request.method == "POST":
        print(request.json)
        to_predict = request.json
        location = request.json["location"]
        del to_predict["location"]

        to_predict_list = list(to_predict.values())

        # Use the OpenWeatherMap API to get the weather forecast for the next 15 days
        api_key = "2cbb6bbbc606b90a6e61379b59074598"
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}"
        response = requests.get(url)
        weather_data = response.json()

        print((float(weather_data["list"][0]["main"]["temp"]) - 273.15))
        Temp = float(weather_data["list"][0]["main"]["temp"]) - 273.15
        Hum = weather_data["list"][0]["main"]["humidity"]
        to_predict_list.append(Temp)
        to_predict_list.append(Hum)
        print(to_predict_list)

        to_predict_list = list(map(int, to_predict_list))

        ans = FertilizerPredictor(to_predict_list)

        fertilizer_info = {"name": "", "img": ""}
        if ans == 0:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Compost"})
            # response = openai.Image.create(
            #     prompt="compost from food scraps, yard waste",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "Compost",
                # "img": response["data"][0]["url"],
                "how_to_use": "Compost is easy to make at home using food scraps, yard waste, and other organic materials. You can also purchase compost at garden centers and nurseries. To use compost as a fertilizer, simply mix it into the soil before planting or use it as a top dressing around established plants. \nThat being said, it's always a good idea to do a soil test to determine the specific nutrient needs of your plants and soil. This can help you choose the right organic fertilizer and ensure that your plants are getting the nutrients they need to grow and thrive.",
                
                "buylink" : "https://www.bombaygreens.com/products/organic-manure-fertilizer-for-vegetable-plants-with-3-4-4-npk-900-grams",
                "image" : "https://www.kribhco.net/assets/img/product/compost.jpg"
            }
        elif ans == 1:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer"})

            # response = openai.Image.create(
            #     prompt="Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer",
                # "img": response["data"][0]["url"],
                "how_to_use": "Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer organic components: Fish bone meal | Alfalfa meal | Feather meal | Soft rock phosphate | Mined potassium sulfate | Seaweed extract | Beneficial soil microbes",
                "buylink" : "https://www.amazon.in/Earth-Organic-Tomato-Vegetable-Fertilizer/dp/B000VZRV4C",
                "image" : "https://m.media-amazon.com/images/I/718ps0sVhEL._SX679_.jpg"
            }
        elif ans == 2:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Dr. Earth All Purpose Fertilizer"})

            # response = openai.Image.create(
            #     prompt="Dr. Earth All Purpose Fertilizer",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "Dr. Earth All Purpose Fertilizer",
                # "img": response["data"][0]["url"],
                "how_to_use": "Dr. Earth All Purpose Fertilizer is applied by sprinkling it evenly over prepared soil, then incorporating it gently with a rake or garden tool. Water the area thoroughly afterward. It helps plants by providing essential nutrients, improving soil health, reducing erosion, and enhancing crop yield and quality.",
                "buylink": "https://www.amazon.in/Dr-Earth-736P-Organic-Fertilizer/dp/B001F9SGKG",
                "image" : "https://m.media-amazon.com/images/I/815h9b8FY-S.jpg"
            }
        elif ans == 3:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Jobe's Organics All-Purpose Fertilizer"})

            # response = openai.Image.create(
            #     prompt="Jobe's Organics All-Purpose Fertilizer",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "Jobe's Organics All-Purpose Fertilizer",
                # "img": response["data"][0]["url"],
                "how_to_use": "Jobe's Organics All-Purpose Fertilizer organic composition: Feather meal | Bone meal | Sulfate of potash | Kelp meal | Alfalfa meal | Humic acid",
                "buylink" : "https://www.amazon.in/Jobes-Organics-Purpose-Fertilizer-Granular/dp/B003SHEDQY",
                "image" : "https://m.media-amazon.com/images/I/41ruFc3dXLL._SY445_SX342_QL70_FMwebp_.jpg"
            }
        elif ans == 4:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Dr. Earth Organic Nitrogen Fertilizer"})

            # response = openai.Image.create(
            #     prompt="Dr. Earth Organic Nitrogen Fertilizer",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "Dr. Earth Organic Nitrogen Fertilizer",
                # "img": response["data"][0]["url"],
                "how_to_use": "Dr. Earth Organic Nitrogen Fertilizer organic composition: Soybean meal | Alfalfa meal | Fishbone meal | Feather meal | Seabird guano | Blood meal | Kelp meal | Potassium sulfate | Humic acid",
                "buylink" : "https://raywiegandsnursery.com/products/dr-earth%C2%AE-organic-and-natural-nitrogen-all-purpose-fertilizer",
                "image" : "https://m.media-amazon.com/images/I/81pCVjmGxkL.__AC_SX300_SY300_QL70_FMwebp_.jpg"
            }
        elif ans == 5:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Espoma Organic Lawn Food"})

            # response = openai.Image.create(
            #     prompt="Espoma Organic Lawn Food",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "Espoma Organic Lawn Food",
                #"img": response["data"][0]["url"],
                "how_to_use": "Espoma Organic Lawn Food organic composition: Corn gluten meal | Feather meal | Soybean meal | Potassium sulfate | Humates | Iron",
                "buylink" : "https://www.espoma.com/product-lines/organic-lawn-fertilizers-2/",
                "image": "https://www.espoma.com/wp-content/uploads/2015/03/espoma_organic-lightning-lime.png"
            }
        else:
            # test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"FoxFarm"})

            # response = openai.Image.create(
            #     prompt="FoxFarm",
            #     n=1,
            #     size="256x256",
            # )
            return {
                "name": "FoxFarm",
                # "img": response["data"][0]["url"],
                "how_to_use": "FoxFarm organic composition: Earthworm castings | Bat guano | Fish meal | Bone meal | Blood meal | Feather meal | Kelp meal",
                "buylink" : "https://www.ubuy.co.in/brand/foxfarm",
                "image" : "https://foxfarm.com/wp-content/uploads/2019/02/fforiginalplantingmix2019.png"            
            }

@app.route("/fertilizer-predict", methods=["GET"])
def get_fertilizer_predict():
    # data = test_db["fertilizer_recommendation"].find()
    fertilizers = ["Compost","Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer","Dr. Earth All Purpose Fertilizer","Jobe's Organics All-Purpose Fertilizer","Dr. Earth Organic Nitrogen Fertilizer","Espoma Organic Lawn Food","FoxFarm"]
    results = []
    for fertilizer in fertilizers:
        # i = test_db["fertilizer_recommendation"].count_documents({"output":fertilizer})
        results.append(i if i is not None else 0)
    result = {
        "labels":fertilizers,
        "datasets":results,
        "xlabel":"Fertilizers",
        "ylabel":"count"
    }
    result = jsonify(result)
    return result


@app.route("/weather-predict", methods=["POST"])
def result3():
    if request.method == "POST":
        # print(request.json)
        to_predict_list = request.json
        # to_predict_list = list(to_predict_list.values())
        to_predict_list = list(to_predict_list.values())
        weather = WeatherPredictor(to_predict_list)
        result = {"data": weather}
        return result


@app.route("/disease-predict", methods=["POST"])
def disease_prediction():
    title = " Disease Detection"

    if request.method == "POST":
        print(request.files)
        if "file" not in request.files:
            return "file not found"
        file = request.files.get("file")
        if not file:
            return "plz send image"
        try:
            img = file.read()
            print(file)

            prediction = DiseasesPredictor(img)
            print(prediction)

            prediction = Markup(str(disease_dic[prediction]))
            return {"prediction": prediction, "title": title}
        except:
            pass
    return render_template("disease.html", title=title)


@app.route("/disease-predict", methods=["GET"])
def get_disease_predict():
    # data = test_db["fertilizer_recommendation"].find()
    results = []
    for disease in range(0,38):
        try:
            i = test_db["disease_predict"].count_documents({"disease":disease})
        except:
            pass
        results.append(i if i is not None else 0)
    result = {
        "labels":disease_classes,
        "datasets":results,
        "xlabel":"Diseases",
        "ylabel":"count"
    }
    result = jsonify(result)
    return result

@app.route("/disease-predict2", methods=[ "POST"])
def submit():
    if request.method == "POST":
        image = request.files["file"]
        filename = image.filename
        # file_path = os.path.join("static/uploads", filename)
        # image.save(file_path)
        # print(file_path)
        pred = prediction(image)
        print(pred)
        title = disease_info["disease_name"][pred]
        description = disease_info["description"][pred]
        prevent = disease_info["Possible Steps"][pred]
        image_url = disease_info["image_url"][pred]
        supplement_name = supplement_info["supplement name"][pred]
        supplement_image_url = supplement_info["supplement image"][pred]
        supplement_buy_link = supplement_info["buy link"][pred]
        print(pred)
        pf = f"how to use {supplement_name}"
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content(pf)
        print(response.text)
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        # instructions = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=f"how to use {supplement_name}",
        #     max_tokens=200,
        #     temperature=0,
        # )
        # print(instructions)
        # instructions = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=f"Disease {title} description {description} supplement name {supplement_name} prevention methods {prevent}",
        #     max_tokens=200,
        # )
        # print(instructions)
        # instructions = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=f"Tell me more about this",
        #     max_tokens=200,
        # )
        # print(instructions)
        print(pred)
        # crop , disease = title.split(":")
        # test_db["disease_predict"].insert_one({"disease": int(pred)})

        return {
            "title": title,
            "desc": description,
            "prevent": prevent,
            # "image_url": image_url,
            # "pred": pred,
            "sname": supplement_name,
            "simage": supplement_image_url,
            "buy_link": supplement_buy_link,
            "how_to_use": (response.text),
        }

@app.route("/more-info",methods = ["GET"])
def get_info():
    prev_in = request.args.get("prev_in")
    user_in = request.args.get("user_in") if request.args.get("user_in")!=None or str(request.args.get("user_in"))!=""  else "describe in more detail following steps"
    print(prev_in,user_in)
    pf=str(user_in) + "in context of" + str(prev_in),
    model = genai.GenerativeModel('gemini-1.0-pro')
    response = model.generate_content(pf)
    print(response.text)
    
    
    
    
    print(response.text)
    return jsonify({
        "instructions":response.text
    })    


@app.route("/price-predict", methods=["POST"])
def result4():
    if request.method == "POST":
        print(request.json)
        to_predict_list = request.json
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = CropPredictor(to_predict_list)
        return result


@app.route("/forecast", methods=["POST"])
def forecast():
    today = date.today()
    # Get the user's location from the form
    location = request.json["location"]

    # Use the OpenWeatherMap API to get the weather forecast for the next 15 days
    # api_key = os.getenv("OPEN_WEATHER_API_KEY")
    api_key = "8fd236fa6ea28af581da0cd3705c9d67"
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}"
        
    response = requests.get(url)
    weather_data = response.json()

    forecast=[]

    for reading in weather_data["list"]:
        if reading["dt_txt"].endswith("12:00:00"):
            forecast.append({
                "date": reading["dt_txt"][:10],
                "weather": reading["weather"][0]["description"],
                "temperature": reading["main"]["temp"],
                "humidity": reading["main"]["humidity"],
                "wind_speed": reading["wind"]["speed"],
                "rain": reading["weather"][0]['description'].title()
        })

    print(forecast[0]['rain'])

    month = datetime.datetime.now().month
    hemisphere = "north"

    # Determine the season based on the month and hemisphere
    if (month >= 3 and month <= 6) and hemisphere == "north":
        climate = "summer"
    elif (month >= 7 and month <= 10) and hemisphere == "north":
        climate = "rainy"
    elif (
        month == 11 or month == 12 or month == 1 or month == 2
    ) and hemisphere == "north":
        climate = "winter"

    temperature = forecast[0]["temperature"]
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    #GEMINI-PRO-WEATHER

    prompt=f" give me in short agricultural conditions based on {temperature} kelvin and {climate} climate in 100 words"
    model = genai.GenerativeModel('gemini-1.0-pro')

    response = model.generate_content(prompt)
    analysis = response.text
    # max_words = 100 
    # words = analysis.split()
    # truncated_analysis = ' '.join(words[:max_words])
    forecast = json.dumps(forecast)
    # Return the forecast to the user
    return [forecast, analysis]


@app.route("/getnews", methods=["GET"])
def getnews():
    api_key = "5e1392e4a78241adbf27393420e62ec2"
    base_url = "https://newsapi.org/v2/everything?"
    query = "agriculture+India"
    sources = "bbc-news,the-hindu,the-times-of-india,ndtv"
    language = "en"
    sortBy = "relevancy"
    pageSize = 100

    complete_url = f"{base_url}q={query}&sources={sources}&language={language}&sortBy={sortBy}&pageSize={pageSize}&apiKey={api_key}"

    response = requests.get(complete_url)
    news_data = response.json()
    articles = news_data.get("articles")

    return articles

@app.route('/voice_input', methods=['POST'])
def process_voice_input():
    try:
        voice_file = request.files['voice']
        recognizer = sr.Recognizer()

        with sr.AudioFile(voice_file) as source:
            audio_data = recognizer.record(source)

        user_input = recognizer.recognize_google(audio_data)
        print("hsgygdsyfgydgfygdsyf")
        # Now, you can use the user_input as needed in your application logic

        return jsonify({"user_input": user_input})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_message = request.json.get('userMessage', '')
    print(user_message)
    
    # Use Gem AI to generate a response
    pf = f'''you are a chatbot to help a farmer with his queries within 50 words.
            

            Sentence = {user_message}'''
    model = genai.GenerativeModel('gemini-1.0-pro')
    response = model.generate_content(pf)
    generated_text = response.text
    print(generated_text)

    items = generated_text.split(',')

    
    # Return the generated response from Gem AI
    return jsonify({'botResponse': generated_text})



if __name__ == "__main__":
    app.run(port=5000 , debug=True)
