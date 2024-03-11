from flask import Flask , render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)



# Load the pickle model 
# Open the file in read binary mode
with open("model.pkl", "rb") as file:
  #Load the model from the file using pkl.load()
  model = pickle.load(file)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods =["POST"])
def getvalue():
    age = request.form['age']
    gender = request.form['gender']
    air_population = request.form['air_pollution']
    alcohol_use = request.form['alcohol_use']
    dust_allergy = request.form['dust_allergy']
    occupational_hazards= request.form['occupational_hazards']
    genetic_risk = request.form['genetic_risk']
    chronic_Lung_Disease = request.form['chronic_Lung_Disease']
    Smoking = request.form['Smoking']
    Snoring = request.form['Snoring']
    a = np.array([[age, gender, air_population, alcohol_use, dust_allergy, occupational_hazards,genetic_risk,chronic_Lung_Disease,Smoking,Snoring]])
    a  =pd.DataFrame(a)
    a = model.predict(a)
    return render_template("final.html",a = a[0])



if __name__ == '__main__':
    app.run(debug=True)