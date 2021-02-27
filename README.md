# Abalone Age Prediction With Deployment

### Goal
The goal of this project is to generate a ANN model to predict an Abalone's age AND then deploy it to server so we can make predictions on single example.

### Data Description
* Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. 

### Modeling
* A simple Artificial Neural Network model is used to make predictions. 

### Deployment
* A Flask app/script is created. After training finished, the model and scaler are saved and then used in this Flask app.
* You can reach the web link from here: [](https://abalone-prediction-deployment.herokuapp.com/)
* First enter the pysical measures of abalone and then press "Predict" button, my model will calculate it's prediction and will display it in a new page.

### Files
* abalone.csv - Dataset
* abalone.ipynb - Project Notebook
* abalone_model.h5 - Saved model
* abalone_scaler.pkl - Saved scaler
* app.py - Flask app to deploy this into web
* Procfile and requirements - files need for Heroku deployment

### Libraries Used
* pandas, numpy
* matplotlib, seaborn
* sklearn
* tensorflow
* Flask
