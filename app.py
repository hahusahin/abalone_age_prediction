from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
from tensorflow.keras.models import load_model
import joblib

def return_prediction(model, scaler, sample_abalone):
    abalone = [[x for x in sample_abalone.values()]]
    abalone = scaler.transform(abalone)
    prediction = model.predict(abalone)[0][0]
    return prediction

app = Flask(__name__)

# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'abalone'

# LOAD THE MODEL AND THE SCALER
abalone_model = load_model("abalone_model.h5")
abalone_scaler = joblib.load("abalone_scaler.pkl")

# Create a WTForm Class
class AbaloneForm(FlaskForm):
    length = TextField('Length')
    diameter = TextField('Diameter')
    height = TextField('Height')
    whole_wgt = TextField('Whole weight')
    shucked_wgt = TextField('Shucked weight')
    viscera_wgt = TextField('Viscera weight')
    shell_wgt = TextField('Shell weight')

    submit = SubmitField('Predict')


@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = AbaloneForm()
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data from the breed on the form.
        session['length'] = form.length.data
        session['diameter'] = form.diameter.data
        session['height'] = form.height.data
        session['whole_wgt'] = form.whole_wgt.data
        session['shucked_wgt'] = form.shucked_wgt.data
        session['viscera_wgt'] = form.viscera_wgt.data
        session['shell_wgt'] = form.shell_wgt.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['length'] = float(session['length'])
    content['diameter'] = float(session['diameter'])
    content['height'] = float(session['height'])
    content['whole_wgt'] = float(session['whole_wgt'])
    content['shucked_wgt'] = float(session['shucked_wgt'])
    content['viscera_wgt'] = float(session['viscera_wgt'])
    content['shell_wgt'] = float(session['shell_wgt'])

    results = return_prediction(model=abalone_model,scaler=abalone_scaler,sample_abalone=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)