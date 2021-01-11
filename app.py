from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np

pipe=pickle.load(open("E:\Python\Python\Projects\Magic Brick\MagicBrick.pkl","rb"))
df=pd.read_csv('E:\Python\Python\Projects\Magic Brick\Magic_Brick.csv')
app=Flask(__name__)


@app.route('/')
def index():
    locations=sorted(df['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    
    sqft=float(request.form.get('area'))
    bhk=float(request.form.get('bhk'))
    bath=float(request.form.get('bathroom'))
    locations=request.form.get('location')
    print(locations,bhk,bath,sqft)
    
    input=pd.DataFrame([[locations,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    pred=pipe.predict(input)[0]* 100000
    return str(np.round(pred,2))

if __name__=='__main__':
    app.run(debug=True)
    