# DeployMLmodel

Deploy ML model:
https://thecleverprogrammer.com/2020/07/14/deploy-a-machine-learning-model/#site-header


Train and Deploy a Machine Learning Model
When you train a machine learning model, also think about how you will deploy a machine learning model to serve your trained model to the available users. You will get a lot of websites who are teaching to train a machine learning model but nobody goes beyond to deploy a machine learning model. Because training and deploying a machine learning model are very different from each other. But it’s not difficult.

Training a model is the most important part in machine learning. But deploying a model is a different art because you have to think a lot in the process how you will make your machine learning application to your users. Let’s do this step by step.


 
Training a Machine Learning Model
I will train a SMS Spam detection machine learning model. I will not explain the procedure of training a machine learning model as you will get a lot of articles in this website about training model. My primary focus for today is to teach you how to deploy a machine learning model.

You can download the dataset I will use in training my machine learning model below:

DOWNLOAD
Now let’s train our model for sms spam detection:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
​
df = pd.read_csv('spam.csv', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
Image for post
Saving our Machine Learning Model
After you train your model, you must have wondered how could you use that model without training the model. So you can save the model for you future use, there is no need of training the model every time while you the model. You can save your machine learning model as follows:


 
from sklearn.externals import joblib
joblib.dump(clf, 'NB_spam_model.pkl')
And the next time you need the same model, you can load this machine learning model as follows:

NB_spam_model = open('NB_spam_model.pkl','rb')
clf = joblib.load(NB_spam_model)
Deploying a Machine Learning Model into a Web Application
After training your sms spam detection classification, it’s time to deploy our machine learning model. I will create a simple web application which will consist a web page which will look like a form where we can write or paste our sms. The web page will consist a submit button, when we will click the submit button it will use our machine learning model to classify whether the sms is spam or ham (not spam).

First, I will create a folder for this project called SMS-Message-Spam-Detector , this is the directory tree inside the folder:

spam.csv
app.py
templates/
       home.html
       result.html
static/
     style.css
Deploy a Machine Learning Model
The sub-directory templates is the directory in which Flask will look for static HTML files for rendering in the web browser, in our case, we have two html files: home.html and result.html.

app.py
The app.py file will contain the main code that will be executed by the python to run our Flask web application. It includes the Machine Learning code for classifying our sms messages as spam or ham.


 
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
​
​
app = Flask(__name__)
​
@app.route('/')
def home():
    return render_template('home.html')
​
@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
​
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)
​
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)
​
​
​
if __name__ == '__main__':
    app.run(debug=True)
​
home.html
The following are the contents of the home.html file that will create a text form where a user can enter a message:


 
<!DOCTYPE html>
<html>
<head>
    <title>Home</title>
    <!-- <link rel="stylesheet" type="text/css" href="../static/css/styles.css"> -->
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
​
    <header>
        <div class="container">
        <div id="brandname">
            Machine Learning App with Flask
        </div>
        <h2>Spam Detector For SMS Messages</h2>
        
    </div>
    </header>
​
    <div class="ml-container">
​
        <form action="{{ url_for('predict')}}" method="POST">
        <p>Enter Your Message Here</p>
        <!-- <input type="text" name="comment"/> -->
        <textarea name="message" rows="4" cols="50"></textarea>
        <br/>
​
        <input type="submit" class="btn-info" value="predict">
        
    </form>
        
    </div>
​
    
    
​
</body>
</html>
style.css
In the header section of or html file (home.html), we loaded style.css file. CSS is to determine the appearance and style of HTML documents. style.css has to be saved in a sub-directory called static, which is the default directory where Flask looks for static files such as CSS.


 
body{
    font:15px/1.5 Arial, Helvetica,sans-serif;
    padding: 0px;
    background-color:#f4f3f3;
}
​
.container{
    width:100%;
    margin: auto;
    overflow: hidden;
}
​
header{
    background:#03A9F4;#35434a;
    border-bottom:#448AFF 3px solid;
    height:120px;
    width:100%;
    padding-top:30px;
​
}
​
.main-header{
            text-align:center;
            background-color: blue;
            height:100px;
            width:100%;
            margin:0px;
        }
#brandname{
    float:left;
    font-size:30px;
    color: #fff;
    margin: 10px;
}
​
header h2{
    text-align:center;
    color:#fff;
​
}
​
​
​
.btn-info {background-color: #2196F3;
    height:40px;
    width:100px;} /* Blue */
.btn-info:hover {background: #0b7dda;}
​
​
.resultss{
    border-radius: 15px 50px;
    background: #345fe4;
    padding: 20px; 
    width: 200px;
    height: 150px;
}
​
result.html
Now, I will create a result.html file that will give us a result of our model prediction. It is the final step when we deploy a machine learning model.


 
<!DOCTYPE html>
<html>
<head>
    <title></title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
​
    <header>
        <div class="container">
        <div id="brandname">
            ML App
        </div>
        <h2>Spam Detector For SMS Messages</h2>
        
    </div>
    </header>
    <p style="color:blue;font-size:20;text-align: center;"><b>Results for Comment</b></p>
    <div class="results">
​
​
        
    {% if prediction == 1%}
    <h2 style="color:red;">Spam</h2>
    {% elif prediction == 0%}
    <h2 style="color:blue;">Not a Spam (It is a Ham)</h2>
    {% endif %}
​
    </div>
​
</body>
</html>
After following all the steps to deploy a machine learning model, now you can simply run this program using your app.py file.

You will see you output as follows:
