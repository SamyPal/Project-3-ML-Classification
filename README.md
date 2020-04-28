**Risk Prediction for Coronary Heart Disease**

This project utilizes classification algorithms in machine learning  - Logistic Regression, K-Nearest Neighbors, Naive Bayes, Decision Tree, and Random Forest ensemble technique) to evaluate the risk to getting coronary heart disease. For this, i used the teaching dataset from the Framinham Heart Disease study. 
The features used for the evaluation was: Age, Mean Blood Pressure, Total Cholesterol, Body Mass Index, total cigarettes smoked per day. EDA was performed on the data set, followed by testing the effectiveness of each vanilla model on the test set, finally grid search was used to evaluate the hyperparameters for the top two performing models. Random forest turned out to be the best model for this scenario. 

I built a flask app with javascript, that takes in patient particulars and evaluates the risk for CHD for the person, to demonstrate the functioning of the model. The app is live here: https://chdrisk-classifier-app.herokuapp.com/


```
Template Organization
---------------------
.
├── Code
│   ├── Samy_Project3_Data_PSQL.ipynb
│   ├── Samy_Project3_ModelDev.ipynb
│   └── __init__.py
├── Data
│   ├── framingham.csv
│   └── preprocessed_data.csv
├── Project3-Presentation.pdf
├── README.md
└── Results
    ├── Local_Flask_App
    │   ├── __pycache__
    │   │   └── make_prediction2.cpython-37.pyc
    │   ├── main2.py
    │   ├── make_prediction2.py
    │   ├── model_summary2.p
    │   └── templates
    │       └── index2.html
    ├── Tableau
    │   └── viz.twb
    └── Web_App_Deployment
        ├── Procfile
        ├── __pycache__
        │   ├── predictor_api.cpython-37.pyc
        │   └── predictor_app.cpython-37.pyc
        ├── predictor_api.py
        ├── predictor_app.py
        ├── requirements.txt
        ├── static
        │   └── models
        │       └── model_summary.p
        └── templates
            └── index.html
```



Enjoy!    
Samy Palaniappan
