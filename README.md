**Risk Prediction for Coronary Heart Disease**

This project utilizes classification algorithms in machine learning  - Logistic Regression, K-Nearest Neighbors, Naive Bayes, Decision Tree, and Random Forest ensemble technique) to evaluate the risk to getting coronary heart disease. For this, i used the teaching dataset from the Framinham Heart Disease study. 
The features used for the evaluation was: Age, Mean Blood Pressure, Total Cholesterol, Body Mass Index, total cigarettes smoked per day. EDA was performed on the data set, followed by testing the effectiveness of each vanilla model on the test set, finally grid search was used to evaluate the hyperparameters for the top two performing models. Random forest turned out to be the best model for this scenario. 

I built a flask app with javascript, that takes in patient particulars and evaluates the risk for CHD for the person, to demonstrate the functioning of the model.


```
Template Organization
---------------------
.
├── README.md
└── template_project
    ├── LICENSE.txt
    ├── README.txt
    ├── code
    │   ├── __init__.py
    │   └── hello_script.py
    ├── data
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   └── raw
    │       └── example_data.txt
    ├── docs
    ├── example_notebook.ipynb
    ├── references
    └── reports
        └── figures
```



Enjoy!    
Samy Palaniappan
