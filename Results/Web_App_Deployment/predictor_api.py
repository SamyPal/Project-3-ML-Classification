import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# read in the model
model_summary = pickle.load(open("static/models/model_summary.p","rb"))
my_model = model_summary['model']
std_scaler = model_summary['scaler']
# create a function to take in user-entered amounts and apply the model
def cvd_or_not(amounts_float):
    '''
    function that takes arguments
    amounts_float as input from user
    '''
    cont_var = []
    trans_var = []
    for i in range(len(amounts_float)-2):
        cont_var.append(amounts_float[i])
    trans_var = std_scaler.transform([cont_var])
    trans_varb = np.append(trans_var, [amounts_float[len(amounts_float)-2], amounts_float[len(amounts_float)-1]])
    input_df = [trans_varb]
    # make a prediction
    prediction = my_model.predict(input_df)[0]

    # return a message
    message_array = ["Congrats! No Risk for CHD.",
                     "Please, Check with a Doctor."]

    return message_array[prediction]
