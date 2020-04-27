import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# read in the model
model_summary = pickle.load(open("model_summary2.p","rb"))
my_model = model_summary['model']
std_scaler = model_summary['scaler']
# create a function to take in user-entered amounts and apply the model
def cvd_or_not(amounts_float, model=my_model, std=std_scaler):
    cont_var = []
    trans_var = []
    for i in range(len(amounts_float)-2):
        cont_var.append(amounts_float[i])
    trans_var = std.transform([cont_var])
    trans_varb = np.append(trans_var, [amounts_float[len(amounts_float)-2], amounts_float[len(amounts_float)-1]])
    # # put everything in terms of tablespoons
    # # flour, milk, sugar, butter, eggs, baking powder, vanilla, salt
    # multipliers = [16, 16, 16, 16, 3, .33, .33, .33]
    #
    # # sum up the total values to get the total number of tablespoons in the batter
    # total = np.dot(multipliers, amounts_float)
    #
    # # note the proportion of flour and sugar
    # flour_cups_prop = multipliers[0] * amounts_float[0] * 100.0 / total
    # sugar_cups_prop = multipliers[2] * amounts_float[2] * 100.0 / total
    #
    # # inputs into the model
    # input_df = [[flour_cups_prop, sugar_cups_prop]]
    input_df = [trans_varb]
    # make a prediction
    prediction = my_model.predict(input_df)[0]

    # return a message
    message_array = ["Congrats! No Risk for CVD",
                     "Check with a Doctor"]

    return message_array[prediction]
