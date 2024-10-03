import streamlit as st
import pandas as pd
import numpy as np
# import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Loading in the Model to predict on the data
# pickle_in = open('rf_model.pkl','rb')
# rf_model = pickle.load(pickle_in)


def welcome():
    return 'welcome all'

# defining the function which will make the prediction using the data which the user inputs
def prediction(temperature,exhaust_vacuum,amb_pressure,r_humidity):

    data=pd.read_csv("C:\\Users\\jalli\\Downloads\\Copy of energy_production (1).csv",sep=';')
    data_cleaned = data.drop_duplicates()
    # data_cleaned.to_csv('cleaned_file_no_duplicates.csv', index=False)
    # print(data_cleaned.head())


    # Function to calculate the lower and upper bounds for outliers
    def calculate_bounds(data_column):
        Q1 = np.percentile(data_column, 25)
        Q3 = np.percentile(data_column, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    # Remove outliers from each column
    for column in data.columns:
        lower_bound, upper_bound = calculate_bounds(data[column])
        # Filter the data to remove outliers
        data_cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    

    scaler = StandardScaler()

    # Apply scaling to all the features
    features = ['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']
    data_scaled = data_cleaned.copy()
    data_scaled[features] = scaler.fit_transform(data_cleaned[features])


    X = data_cleaned[['exhaust_vacuum', 'amb_pressure', 'r_humidity']]
    y = data_cleaned['energy_production']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    # y_pred_rf = rf_model.predict(X_test)
    prediction = rf_model.predict(
        [[temperature,exhaust_vacuum,amb_pressure,r_humidity]])
    # print(prediction)
    return prediction


# this is the main function in which we define our webpage 
def main():
    # giving the webpage a title
    st.title("ENERGY PREDICTOR")
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:tomato;padding:10px">
    <h2 style ="color:black;text-align:center;">Streamlit Energy Predictor ML App </h1>
    </div>
    """
    
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    temperature = st.text_input("Temperature", "Type Here")
    exhaust_vacuum=st.text_input("exhaust_vacuum", "Type Here")
    amb_pressure = st.text_input("amb_pressure", "Type Here")
    r_humidity= st.text_input("r_humidity", "Type Here")

    result =""

    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(temperature,exhaust_vacuum,amb_pressure,r_humidity)
    st.success('Energy Production is {}'.format(result))

if __name__=='__main__':
    main()