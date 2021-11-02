import time
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import hydralit_components as hc
import hydralit_components as hc


from PIL import Image
from cryptocmd import CmcScraper
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,Dropout




main_bg = "bg3.png"
main_bg_ext = "png"


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)


selected_ticker = st.sidebar.selectbox("Choose type of Crypto (i.e. BTC, ETH, BNB, XRP)",options=["BTC", "ETH", "BNB", "XRP"] )


# INITIALIZE SCRAPER
@st.cache
def load_data(selected_ticker):
    
    init_scraper = CmcScraper(selected_ticker)
    df = init_scraper.get_dataframe()
    #min_date = pd.to_datetime(min(df['Date']))
    #max_date = pd.to_datetime(max(df['Date']))


    return df

### LOAD THE DATA
df = load_data(selected_ticker)

### Initialise scraper without time interval
scraper = CmcScraper(selected_ticker)

##############################################################################################


data = scraper.get_dataframe()
data['Date'] = pd.to_datetime(data['Date']).dt.date


st.subheader(f'Historical data of {selected_ticker}') #display
st.write(data.head(5)) # display data frame

####################################################################################################################

#DISPLAY RAW DATA TABLE
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)


plot_log = st.checkbox("Plot log scale")
if plot_log:
	plot_raw_data_log()
else:
	plot_raw_data()



###########################################################################################
rev_data=data.reindex(index=data.index[::-1])

#data['Date']=data.index ##### create date column
closed_prices_data=rev_data[['Close']].values.reshape(-1, 1) ##### get closed prices from data

Scale=MinMaxScaler()

Scaled_data=Scale.fit(closed_prices_data)#datascaler
X=Scaled_data.transform(closed_prices_data) #####normalizing the data
X=X.reshape(X.shape[0],)


#samples split
X_samples=list()
y_samples=list()

NumRows=len(X)
prediction_days=int(st.sidebar.number_input('Input range of days for prediction:', min_value=0, max_value=365, value=60, step=1)) #next day's prediction (based on the last how many days price)
Future_Steps=int(st.sidebar.number_input('Input how many days the application will predict:', min_value=0, max_value=365, value=1, step=1)) # predicting x days from based days

if st.button("Predict"):

###########################################################################################

# TRAINING OF DATA
    for i in range(prediction_days,NumRows-Future_Steps,1):
    
        x_sample_data=X[i-prediction_days:i]
        y_sample_data=X[i:i+Future_Steps]
        X_samples.append(x_sample_data)
        y_samples.append(y_sample_data)

#reshape input as 3D
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1],1)


#y data is a single column only
    y_data=np.array(y_samples)


# num of testing data records
    test_record=5
#split data to train and test
    X_train=X_data[:-test_record]
    X_test=X_data[-test_record:]
    y_train=y_data[:-test_record]
    y_test=y_data[-test_record:]

#define inputs for LSTM
#SampleNum=1
    Steps=X_train.shape[1]
    Features=X_train.shape[2]
#X_test=X_test.reshape(SampleNum,Steps,Features)


###################################################################################################
# LSTM MODEL
    model = Sequential()

#first hidden layer and LSTM layer
    model.add(LSTM(units=50, activation='relu', input_shape=(Steps,Features),return_sequences=True))  

#second layer
    model.add(LSTM(units=25, activation='relu', input_shape=(Steps,Features),return_sequences=True))

#third layer
    model.add(LSTM(units=25, activation='relu',return_sequences=False))

#Output layer
    model.add(Dense(units=Future_Steps))# change to 5 later for multi deep learning

#complile RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    import time
# measure time taken for model to train
    StartTime=time.time()

#fit the RNN to Training set
    model.fit(X_train, y_train, epochs=25, batch_size=32)
    EndTime=time.time()
##############################################################################################

# TESTING OF DATA


    print("##Total time taken:" ,round((EndTime-StartTime)/60),"Minutes ##")

#make predictions on testing data
    predicted_Price=model.predict(X_test)
    predicted_Price=Scaled_data.inverse_transform(predicted_Price)


#get the original price for testing data
    original=y_test
    original=Scaled_data.inverse_transform(y_test)




#generate predictions on full data
    TrainPredictions=Scaled_data.inverse_transform(model.predict(X_train))
    TestPredictions=Scaled_data.inverse_transform(model.predict(X_test))

    Full_Data_Predictions=np.append(TrainPredictions,TestPredictions)
    Full_Orig_Data=closed_prices_data[Steps:]




    P_Data = pd.Series(predicted_Price.ravel('F'))#DataFrame(predicted_Price)
    
    rev_predictions=P_Data.reindex(index=P_Data.index[::-1])


# Visualising the results
    plt.title('### Accuracy of the predictions:'+ str(100 - (100*(abs(original-predicted_Price)/original)).mean().round(2))+'% ###')

    plt.plot(Full_Data_Predictions, color = 'blue', label = 'Predicted price')
    plt.plot(Full_Orig_Data, color = 'lightblue', label = 'Original price')
    plt.legend()
    plt.show()

#############################################################################################

# PREDICT NEXT DAY
    for i in range(prediction_days,NumRows,1):

        X_days=rev_data[i-prediction_days:]
        
    Last_X_Days_Prices=closed_prices_data[-prediction_days:]
 
# Reshaping the data to (-1,1 )because its a single entry
    Last_X_Days_Prices=Last_X_Days_Prices.reshape(-1, 1)
 
# Scaling the data on the same level on which model was trained
    X_test=Scaled_data.transform(Last_X_Days_Prices)
 
    NumberofSamples=1
    TimeSteps=X_test.shape[0]
    NumberofFeatures=X_test.shape[1]

# Reshaping the data as 3D input
    X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)
 
# Generating the predictions for next X days
    NextXDaysPrice = model.predict(X_test)
 

# Generating the prices in original scale
    NextXDaysPrice = Scaled_data.inverse_transform(NextXDaysPrice)

    P_Mul_Data = pd.DataFrame(NextXDaysPrice)
    #print(P_Mul_Data)

    print(rev_predictions)
####################################################################################################################

    
    st.subheader(f'Plot data using {prediction_days} days from historical data')
#st.write(.head())
    st.write(X_days.head(prediction_days))


    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=X_days['Close'], x=X_days['Date']))
    fig1.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)




    st.subheader("Forecast components")

    st.subheader(f'Predicted values for {Future_Steps} days')
    st.write(P_Mul_Data.head())

    st.line_chart(P_Mul_Data)

    data['month'] = data['Date'].apply(lambda x: x.month)
    data['year'] = data['Date'].apply(lambda x: x.year)
    data['day'] = data['Date'].apply(lambda x: x.day)


    month=pd.DataFrame(data.groupby('month'))

#monthly=month['Market Cap']

    st.subheader('Monthly data')
#st.write(month.head(5))

#fig2 = go.Figure()
#fig2.add_trace(go.Scatter(y=month['Close'], x=data['Date']))
#fig2.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
#st.plotly_chart(fig2)


#st.line_chart(month)


    monthly=data.groupby('month').agg('mean')
    plt.title('Monthly')
    plt.xlabel('Month')
    plt.legend(loc='upper right')


    yearly=data.groupby('year').agg('mean')
    plt.title('Yearly')
    plt.xlabel('Year')
    plt.legend(loc='upper right')

    daily=data.groupby('day').agg('mean')
    plt.title('Daily')
    plt.xlabel('Days')
    plt.legend(loc='upper right')
    
    #st.line_chart(monthly)
    #st.subheader('Yearly data')
    #st.line_chart(yearly)
    #st.subheader('Daily data')
    #st.line_chart(daily)



#################################################################################
### trends seasonality etc
data.set_index('Date', inplace=True)

analysis = data[['Close']].copy()


#decompose_result_mult = seasonal_decompose(analysis, model="additive", period=1)
#trend = decompose_result_mult.trend
#seasonal = decompose_result_mult.seasonal
#residual = decompose_result_mult.resid
#decompose_result_mult.plot();

#st.line_chart(trend)
#st.line_chart(seasonal)