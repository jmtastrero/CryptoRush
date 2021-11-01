import streamlit as st
import base64
#import pandas as pd
import streamlit.components.v1 as components

from PIL import Image

main_bg = "bg3.png"
main_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container{{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})    
        }}
    </style>
    
    """,
    unsafe_allow_html=True
)
########################################################################################


st.sidebar.title('CryptoRush')
page_names = ['Home', "Predict"]
pn = st.sidebar.radio('navigation', page_names)

if pn == 'Home':
        image = Image.open('logo.png')
        st.image(image, width=500)
        st.title('CryptoRush')
        st.markdown('A Web Application that enables you to predict and forecast the future value of any cryptocurrency on a daily, weekly, and monthly and yearly basis.')
        
        st.header("CONVERSION")
        HtmlFile = open("converter.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code,height=250,scrolling=False)

        st.header("PRICES IN PHP")
        HtmlFile = open("yes.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code,height=850,scrolling=False)    
else:
    exec(open('CRF.py').read())
    #exec(open('predict.py').read())

#st.header("PRICES")

#df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')

# Custom function for rounding values
def round_value(input_value):
    if input_value.values > 1:
        a = float(round(input_value, 2))
    else:
        a = float(round(input_value, 8))
    return a

col1, col2, col3 = st.columns(3)

# Widget (Cryptocurrency selection box)
#col1_selection = st.sidebar.selectbox('Price', df.symbol, list(df.symbol).index('BTCBUSD') )


# DataFrame of selected Cryptocurrency
#col1_df = df[df.symbol == col1_selection]
#col2_df = df[df.symbol == col1_selection]
#col3_df = df[df.symbol == col1_selection]
#col4_df = df[df.symbol == col1_selection] 


# Apply a custom function to conditionally round values
#col1_price = round_value(col1_df.weightedAvgPrice)
#col2_price = round_value(col2_df.weightedAvgPrice)
#col3_price = round_value(col3_df.weightedAvgPrice)
#col4_price = round_value(col4_df.weightedAvgPrice)


# Select the priceChangePercent column
#col1_percent = f'{float(col1_df.priceChangePercent)}%'
#col2_percent = f'{float(col2_df.priceChangePercent)}%'
#col3_percent = f'{float(col3_df.priceChangePercent)}%'
#col4_percent = f'{float(col4_df.priceChangePercent)}%'


# Create a metrics price box
#col1.metric(col1_selection, col1_price, col1_percent)
#col3.metric(col1_selection, col2_price, col2_percent)
#col3.metric(col1_selection, col3_price, col3_percent)
#col1.metric(col1_selection, col4_price, col4_percent)

#test_input=int(st.sidebar.number_input('input value:',value=5,))

#pick=st.sidebar.selectbox("Crypto",options=["BTC","ETH","BNB","XRP"] )

### if and else statement
### get USD to PHP convert
### output php

#expected=test_input*pick

#st.sidebar.markdown(expected)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
