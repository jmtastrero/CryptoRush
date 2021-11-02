import streamlit as st
import base64
import pandas as pd
import hydralit_components as hc
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

col1, col2, col3 = st.columns(3)

with col2:
    image = Image.open('logo.png')
    st.image(image, width=500)


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


menu_data = [
        {'icon': "far fa-chart-bar",'label':"Predict"},
]


#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#808080'}
#menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)
if menu_id == 'Home':
        st.markdown("<h3 style='text-align:center;'>A Web Application that enables you to predict and forecast the future value of <br> any cryptocurrency on a daily, weekly, and monthly and yearly basis.</h3>", unsafe_allow_html=True)
        st.header("")
        st.header("")
        
        
        
        
        st.subheader("CONVERSION")
        HtmlFile = open("converter.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code,height=200, scrolling=False)    
        
        
        st.subheader("PRICES IN USD")
        HtmlFile = open("USD.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code,height=210, scrolling=True) 
        
        st.header("")
        
        st.subheader("PRICES IN PHP")
        HtmlFile = open("PHP.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        print(source_code)
        components.html(source_code,height=210, scrolling=True)
else:
    exec(open('CRF.py').read())




#hide_streamlit_style = """
            #<style>
            #MainMenu {visibility: hidden;}
            #footer {visibility: hidden;}
            #</style>
            #"""
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)
