import streamlit as st
import pandas as pd
# import altair as alt
import requests
import json
import numpy as np
from dotenv import load_dotenv
import os

from urllib.error import URLError


st.set_page_config(
    page_title = 'Propensity Score App',
    page_icon = 'random',
    layout = 'centered',
    menu_items = {
        'Get Help':'https://github.com/joaomj/health_insurance_cross_sell',
    }
)


# Helper Functions
@st.cache
def get_data(bucket_name:str, key:str, percentage:float):
    # AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    # df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    # return df.set_index("Region")

    data = {
        'bucket':bucket_name,
        'key':key,
        'percentage':percentage,
    }

    headers = {
        'Content-type': "application/json"
    }

    # Main code for post HTTP request
    load_dotenv()
    url = os.environ.get('URL_DEPLOY') 
    response = requests.request("POST", url, headers=headers, data=json.dumps(data))

    # viewing the response as a dataframe object
    lambda_predictions = np.array(response.json())

    columns = [    
        'id', 
        'gender', 
        'age',
        'region_code',
        'policy_sales_channel',
        'driving_license',
        'vehicle_age',
        'vehicle_damage',
        'previously_insured',
        'annual_premium',
        'vintage',
        'score'
        ]

    df = pd.DataFrame(lambda_predictions, columns = columns)

    return df

# convert dataframe to csv file
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


try:
    
    percentage = st.slider(
        label = 'Dataset Percentage', 
        min_value = 0.0, 
        max_value = 100.0, 
        value = 40.0,
        step = 1.0,
        help = 'Select the top %\n of clients with highest propensity to buy another insurance.',
        )

    if percentage > 0.0 :

        percentage = percentage/100
        df = get_data(
            bucket_name = 'joaomj-lambda-buckets-2022', 
            key = 'validation/df_test.joblib', 
            percentage = percentage
            )

        st.write('### Top Percentage of Clients with Highest Propensity to Buy a Car Insurance:')#, df)
        st.dataframe(df)

        csv = convert_df(df)

        st.download_button(
            label = 'Download this data as a .csv file',
            data = csv,
            file_name = 'top_clients.csv',
            mime = 'text/csv'
        )
    
    else:
        st.error("Please select a percentage greather than 0.0") 

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )