from io import BytesIO
import json
import boto3
import joblib
import logging
import pickle
import numpy as np
from healthinsurance.HealthInsurance import HealthInsurance


# Define logger class
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Helper Function to download object from S3 Bucket
def DownloadFromS3(bucket:str, key:str):
    s3 = boto3.client('s3')

    with BytesIO() as f:
        s3.download_fileobj(Bucket = bucket, Key = key, Fileobj = f)
        f.seek(0)
        df_test = joblib.load(f)
    
    return df_test

# Load model into memory
logger.info('Loading model from file...')

model = pickle.load(open('./model_random_forest.pkl', 'rb')) # where the model is stored

logger.info('Model loaded from file.')

# Lambda Function
def lambda_handler(event, context):

    # read JSON data packet containing S3 Bucket specs to access test dataset and dataset percentage of interest
    data = json.loads(event['body'])
    bucket = data['bucket']
    key = data['key']
    percentage = data['percentage']

    # load test data from S3
    logger.info(f'Loading data from{bucket}/{key}')
    df_test_raw = DownloadFromS3(bucket, key)
    logger.info(f'Loaded {type(key)} from S3...')

    # ========================================
    # code from 'handler.py'

    # instantiate HealthInsurance class
    logger.info('Instantiating HealthInsurance class...')
    pipeline = HealthInsurance()
    logger.info('HealthInsurance class instantiated...')

    df_test = df_test_raw.copy()

    # data cleaning
    logger.info('Data cleaning...')
    df1 = pipeline.data_cleaning(df_test)

    # feature engineering
    logger.info('Feature engineering...')
    df2 = pipeline.feature_engineering(df1)

    # data preparation
    logger.info('Data preparation...')    
    df3 = pipeline.data_preparation(df2)

    # prediction
    logger.info('Making prediction...')
    prediction = pipeline.get_prediction(model, df3)

    # join prediction with test data
    logger.info('Joining prediction with test data...')

    # join prediction with original test data
    logger.info('Joining prediction with test data...')
    df_test_raw.rename(columns = {'response':'score'}, inplace = True)
    df_test_raw['score'] = prediction

    # use only the specified percentage of dataset
    df_test_raw.sort_values(by = 'score', ascending = False, inplace = True, ignore_index = True)
    a, b, c = np.split(df_test_raw, [int(percentage*len(df_test)), int((1-percentage)*len(df_test))])

    # return json to be used by API
    logger.info('Preparing response as JSON file...')
    response = json.dumps(a.values.tolist(), separators=(',', ':')) # convert dataframe to list
    # size_obj1 = len(response.encode('utf-8'))
    # print('The size is: {} MB'.format(size_obj1/(1024*1024))) # size in megabytes

    return {
        'statusCode': 200,
        'headers':{
            'Content-type':'application/json'
        },
        'body':response 
    }