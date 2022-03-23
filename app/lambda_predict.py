from io import BytesIO
import json
import boto3
import joblib
import logging
import pickle
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

    # read JSON data packet containing S3 Bucket specs to access test dataset
    data = json.loads(event['body'])
    bucket = data['bucket']
    key = data['key']

    # load test data from S3
    logger.info(f'Loading data from{bucket}/{key}')
    df_test = DownloadFromS3(bucket, key)
    logger.info(f'Loaded {type(key)} from S3...')

    # ========================================
    # code from 'handler.py'

    
    # instantiate HealthInsurance class
    logger.info('Instantiating HealthInsurance class...')
    pipeline = HealthInsurance()
    logger.info('HealthInsurance class instantiated...')

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
    df_test.rename(columns = {'response':'score'}, inplace = True)
    df_test['score'] = prediction

    # return json to be used by API
    logger.info('Preparing response as JSON file...')
    response = df_test.to_json(orient = 'records', date_format = 'iso')

    size_obj = len(response.encode('utf-8'))

    

    # need to do this:
    # Query the data and build a JSON file with all the rows in /tmp (Up to 512MB) directory
    # inside Lambda, upload it to S3 and return a CloudFront Signed URL to access the data.
    # source: https://stackoverflow.com/questions/46298060/aws-lambda-response-error/46298912#46298912

    return {
        'statusCode': 200,
        'headers':{
            'Content-type':'application/json'
        },
        'body':response 
    }

    # ========================================


    # # make predictions and return them as JSON
    # logger.info(f'Performing predictions...')
    # predictions = model.predict(df_test)
    # response = json.dumps(predictions.tolist())

    # return {
    #     'statusCode': 200,
    #     'headers':{
    #         'Content-type':'application/json'
    #     },
    #     'body':response 
    # }