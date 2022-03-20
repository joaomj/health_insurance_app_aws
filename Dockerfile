# Install Python
# currently there is no official AWS Lambda image with Python 3.10
# See more: https://github.com/aws/aws-lambda-base-images/issues/31
FROM public.ecr.aws/lambda/python:3.9

# Setup Python environment
# Install PYTHON requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy required files
COPY app ./

# Set entry point
CMD ["lambda_predict.lambda_handler"]