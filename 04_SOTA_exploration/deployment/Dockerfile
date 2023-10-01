
FROM python:3.7-slim-buster

COPY multimodal_libraries /multimodal_libraries
COPY processing_data_collection /processing_data_collection
COPY models /models
COPY requirements.txt /requirements.txt



RUN pip install -r requirements.txt

#Run the app.py file
CMD ["python", "multimodal_libraries/app.py"]
