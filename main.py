import uvicorn, json, os
from fastapi import FastAPI, File, UploadFile, Request, Form

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
import sparkocr
from sparkocr import start
from dicom_pipeline import dicom_deidentifier

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    with open('/app/4.3.1_ocr.json') as f:
        license_keys = json.load(f)
    global spark
    spark = sparkocr.start(secret=license_keys['SPARK_OCR_SECRET'], 
                       nlp_version=license_keys['PUBLIC_VERSION'],
                       nlp_secret=license_keys['SECRET'],
                       nlp_internal=license_keys['JSL_VERSION']
                       )

@app.post("/deiddicom")
async def deiddicom(file: UploadFile = File(...), outputFile: str = Form(...), entities: List[str] = Form(...)):


    # Getting and saving DICOM file from the request
    dicom_path= "/app/dicom_files/" + file.filename
    contents = file.file.read()
    with open(dicom_path, 'wb') as f:
        f.write(contents)

    # Creating the DEID (start spark session, precise DICOM input file path and user's required entities
    deid = dicom_deidentifier(spark, input_file_path=dicom_path, output_file_path=outputFile, requested_labels= entities)

    # Getting the Deid DICOM file & statistics
    result, numberOfPages, numberOfDeidEntities, numberOfChars = deid.get_result()

    # Checking if there is result
    if result:
        status = 1
    else:
        status = 0

    # Building the JSON output result
    jsonResult = {
        "status": status,
        "numberOfPages": numberOfPages,
        "numberOfDeidEntities": numberOfDeidEntities,
        "numberOfChars": numberOfChars
    }

    return jsonResult


if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=5000)
    