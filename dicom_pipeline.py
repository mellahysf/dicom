import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
import sparkocr


import txt_pipeline
from sparknlp_jsl.utils.ocr_nlp_processor import ocr_entity_processor

class dicom_deidentifier:

    def __init__(self, spark, input_file_path, output_file_path=None, requested_labels= ["NAME", "AGE", "SSN"]):

        """
        This class is used to deidentify the pdf file.

        Parameters
        ----------
        spark : SparkSession
            SparkSession object
        
        input_file_path : str
            Path of the input pdf file
        
        requested_labels : list
            List of labels to be de-identified from the pdf file

        Returns
        -------
        De-identified pdf file. 
        """

        self.spark = spark
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.requested_labels = requested_labels

    
    def deidentify_dicom(self):

        ner_model= txt_pipeline.txt_deidentifier(self.spark, self.requested_labels).ner_pipeline()

        # Read dicom as image
        dicom_to_image = DicomToImage() \
            .setInputCol("content") \
            .setOutputCol("image_raw") \
            .setMetadataCol("metadata") \
            .setDeIdentifyMetadata(True)

        adaptive_thresholding = ImageAdaptiveThresholding() \
            .setInputCol("image_raw") \
            .setOutputCol("corrected_image") \
            .setBlockSize(47) \
            .setOffset(4) \
            .setKeepInput(True)

        # Extract text from image
        ocr = ImageToText() \
            .setInputCol("corrected_image") \
            .setOutputCol("text")

        # Found coordinates of sensitive data
        position_finder = PositionFinder() \
            .setInputCols("ner_chunk") \
            .setOutputCol("coordinates") \
            .setPageMatrixCol("positions") \
            .setMatchingWindow(100) \
            .setPadding(1)

        # Found sensitive data using DeIdentificationModel
        deidentification_rules = DeIdentificationModel.pretrained("deidentify_rb_no_regex", "en", "clinical/models") \
            .setInputCols(["metadata_sentence", "metadata_token", "metadata_ner_chunk"]) \
            .setOutputCol("deidentified_metadata_raw")

        finisher = Finisher() \
            .setInputCols(["deidentified_metadata_raw"]) \
            .setOutputCols("deidentified_metadata") \
            .setOutputAsArray(False) \
            .setValueSplitSymbol("") \
            .setAnnotationSplitSymbol("")

        # Draw filled rectangle for hide sensitive data
        drawRegions = ImageDrawRegions() \
            .setInputCol("image_raw") \
            .setInputRegionsCol("coordinates") \
            .setOutputCol("image_with_regions") \
            .setFilledRect(True) \
            .setRectColor(Color.black)

        # Store image back to Dicom document
        imageToDicom = ImageToDicom() \
            .setInputCol("image_with_regions") \
            .setOutputCol("dicom")

        # OCR pipeline with imageToDicom
        deid_pipeline_imgtodicom = PipelineModel(stages=[
            dicom_to_image,
            adaptive_thresholding,
            ocr,
            ner_model,
            position_finder,
            drawRegions,
            imageToDicom
        ])

        # OCR pipeline without imageToDicom
        deid_pipeline = PipelineModel(stages=[
            dicom_to_image,
            adaptive_thresholding,
            ocr,
            ner_model,
            position_finder,
            drawRegions,
        ])

        #Training the pipelines
        deid_imgtodicom_model = deid_pipeline.fit(self.spark.createDataFrame([[""]]).toDF("text"))
        deid_model = deid_pipeline.fit(self.spark.createDataFrame([[""]]).toDF("text"))

        return deid_imgtodicom_model, deid_model

    def get_result(self):
        """
        This function returns the deidentified dicom & statistics
        """
        deid_pipeline_imgtodicom, deid_model = self.deidentify_dicom()

        dicom_df = spark.read.format("binaryFile").load(self.input_file_path)

        deid_imgtodicom_results = deid_pipeline_imgtodicom.transform(dicom_df).cache()
        deid_results = deid_model.transform(dicom_df).cache()

        # storing dicom result
        for r in deid_imgtodicom_results.select("dicom", "path").collect():
            #path, name = os.path.split(r.path)
            #filename_split = os.path.splitext(name)
            #file_name = os.path.join(filename_split[0] + ".dcm")
            print(f"Storing to {self.output_file_path}")
            with open(output_file_path, "wb") as file:
                file.write(r.dicom)

        # getting some statistics
        counts = deid_results.select(F.explode(F.arrays_zip(deid_results.ner_chunk.result,
                                                            deid_results.ner_chunk.metadata)).alias("cols")) \
            .select(F.expr("cols['0']").alias("chunk"),
                    F.expr("cols['1']['entity']").alias("ner_label"))

        pandas_df = counts.toPandas()
        totalProcessedEntities = pandas_df.shape[0]
        numberOfCharacters = len(''.join(pandas_df["chunk"].values.tolist()))
        # Get number of pages
        numberOfPages = deid_results.count()

        return deid_imgtodicom_results, numberOfPages, totalProcessedEntities, numberOfCharacters
        

    


        
