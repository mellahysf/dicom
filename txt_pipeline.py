import os
import json
import pandas as pd
import numpy as np

from pyspark.sql.types import StructType, StructField, StringType

from sparknlp_jsl.annotator import *
from sparknlp.base import *

from pyspark.sql import functions as F
from sparknlp_jsl.annotator.deid.deIdentification import DeIdentification

class txt_deidentifier:

    def __init__(self, spark, deid_json_path, input_file_path=None, requested_labels= ["NAME", "AGE", "SSN"]):
        self.spark= spark
        self.deid_json_path = deid_json_path
        self.input_file_path = input_file_path
        self.requested_labels = requested_labels

        """
        This class is used to deidentify the given text file based on the deid json file and it returns pyspark data frame.

        Parameters
        ----------
        spark : SparkSession
            SparkSession object
        
        deid_json_path : str
            Path to the deid json file

        input_file_path : str
            Path to the input text file

        Returns
        -------
        txt file
            A deidentified txt file.
        """


    # ------------------------- Load pretrained model from the cache pretrained folder ------------------------- #
    def load_model_from_cache(self, target_model_name):

        # cache directory path
        directory_path = "/app/cache_pretrained"

        # get a list of all the directories in the directory_path
        dir_list = os.listdir(directory_path)

        # filter the directories that contain the target ner model name in their name
        filtered_dirs = [d for d in dir_list if target_model_name in d]

        # if there are no directories that contain target_model_name in their name, print an error message and exit
        if len(filtered_dirs) == 0:
            print(f"Error: no directory found containing {target_model_name} in its name")
            exit()

        # if there is only one directory that contains target_model_name in its name, assign its name to a variable
        if len(filtered_dirs) == 1:
            model_folder_name = directory_path+"/"+filtered_dirs[0]
        

        # if there are multiple directories that contain target_model_name in their name, print a message asking the user to choose one and assign its name to a variable
        if len(filtered_dirs) > 1:
            print(f"Multiple directories found containing {target_model_name} in their name. Please choose one:")
            for i, d in enumerate(filtered_dirs):
                print(f"{i+1}. {d}")
            choice = int(input())
            model_folder_name = filtered_dirs[choice-1]
            
        return model_folder_name
    

    # ------------------------- Load base pipeline ------------------------- #
    def base_pipeline(self):
        """
        This function is used to load the base pipeline for the deidentification process.
        """

        embedding_path= self.load_model_from_cache("embeddings_clinical")

        document_assembler = DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")
                
        sentence_detector = SentenceDetector()\
            .setInputCols(["document"])\
            .setOutputCol("sentence")

        tokenizer = Tokenizer()\
            .setInputCols(["sentence"])\
            .setOutputCol("token")

        word_embeddings = WordEmbeddingsModel.load(embedding_path)\
            .setInputCols(["sentence", "token"])\
            .setOutputCol("embeddings")
        

        base_pipeline= [document_assembler, sentence_detector, tokenizer, word_embeddings]

        return base_pipeline
    



    # ------------------------- Defining contextual parser approaches ------------------------- #

    # CP for social security number
    ssn = {
    "entity": "SSN",
    "ruleScope": "sentence",
    "regex": "\d{3}.?\d{2}.?\d{4}",
    "matchScope": "token",
    "prefix": ["ssn", "social", "security", "ssns", "ss#", "ssn#", "ssid", "ss #", "ssn #"],
    "contextLength": 20
    }

    with open('/app/ssn.json', 'w') as f:
        json.dump(ssn, f)

    global ssn_parser
    ssn_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("SSN") \
            .setJsonPath("/app/ssn.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)

    
    #CP for Account ID
    account = {
    "entity": "ACCOUNT",
    "ruleScope": "sentence",
    "regex": "[0-9]{7,17}|[0123678]\d{3}.?\d{4}.?\d",
    "matchScope": "token",  
    "prefix": ["check", "account", "account#", "acct", "routing", "acct#", "save", "debit", "bank", "aba", "aba routing",\
                "abarouting", "association", "bankrouting"],
    "contextLength": 25
    }

    with open('/app/account.json', 'w') as f:
        json.dump(account, f)

    global account_parser
    account_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("ACCOUNT") \
            .setJsonPath("/app/account.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)
    

    # CP for License Number
    license = {
    "entity": "LICENSE",
    "ruleScope": "sentence",
    "regex": "\S+\d\S+",
    "matchScope": "token",  
    "prefix": ["license", "License", "LICENSE", "lic", "Lic", "LIC", "licence", "Licence", "LICENCE", "lic#", "Lic#", "LIC#", "license#", "License#", "LICENSE#", "licence#", "Licence#", "LICENCE#", "certificate", "Certificate", "CERTIFICATE", "cert", "Cert", "CERT", "certificate#", "Certificate#", "CERTIFICATE#", "cert#", "Cert#", "CERT#"],
    "suffix": ["license", "License", "LICENSE", "lic", "Lic", "LIC", "licence", "Licence", "LICENCE", "lic#", "Lic#", "LIC#", "license#", "License#", "LICENSE#", "licence#", "Licence#", "LICENCE#", "certificate", "Certificate", "CERTIFICATE", "cert", "Cert", "CERT", "certificate#", "Certificate#", "CERTIFICATE#", "cert#", "Cert#", "CERT#"], 
    "contextLength": 20
    }

    with open('/app/license.json', 'w') as f:
        json.dump(license, f)

    global license_parser
    license_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("LICENSE") \
            .setJsonPath("/app/license.json") \
            .setCaseSensitive(True) \
            .setPrefixAndSuffixMatch(False)
    


    # CP for Vehicle Identity Number
    vin_cp = {
    "entity": "VIN",
    "ruleScope": "sentence",
    "matchScope":"token",
    "regex":"(^\d[A-Za-z]{4}\d{2}[A-Za-z]{4}\d{6}$)|(^\d[A-Za-z]\d[A-Za-z]{2}\d{5}[A-Za-z]\d{6}$)",
    "prefix":['VIN'],
    "contextLength": 5,
    }

    with open('/app/vin_cp.json', 'w') as f:
        json.dump(vin_cp, f)

    global vin_parser
    vin_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("VIN") \
            .setJsonPath("/app/vin_cp.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)
    

    # CP for Age
    age = {
    "entity": "AGE",
    "ruleScope": "sentence",
    "matchScope":"sub-token",
    "regex":"(\d{1,2})(\s?-?1/2)?",
    "suffix": ["-years-old",
                "years-old",
                "-year-old",
                "year-old",
                "-months-old",
                "months-old",
                "-month-old",
                "month-old",
                "-day-old",
                "day-old",
                "-days-old",
                "days-old",
                "-week-old",
                "week-old",
                "-weeks-old",
                "weeks-old",
                "month old",
                "days old",
                "year old",
                "years old", 
                "week old",
                "weeks old",
                "years of age",
                "months of age", 
                "weeks of age",
                "days of age",
                ],
    "contextLength": 20
    }

    with open('/app/age.json', 'w') as f:
        json.dump(age, f)

    global age_parser
    age_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("AGE") \
            .setJsonPath("/app/age.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)


    # CP for Phone
    phone = {
    "entity": "PHONE",
    "ruleScope": "document",
    "matchScope":"sub-token",
    "regex":"(?<!\d\s)\([1-9]\d{2}\)\.?\-?\s?\d{3}\.?\-?\s?\d{4}(?!\d)",
    "prefix": ["call", "phone"],
    "contextLength": 10
    }

    with open('/app/phone.json', 'w') as f:
        json.dump(phone, f)

    global phone_parser
    phone_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("PHONE") \
            .setJsonPath("/app/phone.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)
    

    # CP for IDNUM
    ids = {
    "entity": "IDNUM",
    "ruleScope": "sentence",
    "matchScope":"sub-token",
    "regex":"(?<=\#)(\d{4,5})",
    "prefix": ["CVS/PHARMACY","Wal-Mart Pharmacy"],
    "contextLength": 20
    }

    with open('/app/ids.json', 'w') as f:
        json.dump(ids, f)

    global ids_parser
    ids_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("IDNUM") \
            .setJsonPath("/app/ids.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)
    

    # CP for ZIP codes
    zip_cp = {
    "entity": "ZIP",
    "ruleScope": "sentence",
    "matchScope":"token",
    "regex":"(^\d{5}$)|(^\d{5}-\d{4}$)",
    "prefix":['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'Alabama', 'Maine', 'Pennsylvania', 'Alaska', 'Maryland', 'Rhode', 'Island', 'Arizona', 'Massachusetts', 'South', 'Carolina', 'Arkansas', 'Michigan', 'South', 'Dakota', 'California', 'Minnesota', 'Tennessee', 'Colorado', 'Mississippi', 'Texas', 'Connecticut', 'Missouri', 'Utah', 'Delaware', 'Montana', 'Vermont', 'District', 'of', 'Columbia', 'Nebraska', 'Virginia', 'Florida', 'Nevada', 'Washington', 'Georgia', 'New', 'Hampshire', 'West', 'Virginia', 'Hawaii', 'New', 'Jersey', 'Wisconsin', 'Idaho', 'New', 'Mexico', 'Wyoming', 'Illinois', 'New', 'York', 'American', 'Samoa', 'Indiana', 'North', 'Carolina', 'Guam', 'Iowa', 'North', 'Dakota', 'Northern', 'Mariana', 'Islands', 'Kansas', 'Ohio', 'Palau', 'Kentucky', 'Oklahoma', 'Puerto', 'Rico', 'Louisiana', 'Oregon', 'Virgin', 'Islands',
            'ALABAMA', 'MAINE', 'PENNSYLVANIA', 'ALASKA', 'MARYLAND', 'RHODE', 'ISLAND', 'ARIZONA', 'MASSACHUSETTS', 'SOUTH', 'CAROLINA', 'ARKANSAS', 'MICHIGAN', 'SOUTH', 'DAKOTA', 'CALIFORNIA', 'MINNESOTA', 'TENNESSEE', 'COLORADO', 'MISSISSIPPI', 'TEXAS', 'CONNECTICUT', 'MISSOURI', 'UTAH', 'DELAWARE', 'MONTANA', 'VERMONT', 'DISTRICT', 'OF', 'COLUMBIA', 'NEBRASKA', 'VIRGINIA', 'FLORIDA', 'NEVADA', 'WASHINGTON', 'GEORGIA', 'NEW', 'HAMPSHIRE', 'WEST', 'VIRGINIA', 'HAWAII', 'NEW', 'JERSEY', 'WISCONSIN', 'IDAHO', 'NEW', 'MEXICO', 'WYOMING', 'ILLINOIS', 'NEW', 'YORK', 'AMERICAN', 'SAMOA', 'INDIANA', 'NORTH', 'CAROLINA', 'GUAM', 'IOWA', 'NORTH', 'DAKOTA', 'NORTHERN', 'MARIANA', 'ISLANDS', 'KANSAS', 'OHIO', 'PALAU', 'KENTUCKY', 'OKLAHOMA', 'PUERTO', 'RICO', 'LOUISIANA', 'OREGON', 'VIRGIN', 'ISLANDS'],
    "contextLength": 5,
    }

    with open('/app/zip_cp.json', 'w') as f:
        json.dump(zip_cp, f)

    global zip_parser
    zip_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("ZIP") \
            .setJsonPath("/app/zip_cp.json") \
            .setCaseSensitive(True) \
            .setPrefixAndSuffixMatch(False)


    # CP for MEDICALRECORD
    med_cp = {
    "entity": "MEDICALRECORD",
    "ruleScope": "sentence",
    "matchScope":"token",
    "regex":"(^\d{5}$)",
    "prefix":['ICU Admission'],
    "contextLength": 15,
    }

    with open('/app/med_cp.json', 'w') as f:
        json.dump(med_cp, f)

    global med_parser
    med_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("MEDICALRECORD") \
            .setJsonPath("/app/med_cp.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)
    

    # CP for EMAIL
    email_pattern='''(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'''

    email_cp = {
    "entity": "EMAIL",
    "ruleScope": "sentence",
    "matchScope":"token",
    "regex":email_pattern,
    "prefix":["EMAIL", "email", "E-MAIL", "e-mail", "E-MAIL ADDRESS", "e-mail address", "EMAIL ADDRESS", "email address", "MAIL", "mail"],
    "suffix":["EMAIL", "email", "E-MAIL", "e-mail", "E-MAIL ADDRESS", "e-mail address", "EMAIL ADDRESS", "email address", "MAIL", "mail"],
    "contextLength": 60,
    }

    with open('/app/email_cp.json', 'w') as f:
        json.dump(email_cp, f)

    global email_parser
    email_parser = ContextualParserApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("EMAIL") \
            .setJsonPath("/app/email_cp.json") \
            .setCaseSensitive(False) \
            .setPrefixAndSuffixMatch(False)



    # ------------------ 3. Create the NER pipeline ------------------ #

    def ner_pipeline(self):

        """
        This function creates the NER pipeline
        """

        # Defining ner labels and models to be used
        ner_labels= {
             "NAME": "ner_deid_generic_augmented",
             "PATIENT": "ner_deid_subentity_augmented_en",
             "DOCTOR": "ner_deid_subentity_augmented_en",
             "STREET": "ner_deid_subentity_augmented_en",
             "LOCATION": "ner_deid_generic_augmented",
             "CITY": "ner_deid_subentity_augmented_en",
             "COUNTRY": "ner_deid_subentity_augmented_en",
             "ZIP": "ner_deid_subentity_augmented_en",
             "STATE": "ner_deid_subentity_augmented_en",
             "DATE": "ner_deid_generic_augmented",
             "AGE": "ner_deid_generic_augmented",
             "ID": "ner_deid_generic_augmented",
             "PHONE": "ner_deid_subentity_augmented_en",
             "FAX": "ner_deid_subentity_augmented_en",
             "EMAIL": "ner_deid_subentity_augmented_en",
             "MEDICALRECORD": "ner_deid_subentity_augmented_en",
             "DEVICE": "ner_deid_subentity_augmented_en",
             "URL": "ner_deid_subentity_augmented_en",
             "IDNUM": "ner_deid_subentity_augmented_en",
             "HOSPITAL": "ner_deid_subentity_augmented_en",
             "BIOID": "ner_deid_subentity_augmented_en",
             "GENDER": "ner_jsl_enriched",
             "RACE_ETHNICITY": "ner_jsl_enriched",
             "SEXUAL_ORIENTATION": "ner_sdoh_wip_en",
             "SPIRITUAL_BELIEFS": "ner_sdoh_demographics_wip",
             "SSN": ssn_parser,
             "ACCOUNT": account_parser,
             "LICENSE": license_parser,
             "VIN": vin_parser,
             "AGE": age_parser,
             "PHONE": phone_parser,
             "ZIP": zip_parser,
             "MEDICALRECORD": med_parser,
             "EMAIL": email_parser}
        
        # Changing some entity names in NerConverterInternal()
        ner_converter_change= {"CONTACT": "PHONE",
                                "ID": "IDNUM", 
                                "BIOID": "IDNUM",
                                "SPIRITUAL_BELIEFS": "RELIGION"}
        

        #base pipeline
        ner_base_pipeline= self.base_pipeline()

        #ner and CP models to use from the dictionary
        ner_model_list= set(list(ner_labels.values()))
        
        chunk_merger_inputs= []

        for ner_model in ner_model_list: 

            #adding ner models to the base pipeline
            if type(ner_model)==str:
                ner_model_path= self.load_model_from_cache(ner_model)
                tmp_model = MedicalNerModel.load(ner_model_path).setInputCols(["sentence","token","embeddings"]).setOutputCol(ner_model).setLabelCasing('upper')
                tmp_converter= NerConverterInternal().setInputCols(["sentence", "token", ner_model]).setOutputCol(f"{ner_model}_chunks").setReplaceLabels(ner_converter_change)

                chunk_merger_inputs.append(f"{ner_model}_chunks")
                ner_base_pipeline.append(tmp_model)
                ner_base_pipeline.append(tmp_converter)
                
            
            #adding CP models to the base pipeline
            else:
                tmp_model= ner_model
                label= ner_model.getOutputCol()

                tmp_cp_chunk= ChunkConverter().setInputCols(label).setOutputCol(f"{label}_chunks")

                chunk_merger_inputs.append(f"{label}_chunks")
                ner_base_pipeline.append(tmp_model)
                ner_base_pipeline.append(tmp_cp_chunk)


            #defining ChunkMergeApproach and filtering entities
        chunk_merger = ChunkMergeApproach()\
                                .setInputCols(chunk_merger_inputs)\
                                .setOutputCol('ner_chunk')\
                                .setOrderingFeatures(["ChunkPrecedence"])\
                                .setWhiteList(self.requested_labels)\
                                .setChunkPrecedenceValuePrioritization(["SSN_chunks,SSN",
                                                                        "ACCOUNT_chunks,ACCOUNT",
                                                                        "LICENSE_chunks,LICENSE",
                                                                        "VIN_chunks,VIN",
                                                                        "AGE_chunks,AGE",
                                                                        "PHONE_chunks,PHONE",
                                                                        "ZIP_chunks,ZIP",
                                                                        "MEDICALRECORD_chunks,MEDICALRECORD",
                                                                        "EMAIL_chunks,EMAIL"]) 
        

        ner_base_pipeline.append(chunk_merger)

        nlp_pipeline= Pipeline(stages=ner_base_pipeline)
        pipelineModel= nlp_pipeline.fit(self.spark.createDataFrame([[""]]).toDF("text"))

        return pipelineModel
    

    def deidentify(self):
        """
        This function deidentifies the text by usin the NER pipeline
        """


        deid = DeIdentification() \
                    .setInputCols(["sentence", "token", "ner_chunk"]) \
                    .setOutputCol("deidentified") \
                    .setMode("obfuscate")\
                    .setOutputAsDocument(True)\
                    .setSelectiveObfuscationModesPath(self.deid_json_path)\
                    .setSameLengthFormattedEntities(["PHONE", "ID"])\

        deid_pipeline= Pipeline(stages=[self.ner_pipeline(), deid])

        deid_model= deid_pipeline.fit(self.spark.createDataFrame([[""]]).toDF("text"))

        return deid_model
    

    def get_result(self):
        """
        This function returns the deidentified text
        """

        schema = StructType([StructField("text", StringType(), True)])
        df = self.spark.read.format("text").schema(schema).load(self.input_file_path)

        deid_model= self.deidentify()
        deid_result= deid_model.transform(df)

        output_df= deid_result.select(F.explode(F.arrays_zip(deid_result.deidentified.result)).alias("cols")) \
                              .select(F.expr("cols['0']").alias("deidentified"))
        
        output_df.write.mode("overwrite").format("text").option("header", "false").option("delimiter", " ").save("deid_result.txt")

        return deid_result
    

    def count_entities(self):
        """
        This function counts de-identified entities in the text
        """

        deid_result= self.get_result()

        counts= deid_result.select(F.explode(F.arrays_zip(deid_result.ner_chunk.result, 
                                                  deid_result.ner_chunk.metadata)).alias("cols")) \
                    .select(F.expr("cols['0']").alias("chunk"),
                            F.expr("cols['1']['entity']").alias("ner_label"))\
                            .groupBy('ner_label').count().orderBy('count', ascending=False)

        return counts
    
    
        









        

        

        





    

    
    



