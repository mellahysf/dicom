#!/bin/bash

export_json () {
for s in $(echo $values | jq -r 'to_entries|map("\(.key)=\(.value|tostring)")|.[]' $1 ); do
    export $s
done
}

export_json "/app/4.3.1_ocr.json"

pip install --upgrade spark-nlp==$PUBLIC_VERSION
pip install --upgrade -q spark-nlp-jsl==$JSL_VERSION  --extra-index-url https://pypi.johnsnowlabs.com/$SECRET
pip install --upgrade -q spark-ocr==$OCR_VERSION --extra-index-url=https://pypi.johnsnowlabs.com/$SPARK_OCR_SECRET

if [ $? != 0 ];
then
exit 1
fi

python3 /app/main.py