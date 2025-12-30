from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

app = FastAPI()

# Modelo multilingüe mejor adaptado a español
model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Entrada esperada
class TextRequest(BaseModel):
    text: str

# Expresiones regulares complementarias
EMAIL_RE = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_RE = r"(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}"  # flexible
DOCUMENT_RE = r"\b(?:V|E|G|J|P|N|M|R|D|X|Y|Z)?-?\d{6,10}(?:-\d)?\b"
ADDRESS_RE = r"\b(?:calle|av(enida)?|urb(\.|anización)?|sector|mz|cll|cra|manzana|bloque|edificio|apto|piso|#)\s?[\w\-\d]+(?:[\s,#-]+\w+)*\b"

@app.get("/health")
def detect_pii():
	return {'Hello':'PII Service'}

@app.post("/detect")
def detect_pii(body: TextRequest):
    text = body.text
    response = {}

    # Ejecutar modelo NER
    entities = ner_pipeline(text)
    for ent in entities:
        word = ent['word']
        group = ent['entity_group']

        if group == 'PER' and 'name' not in response:
            response['name'] = word
        elif group == 'ORG' and 'business_name' not in response:
            response['business_name'] = word
        elif group in ('LOC', 'GPE') and 'address' not in response:
            response['address'] = word

    # Detectar email
    email_match = re.search(EMAIL_RE, text)
    if email_match:
        response['email'] = email_match.group()

    # Detectar teléfono
    phone_match = re.search(PHONE_RE, text)
    clean_phone = None
    if phone_match:
        digits = re.sub(r"\D", "", phone_match.group())
        if len(digits) >= 8:
            clean_phone = "+" + digits
            response['phone'] = clean_phone

    # Detectar documento, con lógica mejorada de validación
    #document_match = re.search(DOCUMENT_RE, text, re.IGNORECASE)
    #if document_match:
    #    doc = document_match.group()
    #    digits_doc = re.sub(r"\D", "", doc)
    #    prefix_match = re.match(r'^[VEGJP][\-]?', doc, re.IGNORECASE)
    #    if not clean_phone or prefix_match or digits_doc != clean_phone.lstrip('+'):
    #        response['document_id'] = doc
    #        doc_prefix = doc[0].upper()
    #        if doc_prefix in ('V', 'E', 'G', 'J', 'P', 'N', 'M', 'R', 'D', 'X', 'Y', 'Z'):
    #            response['document_id_type'] = doc_prefix

    # Detectar dirección más explícita con RegEx
    if 'address' not in response:
        address_match = re.search(ADDRESS_RE, text, re.IGNORECASE)
        if address_match:
            response['address'] = address_match.group()

    return response




