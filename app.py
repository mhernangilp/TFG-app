import os
import torch
from flask import Flask, request, jsonify
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from peft import PeftModel
import email
from email import policy
from email.parser import BytesParser

# ==========================
#  API Configuration
# ==========================
app = Flask(__name__)

# Force GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================
#  Load Tokenizer & Model
# ==========================
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

base_model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2
)
model = PeftModel.from_pretrained(base_model, './final_model')
model.to(device)
model.eval()


# ==========================
#  Helper Functions
# ==========================
def parse_raw_email(raw_email: bytes) -> (str, str):
    msg = BytesParser(policy=policy.default).parsebytes(raw_email)

    subject = msg['subject'] or ''

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get_content_disposition()
            if content_type == 'text/plain' and content_disposition != 'attachment':
                payload = part.get_payload(decode=True)
                try:
                    body = payload.decode(part.get_content_charset(failobj='utf-8'), errors='replace')
                except:
                    body = payload.decode('utf-8', errors='replace')
                break
    else:
        payload = msg.get_payload(decode=True)
        try:
            body = payload.decode(msg.get_content_charset(failobj='utf-8'), errors='replace')
        except:
            body = payload.decode('utf-8', errors='replace')

    return subject.strip(), body.strip()


def predict_phishing_probability(subject: str, body: str) -> float:
    text = (subject + ' ' + body).strip()
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        # Index 1 → “phishing” class
        proba_phishing = probs[:, 1].item()

    return proba_phishing


# ==========================
#  API Endpoints
# ==========================
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Expected a JSON payload with key 'raw_email'."}), 400

    data = request.get_json()
    raw_email_str = data.get('raw_email')
    if raw_email_str is None:
        return jsonify({"error": "Missing 'raw_email' in JSON payload."}), 400

    raw_email_bytes = raw_email_str.encode('utf-8')

    try:
        subject, body = parse_raw_email(raw_email_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to parse the raw email: {str(e)}"}), 500

    try:
        proba = predict_phishing_probability(subject, body)
    except Exception as e:
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500

    percentage = round(proba * 100, 2)
    is_phishing = proba >= 0.5

    return jsonify({
        "phishing_probability": percentage,
        "is_phishing": is_phishing
    })


@app.route('/', methods=['GET'])
def home():
    return "phishing-detection API is up and running."


# ==========================
#  Run the server
# ==========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
