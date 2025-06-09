import sys
import os
import requests
import json

def main():
    """
    Usage:
        python send_api.py email.txt

    - email.txt: must exist and contain the entire raw email (RFC-822).
      It may include double quotes, backslashes, newlines, etc.
    - This script reads the file content and sends it to /predict,
      then prints out the JSON response (percentage + boolean).
    """
    if len(sys.argv) != 2:
        print("Usage: python send_api.py <raw_email_file.txt>")
        sys.exit(1)

    email_path = sys.argv[1]
    if not os.path.isfile(email_path):
        print(f"Error: File not found: '{email_path}'")
        sys.exit(1)

    # 1) Read the entire file as text. Use utf-8 and ignore errors.
    with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_email = f.read()

    payload = {
        "raw_email": raw_email
    }

    url = "http://localhost:5000/predict"

    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the server: {e}")
        sys.exit(1)

    # 3) Check HTTP status
    if response.status_code != 200:
        print(f"Server returned HTTP {response.status_code}:")
        print(response.text)
        sys.exit(1)

    # 4) Parse and print JSON response
    try:
        result = response.json()
    except ValueError:
        print("Received a non-JSON response:")
        print(response.text)
        sys.exit(1)

    if "phishing_probability" in result and "is_phishing" in result:
        print(json.dumps(result, indent=4))
    else:
        print("Unexpected JSON response:")
        print(result)


if __name__ == "__main__":
    main()
