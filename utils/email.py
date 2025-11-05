# boilerplate
import os
import base64
from dotenv import load_dotenv
load_dotenv()
# end boilerplate

from mailjet_rest import Client

mailjet_api_key = os.environ.get('MAILJET_API_KEY')
mailjet_secret_key = os.environ.get('MAILJET_SECRET_KEY')
mailjet_from_email = os.environ.get('MAILJET_FROM_EMAIL')
mailjet_from_name = os.environ.get('MAILJET_FROM_NAME')
mailjet_to_email = os.environ.get('MAILJET_TO_EMAIL')
mailjet_to_name = os.environ.get('MAILJET_TO_NAME')

mailjet = Client(auth=(mailjet_api_key, mailjet_secret_key), version='v3.1')

def send_email_notification(subject, text_content, html_content, custom_recipient=None, attachment_path=None):
    # Prepare the base data for the email
    data = {
        'Messages': [
            {
                "From": {
                    "Email": mailjet_from_email,
                    "Name": mailjet_from_name
                },
                "To": [
                    {
                        "Email": custom_recipient or mailjet_to_email,
                        "Name":  mailjet_to_name
                    }
                ],
                "Subject": subject,
                "TextPart": text_content,
                "HTMLPart": html_content
            }
        ]
    }

    # If an attachment is provided, add it to the email data
    if attachment_path:
        with open(attachment_path, "rb") as file:
            attachment_content = file.read()
            data['Messages'][0]['Attachments'] = [
                {
                    "ContentType": "image/png",
                    "Filename": os.path.basename(attachment_path),
                    "Base64Content": base64.b64encode(attachment_content).decode('utf-8')
                }
            ]

    # Send the email
    result = mailjet.send.create(data=data)
    if result.status_code == 200:
        print("Email sent successfully.")
    else:
        print(f"Failed to send email. Status code: {result.status_code}, Error: {result.json()}")
