import requests
import json

#import socketio from main here

#For communicating with NLP over HTTP requests
request_url = "http://127.0.0.1:5000/parse_text"

#To be called once the model has found a new letter
class Communication:
    string = ""

    def new_letter(self, letter):
        self.string = self.string + letter

        input = {
            "text": self.string,
            "index": 0,
            "truncate": True,
        }
        json_input = json.dumps(input)

        response = requests.get(request_url, json=json_input)
        print(response)

    def upload_to_interface(self, text_to_display):
        print("Uploading text to the HoloLens")