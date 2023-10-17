import requests

#For communicating with NLP over HTTP requests
request_url = "http://127.0.0.1:5000/parse_text"

#To be called once the model has found a new letter
class Communication:
    string = ""

    def new_letter(letter):
        string = string + letter
        requests.get(request_url, json={
            "text": string,
            "index": 0,
            "truncate": True,
        })