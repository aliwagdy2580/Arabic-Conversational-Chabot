from Building_dnn_model.load_model import DNN_response
from chatbot_BERT2BERT_model.BERT2BERT_model import BERT_response

def question(sent):
    responce=DNN_response(sent)
    if responce=='sorry':
        return BERT_response(sent)
    else:
        return responce

#print(question("ماهى وحبتك المفضله"))



from gtts import gTTS
from playsound import playsound

import speech_recognition as sr
import os
# Record Audio

tts = gTTS(text="مرحبا ياعزيزي أنا صديقك الجديد", lang='ar',slow=False)
tts.save("good.mp3")
playsound("good.mp3")
os.remove("good.mp3")
said=""
while said!= "باي":
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("قل شيئا!")
        audio = r.listen(source, phrase_time_limit=6)

    # Speech recognition using Google Speech Recognition

    said = r.recognize_google(audio, language="ar-AR")
    try:
        print("انت قلت: " + said)
    except sr.UnknownValueError:
        print("انه لا يفهمك")


    tts = gTTS(text=question(said), lang='ar',slow=False)

    tts.save("good.mp3")
    playsound("good.mp3")
    os.remove("good.mp3")
    #print(question(said))









'''
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/chat', methods=['GET', 'POST'])
def chatBot():
    chatInput = request.json['chatInput']
    print(chatInput)
    return jsonify(chatBotReply= question(chatInput))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    '''
