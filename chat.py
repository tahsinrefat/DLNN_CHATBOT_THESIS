import random
import json
import torch
import time

import speech_recognition as sr     #for s2t
import mutagen
from mutagen.mp3 import MP3

from cgitb import text      
from gtts import gTTS       #for t2s
import  os
import math                      

from model import NeuralNet
from nltk_utils import bag_of_words,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('dataset.json','r') as f:
    dataset = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]



model = NeuralNet(input_size,hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "BuddyBot"
print("Let's Chat! Type 'Quit' to exit")
while True:
    r = sr.Recognizer()
    text = ""
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Speak Now...")
        
        audio = r.listen(source)
        
        try:
            text = r.recognize_google(audio)
            print("You have said: \n"+text)
        except Exception as e:
            print("Error: "+ str(e))
    sentence = text
    print("You: "+text)
    if sentence.lower() == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    out_text = ""
    
    if prob.item()>0.75:
        for intent in dataset["intents"]:
            if tag == intent["tag"]:
                out_text = random.choice(intent['responses'])
                print(f"{bot_name}: {out_text}")
    else:
        out_text = "Sorry but I couldn't quite catch that"
        print(f'{bot_name}: I do not understand...')
    language = "en"

    output = gTTS (text = out_text, lang = language, slow = False)
    output.save("output.mp3")
    audio = MP3("output.mp3")
    os.system("xdg-open output.mp3")
    time.sleep(math.ceil(audio.info.length))