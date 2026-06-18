import json
from model import *

file = open("train.json", "r")

Text = []
Label = []
for line in file:
    data = json.loads(line)
    Text.append(data["words"])
    Label.append(data["labels"])
file.close()

tagger = POSTagger()

tagger.fit(Text, Label)

tagger.save()