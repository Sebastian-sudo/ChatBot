import random
import torch
import json

from model import Net
from text_preprocessing import *

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

with open("intents.json", 'r') as f:
	intents = json.load(f)

data = torch.load("model.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = Net(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

class Bot:

	def __init__(self, model):
		self.model = model

	def chat(self):

		print("Hi, q to exit")
		while (True):

			temp = input(": ")
			if temp == 'q':
				break
			temp = tokenize(temp)
			X = bag_of_words(temp, all_words)
			X = X.reshape(1, X.shape[0])
			X = torch.from_numpy(X)

			output = model(X)
			_, preds = torch.max(output, dim=1)
			tag = tags[preds.item()]

			for intent in intents["intents"]:
				if tag == intent["tag"]:
					print(f"Bot: {random.choice(intent['responses'])}")



if __name__ == '__main__':

	Bot = Bot(model)
	Bot.chat()

