import json
from text_preprocessing import tokenize, stem, bag_of_words
from model import Net
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

with open('intents.json', 'r') as f:
	intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:

	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w, tag))

ignore_words = ['?', '.', ',', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:

	bag = bag_of_words(pattern_sentence, all_words)
	X_train.append(bag)

	label = tags.index(tag)
	y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)

class dataset(Dataset):

	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = y_train

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples

dataset = dataset()
train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)

input_size, hidden_size, output_size = len(X_train[0]), 512, len(tags)

model = Net(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
	for (words, labels) in train_loader:
		words = words.to(device)
		labels = labels.to(device)

		y_pred = model(words)
		loss = criterion(y_pred, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if epoch % 25 == 0:

		tb.add_scalar("loss", loss.item(), epoch)
		tb.add_graph(model, words)

		print(f'epoch {epoch}, loss={loss.item()}')

print(f'loss={loss.item()}')

data = {
	"model_state": model.state_dict(),
	"input_size": input_size,
	"output_size": output_size,
	"hidden_size": hidden_size,
	"all_words": all_words,
	"tags": tags,
}

file = "model.pth"
torch.save(data, file)

print("model saved")

tb.close()