import torch
import torch.nn as nn

class Net(nn.Module):

	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, 2 * hidden_size)
		self.l3 = nn.Linear(2 * hidden_size, hidden_size)
		self.l4 = nn.Linear(hidden_size, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.l1(x)
		x = self.relu(x)
		x = self.l2(x)
		x = self.relu(x)
		x = self.l3(x)
		x = self.relu(x)
		x = self.l4(x)
		return x