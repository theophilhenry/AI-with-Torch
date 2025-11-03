import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS
from utils import load_data, line_to_tensor, random_training_example

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, n_class):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size

    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, n_class)
    self.softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, input_tensor, hidden_tensor):
    input_tensor = input_tensor.unsqueeze(0) # Say that it's 1 batch
    combined = torch.cat((input_tensor, hidden_tensor), 1)
    
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    output = self.softmax(output)
    return output, hidden
  
  def init_hidden(self):
    return torch.zeros(1, self.hidden_size)

category_lines, all_categories = load_data()
n_class = len(all_categories)
n_hidden = 128
# len(ALL_LETTERS) : 55
rnn = RNN(len(ALL_LETTERS), n_hidden, n_class)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.005)

sum_of_losses = 0
all_losses = []

for i in range(10000):
  # "English", "Brudah", "tensor(2)", "[[0, 0, 0, ...], [], []]"<6 (brudah), 55 (all_chars)>
  category, line, category_idx_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories=all_categories)

  # Reset hidden to list of 1s
  # [[1, 1, 1, 1, 1, ...<128>]]
  hidden = rnn.init_hidden()

  # Loop through each alphabet
  for t in range(line_tensor.size()[0]):
    alphabet_tensor = line_tensor[t]
    raw_softmax_output, new_hidden = rnn(alphabet_tensor, hidden)
    hidden = new_hidden # Update the hidden layer values

  loss = criterion(raw_softmax_output, category_idx_tensor)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  sum_of_losses += loss.item()
  
  if (i+1) % 1000 == 0:
    all_losses.append(sum_of_losses / 1000)
    sum_of_losses = 0

  if (i+1) % 5000 == 0:
    category_idx = torch.argmax(raw_softmax_output).item()
    guess = all_categories[category_idx]
    correct = "CORRECT" if guess == category else f"WRONG ({category})"
    print(f"{i+1} {(i+1)/10000*100} {loss:.4f} {line} / {guess} {correct}")

print(all_losses)
# plt.figure()
# plt.plot(all_losses)
# plt.show()

def predict(sentence):
  with torch.no_grad():
    line_tensor = line_to_tensor(sentence)
    hidden = rnn.init_hidden()

    for t in range(line_tensor.shape[0]):
      alphabet_tensor = line_tensor[t]
      raw_softmax_output, new_hidden = rnn(alphabet_tensor, hidden)
      hidden = new_hidden

    category_idx = torch.argmax(raw_softmax_output).item()
    guess = all_categories[category_idx]
    print(guess)

while True:
  sentence = input("Input name: ")
  if sentence == "quit":
    break
  predict(sentence)