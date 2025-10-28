import io
import os
import unicodedata
import string
import glob

import torch
import random

ALL_LETTERS = string.ascii_letters + ".,;"

def unicode_to_ascii(s):
  characters = unicodedata.normalize('NFD', s)
  ascii = []
  for c in characters:
    if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS:
      ascii.append(c)
  return ''.join(ascii)

# Returns
# (
#   {"Arabic": ["Khouri", "Nahas", "..."], "English": ["..."]}
#   ["Arabic", "English", "..."], (all_categories)
# )
def load_data():
  category_lines = {}
  all_categories = []

  def find_files(path):
    return glob.glob(path)
  
  def read_lines(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]
  
  for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0] # Arabic, English, etc...
    all_categories.append(category)

    lines = read_lines(filename)
    category_lines[category] = lines

  return category_lines, all_categories


# Represent single letter to one-hot-vector, ex:
# "a" -> "1 0 0 0 0 ..."
# "b" -> "0 1 0 0 0 ..."
# 1 is decided based on it's position int he ALL_LETTERS
# To make the word, we join them into 2d matrix <line_length x 1 x 26 (n letters)>
# Extra 1 dimension? PyTorch assume everything in batches, we use batch size of 1 here.

def _letter_to_index(letter):
  return ALL_LETTERS.find(letter)

# This will return
# tensor( [ [0, 0, 0, 0, 0, 1 ...] ] )
# DEMONSTRATION PURPOSES
# def letter_to_tensor(letter):
#   tensor = torch.zeros(1, len(ALL_LETTERS))
#   tensor[0][letter_to_index(letter)] = 1
#   return tensor

def line_to_tensor(line):
  tensor = torch.zeros(len(line), len(ALL_LETTERS))
  for i, letter in enumerate(line):
    tensor[i][_letter_to_index(letter)] = 1
  return tensor

def random_training_example(category_lines, all_categories):
  def random_choice(a):
    random_idx = random.randint(0, len(a) - 1)
    return a[random_idx]

  category = random_choice(all_categories)
  category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
  
  line = random_choice(category_lines[category])
  line_tensor = line_to_tensor(line)

  return category, line, category_tensor, line_tensor

if __name__ == '__main__':
  # print(unicode_to_ascii('Ślusàrski'))
  # a, c = load_data()
  # category, line, category_tensor, line_tensor = random_training_example(a, c)
  # print(line)
  # print(line_tensor.shape)
  
  print(unicode_to_ascii("Mister Good"))