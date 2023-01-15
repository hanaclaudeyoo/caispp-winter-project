# -*- coding: utf-8 -*-
"""Winter Project News Sarcasm Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11jpjxnV4lb5WvJ8XDWDnNlyu0flh_YSh

**LOAD DATA**
"""

# setup for data import
! pip install kaggle  # install kaggle library
! mkdir ~/.kaggle # make directory
! cp kaggle.json ~/.kaggle/ #copy kaggle.json into dir
! chmod 600 ~/.kaggle/kaggle.json #allocate permissions

# download dataset
! kaggle datasets download rmisra/news-headlines-dataset-for-sarcasm-detection

! unzip news-headlines-dataset-for-sarcasm-detection.zip

import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

"""**DATA PREPROCESSING**"""

import pandas

dataframe = pandas.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
dataframe.head()

headlines = dataframe.headline.values
labels = dataframe.is_sarcastic.values

"""TOKENIZATION"""

!pip install transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(' Original: ', headlines[0])
print('Tokenized: ', tokenizer.tokenize(headlines[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(headlines[0])))

import torch

input_ids = []
attention_masks = []

# find maximum length headline
max_len = 0

print(len(headlines[0]))

for hl in headlines:
    max_len = max(max_len, len(hl))

print('Max sentence length: ', max_len)   #gives letter count, not word count?

#fill input_ids

for hl in headlines:
  encoded_dict = tokenizer.encode_plus(
        hl,
        add_special_tokens = True,
        max_length = max_len,
        padding='max_length',
        return_attention_mask = True,
        return_tensors = 'pt'
    )
  input_ids.append(encoded_dict['input_ids'])
  attention_masks.append(encoded_dict['attention_mask'])

#convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

print('Original: ', headlines[0])
print('Token IDs:', input_ids[0])

"""**HYPERPARAMETERS**"""

data_portion = 0.2
learning_rate = 1e-5
batch_size = 16
epochs = 2

"""DATA SPLITTING"""

from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(input_ids, attention_masks, labels)

#reduce dataset size
dataset, part = random_split(dataset, [data_portion, 1-data_portion])
print("total: ", len(dataset), "samples")

#create 90-10 train-validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

#randomly divide dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

"""**LOAD BERT**"""

#iterator for dataset - saves memory during training
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

train_dataloader = DataLoader(
      train_dataset,
      sampler = RandomSampler(train_dataset),
      batch_size = batch_size
  )
validation_dataloader = DataLoader(
      val_dataset,
      sampler = SequentialSampler(val_dataset),
      batch_size = batch_size
  )

from transformers import  BertForSequenceClassification, AdamW

model =  BertForSequenceClassification.from_pretrained("bert-base-uncased")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

from transformers import get_linear_schedule_with_warmup

total_steps = batch_size * epochs   # num batches x num epochs

scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps = 0,
      num_training_steps = total_steps
  )

"""HELPER FUNCTIONS"""

#helper function: compute accuracy
import numpy as np

def flat_accuracy(predictions, labels):
  pred_flat = np.argmax(predictions, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

#helper function: formatting time
import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

"""**TRAINING**"""

import random
import numpy as np

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
  #full pass over training set
  print("==== Epoch ", epoch_i+1, "====")

  t0 = time.time()
  total_train_loss = 0
  model.train()

  #for each batch
  for step, batch in enumerate(train_dataloader):
    #   TRAINING
    #progress update
    if step%40==0 and not step==0:
      elapsed = format_time(time.time() - t0)
      print("  Batch ", step, "of ", len(train_dataloader), ", elapsed: ", elapsed)
    
    #unpack training batch
    b_input_ids = batch[0]
    b_input_mask = batch[1]
    b_labels = batch[2]

    model.zero_grad()

    #forward pass
    model = model.to(device)
    result = model(b_input_ids.cuda(),
                   token_type_ids = None,
                   attention_mask = b_input_mask.cuda(),
                   labels = b_labels.cuda(),
                   return_dict = True)
    loss = result.loss
    logits = result.logits
    total_train_loss += loss.item()

    #backwards pass
    loss.backward()

    #prevent exploding gradients: clip gradient norm to 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #update
    optimizer.step()
    scheduler.step()
  
  avg_train_loss = total_train_loss / len(train_dataloader)
  training_time = format_time(time.time() - t0)
  print("Avg training loss: ", avg_train_loss)
  print("Training epoch took: ", training_time)

  # VALIDATION
  print("  validation")
  t0 = time.time()
  model.eval()

  total_eval_accuracy = 0
  total_eval_loss = 0
  nb_eval_steps = 0

  for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
      #forward pass
      model = model.to(device)
      result = model(b_input_ids,
                     token_type_ids = None,
                     attention_mask = b_input_mask,
                     labels = b_labels,
                     return_dict = True)
    loss = result.loss
    logits = result.logits

    total_eval_loss += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    total_eval_accuracy += flat_accuracy(logits, label_ids)
  
  avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
  avg_val_loss = total_eval_loss / len(validation_dataloader)
  validation_time = format_time(time.time() - t0)
  print("Accuracy: ", avg_val_accuracy)
  print("Validation Loss: ", avg_val_loss)
  print("Validation took: ", validation_time)

  training_stats.append({
        'epoch': epoch_i + 1,
        'Training Loss': avg_train_loss,
        'Valid. Loss': avg_val_loss,
        'Valid. Accur.': avg_val_accuracy,
        'Training Time': training_time,
        'Validation Time': validation_time
    })

print("COMPLETE")
print("total training time: ", format_time(time.time() - total_t0))

import pandas as pd

pd.set_option('precision', 3)

df_stats = pd.DataFrame(data=training_stats)

df_stats = df_stats.set_index('epoch')

df_stats