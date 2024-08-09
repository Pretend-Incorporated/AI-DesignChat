import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from flask import Flask, render_template, request, jsonify

nltk.download('punkt')

stemmer = PorterStemmer()

# Load dataset
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Tokenization and stemming
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

all_words = []
tags = []
xy = []

# Process intents
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
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

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
model = NeuralNet(input_size, hidden_size, output_size)

# Training parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        outputs = model(torch.from_numpy(X_batch))
        loss = criterion(outputs, torch.from_numpy(y_batch).long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')
torch.save(model.state_dict(), 'model.pth')

# Define Flask app
app = Flask(__name__)

# Define function to mask PII
def mask_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b'
    cc_pattern = r'\b\d{4}(| |-)\d{4}\1\d{4}\1\d{4}\b'

    text = re.sub(email_pattern, '[EMAIL REDACTED]', text)
    text = re.sub(phone_pattern, '[PHONE REDACTED]', text)
    text = re.sub(cc_pattern, '[CREDIT CARD REDACTED]', text)
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json['message']
    sanitized_msg = mask_pii(msg)
    
    tokenized_msg = tokenize(sanitized_msg)
    X = bag_of_words(tokenized_msg, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    for intent in intents['intents']:
        if tag == intent['tag']:
            response = random.choice(intent['responses'])
            break

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)