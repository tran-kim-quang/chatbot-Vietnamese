import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    return text

# Hàm token hóa văn bản
def tokenize(text):
    return text.split()

# Tạo từ điển (Vocabulary)
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

# Đọc dữ liệu từ file JSON
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Khởi tạo từ điển và thêm các token cần thiết
vocab = Vocabulary()
vocab.add_word('<unk>')

# Duyệt qua intents để xây dựng từ điển
for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = preprocess_text(pattern)
        for word in tokenize(pattern):
            vocab.add_word(word)

# Tạo nhãn cho mỗi intent
intent_labels = {intent['tag']: i for i, intent in enumerate(data['intents'])}
label_to_intent = {i: tag for tag, i in intent_labels.items()}

# Tạo dữ liệu huấn luyện
train_data = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        words = [vocab.word2idx.get(w, vocab.word2idx['<unk>']) 
                 for w in tokenize(preprocess_text(pattern))]
        train_data.append((words, intent_labels[intent['tag']]))

# Định nghĩa mô hình phân loại Intent
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_classes):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)      # [batch_size, seq_length, embedding_dim]
        _, (hidden, _) = self.rnn(embedded)
        # hidden[-1]: [batch_size, hidden_size]
        output = self.fc(hidden[-1])      # [batch_size, num_classes]
        return output

# Các thông số mô hình
input_size = len(vocab)
hidden_size = 128
embedding_dim = 64
num_classes = len(intent_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IntentClassifier(input_size, hidden_size, embedding_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for sentence, label in train_data:
        # sentence là list chỉ mục, ta cần chuyển sang Tensor và thêm batch dimension
        sentence_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(device)
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)

        # Forward
        output = model(sentence_tensor)
        loss = criterion(output, label_tensor)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_data)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Hàm dự đoán intent và chọn phản hồi ngẫu nhiên
def predict_intent(sentence):
    model.eval()
    # Tiền xử lý, chuyển thành chỉ mục
    sentence = preprocess_text(sentence)
    sentence_indices = [vocab.word2idx.get(w, vocab.word2idx['<unk>']) 
                        for w in tokenize(sentence)]
    sentence_tensor = torch.tensor(sentence_indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sentence_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # Tìm intent tương ứng và trả về câu trả lời
    intent_tag = label_to_intent[predicted_label]
    for intent in data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Xin lỗi, tôi không hiểu."

# Giao tiếp với Chatbot
print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() == 'exit':
        break
    response = predict_intent(user_input)
    print("Chatbot:", response)
