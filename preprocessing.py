import json
import re

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()                  # Chuyển thành chữ thường
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

def load_data_and_build_vocab(json_path):
    """
    Đọc file data.json, trả về:
      - data (toàn bộ dữ liệu raw)
      - vocab (đối tượng Vocabulary)
      - intent_labels (dict: intent -> index)
      - label_to_intent (dict: index -> intent)
      - train_data (danh sách (list of tuples): (câu đã chuyển thành token-index, nhãn))
    """
    # Đọc file JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tạo đối tượng Vocabulary
    vocab = Vocabulary()
    vocab.add_word('<unk>')  # Thêm token <unk>

    # Duyệt qua tất cả patterns để xây dựng từ điển
    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern_processed = preprocess_text(pattern)
            for word in tokenize(pattern_processed):
                vocab.add_word(word)

    # Tạo nhãn cho mỗi intent
    intent_labels = {intent['tag']: i for i, intent in enumerate(data['intents'])}
    label_to_intent = {v: k for k, v in intent_labels.items()}

    # Tạo dữ liệu huấn luyện: (list of (input_indices, label))
    train_data = []
    for intent in data['intents']:
        label_id = intent_labels[intent['tag']]
        for pattern in intent['patterns']:
            pattern_processed = preprocess_text(pattern)
            sentence_indices = [
                vocab.word2idx.get(word, vocab.word2idx['<unk>'])
                for word in tokenize(pattern_processed)
            ]
            train_data.append((sentence_indices, label_id))

    return data, vocab, intent_labels, label_to_intent, train_data
