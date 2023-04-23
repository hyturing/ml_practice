# Load the pre-trained BERT model
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Move the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare the data


df = pd.read_csv('preprocess.csv')
df = df.head(5000)
X = df['PRODUCT'].tolist()
y = df['PRODUCT_LENGTH'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tokenize the input
encoded_train = tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')
encoded_test = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')

# Prepare the input
input_ids_train = encoded_train['input_ids'].to(device)
attention_masks_train = encoded_train['attention_mask'].to(device)
segment_ids_train = encoded_train['token_type_ids'].to(device)

input_ids_test = encoded_test['input_ids'].to(device)
attention_masks_test = encoded_test['attention_mask'].to(device)
segment_ids_test = encoded_test['token_type_ids'].to(device)

# Convert y_train to tensor and move to GPU
y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)

# Move optimizer to GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model on the GPU
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids_train, attention_masks_train, segment_ids_train, labels=y_train_tensor)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Evaluate the model on the GPU


model.eval()
with torch.no_grad():
    outputs = model(input_ids_test, attention_masks_test, segment_ids_test)
    y_pred = outputs.logits.squeeze(1).tolist()
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

mape = mean_absolute_percentage_error(y_test, y_pred)
print('MAPE:', mape)
