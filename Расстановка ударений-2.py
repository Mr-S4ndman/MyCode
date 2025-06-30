import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pymorphy2
from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv("silero-stress-predictor//train.csv")
test_df = pd.read_csv("silero-stress-predictor//test.csv")

train_df.dropna(subset=["word", "stress", "num_syllables"], inplace=True)
test_df.dropna(subset=["word", "num_syllables"], inplace=True)
y = train_df["stress"] - 1

def extract_basic_numeric_features(df):
    df = df.copy()
    df["word_length"] = df["word"].apply(len)
    vowels = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
    df["vowel_count"] = df["word"].apply(lambda w: sum(ch in vowels for ch in w))
    df["vowel_ratio"] = df["vowel_count"] / df["word_length"]
    df["is_monosyllabic"] = (df["num_syllables"] == 1).astype(int)
    return df

train_df = extract_basic_numeric_features(train_df)
test_df = extract_basic_numeric_features(test_df)

morph = pymorphy2.MorphAnalyzer()

def extract_morph_features(df):
    df = df.copy()
    pos_list, gender_list, number_list, case_list, person_list = [], [], [], [], []
    for word in df["word"]:
        p = morph.parse(word)[0]
        tag = p.tag
        pos_list.append(tag.POS or "UNKN")
        if "masc" in tag:
            gender_list.append("masc")
        elif "femn" in tag:
            gender_list.append("femn")
        elif "neut" in tag:
            gender_list.append("neut")
        else:
            gender_list.append("UNKN")
        if "sing" in tag:
            number_list.append("sing")
        elif "plur" in tag:
            number_list.append("plur")
        else:
            number_list.append("UNKN")
        c = "UNKN"
        for c_try in ["nomn","gent","datv","accs","ablt","loct","voct","gen2","acc2","loc2"]:
            if c_try in tag:
                c = c_try
                break
        case_list.append(c)
        if "1per" in tag:
            person_list.append("1per")
        elif "2per" in tag:
            person_list.append("2per")
        elif "3per" in tag:
            person_list.append("3per")
        else:
            person_list.append("UNKN")
    df["morph_pos"] = pos_list
    df["morph_gender"] = gender_list
    df["morph_number"] = number_list
    df["morph_case"] = case_list
    df["morph_person"] = person_list
    return df

train_df = extract_morph_features(train_df)
test_df = extract_morph_features(test_df)

morph_cols = ["morph_pos", "morph_gender", "morph_number", "morph_case", "morph_person"]
combined_df = pd.concat([train_df, test_df], keys=["train","test"])
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_morph = encoder.fit_transform(combined_df[morph_cols])
encoded_morph_df = pd.DataFrame(encoded_morph, columns=encoder.get_feature_names_out(morph_cols), index=combined_df.index)
combined_df = pd.concat([combined_df, encoded_morph_df], axis=1)
train_df = combined_df.loc["train"].copy()
test_df = combined_df.loc["test"].copy()

base_numeric_cols = ["word_length", "vowel_count", "vowel_ratio", "is_monosyllabic"]
morph_ohe_cols = list(encoded_morph_df.columns)
all_numeric_cols = base_numeric_cols + morph_ohe_cols
X_train_numeric = train_df[all_numeric_cols].values
X_test_numeric = test_df[all_numeric_cols].values

all_text = "".join(train_df["word"].astype(str).values)
vocab = sorted(list(set(all_text)))
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
vocab = [PAD_TOKEN, UNK_TOKEN] + vocab
char2idx = {ch: i for i, ch in enumerate(vocab)}

max_len = max(train_df["word"].apply(len).max(), test_df["word"].apply(len).max())

def encode_word(word):
    indices = []
    for ch in word:
        indices.append(char2idx[ch] if ch in char2idx else char2idx[UNK_TOKEN])
    if len(indices) < max_len:
        indices += [char2idx[PAD_TOKEN]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

X_encoded = np.array([encode_word(w) for w in train_df["word"]])
X_test_encoded = np.array([encode_word(w) for w in test_df["word"]])

X_train_seq, X_val_seq, X_train_num, X_val_num, y_train, y_val = train_test_split(
    X_encoded, X_train_numeric, y, test_size=0.2, random_state=42, stratify=y
)

num_classes = 6
embedding_dim = 64
conv_filters = 64
kernel_size = 3
rnn_units_1 = 128
rnn_units_2 = 128
dense_numeric = 32
dense_merged_1 = 64
dense_merged_2 = 32
dropout_rate = 0.2
batch_size = 64
epochs = 23

input_seq = keras.Input(shape=(max_len,), name="char_seq")
x = layers.Embedding(input_dim=len(vocab), output_dim=embedding_dim, input_length=max_len)(input_seq)
x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Bidirectional(layers.LSTM(rnn_units_1, return_sequences=True))(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Bidirectional(layers.LSTM(rnn_units_2, return_sequences=False))(x)
x = layers.Dropout(dropout_rate)(x)

input_feats = keras.Input(shape=(X_train_num.shape[1],), name="numeric_feats")
y2 = layers.Dense(dense_numeric, activation="relu")(input_feats)
y2 = layers.Dropout(dropout_rate)(y2)
merged = layers.Concatenate()([x, y2])
z = layers.Dense(dense_merged_1, activation="relu")(merged)
z = layers.Dropout(dropout_rate)(z)
z = layers.Dense(dense_merged_2, activation="relu")(z)
z = layers.Dropout(dropout_rate)(z)
z = layers.Dense(num_classes, activation="softmax")(z)
model = keras.Model(inputs=[input_seq, input_feats], outputs=z)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    [X_train_seq, X_train_num],
    y_train,
    validation_data=([X_val_seq, X_val_num], y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

plt.plot(history.history["loss"], marker="o", label="Train Loss")
plt.plot(history.history["val_loss"], marker="o", label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], marker="o", label="Train Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Val Accuracy")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

val_preds = model.predict([X_val_seq, X_val_num]).argmax(axis=1)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")

test_preds = model.predict([X_test_encoded, X_test_numeric]).argmax(axis=1)
test_preds = test_preds + 1
submission = pd.DataFrame({"id": test_df["id"], "stress": test_preds})
submission.to_csv("silero-stress-predictor//submission.csv", index=False)
print("Файл с предсказаниями сохранён в silero-stress-predictor//submission.csv")
