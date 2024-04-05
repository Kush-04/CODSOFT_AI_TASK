import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import pickle

# Load pre-trained VGG16 model without top (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Extract features from images using VGG16
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    features = np.reshape(features, features.shape[1:])
    return features

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load maximum sequence length
max_sequence_length = 20

# Define LSTM model
inputs1 = Input(shape=(base_model.output_shape[1],))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_sequence_length,))
se1 = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = tf.keras.layers.add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(len(tokenizer.word_index)+1, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Load weights
model.load_weights('model_weights.h5')

# Generate caption for given image
def generate_caption(image_path):
    in_text = 'startseq'
    for i in range(max_sequence_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_sequence_length)
        yhat = model.predict([extract_features(image_path).reshape(1,-1), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ''
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.split()[1:-1]  # Remove 'startseq' and 'endseq' tokens

# Example usage
image_path = 'example_image.jpg'
caption = generate_caption(image_path)
print(' '.join(caption))
