import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.feature_extraction.text as sk
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def first_syn_test():
    _, x_train, x_test, y_train, y_test = load_syn_dataset()
    yy_test = y_test
    y_train = to_categorical(y_train, 7)
    y_test = to_categorical(y_test, 7)
    x_train = to_categorical(x_train, 7)
    x_test = to_categorical(x_test, 7)
    print(x_train.shape, y_train.shape)
    model = Sequential()
    model.add(Dense(7, activation='relu'))
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=20, epochs=5, verbose=1, validation_data=(x_test, y_test))
    print(history.history.get('acc'))
    model.predict_classes(x_test, batch_size=32)
    print(model.summary())
    print(model.evaluate(x_test, y_test, verbose=1))


def create_model(max_words, num_classes):
    model = Sequential()
    model.add(Dense(400, input_shape=(max_words,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(350, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def save_model(model):
    ff = model.to_json()
    with open("model.json", "w") as file:
        file.write(ff)
    model.save_weights("model.h5")


def tokenize_transform(max_words, x_train, x_test, y_train, y_test):
    tokenize = Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(x_train)
    x_train_mat = tokenize.texts_to_matrix(x_train)
    x_test_mat = tokenize.texts_to_matrix(x_test)
    labeler = LabelEncoder()
    labeler = labeler.fit(y_train)
    y_train_tran = labeler.transform(y_train)
    y_test_tran = labeler.transform(y_test)
    num_classes = np.max(y_train_tran) + 1
    # convert the vector into binary matrix
    y_train_mat = to_categorical(y_train_tran, num_classes)
    y_test_mat = to_categorical(y_test_tran, num_classes)
    return x_test_mat, x_train_mat, y_test_mat, y_train_mat, num_classes


def graph_conf_mat(predicted, y_test_mat):
    '''
    Creates the confusion matrix for the neural network
    :param predicted: What the model generated
    :param y_test_mat: The test cases
    :return:
    '''
    conf_mat = confusion_matrix(y_test_mat.argmax(axis=1), predicted.argmax(axis=1))
    military_labels = {'BRT': 0, 'BWT': 1, 'SUN': 5, 'ISG': 2, 'RIC': 3, 'SUN/ISG': 6, 'SCT': 4}
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=military_labels, yticklabels=military_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("neural network")
    plt.show()


def run_network():
    '''
    Train and predict using the neural network against the dataset.
    :return:
    '''
    file_loc = "insert_path_to_data"
    df = pd.read_excel(file_loc, sheet_name='Sheet1', na_values=['NA'], usecols="B,E")
    df.fillna('NA', inplace=True)
    df['Thread ID'] = df['Thread ID'].str.replace('\d+', '')
    x_train, x_test, y_train, y_test = train_test_split(df['Unstructured Text'].values, df['Thread ID'], test_size=0.25,
                                                        random_state=1000)
    max_words = 5000
    x_test_mat, x_train_mat, y_test_mat, y_train_mat, num_classes = tokenize_transform(max_words, x_train, x_test,
                                                                                       y_train, y_test)
    model = create_model(max_words, num_classes)
    model.fit(x_train_mat, y_train_mat,
              batch_size=64,
              epochs=5,
              verbose=1, validation_data=(x_test_mat, y_test_mat))
    print(model.evaluate(x_test_mat, y_test_mat, batch_size=64, verbose=1))
    save_model(model)
    pred = model.predict(x_test_mat, batch_size=64)
    graph_conf_mat(pred, y_test_mat)


def load_syn_dataset():
    # returns the label
    file_loc = "insert_path_to_data"
    df = pd.read_excel(file_loc, sheet_name='Sheet1', na_values=['NA'], usecols="B,E")
    df.fillna('NA', inplace=True)

    df['Thread ID'] = df['Thread ID'].str.replace('\d+', '')
    labeler = LabelEncoder()
    tmp = df['Thread ID']
    df['Thread ID'] = labeler.fit_transform(df['Thread ID'])
    print(dict(zip(tmp, df['Thread ID'])))
    vec = sk.TfidfVectorizer(stop_words='english', sublinear_tf=True)
    testy = df['Unstructured Text'].tolist()

    X = vec.fit_transform(testy)
    # return the labels , then the data
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), df['Thread ID'], test_size=0.3, random_state=10)
    return X, X_train, X_test, y_train, y_test


first_syn_test()
run_network()
