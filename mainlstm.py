from flask import Flask, render_template, request
tagger = MeCab.Tagger()
book_list = [
    "dogura.txt", "ginga.txt", "chijin.txt", "lemon.txt", "kappa.txt",
    "sangetsu.txt"
]
img_list = [
    "dogura.png", "ginga.png", "ai.png", "lemon.png", "kappa.png", "tora.png"
]

import re
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import LambdaCallback


#テキスト前処理
def text_change(text_path):
    with open(text_path, mode="r") as f:  # ファイルの読み込み
        text_original = f.read()

    # 本文前の注釈にタグを埋め込んで、そこを元に本文を抽出
    text = re.sub(r'--+', 'タグを埋め込みます', text_original)
    text = text.split('タグを埋め込みます')[-1]

    #前処理として不要な文字を削除する
    text = re.sub("《[^》]+》", "", text)  # ルビの削除
    text = re.sub(r"［.+?］", "", text)  # （追加）本文中にある注釈や解説を削除
    text = re.sub("［[^］]+］", "", text)  # 読みの注意の削除
    text = re.sub("[｜ 　]", "", text)  # | と全角半角スペースの削除
    text = text.split("底本")[0]  # （追加）出版社や作成日などの情報を削除
    return text


def texts_change(a, b):
    a,b=os.path.abspath(a),os.path.abspath(b)
    text1 = text_change(a)
    text2 = text_change(b)
    text = text1 + text2
    return text


#print("文字数", len(text))  # len() で文字列の文字数も取得可能
#LSTMの設定
n_rnn = 8  # 時系列の数
batch_size = 128
epochs = 60  #epochsは、多いほど、精密に学習するが、重くなるため今回は小さくしている
n_mid = 256  # 中間層のニューロン数


#文字のベクトル化
def change_vector(text, n_rnn):
    chars = sorted(list(set(text)))  # setで文字の重複をなくし、各文字をリストに格納する
    #print("文字数（重複無し）", len(chars))
    char_indices = {}  # 文字がキーでインデックスが値
    for i, char in enumerate(chars):
        char_indices[char] = i
    indices_char = {}  # インデックスがキーで文字が値
    for i, char in enumerate(chars):
        indices_char[i] = char

    # 時系列データと、それから予測すべき文字を取り出す
    time_chars = []
    next_chars = []
    for i in range(0, len(text) - n_rnn):
        time_chars.append(text[i:i + n_rnn])
        next_chars.append(text[i + n_rnn])

    # 入力と正解をone-hot表現で表す。１文字毎に0,1のベクトルをフルイメージです。
    x = np.zeros((len(time_chars), n_rnn, len(chars)), dtype=np.bool)
    t = np.zeros((len(time_chars), len(chars)), dtype=np.bool)
    for i, t_cs in enumerate(time_chars):
        t[i, char_indices[next_chars[i]]] = 1  # 正解をone-hot表現で表す
        for j, char in enumerate(t_cs):
            x[i, j, char_indices[char]] = 1  # 入力をone-hot表現で表す
    return x, t, chars, char_indices, indices_char


#print("xの形状", x.shape)
#print("tの形状", t.shape)
#LSTMのモデル構築
def make_model(chars):
    model_lstm = Sequential()
    model_lstm.add(LSTM(n_mid, input_shape=(n_rnn, len(chars))))
    model_lstm.add(Dense(len(chars), activation="softmax"))
    model_lstm.compile(loss='categorical_crossentropy', optimizer="adam")
    return model_lstm


#print(model_lstm.summary())

#文章生成用の関数定義

#学習開始

#elapsed_time = time.time() - start
#print ("学習開始 elapsed_time:{0}".format(elapsed_time) + "[sec]")

# Faskのインスタンスを作成
app = Flask(__name__, static_folder='./templates/images')


# ルーティングの指定
@app.route('/')
def index():
    return render_template("index.html")


# ルーティングの指定
@app.route(
    '/output', )
def output():
    book1 = request.args.get("book1")
    book2 = request.args.get("book2")
    book1 = int(book1)
    book2 = int(book2)
    if book1 == book2:
        book2 = 0
    #fusion_books=[book_list[book1],book_list[book2]]
    text = texts_change(book_list[book1], book_list[book2])
    x, t, chars, char_indices, indices_char = change_vector(text, n_rnn)
    model_lstm = make_model(chars)
    model = model_lstm

    def on_epoch_end(epoch, logs):
        #print("エポック: ", epoch)

        #elapsed_time = time.time() - start
        #print ("on_epoch_end  elapsed_time:{0}".format(elapsed_time) + "[sec]")

        beta = 4  # 確率分布を調整する定数
        prev_text = text[0:n_rnn]  # 入力に使う文字
        created_text = prev_text  # 生成されるテキスト

        #print("シード: ", created_text)

        for i in range(500):
            # 入力をone-hot表現に
            x_pred = np.zeros((1, n_rnn, len(chars)))
            for j, char in enumerate(prev_text):
                x_pred[0, j, char_indices[char]] = 1

            # 予測を行い、次の文字を得る
            # yの形状は、1列 1049行(文字数=出力層の数)になっている
            y = model.predict(x_pred)
            #print(y.shape )
            p_power = y[0]**beta  # 確率分布の調整(1049個の配列の中から、確率が高い文字を取得しようとしている　)
            next_index = np.random.choice(len(p_power),
                                          p=p_power / np.sum(p_power))
            next_char = indices_char[next_index]

            created_text += next_char
            prev_text = prev_text[1:] + next_char
        #return created_text
        #print(created_text)
        #print()

    epock_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    history_lstm = model_lstm.fit(x,
                                  t,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[epock_end_callback])
    # ... and generate from model.
    # 10回くりかえす。（max10行）
    output_list = history_lstm
    img1 = "images/" + img_list[book1]
    img2 = "images/" + img_list[book2]

    return render_template("output.html",
                           output_list=output_list,
                           img1=img1,
                           img2=img2)


# デバッグモードでサーバを起動させる
app.run(debug=True, host='0.0.0.0')
