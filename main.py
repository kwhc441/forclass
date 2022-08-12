from flask import Flask, render_template, request
import markovify


from glob import iglob
import re
import os
import MeCab
tagger = MeCab.Tagger()
book_list=["dogura.txt","ginga.txt","chijin.txt","lemon.txt","kappa.txt","sangetsu.txt"]
img_list=["dogura.png","ginga.png","ai.png","lemon.png","kappa.png","tora.png"]

def load_from_file(files_pattern):
    """read and merge files which matches given file pattern, prepare for parsing and return it.
    """

    # read text
    text = ""
    for path in iglob(files_pattern):
        with open(path, 'r') as f:
            text += f.read().strip()

    # delete some symbols
    unwanted_chars = ['\r', '\u3000', '-', '｜']
    for uc in unwanted_chars:
        text = text.replace(uc, '')

    # delete aozora bunko notations
    unwanted_patterns = [re.compile(r'《.*》'), re.compile(r'［＃.*］')]
    for up in unwanted_patterns:
        text = re.sub(up, '', text)

    return text
def load_from_files(a,b):
    """read and merge files which matches given file pattern, prepare for parsing and return it.
    """
    files=[os.path.abspath(a),os.path.abspath(b)]
    """
    for i in range(2):
       t=file_list[i]
       file_path=os.path.abspath(t)
       files.append(file_path)
"""
      # read text
    text = ""
    for path in files:
        with open(path, 'r') as f:
            text += f.read().strip()

    # delete some symbols
    unwanted_chars = ['\r', '\u3000', '-', '｜']
    for uc in unwanted_chars:
        text = text.replace(uc, '')

    # delete aozora bunko notations
    unwanted_patterns = [re.compile(r'《.*》'), re.compile(r'［＃.*］')]
    for up in unwanted_patterns:
        text = re.sub(up, '', text)

    return text


def split_for_markovify(text):
    """split text to sentences by newline, and split sentence to words by space.
    """
    # separate words using mecab
    mecab = MeCab.Tagger()
    splitted_text = ""

    # these chars might break markovify
    # https://github.com/jsvine/markovify/issues/84
    breaking_chars = [
        '(',
        ')',
        '[',
        ']',
        '"',
        "'",
    ]

    # split whole text to sentences by newline, and split sentence to words by space.
    for line in text.split():
        mp = mecab.parseToNode(line)
        while mp:
            try:
                if mp.surface not in breaking_chars:
                    splitted_text += mp.surface    # skip if node is markovify breaking char
                if mp.surface != '。' and mp.surface != '、':
                    splitted_text += ' '    # split words by space
                if mp.surface == '。':
                    splitted_text += '\n'    # reresent sentence by newline
            except UnicodeDecodeError as e:
                # sometimes error occurs
                print(line)
            finally:
                mp = mp.next

    return splitted_text


def main(a,b):
    # load text
    #file_list=[a,b]
    #rampo_text = load_from_file("*txt")
    rampo_text = load_from_files(a,b)

    # split text to learnable form
    splitted_text = split_for_markovify(rampo_text)

    # learn model from text.
    text_model = markovify.NewlineText(splitted_text, state_size=3)

    # ... and generate from model.
    sentence = text_model.make_sentence()
    print(''.join(sentence.split()))    # need to concatenate space-splitted text

    # save learned data
    with open('task_learned_data.json', 'w') as f:
        f.write(text_model.to_json())

    # later, if you want to reuse learned data...
    """
    with open('learned_data.json') as f:
        text_model = markovify.NewlineText.from_json(f.read())
    """

# Faskのインスタンスを作成
app = Flask(__name__,static_folder='./templates/images')

# ルーティングの指定
@app.route('/')
def index():
  return render_template("index.html")

# ルーティングの指定
@app.route('/output',)
def output():
  book1=request.args.get("book1")
  book2=request.args.get("book2")
  book1=int(book1)
  book2=int(book2)
  if book1==book2:
    book2=0
  #fusion_books=[book_list[book1],book_list[book2]]
  main(book_list[book1],book_list[book2])
  with open('task_learned_data.json') as f:
    text_model = markovify.NewlineText.from_json(f.read())

# ... and generate from model.
# 10回くりかえす。（max10行）
  output_list=[]
  img1 = "images/"+img_list[book1]
  img2 = "images/"+img_list[book2]
  for i in range(10):
    sentence = text_model.make_sentence()
    if sentence != None:
      output_list.append(''.join(sentence.split()))
  return render_template("output.html",output_list=output_list,img1=img1,img2=img2)

# デバッグモードでサーバを起動させる
app.run(debug=True, host='0.0.0.0')