from flask import Flask
from flask import request
app = Flask(__name__)

import autospace_crf
import pycrfsuite

model_file = 'corpus/MDM001_FullKorean2.crfsuite'
tagger = pycrfsuite.Tagger()
tagger.open(model_file)

def run_demo(text):
    if not text:
        return 'err: ' + text

    one_corpus = autospace_crf.raw2corpus(text)
    one_sent = autospace_crf.corpus2sent(one_corpus)
    one_crf_input = autospace_crf.sent2crfinput(one_sent)

    xseq, _ = one_crf_input
    pred_y = tagger.tag(xseq)

    result = ''
    for x, y in zip(text.replace(' ', ''), pred_y):
        if y == 'B':
            result += ' {}'.format(x)
        else:
            result += x
    result = result.strip()
    return '원문: {}<br>\n결과: {}'.format(text, result)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/demo', methods=['GET'])
def demo():
    try:
        #text = request.form['text']
        text = request.args.get('text', '')
        return run_demo(text)
    except:
        return 'err'

if __name__ == '__main__':
    app.run(host='0.0.0.0')

