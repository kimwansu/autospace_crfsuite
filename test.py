# -*- coding: utf-8 -*-

import re
import codecs
import sys

def raw2corpus_one(raw_sentence):
    taggeds = []
    text = re.sub(r'(\ )+', ' ', raw_sentence).strip()
    for i in range(len(text)):
        if i == 0:
            taggeds.append('{}/B'.format(text[i]))
        elif text[i] != ' ':
            successor = text[i - 1]
            if successor == ' ':
                taggeds.append('{}/B'.format(text[i]))
            else:
                taggeds.append('{}/I'.format(text[i]))

    return ' '.join(taggeds)
#

def raw2corpus(raw_path, corpus_path):
    raw = codecs.open(raw_path, encoding='utf-8')
    raw_sentences = raw.read().split('\n')
    #print('raw_sent[0]=' + raw_sentences[0])
    corpus = codecs.open(corpus_path, 'w', encoding='utf-8')
    sentences = []

    for raw_sentence in raw_sentences:
        if not raw_sentences:
            continue

        taggeds = raw2corpus_one(raw_sentence)
        sentences.append(taggeds)
    corpus.write('\n'.join(sentences))
#

def corpus2raw_one(corpus_sentence):
    text = ''
    taggeds = corpus_sentence.split(' ')
    len_taggeds = len(taggeds)
    for tagged in taggeds:
        try:
            word, tag = tagged.split('/')
            if word and tag:
                if tag == 'B':
                    text += ' ' + word
                else:
                    text += word
        except:
            pass

    return text.strip()
#

def corpus2raw(corpus_path, raw_path):
    corpus = codecs.open(corpus_path, encoding='utf-8')
    corpus_sentences = corpus.read().split('\n')
    raw = codecs.open(raw_path, 'w', encoding='utf-8')
    sentences = []
    for corpus_sentence in corpus_sentences:
        text = corpus2raw_one(corpus_sentence)
        sentences.append(text)

    raw.write('\n'.join(sentences))
#

def corpus2sent_one(raw):
    tokens = raw.split(' ')
    sentence = []
    for token in tokens:
        try:
            word, tag = token.split('/')
            if word and tag:
                sentence.append([word, tag])
        except:
            pass

    return sentence
#

def corpus2sent(path):
    corpus = codecs.open(path, encoding='utf-8').read()
    raws = corpus.split('\n')
    sentences = []
    for raw in raws:
        sentence = corpus2sent_one(raw)
        sentences.append(sentence)
    return sentences
#

def index2feature(sent, i, offset):
    word,tag = sent[i + offset]
    if offset < 0:
        sign = ''
    else:
        sign = '+'
    return '{}{}:word={}'.format(sign, offset, word)
#

def index2feature_v2(sent, i, offsets):
    keys = []
    vals = []

    for offset in offsets:
        word, _ = sent[i + offset]
        if offset < 0:
            sign = '-'
        else:
            sign = '+'



    return '{}={}'.format('|'.join(keys), '|'.join(vals))
#

def word2features(sent, i):
    L = len(sent)
    features = ['bias']
    features.append(index2feature(sent, i, 0))
    if i > 1:
        features.append(index2feature(sent, i, -2))

    if i > 0:
        features.append(index2feature(sent, i, -1))
    else:
        features.append('bos')

    if i < L - 2:
        features.append(index2feature(sent, i, 2))

    if i < L - 1:
        features.append(index2feature(sent, i, 1))
    else:
        features.append('eos')

    return features
#

def word2features_v2(sent, i):
    length = len(sent)
    features = []
    for a in range(9):
        features.append(['bias'])

    f = index2feature(sent, i, 0)
    for a in [1, 4, 5, 6, 7, 8]:
        features[a].append(f)

    if i > 1:
        f = index2feature(sent, i, -2)
        for a in [3, 6]:
            features[a].append(f)
    else:
        for a in [3, 6]:
            features[a].append('bos')

    if i > 0:
        f = index2feature(sent, i, -1)
        for a in [0, 3, 4, 6, 7]:
            features[a].append(f)
    else:
        for a in [3, 4, 6, 7]:
            features[a].append('bos')

    if i < length - 2:
        f = index2feature(sent, i, 2)
        features[8].append(f)
    else:
        features[8].append('eos')

    if i < length - 1:
        f = index2feature(sent, i, 1)
        for a in [2, 5, 7, 8]:
            features[a].append(f)
    else:
        for a in [5, 7, 8]:
            features[a].append('eos')

    features = list(filter(lambda x: x != ['bias'], features))
    return features
#

def sent2words(sent):
    return [word for word, tag in sent]
#

def sent2tags(sent):
    return [tag for word, tag in sent]
#

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
#

def sent2features_v2(sent):
    return [word2features_v2(sent, i) for i in range(len(sent))]
#

raw2corpus('raw_train.txt', 'train.txt')
raw2corpus('raw_test.txt', 'test.txt')


import pycrfsuite

def train(crf_file, model_file):
    train_sents = corpus2sent(crf_file)
    #train_x = [sent2features(sent) for sent in train_sents]
    train_x = [sent2features_v2(sent) for sent in train_sents]
    train_y = [sent2tags(sent) for sent in train_sents]
    trainer = pycrfsuite.Trainer()

    for x, y in zip(train_x, train_y):
        trainer.append(x, y)

    trainer.train(model_file)
#

def flush_one(x, y):
    return ' '.join(['{}/{}'.format(feature[1].split('=')[1], tag) for feature, tag in zip(x, y)])
#

def flush(path, X, Y):
    result = codecs.open(path, 'w', encoding='utf-8')
    for x, y in zip(X, Y):
        result.write(flush_one(x, y))
        result.write('\n')
    result.close()
#


from itertools import chain
'''
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def report(test_y, pred_y):
    lb = LabelBinarizer()
    test_y_combined = lb.fit_transform(list(chain.from_iterable(test_y)))
    pred_y_combined = lb.transform(list(chain.from_iterable(pred_y)))
    tagset = sorted(set(lb.classes_))
    class_indicies = {cls: idx for idx, cls in enumerate(tagset)}
    print(classification_report(test_y_combined, pred_y_combined, labels=[class_indicies[cls] for cls in tagset], target_names=tagset))
#'''

def test(in_file, model_file):
    crf_file = in_file[:-4] + '_crf.txt'
    pred_file = in_file[:-4] + '_pred.txt'

    raw2corpus(in_file, crf_file)
    test_sents = corpus2sent(crf_file)
    #test_x = [sent2features(sent) for sent in test_sents]
    test_x = [sent2features_v2 for sent in test_sents]
    test_y = [sent2tags(sent) for sent in test_sents]
    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)

    pred_y = [tagger.tag(x) for x in test_x]
    flush(pred_file, test_x, pred_y)
    corpus2raw(pred_file, pred_file.replace('_test_pred', '_test_pred_raw'))

    #report(test_y, pred_y)
#

def usage():
    print('usage: python3 test.py (train|test) input_file model_file')
    print('       python3 test.py demo model_file')
#

def demo(model_file):
    while True:
        raw_sentence = input('Input: ')
        if len(raw_sentence) == 0 or raw_sentence.lower()[:3] == 'end':
            print('end.')
            break

        taggeds = raw2corpus_one(raw_sentence)
        sent = corpus2sent_one(taggeds)
        print(sent)

        test_x = sent2features(sent)
        test_y = sent2tags(sent)
        print(test_x)
        print(test_y)

        tagger = pycrfsuite.Tagger()
        tagger.open(model_file)

        pred_y = tagger.tag(test_x)
        corpus_y = flush_one(test_x, pred_y)
        result = corpus2raw_one(corpus_y)
        print(result)
#

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    if sys.argv[1] == 'train' and len(sys.argv) >= 4:
        in_file = sys.argv[2]
        conv_file = in_file[:-4] + '_crf.txt'
        model_file = sys.argv[3]

        raw2corpus(in_file, conv_file)
        sent0 = corpus2sent(conv_file)
        train(conv_file, model_file)
    elif sys.argv[1] == 'test' and len(sys.argv) >= 4:
        in_file = sys.argv[2]
        model_file = sys.argv[3]

        test(in_file, model_file)
    elif sys.argv[1] == 'demo' and len(sys.argv) >= 3:
        model_file = sys.argv[2]
        demo(model_file)
    else:
        usage()
        sys.exit(0)

