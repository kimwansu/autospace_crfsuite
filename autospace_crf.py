# -*- coding: utf-8 -*-

import codecs
import sys
import re

from multiprocessing import cpu_count, Pool

import pycrfsuite


def read_file(in_file, enc='utf-8'):
    lines = []
    with codecs.open(in_file, 'r', encoding=enc) as rfh:
        for line in rfh:
            lines.append(line)

    return lines


def write_file(out_file, lines, enc='utf-8', line_end='\n'):
    with codecs.open(out_file, 'w', encoding=enc) as wfh:
        wfh.write(line_end.join(lines))


def raw2corpus(raw_sentence):
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


def corpus2sent(line):
    sent = []
    tokens = line.split(' ')
    for token in tokens:
        if '/' not in token:
            continue

        word, tag = token.split('/')
        sent.append((word, tag))

    return sent


def index2feature(sent, i, offsets):
    keys = []
    vals = []

    for offset in offsets:
        if i + offset < 0 or i + offset >= len(sent):
            return None

        word, _ = sent[i + offset]
        keys.append('w[{}]'.format(offset))
        vals.append(word)

    return '{}={}'.format('|'.join(keys), '|'.join(vals))


def sent2crfinput(sent):
    xseq = []
    yseq = []

    for i in range(len(sent)):
        word, tag = sent[i]

        features = []
        features.append(index2feature(sent, i, [-1]))
        features.append(index2feature(sent, i, [0]))
        features.append(index2feature(sent, i, [1]))
        features.append(index2feature(sent, i, [-2, -1]))
        features.append(index2feature(sent, i, [-1, 0]))
        features.append(index2feature(sent, i, [0, 1]))
        features.append(index2feature(sent, i, [-2, -1, 0]))
        features.append(index2feature(sent, i, [-1, 0, 1]))
        features.append(index2feature(sent, i, [0, 1, 2]))

        if i == 0:
            features.append('__BOS__')
        elif i == len(sent) - 1:
            features.append('__EOS__')

        features = list(filter(None, features))

        xseq.append(features)
        yseq.append(tag)

    return xseq, yseq


def train(in_file, model_file):
    lines = read_file(in_file)
    '''
    corpus = [raw2corpus(line) for line in lines]
    sent = [corpus2sent(line) for line in corpus]
    crf_input = [sent2crfinput(s) for s in sent]
    '''
    pool = Pool(cpu_count())
    corpus = pool.map(raw2corpus, lines)
    sent = pool.map(corpus2sent, corpus)
    crf_input = pool.map(sent2crfinput, sent)

    trainer = pycrfsuite.Trainer()
    for c in crf_input:
        xseq, yseq = c
        trainer.append(xseq, yseq)

    trainer.train(model_file)


def test(in_file, model_file):
    pred_file = in_file[:-4] + '_pred2.txt'

    lines = read_file(in_file)
    '''
    corpus = [raw2corpus(line) for line in lines]
    sent = [corpus2sent(line) for line in corpus]
    crf_input = [sent2crfinput(s) for s in sent]
    '''
    pool = Pool(cpu_count())
    corpus = pool.map(raw2corpus, lines)
    sent = pool.map(corpus2sent, corpus)
    crf_input = pool.map(sent2crfinput, sent)

    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)

    pred_y = [tagger.tag(xseq) for xseq, _ in crf_input]

    result_lines = []
    for line, py in zip(lines, pred_y):
        tagged_line = []
        for x, y in zip(line.replace(' ', ''), py):
            if y == 'B':
                tagged_line.append(' ')

            tagged_line.append(x)

        result_lines.append(''.join(tagged_line).strip())

    write_file(pred_file, result_lines)


def demo(model_file):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)
    while True:
        text = input('Input: ')
        if not text:
            break

        one_corpus = raw2corpus(text)
        one_sent = corpus2sent(one_corpus)
        one_crf_input = sent2crfinput(one_sent)

        xseq, _ = one_crf_input
        pred_y = tagger.tag(xseq)

        result = ''
        for x, y in zip(text.replace(' ', ''), pred_y):
            if y == 'B':
                result += ' {}'.format(x)
            else:
                result += x
        result = result.strip()
        print(result)


def usage():
    print('usage: python3 test.py (train|test) input_file model_file')
    print('       python3 test.py demo model_file')


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    if sys.argv[1] == 'train' and len(sys.argv) >= 4:
        in_file = sys.argv[2]
        model_file = sys.argv[3]

        train(in_file, model_file)
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


if __name__ == '__main__':
    main()
