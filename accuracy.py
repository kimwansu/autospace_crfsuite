# -*- coding: utf-8 -*-

import codecs
import sys
import re

def read_file(in_file, enc='utf-8'):
    lines = []
    with codecs.open(in_file, 'r', encoding=enc) as rfh:
        for line in rfh:
            lines.append(re.sub(r'(\ )+', ' ', line).strip())

    return list(filter(None, lines))

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

def count_word_accuracy(original_line, test_line):
    total_word_count = 0
    same_word_count = 0
    original_words = original_line.split(' ')
    test_words = test_line.split(' ')

    total_word_count += len(original_words)
    for w in test_words:
        if w in original_words:
            same_word_count += 1

    return total_word_count, same_word_count

def count_space_accuracy(original_corpus, test_corpus):
    B_total = 0
    B_same = 0
    I_total = 0
    I_same = 0

    original_taggeds = [o.split('/')[1] for o in original_corpus.split(' ')]
    test_taggeds = [t.split('/')[1] for t in test_corpus.split(' ')]

    for o, t in zip(original_taggeds, test_taggeds):
        if o == 'B':
            B_total += 1
            if o == t:
                B_same += 1
        elif o == 'I':
            I_total += 1
            if o == t:
                I_same += 1

    return B_total, B_same, I_total, I_same

def usage():
    print('usage: python3 accuracy.py (original_file) (test_file)')

def main():
    if len(sys.argv) < 3:
        usage()
        sys.exit(1)

    original_file = sys.argv[1]
    test_file = sys.argv[2]

    original_lines = read_file(original_file)
    test_lines = read_file(test_file)

    total_wc, same_wc = map(sum, zip(*[count_word_accuracy(ol, tl) for ol, tl in zip(original_lines, test_lines)]))
    wc_accuracy = round(float(same_wc) / total_wc * 100, 2)
    print('어절 단위 정확도: {} / {}\t{} %'.format(same_wc, total_wc, wc_accuracy))

    original_corpus = [raw2corpus(line) for line in original_lines]
    test_corpus = [raw2corpus(line) for line in test_lines]

    B_total, B_same, I_total, I_same = map(sum, zip(*[count_space_accuracy(oc, tc) for oc, tc in zip(original_corpus, test_corpus)]))
    B_accuracy = round(float(B_same) / B_total * 100, 2)
    I_accuracy = round(float(I_same) / I_total * 100, 2)

    sp_same = B_same + I_same
    sp_total = B_total + I_total
    sp_accuracy = round(float(sp_same) / sp_total * 100, 2)
    print('띄어쓰기 정확도\n--------')
    print('B:\t{} / {}\t{} %'.format(B_same, B_total, B_accuracy))
    print('I:\t{} / {}\t{} %'.format(I_same, I_total, I_accuracy))
    print('--------')
    print('전체:\t{} / {}\t{} %'.format(sp_same, sp_total, sp_accuracy))

if __name__ == '__main__':
    main()
