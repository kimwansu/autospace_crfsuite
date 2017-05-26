# -*- coding: utf-8 -*-

import codecs
import re



def main():
    lines = []
    with codecs.open('./ted_7_ErasePunc_FullKorean__test.txt', 'r', encoding='utf-8') as rfh:
        for line in rfh.readlines():
            lines.append(re.sub(r'[\ \n\r]+', '', line).strip())

    buf = ''
    with codecs.open('./test_result2.txt', 'r', encoding='utf-8') as rfh:
        buf = rfh.read()

    out_lines = []
    i = 0
    for line in lines:
        line_y = buf[i:i+len(line)]

        if len(line) != len(line_y):
            print(line)
            print(line_y)
            print()

        result = ''
        for ch, y in zip(line, line_y):
            #print('{}, {}'.format(ch, y))
            if y == '1':
                result += ' ' + ch
            else:
                result += ch

        out_lines.append(result.strip())
        i += len(line)

    with codecs.open('./test_result_restored.txt', 'w', encoding='utf-8') as wfh:
        wfh.write('\n'.join(out_lines))

if __name__ == '__main__':
    main()
