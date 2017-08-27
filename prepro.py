# -*- coding: utf-8 -*-
import json
from collections import Counter


def get_answer_sent(paragraph, posi, answer):
    bias = str.find(paragraph[posi:], answer)
    if bias == -1 or bias > 15:  # if position is wrong too much
        return "_UNK"
    posi += bias
    assert(answer in paragraph[posi:])
    start = posi - 1
    if start < 0:
        start = 0
    end = posi + len(answer)
    if end > len(paragraph):
        end = len(paragraph)

    def is_start(position, context):
        assert(context[position] == ".")
        if context[position+1] != " ":
            return False
        return True

    while True:
        if start == 0:
            break
        ch = paragraph[start]
        if ch != ".":
            start -= 1
        else:
            if is_start(start, paragraph):
                start += 1
                break
            else:
                start -= 1

    def is_end(position, context):
        assert(context[position] == ".")
        if position == len(context) - 1:
            return True
        if context[position+1] != " ":
            return False
        last_word = context[:position+1].split()[-1]
        if last_word.count(".") == 1:
            return True
        else:
            return False

    while True:
        if end == len(paragraph):
            break
        ch = paragraph[end]
        if ch is not ".":
            end += 1
        else:
            if is_end(end, paragraph):
                end += 1
                break
            else:
                end += 1
    answer_sent = paragraph[start:end].strip()
    if answer not in answer_sent:
        print "assertion error:"
        print answer
        print answer_sent
    assert answer in answer_sent
    if str.count(answer_sent, answer) != 1:
        return '_UNK'
    return answer_sent


if __name__ == "__main__":
    questions, answers, paragraphs, answer_posi, answer_sent = [], [], [], [], []
    with open('data/train-v1.1.json') as f:
        data = json.loads(f.read())
    unk_count = 0
    for article in data['data']:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = str.lower(paragraph['context'].encode('utf-8'))
            for qa in paragraph['qas']:
                question = str.lower(qa['question'].encode('utf-8'))
                assert(len(qa['answers']) > 0)
                answer = str.lower(qa['answers'][0]['text'].encode('utf-8'))
                position = qa['answers'][0]['answer_start']
                sent = get_answer_sent(context, position, answer)
                if sent is not "_UNK":
                    answer_posi.append(position)
                    answers.append(answer)
                    answer_sent.append(sent.replace("\n", " "))
                    paragraphs.append(context.replace("\n", " "))
                    questions.append(question)
                else:
                    unk_count += 1
    data = {'question': questions, 'answer': answers, "paragraph": paragraphs, "answer_posi": answer_posi,
            'answer_sent': answer_sent}
    for k, v in data.items():
        with open("data/{}.txt".format(k), "w+") as f:
            for item in v:
                f.write("{}\n".format(item))
            f.close()
    print("data size: {}".format(len(paragraphs)))

    words = []
    for item in paragraphs:
        words += str.lower(item.strip()).split()
    for item in questions:
        words += str.lower(item.strip()).split()
    words = Counter(words).most_common()
    with open('data/vocab.txt', 'w+') as f:
        for word, cnt in words:
            f.write('{}\t{}\n'.format(word, cnt))
    print('vocab size: {}'.format(len(words)))
    print('unk count: {}'.format(unk_count))