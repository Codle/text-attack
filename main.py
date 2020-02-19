""" 基于 TextFool 的文本对抗模型
"""
import argparse
import fasttext
import jieba
import json
from copy import deepcopy
import numpy as np
from gensim.models import KeyedVectors
from preprocessing_module import preprocess_text
from distance_module import measure as dis_utils


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='/tcdata/benchmark_texts.txt',
                    help='输入文件')
parser.add_argument('--model_path', default='data/mini.ftz',
                    help='预先模型文件')
parser.add_argument('--output_file', default='adversarial.txt', help='输出文件')
parser.add_argument(
    '--vector_file', default='data/Tencent_AILab_ChineseEmbedding.txt')


def compute_word_importance(model, text_tokens):
    """ 计算词语的重要性
    """
    importances = []
    temp_text = []
    ori_label, ori_score = model.predict(''.join(text_tokens))

    for i in range(len(text_tokens)):
        new_text = ''.join(text_tokens[:i] + text_tokens[i+1:])
        temp_text.append(new_text)
    labels, scores = model.predict(temp_text)
    for label, score in zip(labels, scores):
        if label[0] == ori_label[0]:
            importances.append(ori_score[0] - score[0])
        else:
            importances.append(
                (ori_score[0] - score[0]) + (score[0] - ori_score[0]))
    return importances


def main():
    # 预先测试模型
    model = fasttext.load_model(args.model_path)
    with open(args.input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    out_lines = []
    for line in lines:
        # 文本清理
        text = preprocess_text(line)
        # 文本分词
        words = [word for word in jieba.cut(text)]
        _temp = []
        for word in words:
            if word in DEFAULT_KEYVEC.vocab:
                _temp.append(word)
            else:
                for ch in word:
                    if ch in DEFAULT_KEYVEC.vocab:
                        _temp.append(ch)
                    else:
                        continue
        words = _temp
        # print(words)
        out = model.predict(text)

        # 第一步：计算文本重要性
        importances = compute_word_importance(model, words)
        importeance_order = reversed(np.argsort(importances))
        # print(importances, importeance_order)

        # 第二步：选出候选单词
        candidates = dict()
        for word in words:
            top_words = DEFAULT_KEYVEC.most_similar(positive=word, topn=300)
            sim_words = []
            for idx, (_word, _score) in enumerate(reversed(top_words)):

                # if _score < 0.03:
                _word = preprocess_text(_word)
                _cutword = [_ for _ in jieba.cut(_word)]
                if len(_word) and len(_cutword) == 1:
                    sim_words.append(_word)
                # else:
                #     break
                if len(sim_words) > 100:
                    break
            candidates[word] = sim_words

        # Jaccord
        for key in candidates.keys():
            _candidate = []
            for candidate in candidates[key]:
                dis = dis_utils.normalized_levenshtein(key, candidate)
                # print(dis)
                if dis > 0.5:
                    _candidate.append(candidate)
            candidates[key] = _candidate

        # print(candidates)
        # break
        # 第三步：词性过滤
        # 第四步：语句相似性

        # 第五步：替换文本
        for order in importeance_order:
            if len(candidates[words[order]]) == 0:
                continue
            temp = []
            for candidate in candidates[words[order]]:
                temp_words = deepcopy(words)
                temp_words[order] = candidate
                temp.append(''.join(temp_words))
            preds = model.predict(temp)
            # 区分如果存在
            preds_order = np.argsort(preds[1].reshape(-1))
            # print(preds_order)
            flag = -1
            for pred_order in preds_order:
                if preds[0][pred_order][0] != out[0][0]:
                    # print(preds[0][pred_order][0], out[0][0])
                    flag = pred_order
                    break
            if flag != -1:
                words[order] = candidates[words[order]][flag]
                break
            else:
                words[order] = candidates[words[order]][0]
        out_lines.append(''.join(words) + '\n')

    target = json.dumps({'text': out_lines}, ensure_ascii=False)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(target)


if __name__ == "__main__":
    args = parser.parse_args()

    # 初始化词典
    EMBEDDING_PATH = args.vector_file
    EMBEDDING_DIM = 300
    DEFAULT_KEYVEC = KeyedVectors.load_word2vec_format(
        EMBEDDING_PATH,
        binary=False,
        encoding='utf-8',
        limit=100000)

    main()
