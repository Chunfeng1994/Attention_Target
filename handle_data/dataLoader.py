import re
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.lower()
    string = re.sub(r"，", ",", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r"^-user-$", "<user>", string)
    string = re.sub(r"^-url-$", "<url>", string)
    string = re.sub(r"^-lqt-$", "\'", string)
    string = re.sub(r"^-rqt-$", "\'", string)
    # 或者
    # string = re.sub(r"^-lqt-$", "\"", string)
    # string = re.sub(r"^-rqt-$", "\"", string)
    string = re.sub(r"^-lrb-$", "\(", string)
    string = re.sub(r"^-rrb-$", "\)", string)
    string = re.sub(r"^lol$", "<lol>", string)
    string = re.sub(r"^<3$", "<heart>", string)
    string = re.sub(r"^#.*", "<hashtag>", string)
    string = re.sub(r"^[0-9]*$", "<number>", string)
    string = re.sub(r"^\:\)$", "<smile>", string)
    string = re.sub(r"^\;\)$", "<smile>", string)
    string = re.sub(r"^\:\-\)$", "<smile>", string)
    string = re.sub(r"^\;\-\)$", "<smile>", string)
    string = re.sub(r"^\;\'\)$", "<smile>", string)
    string = re.sub(r"^\(\:$", "<smile>", string)
    string = re.sub(r"^\)\:$", "<sadface>", string)
    string = re.sub(r"^\)\;$", "<sadface>", string)
    string = re.sub(r"^\:\($", "<sadface>", string)
    return string.strip()


def read_sentence(path,is_train):
    data = []
    sentence_length = Counter()
    feature_dict = Counter()
    label_dict = Counter()
    with open(path,"r",encoding="utf-8") as f:
        sentence = []
        start = -1
        end = -1
        count = 0
        target_flag = False
        label = None
        for s in f:
            s = s.strip()
            if len(s) != 0 and s != "":
                s = s.split(" ")
                word = clean_str(s[0])
                sentence.append(word)
                sentence_length[len(word)] += 1
                count += 1
                if is_train:
                    feature_dict[word] += 1
                if s[1][0] == 'b':
                    start = count - 1
                    target_flag = True
                    label = s[1][2:]
                    if is_train:
                        label_dict[label] += 1
                elif s[1][0] == 'o':
                    if target_flag:
                        end = count - 2
                        target_flag = False
            else:
                sentence_length[len(s)] += 1
                if target_flag:
                    end = count - 2
                if start > end:
                    end = start
                data.append((sentence, start, end, label))
                sentence = []
                count = 0
    if is_train:

        return data,sentence_length,feature_dict,label_dict

    return data,sentence_length



