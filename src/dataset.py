import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path,label_list, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels = [], []
        abstract = ""
        abs_labels = []
        with open(data_path) as data_file:
            data_file_lines = data_file.readlines()
            data_file_lines = list(filter(None,[line.rstrip() for line in data_file_lines]))
            for line in data_file_lines[1:]:  # ignore first line which is ID
                if line.startswith('###'):
                    texts.append(abstract)
                    labels.append(abs_labels)
                    abstract = ""
                    abs_labels = []
                    continue
                label, txt = line.split('\t', 1)
                abstract += txt.lower()+'\n'
                abs_labels.append(label.lower())

        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.label_list = label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ### TODO: Add init and eos tokens here
        # ['<pad>','<unk>','<sos>'] + vocab
        labels = self.labels[index]
        # label_encode = [self.dict.index(label)+3 if label in self.dict else 1 for label in labels]
        label_encode = [self.label_list.index(label)+2 for label in labels] # 0=<pad>, 1=<sos>
        labels = ['<sos>']+labels
        label_encode = [1]+label_encode
        if(len(label_encode)<self.max_length_sentences+1):
            extended_labels = [0 for _ in range(self.max_length_sentences - len(label_encode)+1)]
            label_encode.extend(extended_labels)
        label_encode = label_encode[:self.max_length_sentences+1]
        label_encode = np.stack(arrays=label_encode,axis=0)

        text = self.texts[index]
        # print('text',text)
        # print('sentences',sent_tokenize(text))
        # print('words',word_tokenize(sent_tokenize(text)[0]))
        document_encode = [
            [self.dict.index(word)+2 if word in self.dict else 1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [0 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[0 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        # document_encode += 1

        return document_encode.astype(np.int64), label_encode


if __name__ == '__main__':
    LABEL_LIST = ['background','objective','methods','results','conclusions']
    test = MyDataset(data_path="../data/train.txt", dict_path="../data/glove.6B.50d.txt",label_list=LABEL_LIST)
    item = test.__getitem__(index=1)
    print(item[0], item[1])
