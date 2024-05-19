from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
import csv
import numpy as np


# polje -> tekst + oznaka sentimenta
class Instance:
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __str__(self):
        return f"Instance(text='{self.text}', label='{self.label}')"

    def __iter__(self):
        yield self.text
        yield self.label


# spremanje i dohvaćanje podataka
# nasljeđuje torch.utils.data.Dataset
class NLPDataset(Dataset):
    def __init__(self, instances, text_vocab, label_vocab):
        self.instances = instances
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    # metoda se poziva kao NLPDataset.from_file(...)
    @classmethod
    def from_file(cls, file_path, text_vocab, label_vocab):
        instances = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                text = text.split(" ")
                label = row[1]
                label = label.replace(" ", "")
                instance = Instance(text, label)
                instances.append(instance)
        return cls(instances, text_vocab, label_vocab)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        text = instance.text
        label = instance.label
        numericalized_text = self.text_vocab.encode(text)
        numericalized_label = self.label_vocab.encode([label])
        return numericalized_text, numericalized_label

    def __str__(self):
        return f'NLPDataset with {len(self.instances)} instances'


# pretvorba tekstnih podataka u indekse
# stoi -> string to index
# itos -> index to string
class Vocab:
    # Vaša implementacija vokabulara se treba izgraditi temeljem rječnika frekvencija za neko polje.
    def __init__(self, freq_dict: Dict[str, int], max_size: int = -1, min_freq: int = 1, labelVocab = False):
        self.freq_dict = freq_dict
        if max_size == -1:
            self.max_size = len(freq_dict)
        else:
            self.max_size = max_size
        self.min_freq = min_freq
        # labelVocab ne sadrzi <PAD> i <UNK>
        if labelVocab:
            self.special_tokens = {}
        else:
            self.special_tokens = {0: '<PAD>', 1: '<UNK>'}
        self.itos = list(self.special_tokens.values())
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self.build_vocab()

    def build_vocab(self):
        # sortiramo silazno po broju ponavljanja
        sorted_tokens = sorted(self.freq_dict.keys(), key=lambda x: self.freq_dict[x], reverse=True)
        # ako je max_size -1, onda uzmemo sve tokene, inace uzmemo samo max_size najcescih
        if self.max_size != -1:
            sorted_tokens = sorted_tokens[:self.max_size-len(self.special_tokens)]
        # ne uzimamo u obzir one kojima je frekvencija manja od min_freq
        for index, token in enumerate(sorted_tokens):
            if self.freq_dict[token] < self.min_freq:
                break
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    def __len__(self):
        return len(self.itos)

    # za svaki token daje indeks
    def encode(self, tokens):
        rezultat = []
        for token in tokens:
            if token in self.stoi:
                rezultat.append(self.stoi[token])
        return torch.tensor(rezultat, dtype=torch.int)


# Bitno: u vašoj collate funkciji vraćajte i duljine originalnih instanci (koje nisu nadopunjene).
# nadopuniti duljine instanci znakom punjenja do duljine najdulje instance u batchu
def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = pad_sequence(texts, padding_value=pad_index)
    labels = torch.tensor(labels)
    lengths = lengths.clone().detach()
    return texts.T, labels, lengths


def get_token_freq(file_path, labelVocab=False):
    field_tokens = []
    if labelVocab:
        index = 1
    else:
        index = 0
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[index]
            field_tokens.extend(text.split(" "))
    token_freq = {}
    for token in field_tokens:
        if token == '':
            continue
        if token not in token_freq:
            token_freq[token] = 1
        else:
            token_freq[token] += 1
    #print(token_freq)
    return token_freq


def generate_embedding_matrix(vocabulary, embedding_file=None, embedding_dim=300):# vocabulary je stoi
    # inicijaliziramo normalnu matricu dimenzija V x D
    embedding_matrix = np.random.normal(0, 1, (len(vocabulary), embedding_dim))
    if embedding_file:
        embedding_dict = {}
        with open(embedding_file, 'r') as f:
            for line in f:
                # za svaku rijec koju citamo prebrisemo inicijalnu reprezentaciju u retku i zamijenimo s vektorskom reprezentacijom
                # ako riječi nema onda ostaje normalna razdioba
                word, *vector = line.split(' ')
                embedding_dict[word] = np.array(vector, dtype=np.float32)
        for word in embedding_dict.keys():
            if word in vocabulary.keys():
                index = vocabulary[word]
                embedding_matrix[index] = embedding_dict[word]
        from_file = True
    else:
        from_file = False
    # na indeksu 0 mora biti reprezentacija za posebni znak punjenja
    embedding_matrix[0] = 0
    embedding_matrix = torch.from_numpy(embedding_matrix)
    return embedding_matrix, from_file


def getData(batch_size_train=2, batch_size_test=2, batch_size_valid=2, shuffle=False, max_size=-1, min_freq=0):
    # Bitno: vokabular se izgrađuje samo na train skupu podataka.
    # Jednom izgrađeni vokabular na train skupu postavljate kao vokabular testnog i validacijskog skupa podataka
    filename = 'data/sst_train_raw.csv'
    frequencies = get_token_freq(filename, labelVocab=False)
    text_vocab = Vocab(frequencies, max_size=max_size, min_freq=min_freq)
    frequencies = get_token_freq(filename, labelVocab=True)
    label_vocab = Vocab(frequencies, max_size=max_size, min_freq=min_freq, labelVocab=True)
    train_dataset = NLPDataset.from_file(filename, text_vocab, label_vocab)
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=shuffle, collate_fn=pad_collate_fn)
    filename = 'data/sst_test_raw.csv'
    test_dataset = NLPDataset.from_file(filename, text_vocab,
                                   label_vocab)  # ovaj text_vocab i label_vocab su od train podataka
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=shuffle, collate_fn=pad_collate_fn)
    filename = 'data/sst_valid_raw.csv'
    valid_dataset = NLPDataset.from_file(filename, text_vocab, label_vocab)
    dataloader_valid = DataLoader(dataset=valid_dataset, batch_size=batch_size_valid, shuffle=shuffle, collate_fn=pad_collate_fn)
    return train_dataset, dataloader_train, test_dataset, dataloader_test, valid_dataset, dataloader_valid


def getVocab(max_size=-1, min_freq=1):
    filename = 'data/sst_train_raw.csv'
    frequencies = get_token_freq(filename, labelVocab=False)
    text_vocab = Vocab(freq_dict=frequencies, max_size=max_size, min_freq=min_freq)
    frequencies = get_token_freq(filename, labelVocab=True)
    label_vocab = Vocab(frequencies, max_size=max_size, min_freq=min_freq, labelVocab=True)
    return text_vocab, label_vocab

text_vocab, label_vocab = getVocab()

embedding_matrix, from_file = generate_embedding_matrix(text_vocab.stoi, embedding_file='data/sst_glove_6b_300d.txt')
#print("embedding matrix", embedding_matrix.shape) # embedding matrix torch.Size([14804, 300])
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=from_file)

embedding_matrix2, from_file2 = generate_embedding_matrix(text_vocab.stoi, embedding_file=None)
embedding_layer2 = torch.nn.Embedding.from_pretrained(embedding_matrix2, padding_idx=0, freeze=from_file)


if __name__ == "__main__":
    train_dataset, dataloader_train, test_dataset, dataloader_test, valid_dataset, dataloader_valid = getData()

    print(len(text_vocab.itos))
    print(label_vocab.stoi)

    # Referenciramo atribut klase pa se ne zove nadjačana metoda)
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")

    # Koristimo nadjačanu metodu indeksiranja
    numericalized_text, numericalized_label = train_dataset[3]
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")

    texts, labels, lengths = next(iter(dataloader_train))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")




