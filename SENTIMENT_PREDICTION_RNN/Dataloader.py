import numpy as np
import torch as T
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

class DATA_LOADER():
    def __init__(self):
        with open(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\SENTIMENT_PREDICTION_RNN\reviews.txt', 'r') as f:
            self.reviews = f.read()

        with open(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\SENTIMENT_PREDICTION_RNN\labels.txt', 'r') as f:
            self.labels = f.read()

        from string import punctuation

        self.reviews = self.reviews.lower()
        self.reviews = ''.join(
            [i for i in self.reviews if i not in punctuation])

        # ITS TO BE USED LATER FOR ENCODING THE REVIEWS TO INTEGERS
        self.reviews_words = self.reviews.split('\n')
        self.reviews = ' '.join(self.reviews_words)
        self.reviews = self.reviews.split()

        self.labels = self.labels.split('\n')
        self.labels = [int(i == 'positive') for i in self.labels]

        # REVIEWS CONTAINS ALL THE INDIVIDUAL WORDS. WE HAVE TO FIND THE NO. OF OCCURANCES OF THE WORDS.
        # SORT THE LIST BASED ON THE MAX OCCURANCES, I.E WORD 'THE' HAS THE HIGHEST OCCURANCES AND HENCE ITS GIVEN INDEX OF 1.
        # NOW TAKE THE REVIEW WORDS AND ENCODE THEM WITH THE INDEX VALUE.

        from collections import Counter

        # COUNT THE NO. OF OCCURANCES OF THE WORDS
        self.reviews = Counter(self.reviews)
        # SORTING THE WORD LIST BASED ON MAX OCCURANCES
        self.words = sorted(self.reviews, key=self.reviews.get, reverse=True)

        self.words_to_int = {word: ii for ii, word in enumerate(
            self.words, 1)}   # INDEXING THE WORDS IN THE SORTED ORDER
        self.encoded_review = []                                             # ENCODED VALUES FOR THE REVIEWS

        for review in self.reviews_words:
            self.encoded_review.append(
                [self.words_to_int[word] for word in review.split()])

        # TO REMOVE REVIEWS OF ZERO LENGTH I.E REVIEW AT LAST IS AN EMPTY ARRAY.

        len_of_rev = [len(i) for i in self.encoded_review]
        max_rev_length = max(len_of_rev)                                # TO FIND THE MAX LENGTH REVIEW

        non_zero_idx = [ii for ii, review in enumerate(
            self.encoded_review) if len(review) != 0]

        self.encoded_review = [self.encoded_review[ii] for ii in non_zero_idx]
        self.encoded_labels = np.array(
            [self.labels[ii] for ii in non_zero_idx], dtype=np.int64)

        # TO FEED TO THE MODEL WE NEED TO TRUNCATE OR EXTEND ALL OUR REVIEWS TO A STD LENGTH LET US TAKE MEAN LENGTH
        self.features = self.std_length(self.encoded_review, int(
            np.ceil(np.mean(len_of_rev))))

    def std_length(self, encoded_review, seq_len):
        features = np.zeros((len(encoded_review), seq_len), dtype=np.int64)
        for i, j in enumerate(encoded_review):
            ind = min(len(j), seq_len)
            features[i, -ind:] = j[:ind]
        return features

        
        #print(features.shape)
        #print(features[0], labels[0])

    def sentiment_dataloader(self):
        train_split = int(len(self.features) * 0.8)
        val_split = int(len(self.features) * 0.9)

        train_features, val_features, test_features = self.features[:train_split], \
                                                self.features[train_split:val_split], self.features[val_split:]

        train_labels, val_labels, test_labels = self.encoded_labels[:train_split], \
            self.encoded_labels[train_split:val_split], self.encoded_labels[val_split:]

        train_data = TensorDataset(T.from_numpy(train_features), T.from_numpy(train_labels))
        val_data = TensorDataset(T.from_numpy(val_features), T.from_numpy(val_labels))
        test_data = TensorDataset(T.from_numpy(test_features), T.from_numpy(test_labels))

        train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=50, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True)

        return train_loader, val_loader, test_loader




