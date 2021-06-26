import numpy as np
class DATA_LOADER():
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as f:
            text_data = f.read()

        # TOKENISATION FOR DATA ENCODING
        self.char_in_text = tuple(set(text_data))
        self.n_characters = len(self.char_in_text)

        # DICT FOR KEY AS NOS. AND VALUE AS CHARACTER
        self.I2C = dict(enumerate(self.char_in_text))
        self.C2I = {v: k for k, v in self.I2C.items()}

        # ENCODING THE GIVEN TEXT
        self.encoded_text = np.array([self.C2I[i] for i in text_data])

    def load_data(self):
        """
        RETURNS ENCODED TEXT AND LENGTH OF SEQ
        """
        return self.encoded_text, self.n_characters


    def one_hot(self, inp, length):
        a = np.zeros((inp.size, length))
        a[np.arange(a.shape[0]), inp.flatten()] = 1
        a = a.reshape((*inp.shape, length))
        return a

    #encoded_text, no_of_char = load_data(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\CHARACTER_LEVEL_RNN\data.txt')
    #encoded_text = one_hot(encoded_text, no_of_char)

    def get_batch(self, a, BATCH_SIZE, SEQ_LENGTH):

        PER_BATCH_ELEMENTS = BATCH_SIZE * SEQ_LENGTH    # 400
        N = len(a) // PER_BATCH_ELEMENTS                # 3854

        a = a[:N * PER_BATCH_ELEMENTS]                  # (1541600,)
        a = a.reshape((BATCH_SIZE, -1))                 # (8, 192700)

        for i in range(0, a.shape[1], SEQ_LENGTH):
            
            x = a[:, i : i+SEQ_LENGTH]
            y = np.zeros_like(x)

            try:
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, N + SEQ_LENGTH]
            except:
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            
            yield x, y

    #batches = get_batch(encoded_text, 8, 50)
    #print(get_batch(a, BATCH_SIZE, SEQ_LENGTH).shape) # (SEQ_LEN, BATCH_SIZE, NO.OF.ELEMENTS)
    #x, y = next(batches)
    #print(x.shape, y.shape)
