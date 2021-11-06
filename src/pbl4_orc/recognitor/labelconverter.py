import numpy as np
import torch


class CTCLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(self, character, separator_list={}, dict_pathlist={}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = [
            "[blank]"
        ] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        self.ignore_idx = [0]

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """  # noqa: E501
        length = [len(s) for s in text]
        text = "".join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        index = 0
        for i in length:
            t = text_index[index : index + i]
            # Returns a boolean array where true is when the value is not repeated
            a = np.insert(~((t[1:] == t[:-1])), 0, True)
            # Returns a boolean array
            # where true is when the value is not in the ignore_idx list
            b = ~np.isin(t, np.array(self.ignore_idx))
            # Combine the two boolean array
            c = a & b
            # Gets the corresponding character according to the saved indexes
            text = "".join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += i
        return texts


class AttnLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder.
        # [s] for end-of-sentence token.
        list_token = ["[GO]", "[s]"]  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder.
                [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and
                text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder,
                which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step.
        # batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append("[s]")
            text = [self.dict[char] for char in text]
            batch_text[i][1 : 1 + len(text)] = torch.LongTensor(
                text
            )  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
