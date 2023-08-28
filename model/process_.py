import re
import os
import string
from pkg_utils import (
    is_tiktoken_available,
    is_torch_available,
    torch_version,
    tiktoken_version,
    is_numpy_available,
    is_tensorflow_available,
    PackageNotFoundError
)

class ProcessorBase:
    def base(self, fast: bool):
        self.fast = fast
        if is_tiktoken_available() and self.fast:
            from tiktoken import get_encoding
            try:
                self.E = get_encoding('gpt2')
            except Exception as e:
                print(f'Error in loading encoder and decoder: {e}')
                print('Try ===> pip3 install tiktoken')
                raise e
        else:
            from utils import encoder
            try:
                self.E = encoder.get_encoder()
            except Exception as e:
                print(e)
        return self.E
    # def token(self):
    #     if is_tokenizers_available():
    #         from tokenize import tokenize as tk
class Tokenizer(ProcessorBase):
    def __init__(self, fast : bool = False):
        self.E = self.base(fast=fast)
        if is_torch_available():
            import torch
        else:
            PackageNotFoundError("torch")
        self.torch = torch

    def encode_(self,text: str, return_tensors : bool = False) -> list:
        '''
        Encode a string

        Args:
        - text (str): Input string to be encoded

        Returns:
        - list: Encoded list of integers
        '''
        if text == '':
            raise ValueError("text must be a string, not none") 
        if return_tensors:
            return self.torch.tensor(self.E.encode(text))
        else:
            return self.E.encode(text)

    def decode_(self,encoded_text: list) -> str:
        '''
        Decode a list of integers which are encoded

        Args:
        - encoded_text (list): List of integers to be decoded

        Returns:
        - str: Decoded string
        '''
        if encoded_text is None:
            raise ValueError("Encoded text cannot be empty")
        return self.E.decode(encoded_text)
    
    def encode_file(self, input_file_path: str, output_file_path: str):
        '''
        Encode the text from an input file and save the encoded text to an output file.
        
        Args:
        - input_file_path (str): Path to the input text file.
        - output_file_path (str): Path to the output file to save the encoded text.
        '''
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except FileNotFoundError as e:
            print(f'Input file not found:{input_file_path} ===> {e}')
        except IOError as e:
            print(f'Error reading input file : {e}')
            
        encoded_text = encode_(input_text)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(map(str, encoded_text)))

    # def tokenize_file(input_file_path : os.PathLike, output_file_path : os.PathLike):


    def split_text_into_batches(self, text: str, batch_size: int) -> list:
            """
            Split a text into batches based on a specified batch size.

            Args:
            - text (str): Input text to be split.
            - batch_size (int): Size of each batch.

            Returns:
            - list: List of batches.
            """
            if text is None:
                return [[]]
            words = text.split()
            if batch_size >= len(words):
                return [words]
            else:
                return [words[i:i+batch_size] for i in range(0, len(words), batch_size)]

    def normalize_sentence(self, sentence:str):
            """Lower text and remove punctuation, articles and extra whitespace."""
            return self.white_space_fix(self.remove_articles(self.remove_punc(self.lower(self.replace_contractions(sentence)))))

    def replace_contractions(text: str) -> str:
                """Replace contractions in the text.
                contractions = {"ain't": "is not","aren't": "are not","can't": "cannot","couldn't": "could not","didn't": "did not","doesn't": "does not",
                "don't": "do not","hadn't": "had not","hasn't": "has not","haven't": "have not","he's": "he is","I'd": "I would","I'll": "I will","I'm": "I am","I've": "I have",
                "isn't": "is not","it's": "it is","let's": "let us","mightn't": "might not","mustn't": "must not","shan't": "shall not","she's": "she is","shouldn't": "should not",
                "that's": "that is","there's": "there is","they'd": "they would","they'll": "they will","they're": "they are","they've": "they have","we'd": "we would",
                "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what're": "what are","what's": "what is","what've": "what have","where's": "where is",
                "who's": "who is","who've": "who have","won't": "will not","wouldn't": "would not","you'd": "you would","you'll": "you will","you're": "you are","you've": "you have",}
                """
                contractions = {"ain't": "is not","aren't": "are not","can't": "cannot","couldn't": "could not","didn't": "did not","doesn't": "does not",
                "don't": "do not","hadn't": "had not","hasn't": "has not","haven't": "have not","he's": "he is","I'd": "I would","I'll": "I will","I'm": "I am","I've": "I have",
                "isn't": "is not","it's": "it is","let's": "let us","mightn't": "might not","mustn't": "must not","shan't": "shall not","she's": "she is","shouldn't": "should not",
                "that's": "that is","there's": "there is","they'd": "they would","they'll": "they will","they're": "they are","they've": "they have","we'd": "we would",
                "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what're": "what are","what's": "what is","what've": "what have","where's": "where is",
                "who's": "who is","who've": "who have","won't": "will not","wouldn't": "would not","you'd": "you would","you'll": "you will","you're": "you are","you've": "you have",}

                for contraction, replacement in contractions.items():
                    text = text.replace(contraction, replacement)
                return text
            
    def remove_stopwords(self, text: str, stopwords: list) -> str:
                """Remove stopwords from the text."""
                words = text.split()
                filtered_words = [word for word in words if word.lower() not in stopwords]
                return ' '.join(filtered_words)

    def remove_articles(self, text):
                regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
                return re.sub(regex, " ", text)

    def white_space_fix(self, text):
                return " ".join(text.split())

    def remove_punc(self, text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

    def lower(self, text):
                return text.lower()

    def get_word_frequencies(self, text: str) -> dict:
            """Calculate the frequency of each word in the text."""
            cleaned_text = normalize_sentence(text)
            words = cleaned_text.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            return word_freq

    def count_sentences(self, text: str) -> int:
            """Count the number of sentences in the text."""
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            return len(sentences)

    def get_tokens(self, sentence:str) -> list:
            '''Return a list of tokens'''
            if sentence is None:
                return []
            return normalize_sentence(self.clean_up_tokenization(sentence)).split()

    def clean_up_tokenization(self, out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def compute_match(self, actual, predicted):
            ''' returns 1 if both tokens match ===> checks whether the predictd and actual output is same'''
            return int(normalize_sentence(actual) == normalize_sentence(predicted))
    def clean_code_for_chat(self, result):
        '''
        this code is used to clean the code for chat generations.
        '''
        lines = result.split("\n")
        idx = 0
        while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
            idx += 1
        explanation = "\n".join(lines[:idx]).strip()
        if idx == len(lines):
            return explanation, None

        idx += 1
        start_idx = idx
        while not lines[idx].lstrip().startswith("```"):
            idx += 1
        code = "\n".join(lines[start_idx:idx]).strip()

        return explanation, code

    def clean_code_for_run(self, result):
        '''
        this code is used to process code data from text data.
        '''
        result = f"I will use the following {result}"
        explanation, code = result.split("Answer:")
        explanation = explanation.strip()
        code = code.strip()

        code_lines = code.split("\n")
        if code_lines[0] in ["```", "```py", "```python"]:
            code_lines = code_lines[1:]
        if code_lines[-1] == "```":
            code_lines = code_lines[:-1]
        code = "\n".join(code_lines)

        return explanation, code

    def _is_tensor(self, x)->bool:
            return isinstance(x, self.torch.Tensor)

    def to_tensor(self, encoded_text: list, dtype = None):
        '''
        Convert a list of encoded text to a tensor.

        Args:
        - encoded_text (list): List of integers representing encoded text.
        - dtype: Data type of the tensor.

        Returns:
        - torch.tensor: Tensor representation of the encoded text.
        '''
        if encoded_text is None:
            raise ValueError("Encoded text cannot be empty")
        if dtype is None:
            dtype = self.torch.long
        tensors = self.torch.tensor(encoded_text, dtype=dtype)
        return tensors
    
    def tensor_pad(self, encoded_batch_text: list, dtype):
        '''
        Pad a batch of encoded text tensors.

        Args:
        - encoded_batch_text (list): List of lists of integers representing encoded text.
        - dtype: Data type of the tensor.

        Returns:
        - torch.tensor: Padded tensor representation of the batch of encoded text.
        '''
        if encoded_batch_text is None:
            raise ValueError("Encoded batch text cannot be empty")
        padded_tensors = self.torch.nn.utils.rnn.pad_sequence([self.torch.tensor(encoded_text, dtype=dtype) for encoded_text in encoded_batch_text], batch_first=True, padding_value=0)
        return padded_tensors

    def encode_and_tensor(self, inputs: str):
        '''
        Encode the input text and return a tensor of the encoded value.

        Args:
        - inputs (str): Input text to be encoded.

        Returns:
        - torch.tensor: Tensor representation of the encoded text.
        '''
        if is_torch_available():
            from torch import tensor, long
        encoded_text = encode_(inputs)
        tensors = tensor(encoded_text, dtype=long)
        return tensors

    def is_tensor(self, x):
        """
        Tests if `x` is a tensor-like object from popular numerical libraries.
        """
        torch_available = is_torch_available()
        tensorflow_available = is_tensorflow_available()
        numpy_available = is_numpy_available()

        if torch_available:
            import torch
            if isinstance(x, torch.Tensor):
                return True
        elif tensorflow_available:
            import tensorflow as tf
            if isinstance(x, tf.Tensor):
                return True
        elif numpy_available: 
            import numpy as np
            if isinstance(x, np.ndarray):
                return True
        else:
            return False

if __name__ == '__main__':
    txt = "I love using emojis 😊🚀"
    Tokenizers = Tokenizer()
    print(Tokenizers.encode_(txt, return_tensors=False))
    Tokenize = Tokenizer(fast=True)
    print(Tokenize.encode_(txt, return_tensors=False))

