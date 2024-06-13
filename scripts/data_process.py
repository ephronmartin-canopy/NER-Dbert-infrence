import warnings
import os, re
import time
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from datasets import DatasetDict, load_dataset 
from datasets import Dataset as CreateDataset
from sklearn.model_selection import train_test_split

from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,set_seed, AutoModel
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForTokenClassification, AdamW, Trainer, TrainingArguments
from scipy import stats as st
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


warnings.filterwarnings("ignore")

def tokenize(text):
    # Regular expression to handle the tokenization
    pattern = r"""
    (?P<url><https?://[^\s<>]+>) |              # Matches URLs within <>
    (?P<url2>\(https?://[^\s()]+\)) |           # Matches URLs within ()
    (?P<url3>https?://[^\s]+) |                 # Matches plain URLs
    (?P<email>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b) | # Matches email addresses
    (?P<asterisks>\*{2,}) |                     # Matches sequences of asterisks
    (?P<number_k>\b\d+\.\d+k\b) |               # Matches numbers with 'k' suffix like 69.5262k
    (?P<alphanumeric>\b\w+[-\w]*\w+\b) |        # Matches alphanumeric patterns with hyphens in between
    (?P<timestamp>\b(?:2[0-3]|1\d|\d|24):[0-5]\d(?::[0-5]\d)?(?:am|pm)?\b) |    # Matches timestamps like 10:46 or 10:46:03 or 10:46am
    (?P<date>\b\d{1,2}/\d{1,2}/\d{2,4}\b) |     # Matches dates like 11/30/00
    (?P<decimal>\b\d+(?:\.\d+)+\b) |            # Matches decimal numbers like 3.0 or 3.15.2
    (?P<money>\$\-?\d+(?:,\d{3})*(?:\.\d{2})?) |# Matches monetary values like $-11,196.41
    (?P<word>\w+)|                              # Matches words
    (?P<punct>[^\w\s])                          # Matches punctuation
    """
    tokens = []
    for match in re.finditer(pattern, text, re.VERBOSE):
        if match.group('url'):
            tokens.append(match.group('url'))
        elif match.group('url2'):
            tokens.append(match.group('url2'))
        elif match.group('url3'):
            tokens.append(match.group('url3'))
        elif match.group('email'):
            tokens.append(match.group('email'))
        elif match.group('asterisks'):
            tokens.append(match.group('asterisks'))
        elif match.group('number_k'):
            tokens.append(match.group('number_k'))
        elif match.group('alphanumeric'):
            tokens.append(match.group('alphanumeric'))
        elif match.group('timestamp'):
            tokens.append(match.group('timestamp'))
        elif match.group('date'):
            tokens.append(match.group('date'))
        elif match.group('decimal'):
            tokens.append(match.group('decimal'))
        elif match.group('money'):
            money_value = match.group('money')
            # Split money value as needed
            if "," in money_value:
                tokens.extend(re.split(r'(?<=\d),(?=\d)', money_value))
            else:
                tokens.append(money_value)
        elif match.group('word'):
            tokens.append(match.group('word'))
        elif match.group('punct'):
            tokens.append(match.group('punct'))
    return tokens


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    


def check_alphanum(token):
    return bool(re.search('[a-z]', token, re.IGNORECASE)) and bool(re.search('[0-9]', token, re.IGNORECASE))


def combine_sents_digits(sents):
    comb_sents = []
    cnt = 0
    for i in range(0, len(sents) - 1):
        current_sent = " ".join(sents[i]).strip()
        next_sent = " ".join(sents[i + 1]).strip()
        
        if (i != 0) and (len(sents[i]) == 1) and (sents[i][0].strip().isdigit() or check_alphanum(sents[i][0].strip())):
            comb_sents[-1] = comb_sents[-1].strip() + " " + current_sent
            cnt += 1
        else:
            comb_sents.append(current_sent)
    return comb_sents

punct_tokens = ['-',',',';',':','?','.',"'",'"','(',')','[',']','{','}','@','*','/','&','#','%','^','+','=','>', '|','~']


def clean_punctuations(sentences):
    out_sentences = []
    for word in sentences: 
        if check_alphanum(word):
            word = "alphanum"

        if word.isdigit():
            word = "number"+str(len(word))

        out_sentences.append(word)
        
    return out_sentences



def get_expectional_tokens(numberRange=25):
    expectional_tokens = []
    for i in range(1,numberRange+1):
        expectional_tokens.append(f'number{i}')
    expectional_tokens.append("alphanum")
    return expectional_tokens
    
def tokenize_and_preserve_labels(sentence, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """
    combined_pattern = re.compile(
    r"""
    ([!\"#$%&'()*•+,\-.\/:;<=>?@\[\]^_`{|}~][A-Za-z0-9])|
    ([A-Za-z0-9][!\"#$%&'()*–+,\-.\/:;<=>?@\[\]^_`{|}~])|
    ([!\"#$%&'()*+,\-.\/:;<=>?@\[\]^_`{|}~][!\"#$%&'()*+,\-.\/:;<=>?@\[\]^_`{|}~])|
    (\d[A-Za-z])|([A-Za-z0-9]\’)|([A-Za-z]\d)|(\—+)|\“[A-Za-z0-9]|[A-Za-z]\”
    """,
    re.VERBOSE)
    
    expectional_tokens = get_expectional_tokens()
    tokenized_sentence = []
    for word in sentence:
        exceptionMatch = re.findall(combined_pattern, word)
        if word not in expectional_tokens and exceptionMatch:
            tokenized_sentence.append("[UNK]")
        else:
        # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            if tokenized_word:
                n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
                tokenized_sentence.extend(tokenized_word)
        # Add the same label to the new list of  `n_subwords` times
            else:
                tokenized_sentence.append("[UNK]")
    return tokenized_sentence


def join_sentences_with_delimiter(sentences, tokenizer, delimiter = "[SEP]", max_seq_length=500, overlap_percentage=0.3):
    joined_sentences = []
    joined_tags = []
    current_sentence = []
    current_tags = []
    current_length = 0
    overlap_length = int(max_seq_length * overlap_percentage)

    for sentence in sentences:
        tokenized_sentence,  = tokenize_and_preserve_labels(sentence, tokenizer)
        sentence_length = len(tokenized_sentence) + 1  # +1 for the "CRFLS" delimiter

        # Check if adding the current sentence would exceed the max_seq_length
        while current_length + sentence_length > max_seq_length:

            remain_tokens = max_seq_length - current_length
            current_sentence.extend(tokenized_sentence[:remain_tokens])
            current_tags.extend(tokenized_tags[:remain_tokens])

            tokenized_sentence=tokenized_sentence[remain_tokens:]
            tokenized_tags=tokenized_tags[remain_tokens:]

            joined_sentences.append(current_sentence[-max_seq_length:])
            joined_tags.append(current_tags[-max_seq_length:])

            current_sentence = current_sentence[-overlap_length:]
            current_tags = current_tags[-overlap_length:]
            current_length = len(current_sentence)
            sentence_length = len(tokenized_sentence) + 1

        # Add the current sentence and tags to the ongoing sequence
        current_sentence.extend(tokenized_sentence)
        current_sentence.append(delimiter)  # Add delimiter at the end of each sentence
        current_tags.extend(tokenized_tags)
        current_tags.append("o")
        current_length += sentence_length

    if current_length >= max_seq_length:
        joined_sentences.append(current_sentence[:max_seq_length])
        joined_tags.append(current_tags[:max_seq_length])
        current_sentence = current_sentence[max_seq_length:]
        current_tags = current_tags[max_seq_length:]

    if current_sentence:
        joined_sentences.append(current_sentence)
        joined_tags.append(current_tags)
    return joined_sentences, joined_tags

class NERdataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len, label2id, overlap=0.3):
        self.len = len(sentences)
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        
    def __getitem__(self, index):
        # step 1: tokenize 
        tokenized_sentence = self.sentences[index]  
        tokenized_sentence = tokenize_and_preserve_labels(tokenized_sentence, self.tokenizer)

        # step 2: add special tokens 
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        raw_sent_len = len(tokenized_sentence)
        
        # step 3: truncating/padding
        maxlen = self.max_len
        if (len(tokenized_sentence) > maxlen):    # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
        else:   # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'length':raw_sent_len
        } 
    
    def __len__(self):
        return self.len
    
class BertConfig:
    def __init__(self, config):
        self.model_name = config['model_name']
        self.max_length = config["max_length"]
        self.dropout = config['dropout']
        self.num_tags = config['num_tags']
        self.device = config['device']
        self.label2id = config["label2id"]
        self.id2label = config["id2label"]
        self.task_specific_params = {}  # Add any necessary attributes here


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.distilbert = AutoModel.from_pretrained(config.model_name)  # DistilBERT model
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.num_labels = config.num_tags
        self.num_tokens = 512
        self.hidden_size = 768
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(2)
        self.device = config.device

    def forward(self, input_ids, attention_mask, labels=None):
        loss_fct = nn.CrossEntropyLoss()  
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs[0]
        #sequence_output = self.dropout(sequence_output)
        
        # Apply the classification layer to the sequence representation
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[1:]
        
        if labels is not None:
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

            # Reshape the labels to match logits
            #labels = labels.view(-1)  # Flatten the labels tensor
            outputs = (loss,) + outputs
        else:
            logits = self.softmax(logits)
            outputs = (logits,) + outputs[1:]
            
        
        return outputs
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.distilbert(input_ids, attention_mask)
            sequence_output = outputs[0]
            #sequence_output = self.dropout(sequence_output)

            # Apply the classification layer to the sequence representation
            logits = self.softmax(self.classifier(sequence_output))
            final =torch.argmax(logits, axis=2)

        return final
    
    def can_generate(self):
        return False

def merge_tokens_and_labels(tokens, labels):
    merged_tokens = []
    merged_labels = []
    
    current_token = []
    current_label = None
    current_start_offset = None
    
    for token, (label, offset) in zip(tokens, labels):
        tag = label.strip("I-").strip("B-")
        if tag == 'O':
            if current_token:
                # Finish the current merged entity
                merged_tokens.append(' '.join(current_token))
                merged_labels.append((current_label, [current_start_offset, offset[1]]))
                current_token = []
                current_label = None
                current_start_offset = None
            # Add the 'O' label token as is
            merged_tokens.append(token)
            merged_labels.append((label, offset))
        else:
            if current_label is None:
                # Start a new entity
                current_token = [token]
                current_label = tag
                current_start_offset = offset[0]
            else:
                if tag == current_label:
                    # Continue the current entity
                    current_token.append(token)
                else:
                    # Finish the current entity and start a new one
                    merged_tokens.append(' '.join(current_token))
                    merged_labels.append((current_label, [current_start_offset, offset[1]]))
                    current_token = [token]
                    current_label = tag
                    current_start_offset = offset[0]
    
    # If there's an unfinished entity at the end
    if current_token:
        merged_tokens.append(' '.join(current_token))
        merged_labels.append((current_label, [current_start_offset, offset[1]]))
    
    return merged_tokens, merged_labels