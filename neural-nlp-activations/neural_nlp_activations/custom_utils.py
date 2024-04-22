import itertools
import copy
import os

import sys
if not "/home/ben/code/jupyterlab" in sys.path: sys.path.append("/home/ben/code/jupyterlab")
from ben_utils import *
from ProjectManager import ProjectManager
PM = ProjectManager()

try:
    print(get_ipython())
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

import pandas as pd
import time

def get_sentence_data(activations_index=0, apai=500):
    """
    apai is the number of sentences per activation index
    """
    i1=apai*activations_index
    i2=apai*(activations_index+1)
    df = PM.load_dataset("gpt_input")
    df=df[(i1 <= df["sentence_idx"]) & (df["sentence_idx"] < i2)]
    sentences_in = [list(x[1]) for x in df.groupby("sentence_idx")["model_words"]]
    append_spaces_in = [list(x[1]) for x in df.groupby("sentence_idx")["append_space"]]
    return sentences_in, append_spaces_in


from neural_nlp_activations.models.implementations import model_pool, model_layers, PytorchWrapper
from neural_nlp_activations.models.implementations import _PytorchTransformerWrapper

# Create a new subclass that inherits from _PytorchTransformerWrapper
class ModifiedPytorchTransformerWrapper(_PytorchTransformerWrapper):
        
    # Override the ModelContainer class inside the new subclass
    class ModelContainer(_PytorchTransformerWrapper.ModelContainer):
        
        def __call__(self, sentences=None, layers=[], append_spaces=None, v=0):
            import torch
            self.model.eval()
            additional_tokens = []
            # If the tokenizer has a `cls_token`, we insert a `cls_token` at the beginning of the text
            # and a [SEP] token at the end of the text. For models without a `cls_token`, no tokens are inserted.
            use_special_tokens = self.tokenizer.cls_token is not None
            # print(f"use_special_tokens: {use_special_tokens}")
            if use_special_tokens:
                additional_tokens += [self.tokenizer.cls_token, self.tokenizer.sep_token]
                if len(text) > 0:
                    text[0] = self.tokenizer.cls_token + text[0]
                    text[-1] = text[-1] + self.tokenizer.sep_token
                    
            if isinstance(sentences[0],list):
                sentences = [" ".join(tokens) for tokens in sentences]
                
            num_words = [len(sentence.split()) for sentence in sentences]
            text = copy.deepcopy(sentences)
            default_tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
            
            if append_spaces:
                tokenized_sentences=[convert_tokenized_sentences_to_gpt(x.split(), y, tokenizer=self.tokenizer)
                                     for x,y in zip(sentences,append_spaces)]
                if v:
                    print("**debug pretokenization effects**")
                    for i,(x,y) in enumerate(zip(tokenized_sentences,default_tokenized_sentences)):
                        for j in range(max(len(x),len(y))):
                            t1=x[j] if j<len(x) else ""
                            t2=y[j] if j<len(y) else ""
                            print(f"i={i},j={j} t1:{t1}, t2:{t2}")
            else:
                tokenized_sentences = default_tokenized_sentences
                                                   
            tokenized_sentences = list(itertools.chain.from_iterable(tokenized_sentences))
            tokenized_sentences = np.array(tokenized_sentences)
            
            # mapping from original text to later undo chain
            sentence_indices = [0] + [sum(num_words[:i]) for i in range(1, len(num_words), 1)]
            max_num_words = 512 if not use_special_tokens else 511
            aligned_tokens = self.align_tokens(
                tokenized_sentences=tokenized_sentences, sentences=sentences,
                max_num_words=max_num_words, additional_tokens=additional_tokens, use_special_tokens=use_special_tokens, v=v)
            
            encoded_layers = [[]] * len(self.layer_names)
            sentence_indexes=[]
            token_indexes=[]
            all_context_lengths=[]
            for context_ids, sentence_index, token_index in aligned_tokens:
                # if sentence_index % 100 == 99: time.sleep(1) 
                # if sentence_index % 10 == 9: time.sleep(.1) 
                time.sleep(.01) 
                # print("sentence_index",sentence_index)
                sentence_indexes.append(sentence_index)
                token_indexes.append(token_index)
                # print("   cont")
                # Convert inputs to PyTorch tensors
                
                tokens_tensor = torch.tensor([context_ids])
                # print("tokens_tensor:",tokens_tensor)
                tokens_tensor = tokens_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

                # Predict hidden states features for each layer
                with torch.no_grad():
                    context_encoding, = self.model(tokens_tensor)[-1:]
                # We have a hidden state for all the layers
                assert len(context_encoding) == len(self.layer_names)
                # take only the encoding of the current word index
                context_length=context_encoding[0].shape[1]
                # print(context_length)
                all_context_lengths.append(context_length)
                
                word_encoding = [encoding[:, -1:, :] for encoding in context_encoding]
                # print_dimensions(word_encoding, title="word_encoding1", depth=0, stats=True)
                word_encoding = [PytorchWrapper._tensor_to_numpy(encoding) for encoding in word_encoding]
                # print_dimensions(word_encoding, title="word_encoding2", depth=0, stats=True)
                encoded_layers = [previous_words + [word_layer_encoding] for previous_words, word_layer_encoding
                                  in zip(encoded_layers, word_encoding)]
            encoded_layers = [np.concatenate(layer_encoding, axis=1) for layer_encoding in encoded_layers]
            
            assert all(layer_encoding.shape[1] == sum(num_words) for layer_encoding in encoded_layers)
            # separate into sentences again
            sentence_encodings = [[layer_encoding[:, start:end, :] for start, end in
                                   zip(sentence_indices, sentence_indices[1:] + [sum(num_words)])]
                                  for layer_encoding in encoded_layers]
            # attatch layer names layers
            sentence_encodings = OrderedDict(zip(self.layer_names, sentence_encodings))
            # select layers
            sentence_encodings = OrderedDict([(layer, encoding) for layer, encoding in sentence_encodings.items()
                                              if layer in layers])
            metadata_df=pd.DataFrame({"context_length":all_context_lengths,"input_indexes":sentence_indexes,"token_indexes":token_indexes})
            return sentence_encodings, metadata_df
        
        def align_tokens(self, tokenized_sentences, sentences, max_num_words, additional_tokens, use_special_tokens, v=0):
            # sliding window approach (see https://github.com/google-research/bert/issues/66)
            # however, since this is a brain model candidate, we don't let it see future words (just like the brain
            # doesn't receive future word input). Instead, we maximize the past context of each word
            sentence_index = 0
            sentences_chain = ' '.join(sentences).split()
            previous_indices = []

            for token_index in tqdm(range(len(tokenized_sentences)), desc='token features',disable=0):
                if tokenized_sentences[token_index] in additional_tokens:
                    continue  # ignore altogether
                # combine e.g. "'hunts', '##man'" or "'jennie', '##s'"
                tokens = [
                    # tokens are sometimes padded by prefixes, clear those here
                    word.lstrip('##').lstrip('▁').rstrip('@@')
                    for word in tokenized_sentences[previous_indices + [token_index]]]
                token_word = ''.join(tokens).lower()
                for special_token in self.tokenizer_special_tokens:
                    token_word = token_word.replace(special_token, '')
                try:
                    if v: print(f"{token_word}/{sentences_chain[sentence_index].lower()} = token_word/sentences_chain[sentence_index].lower()")
                    if sentences_chain[sentence_index].lower() != token_word:
                        previous_indices.append(token_index)
                        continue
                except:
                    print("**** Error line 917 in implementations! *****")
                    print("sentence_index:",sentence_index)
                    print("sentences_chain:",sentences_chain)
                    print("token_word:",token_word)
                    print("additional_tokens:",additional_tokens)
                    print("self.tokenizer_special_tokens:",self.tokenizer_special_tokens)
                    continue
                previous_indices = []
                sentence_index += 1

                context_start = max(0, token_index - max_num_words + 1)
                context = tokenized_sentences[context_start:token_index + 1]
                if use_special_tokens and context_start > 0:  # `cls_token` has been discarded
                    # insert `cls_token` again following
                    # https://huggingface.co/pytorch-transformers/model_doc/roberta.html#pytorch_transformers.RobertaModel
                    context = np.insert(context, 0, tokenized_sentences[0])
                context_ids = self.tokenizer.convert_tokens_to_ids(context)
                yield context_ids, sentence_index - 1, token_index
                
def convert_tokenized_sentences_to_gpt(list_of_words, append_space, tokenizer=None, protected_tokens=["'s", "'t", "'ve", "'ll", "'re", "'d", "'m"]):
    tokens=[]
    for w, append_space in zip(list_of_words, append_space):
        if w in protected_tokens:
            w_tokens=[w]
        else:
            w_tokens = tokenizer.tokenize(w)
            if append_space:
                w_tokens[0] = "Ġ" + w_tokens[0]
        tokens+=w_tokens
        # tokens = tokenizer.tokenize(text)
    return tokens

# def initialize_model_impl(model="gpt2"):
#     model_impl = None or model_pool[model]
#     layers = None or model_layers[model]
#     model_impl._model.eval()
#     model_impl.content.__class__ = ModifiedPytorchTransformerWrapper
#     model_impl._model_container.__class__ = ModifiedPytorchTransformerWrapper.ModelContainer
#     print(f"initialized model={model}, num_layers={len(layers)}, layers: {layers}")
#     return model_impl, layers