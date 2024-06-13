from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Pipeline, TokenClassificationPipeline
from .data_process import tokenize, clean_punctuations,merge_tokens_and_labels



class CustomNERPipeline(TokenClassificationPipeline):
    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

    def preprocess(self, inputs, **preprocess_params):
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        
        # Tokenize the input and apply cleaning functions
        tokenized_sentences = [tokenize(sent) for sent in inputs.splitlines()]
        inputs = self.tokenizer(
            self.clean_sents(tokenized_sentences),
            return_tensors='pt',  # Specify the framework as 'pt' for PyTorch
            padding=True,    
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            **tokenizer_params,
        )
        inputs.pop("overflow_to_sample_mapping", None)
        num_chunks = len(inputs["input_ids"])

        for i in range(num_chunks):
            model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            model_inputs["sentence"] = tokenized_sentences[i]
            model_inputs["is_last"] = i == num_chunks - 1

            yield model_inputs

    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")
        
        prediction = self.model.predict(model_inputs["input_ids"], model_inputs["attention_mask"])
 
        return {
            "logits": prediction,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "is_last": is_last,
            **model_inputs,
        }
    def postprocess(self, model_outputs):
        print("\n\n>>>>>> Model output ..........................")
        # Combine the predictions from all batches
        orginal_sentences = []
        last_offset = 0
        newSent, newline = [], []
        
        for batch in model_outputs:
            orginal_sentences.append(batch["sentence"])
            wp_preds = self.FilterValues(ids=batch["input_ids"][0], logits=batch["logits"][0], offset_mapping=batch["offset_mapping"])
            for pair in wp_preds:
                if pair[0].startswith("##") or pair[0] in ['[CLS]', '[PAD]']:
                    # skip prediction
                    continue
                else:
                    if pair[0] != "[SEP]":
                        token, label, (start, end) = pair
                        adjusted_start = start + last_offset
                        adjusted_end = end + last_offset
                        newline.append((token, label, [adjusted_start, adjusted_end]))
                    else:
                        if newline:
                            newSent.append(newline)
                            last_token = newline[-1]
                            last_offset = last_token[2][1]                       
                        newline = []
                        

        # Adding the last line if not ended with [SEP]
        if newline:
            newSent.append(newline)

        # Mapping tokens to their respective labels
        final_output = {}
        for sent_idx, sent_labels in enumerate(newSent):
            tokens = orginal_sentences[sent_idx]  # Original sentence tokens

            # Ensure sent_labels are in the form (label, offset)
            sent_labels = [(label, offset) for token, label, offset in sent_labels]
            
            tokens, sent_labels = merge_tokens_and_labels(tokens, sent_labels)

            assert len(tokens) == len(sent_labels), f"Token length {len(tokens)} doesn't match label length {len(sent_labels)}"

            for token, label in zip(tokens, sent_labels):
                tag = label[0].strip("I-").strip("B-")
                if tag != 'O':
                    if tag not in final_output:
                        final_output[tag] = []
                    final_output[tag].append({"token": token, "offset": label[1]})

        return final_output

    
    @staticmethod
    def clean_sents(sentences):
        results=  [" ".join(clean_punctuations(sent)) for sent in sentences]
        print(results)
        return results
    
    def FilterValues(self, ids, logits, offset_mapping):
        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.id2label[i] for i in logits.squeeze().tolist()]
        padTokens = tokens.count("[PAD]")
        if  padTokens > 0:
            newTokens = tokens[:-padTokens]
            newPred = token_predictions[:-padTokens]
        else: 
            newPred, newTokens = token_predictions,tokens
        wp_preds = list(zip(newTokens, newPred, offset_mapping.squeeze().tolist() ))
        return wp_preds
   