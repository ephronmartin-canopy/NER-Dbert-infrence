from transformers import AutoTokenizer,set_seed, AutoModel
import torch.nn as nn
import torch



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
