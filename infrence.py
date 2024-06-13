from transformers import Pipeline, TokenClassificationPipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
from scripts.data_process import tokenize, merge_tokens_and_labels, clean_punctuations
from scripts.model import BertConfig, BertModel
from scripts.pipeline import CustomNERPipeline
import os, torch
import json
import warnings

warnings.filterwarnings("ignore")

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
config = read_json_file("configs/config.json")
label2id = read_json_file("configs/label2id.json")
model_Path = "models/NER_BERT_modelv2.0.pth"




if __name__ == "__main__":
    
    text = """I am John Doe, a software engineer at Google, visited the Golden Gate Bridge in San Francisco, California last summer.
        Ephron is enginner, with  11 years of experience, He met with Dr. Jane Smith, a renowned AI researcher from Stanford University. They discussed potential collaborations on projects funded     by the National Science Foundation. Later, they attended the AI conference held in Silicon Valley, where Elon Musk delivered the keynote speech."""

    
    print("----------------- Started --------------------")

    id2label = {v:k for k,v in label2id.items()}
    model_Path = "models/NER_BERT_modelv2.0.pth"
    
    config["label2id"] =  label2id
    config["id2label"] = id2label

    bertConfig = BertConfig(config)
    model = BertModel(bertConfig)
    if os.path.isfile(model_Path):
        model.load_state_dict(torch.load(model_Path,map_location=torch.device('cpu')))
        print("Model state Loaded :)")
        
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    # Load model and tokenizer      
    # Initialize the custom pipeline
    custom_pipeline = CustomNERPipeline(model=model, tokenizer=tokenizer)
    
    ner_results = custom_pipeline(text)
    
    print("\n\n")

    print(ner_results)
    
    
