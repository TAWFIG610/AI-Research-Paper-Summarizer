import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from src.config.model_config import ModelConfig

class PegasusSummarizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = PegasusTokenizer.from_pretrained(config.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(config.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate_summary(self, text: str) -> str:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding="longest", 
            max_length=self.config.max_input_length
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=self.config.max_target_length,
            num_beams=self.config.num_beams,
            early_stopping=self.config.early_stopping
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
