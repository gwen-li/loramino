from transformers import GPTNeoXForCausalLM, AutoTokenizer
from .model import Model

class Pythia(Model):
    def __init__(self, num_params: str):
        super().__init__()
        self.model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{num_params}-deduped",
            revision="step3000",
            cache_dir=f"./pythia-{num_params}-deduped/step3000",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            f"EleutherAI/pythia-{num_params}-deduped",
            revision="step3000",
            cache_dir=f"./pythia-{num_params}-deduped/step3000",
        )
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
        




