from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

# Load the tokenizer and the pre-trained RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')


# Define a function to generate paraphrases
def generate_paraphrases(sentence):
    # Tokenize the input sentence
    tokenized = tokenizer.encode(sentence, add_special_tokens=True)

    # Generate paraphrases by replacing one token at a time
    paraphrases = []
    for i in range(1, len(tokenized)-1):
        masked_token = tokenized[i]
        tokenized_copy = tokenized.copy()
        tokenized_copy[i] = tokenizer.mask_token_id
        input_ids = torch.tensor([tokenized_copy])
        logits = model(input_ids)[0]
        mask_logits = logits[0, i]
        top_k_tokens = torch.topk(mask_logits, k=10, dim=0).indices.tolist()
        for token in top_k_tokens:
            tokenized_copy[i] = token
            paraphrase = tokenizer.decode(tokenized_copy)
            paraphrases.append(paraphrase)

    # Return the paraphrases
    return list(set(paraphrases))


# Test the function on a sample sentence
sentence = 'Why do I need to keep informed?'
paraphrases = generate_paraphrases(sentence)
paraphrases = [paraphrase.replace("<s>", "").replace(
    "</s>", "") for paraphrase in paraphrases]
print(paraphrases, len(paraphrases))
