# import pandas as pd
# data = pd.read_csv("BankFAQs.csv")
# import json
# intents = [ {
#       "tag": list(row)[0][2],
#       "patterns": [list(row)[0][0]],
#       "responses": [list(row)[0][1]]
#     }for row in zip(data.values)]

# dict = {'intents':intents}
# json.dump(dict,open("intents.json","w"),indent=5)

import pandas as pd
import json


def pre_process(file_csv):
  data = pd.read_csv(f"{file_csv}")

  def create_tag(pattern):
      tag = pattern.replace(" ", "_")  # Replace spaces with underscores
      return tag.lower()

  intents = [{
      "tag": create_tag(row[0]),       # Use the modified pattern sentence as the tag
      "patterns": [row[0]],            # Use the original pattern sentence as the pattern
      "responses": [row[1]]
  } for row in data.values]

  dict_data = {'intents': intents}

  # Save the JSON data to a file
#   with open(f"{file_csv[:-4]}.json", "w") as json_file:
  with open("intents.json", "w") as json_file:
      json.dump(dict_data, json_file, indent=4)

# import pandas as pd
# import json
# # from transformers import *
# data = pd.read_csv("internet_dataset.csv")

# model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
# tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")


# from parrot import Parrot

# parrot = Parrot()


# def create_unique_tags(pattern):
#     words = pattern.lower().split()  # Split pattern into words and convert to lowercase
#     unique_tags = list(set(words))  # Get unique words as tags
#     return unique_tags



# def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=1, num_beams=5):
#   # tokenize the text to be form of a list of token IDs
#   inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
#   # generate the paraphrased sentences
#   outputs = model.generate(
#     **inputs,
#     num_beams=num_beams,
#     num_return_sequences=num_return_sequences,
#   )
#   # decode the generated sentences using the tokenizer to get them back to text
#   return tokenizer.batch_decode(outputs, skip_special_tokens=True)



# a = 'Hi my name is asad'
# paraphrases = parrot.augment(input_phrase=a)
# if paraphrases:
#   for paraphrase in paraphrases:
#     print(paraphrase)


# import nlpaug.augmenter.sentence as nas
# aug = nas.ContextualWordEmbsForSentenceAug()

# augmented_data = aug.augment('What are some of the warning signs of mental illness?',1)
# print(augmented_data)
# print(get_paraphrased_sentences(model,tokenizer, a, num_return_sequences=1, num_beams=5))



# intents = []

# for _, row in data.iterrows():
#     pattern = row[0]
#     response = row[1]

#     # Paraphrase pattern and response
#     paraphrased_pattern = " ".join(create_unique_tags(pattern))  # Create unique tag-based pattern
#     paraphrased_response = model.generate(tokenizer.encode(response, return_tensors="pt"), max_length=50, num_return_sequences=1)
#     paraphrased_response = tokenizer.decode(paraphrased_response[0], skip_special_tokens=True)

#     # Create unique tags based on pattern words
#     tags = create_unique_tags(pattern)

#     intent = {
#         "tag": "_".join(tags),  # Join unique words as a tag
#         "patterns": [pattern, paraphrased_pattern],
#         "responses": [response, paraphrased_response]
#     }
    
#     intents.append(intent)

# dict_data = {'intents': intents}

# # Save the JSON data to a file
# with open("internet.json", "w") as json_file:
#     json.dump(dict_data, json_file, indent=4)
