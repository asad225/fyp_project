import nlpaug.augmenter.word as naw
# from proof_read import proofread

import nltk

def augment_data(sentence, num_aug):
    augmented_sentences = []
    # Initialize the word augmenter
    aug = naw.SynonymAug(aug_src='wordnet')

    for _ in range(num_aug):
        # Augment the sentence by replacing some words with their synonyms  
        augmented_sentences.append(aug.augment(sentence))

    return augmented_sentences

augment_sentences = augment_data('Why do I need to keep informed?', 5)
print(augment_sentences)

# sens = [proofread(sen[0]) for sen in augment_sentences]
# print(sens)
