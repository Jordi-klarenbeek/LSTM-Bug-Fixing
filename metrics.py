from nltk.translate.bleu_score import sentence_bleu

# BLUE score
def calcBleu(reference, hypothesis):
    reference_strings_lst = []

    # sentence_bleu expects a list with reference, so with only one reference it will get a list with one reference list of tokens
    reference_strings_lst.append(reference)

    return sentence_bleu(reference_strings_lst, hypothesis)

# Top 1 accuracy
def calcMatch(reference, hypothesis):
    # Check if all elements of reference and hypothesis match exactly
    if all(map(lambda x, y: str(x) == str(y), reference, hypothesis)):
        return 1
    else:
        return 0

