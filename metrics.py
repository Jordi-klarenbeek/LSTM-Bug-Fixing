from nltk.translate.bleu_score import sentence_bleu

#TODO: Create different metrics

# BLUE score
def calcBleu(reference, hypothesis, end_token):
    reference_strings = []
    reference_strings_lst = []
    hypothesis_strings = []

    # transform tokens of reference into strings
    for token in reference:
        reference_strings.append(str(token.item()))
        if token.item() == end_token:
            break

    # sentence_bleu expects a list with reference, so with only one refernce it will get a list with one reference list of tokens
    reference_strings_lst.append(reference_strings)

    # transform tokens of hypothesis list into strings
    for token in hypothesis:
        hypothesis_strings.append(str(token))
        if token == end_token:
            break

    return sentence_bleu(reference_strings_lst, hypothesis_strings)

# Top 1 accuracy
def calcMatch(reference, hypothesis):
    # Check if all elements of reference and hypothesis match exactly
    if all(map(lambda x, y: str(x.item()) == str(y), reference, hypothesis)):
        return 1
    else:
        return 0

