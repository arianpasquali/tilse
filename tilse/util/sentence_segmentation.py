import syntok.segmenter as segmenter

def sentence_segmenter(document):
    sentences = []
    for paragraph in segmenter.process(document):
        for sentence in paragraph:
            s_sentence = ""
            for token in sentence:
                # roughly reproduce the input,
                # except for hyphenated word-breaks
                # and replacing "n't" contractions with "not",
                # separating tokens by single spaces
                # print(token.value, end=' ')
                s_sentence += token.value + " "
            # print()  # print one sentence per line
            sentences.append(s_sentence)
        # print()  # separate paragraphs with newlines

    return "\n".join(sentences)