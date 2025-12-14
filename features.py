import re
import numpy as np

def extract_features(text: str):
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s.strip()]

    words = re.findall(r'\w+', text.lower())

    if len(sentences) == 0 or len(words) == 0:
        return np.zeros(6)

    avg_sentence_length = len(words) / len(sentences)
    text_length = len(words)
    unique_word_ratio = len(set(words)) / len(words)

    punctuation_count = len(re.findall(r'[.,!?;]', text))
    punctuation_ratio = punctuation_count / max(len(text), 1)

    avg_word_length = np.mean([len(w) for w in words])

    return [
        avg_sentence_length,
        text_length,
        unique_word_ratio,
        punctuation_ratio,
        avg_word_length,
        len(sentences)
    ]
