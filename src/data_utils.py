from collections import Counter


def read_conll_file(filepath):
    """
    Read data from CoNLL format file and preprocess it into sentences.

    Args:
        filepath (str): Path to the data file

    Returns:
        list: List of sentences, where each sentence is a list of (word, tag) tuples
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    sentences = []
    current_sentence = []

    for line in content.split('\n'):
        line = line.strip()
        if line == '':
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue

        parts = line.split('\t')
        if len(parts) == 2:
            word, tag = parts
            current_sentence.append((word, tag))

    if current_sentence:  # Add the last sentence if file doesn't end with empty line
        sentences.append(current_sentence)

    return sentences


def get_dataset_stats(sentences):
    """
    Get statistics about a dataset.

    Args:
        sentences (list): List of sentences from read_conll_file

    Returns:
        dict: Dictionary containing dataset statistics
    """
    # Basic statistics
    num_sentences = len(sentences)
    num_tokens = sum(len(sentence) for sentence in sentences)

    # Count tags
    tag_counter = Counter()
    for sentence in sentences:
        for _, tag in sentence:
            tag_counter[tag] += 1

    # Count sentence lengths
    sentence_lengths = [len(sentence) for sentence in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentences else 0

    # Count unique words
    unique_words = set()
    for sentence in sentences:
        for word, _ in sentence:
            unique_words.add(word)

    return {
        'num_sentences': num_sentences,
        'num_tokens': num_tokens,
        'unique_words': len(unique_words),
        'avg_sentence_length': avg_sentence_length,
        'tag_counter': tag_counter,
        'sentence_lengths': sentence_lengths
    }


def print_dataset_stats(stats, dataset_name):
    """
    Print statistics about a dataset.

    Args:
        stats (dict): Dictionary containing dataset statistics
        dataset_name (str): Name of the dataset for display
    """
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    print(f"Number of sentences: {stats['num_sentences']}")
    print(f"Number of tokens: {stats['num_tokens']}")
    print(f"Number of unique words: {stats['unique_words']}")
    print(f"Average sentence length: {stats['avg_sentence_length']:.2f} tokens")
    print(f"Number of unique tags: {len(stats['tag_counter'])}")

    # Top 10 most common tags
    print("\nTop 10 most common tags:")
    for tag, count in stats['tag_counter'].most_common(10):
        percentage = (count / stats['num_tokens']) * 100
        print(f"{tag}: {count} ({percentage:.2f}%)")

    # Entity type distribution
    entity_types = [tag for tag in stats['tag_counter'].keys() if tag != 'O']
    print("\nEntity type distribution:")
    for tag in sorted(entity_types):
        count = stats['tag_counter'][tag]
        percentage = (count / stats['num_tokens']) * 100
        print(f"{tag}: {count} ({percentage:.2f}%)")


def extract_word_tag_sequences(filepath):
    """
    Extract separate word and tag sequences from a CoNLL file.
    
    Args:
        filepath (str): Path to the CoNLL format file
        
    Returns:
        tuple: (words_sequences, tags_sequences) where each is a list of lists
    """
    sentences = read_conll_file(filepath)
    
    word_sequences = []
    tag_sequences = []
    
    for sentence in sentences:
        words = [word for word, _ in sentence]
        tags = [tag for _, tag in sentence]
        
        word_sequences.append(words)
        tag_sequences.append(tags)
        
    return word_sequences, tag_sequences
