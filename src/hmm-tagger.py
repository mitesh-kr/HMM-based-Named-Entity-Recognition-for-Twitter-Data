import numpy as np
import pickle
import math
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


class HMMTagger:
    def __init__(self, use_context=False, n_gram=2):
        """
        Initialize the HMM tagger.

        Args:
            use_context (bool): If True, use context (previous tag) for emission probabilities
            n_gram (int): 2 for bigram, 3 for trigram model for transition probabilities
        """
        self.states = set()  # All possible tags
        self.vocab = set()   # All possible words
        self.start_prob = {}  # Initial state probabilities
        self.trans_prob = {}  # Transition probabilities
        self.emit_prob = {}   # Emission probabilities
        self.use_context = use_context
        self.n_gram = n_gram
        self.tag_counts = Counter()

        # For handling unseen words/transitions
        self.smoothing_value = 1e-10

    def _preprocess_data(self, filepath):
        """
        Read data from file and preprocess it into sentences.

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

    def fit(self, train_file):
        """
        Train the HMM model by estimating parameters from the training data.

        Args:
            train_file (str): Path to the training file
        """
        print(f"Training with context: {self.use_context}, n-gram: {self.n_gram}")
        train_data = self._preprocess_data(train_file)

        # Step 1: Find states (all possible tags)
        for sentence in train_data:
            for word, tag in sentence:
                self.states.add(tag)
                self.vocab.add(word)
                self.tag_counts[tag] += 1

        print(f"Found {len(self.states)} unique tags and {len(self.vocab)} unique words")

        # Step 2: Calculate start probabilities
        start_counts = Counter()
        total_sentences = len(train_data)

        for sentence in train_data:
            if sentence:  # Ensure the sentence is not empty
                _, first_tag = sentence[0]
                start_counts[first_tag] += 1

        # Add smoothing for start probabilities
        for tag in self.states:
            self.start_prob[tag] = (start_counts[tag] + self.smoothing_value) / (total_sentences + self.smoothing_value * len(self.states))

        # Step 3: Calculate transition probabilities
        if self.n_gram == 2:  # Bigram
            self._calculate_bigram_transitions(train_data)
        elif self.n_gram == 3:  # Trigram
            self._calculate_trigram_transitions(train_data)

        # Step 4: Calculate emission probabilities
        if self.use_context:
            self._calculate_contextual_emissions(train_data)
        else:
            self._calculate_simple_emissions(train_data)

    def _calculate_bigram_transitions(self, train_data):
        """Calculate transition probabilities for bigram model."""
        transition_counts = defaultdict(Counter)

        for sentence in train_data:
            for i in range(len(sentence) - 1):
                _, current_tag = sentence[i]
                _, next_tag = sentence[i + 1]
                transition_counts[current_tag][next_tag] += 1

        # Calculate probabilities with smoothing
        for prev_tag in self.states:
            total = sum(transition_counts[prev_tag].values()) + self.smoothing_value * len(self.states)
            self.trans_prob[prev_tag] = {}
            for next_tag in self.states:
                self.trans_prob[prev_tag][next_tag] = (transition_counts[prev_tag][next_tag] + self.smoothing_value) / total

    def _calculate_trigram_transitions(self, train_data):
        """Calculate transition probabilities for trigram model."""
        transition_counts = defaultdict(Counter)

        for sentence in train_data:
            if len(sentence) < 3:
                continue

            for i in range(len(sentence) - 2):
                _, tag_minus_1 = sentence[i]
                _, tag_0 = sentence[i + 1]
                _, tag_plus_1 = sentence[i + 2]

                # Store as (tag_minus_1, tag_0) -> tag_plus_1
                transition_counts[(tag_minus_1, tag_0)][tag_plus_1] += 1

        # Calculate probabilities with smoothing
        self.trans_prob = {}
        for tag_pair in [(t1, t2) for t1 in self.states for t2 in self.states]:
            total = sum(transition_counts[tag_pair].values()) + self.smoothing_value * len(self.states)
            self.trans_prob[tag_pair] = {}
            for next_tag in self.states:
                self.trans_prob[tag_pair][next_tag] = (transition_counts[tag_pair][next_tag] + self.smoothing_value) / total

    def _calculate_simple_emissions(self, train_data):
        """Calculate emission probabilities without context."""
        emission_counts = defaultdict(Counter)

        for sentence in train_data:
            for word, tag in sentence:
                emission_counts[tag][word] += 1

        # Calculate probabilities with smoothing
        self.emit_prob = {}
        for tag in self.states:
            total = sum(emission_counts[tag].values()) + self.smoothing_value * len(self.vocab)
            default_value = self.smoothing_value / total

            # Use a regular dictionary instead of defaultdict with lambda
            self.emit_prob[tag] = {}
            # Set a default value attribute that we can use when accessing unseen words
            self.emit_prob[tag]['__default__'] = default_value

            for word in self.vocab:
                self.emit_prob[tag][word] = (emission_counts[tag][word] + self.smoothing_value) / total

    def _calculate_contextual_emissions(self, train_data):
        """Calculate emission probabilities with context (previous tag)."""
        emission_counts = defaultdict(Counter)

        for sentence in train_data:
            prev_tag = "START"  # Special tag for sentence start
            for word, tag in sentence:
                context = (prev_tag, tag)
                emission_counts[context][word] += 1
                prev_tag = tag

        # Calculate probabilities with smoothing
        self.emit_prob = {}
        for prev_tag in list(self.states) + ["START"]:
            for tag in self.states:
                context = (prev_tag, tag)
                total = sum(emission_counts[context].values()) + self.smoothing_value * len(self.vocab)
                default_value = self.smoothing_value / total

                if context not in self.emit_prob:
                    self.emit_prob[context] = {}
                    # Set a default value attribute that we can use when accessing unseen words
                    self.emit_prob[context]['__default__'] = default_value

                for word in self.vocab:
                    self.emit_prob[context][word] = (emission_counts[context][word] + self.smoothing_value) / total

    def viterbi_decode(self, sentence):
        """
        Use the Viterbi algorithm to find the most likely tag sequence.

        Args:
            sentence (list): List of words in the sentence

        Returns:
            list: List of predicted tags
        """
        # Handle empty sentence
        if not sentence:
            return []

        # For numerical stability, use log probabilities
        V = [{}]  # Viterbi matrix
        path = {}  # Best path

        # Initialize base cases (first observation)
        for state in self.states:
            # Calculate emission probability based on context
            if self.use_context:
                emit_dict = self.emit_prob.get(("START", state), {})
                emit_prob = emit_dict.get(sentence[0], emit_dict.get('__default__', self.smoothing_value))
            else:
                emit_dict = self.emit_prob.get(state, {})
                emit_prob = emit_dict.get(sentence[0], emit_dict.get('__default__', self.smoothing_value))

            V[0][state] = math.log(self.start_prob[state]) + math.log(emit_prob)
            path[state] = [state]

        # Run Viterbi for subsequent observations
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_state in self.states:
                max_prob = float('-inf')
                best_prev_state = None

                # Calculate the best previous state
                if self.n_gram == 2:  # Bigram
                    for prev_state in self.states:
                        # Get transition probability
                        trans_prob = self.trans_prob.get(prev_state, {}).get(curr_state, self.smoothing_value)

                        # Get emission probability based on context
                        if self.use_context:
                            emit_dict = self.emit_prob.get((prev_state, curr_state), {})
                            emit_prob = emit_dict.get(sentence[t], emit_dict.get('__default__', self.smoothing_value))
                        else:
                            emit_dict = self.emit_prob.get(curr_state, {})
                            emit_prob = emit_dict.get(sentence[t], emit_dict.get('__default__', self.smoothing_value))

                        prob = V[t-1][prev_state] + math.log(trans_prob) + math.log(emit_prob)

                        if prob > max_prob:
                            max_prob = prob
                            best_prev_state = prev_state

                elif self.n_gram == 3 and t > 1:  # Trigram (for t > 1)
                    for prev_prev_state in self.states:
                        for prev_state in self.states:
                            # Get transition probability from (prev_prev_state, prev_state) to curr_state
                            trans_prob = self.trans_prob.get((prev_prev_state, prev_state), {}).get(curr_state, self.smoothing_value)

                            # Get emission probability based on context
                            if self.use_context:
                                emit_dict = self.emit_prob.get((prev_state, curr_state), {})
                                emit_prob = emit_dict.get(sentence[t], emit_dict.get('__default__', self.smoothing_value))
                            else:
                                emit_dict = self.emit_prob.get(curr_state, {})
                                emit_prob = emit_dict.get(sentence[t], emit_dict.get('__default__', self.smoothing_value))

                            prev_key = (prev_prev_state, prev_state)
                            if prev_key in V[t-1]:
                                prob = V[t-1][prev_key] + math.log(trans_prob) + math.log(emit_prob)

                                if prob > max_prob:
                                    max_prob = prob
                                    best_prev_state = prev_key

                # Special case for t=1 in trigram model
                elif self.n_gram == 3 and t == 1:
                    for prev_state in self.states:
                        # In this case, we only have one previous state
                        trans_prob = self.trans_prob.get(("START", prev_state), {}).get(curr_state, self.smoothing_value)

                        if self.use_context:
                            emit_dict = self.emit_prob.get((prev_state, curr_state), {})
                            emit_prob = emit_dict.get(sentence[t], emit_dict.get('__default__', self.smoothing_value))
                        else:
                            emit_dict = self.emit_prob.get(curr_state, {})
                            emit_prob = emit_dict.get(sentence[t], emit_dict.get('__default__', self.smoothing_value))

                        prob = V[t-1][prev_state] + math.log(trans_prob) + math.log(emit_prob)

                        if prob > max_prob:
                            max_prob = prob
                            best_prev_state = prev_state

                # Update Viterbi matrix and path
                if best_prev_state is not None:
                    if self.n_gram == 2 or t == 1:
                        V[t][curr_state] = max_prob
                        new_path[curr_state] = path[best_prev_state] + [curr_state]
                    elif self.n_gram == 3 and t > 1:
                        prev_prev, prev = best_prev_state
                        V[t][(prev, curr_state)] = max_prob
                        new_path[(prev, curr_state)] = path.get(best_prev_state, [prev_prev, prev]) + [curr_state]

            path = new_path

        # Find the best final state
        if not V[-1]:  # Handle case where no valid path is found
            return ["O"] * len(sentence)  # Default to "O" tag

        if self.n_gram == 2:
            best_final_state = max(V[-1].items(), key=lambda x: x[1])[0]
            return path[best_final_state]
        else:  # Trigram
            if len(sentence) > 2:
                # For trigram, the last state is a pair
                best_final_pair = max(V[-1].items(), key=lambda x: x[1])[0]
                return path[best_final_pair]
            elif len(sentence) == 2:
                # Special case for very short sentences
                best_final_state = max(V[-1].items(), key=lambda x: x[1])[0]
                return path[best_final_state]
            else:
                # Single word sentence
                best_final_state = max(V[0].items(), key=lambda x: x[1])[0]
                return [best_final_state]

    def predict(self, test_file):
        """
        Predict tags for sentences in the test file.

        Args:
            test_file (str): Path to the test file

        Returns:
            tuple: (predicted_tags, actual_tags)
        """
        test_data = self._preprocess_data(test_file)
        all_predicted_tags = []
        all_actual_tags = []

        for i, sentence in enumerate(test_data):
            # Print progress every 100 sentences
            if (i + 1) % 100 == 0:
                print(f"Processing sentence {i+1}/{len(test_data)}")

            words = [word for word, _ in sentence]
            actual_tags = [tag for _, tag in sentence]

            predicted_tags = self.viterbi_decode(words)

            # Ensure predicted tags match the number of words
            if len(predicted_tags) != len(words):
                print(f"Warning: Mismatch in number of predicted tags ({len(predicted_tags)}) and words ({len(words)})")
                # Truncate or pad predicted tags to match words length
                if len(predicted_tags) < len(words):
                    predicted_tags.extend(["O"] * (len(words) - len(predicted_tags)))
                else:
                    predicted_tags = predicted_tags[:len(words)]

            all_predicted_tags.extend(predicted_tags)
            all_actual_tags.extend(actual_tags)

        return all_predicted_tags, all_actual_tags

    def evaluate(self, test_file):
        """
        Evaluate the model on the test file.

        Args:
            test_file (str): Path to the test file

        Returns:
            dict: Dictionary with evaluation metrics
        """
        predicted_tags, actual_tags = self.predict(test_file)

        # Calculate metrics
        accuracy = accuracy_score(actual_tags, predicted_tags)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_tags, predicted_tags, average='weighted')

        report = classification_report(actual_tags, predicted_tags)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }

        return metrics, predicted_tags, actual_tags

    def save_model(self, filepath):
        """Save the trained model to a file."""
        model_data = {
            'states': self.states,
            'vocab': self.vocab,
            'start_prob': self.start_prob,
            'trans_prob': self.trans_prob,
            'emit_prob': self.emit_prob,
            'use_context': self.use_context,
            'n_gram': self.n_gram,
            'tag_counts': self.tag_counts,
            'smoothing_value': self.smoothing_value
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
            # If there's an error, try saving in a simpler format
            np.save(filepath.replace('.pkl', '.npy'), model_data)
            print(f"Model saved in numpy format instead")

    def load_model(self, filepath):
        """Load a trained model from a file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.states = model_data['states']
            self.vocab = model_data['vocab']
            self.start_prob = model_data['start_prob']
            self.trans_prob = model_data['trans_prob']
            self.emit_prob = model_data['emit_prob']
            self.use_context = model_data['use_context']
            self.n_gram = model_data['n_gram']
            self.tag_counts = model_data['tag_counts']
            self.smoothing_value = model_data['smoothing_value']
            print(f"Model successfully loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True

    def save_predictions(self, predictions, filepath):
        """Save predictions to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for tag in predictions:
                f.write(f"{tag}\n")
        print(f"Predictions saved to {filepath}")
