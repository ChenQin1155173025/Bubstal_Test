class NaiveBayesClassifier:
    
    def __init__(self, smoothing_alpha=1.0):
        self.smoothing_alpha = smoothing_alpha
        self.word_counts = {}
        self.total_words = {}
        self.document_counts = {}
        self.priors = {}
        self.vocabulary = set()
        self.denominator = {}
    
    def tokenize(self, text):
        """Standardize text: lowercase, remove punctuation, split"""
        text = text.lower()
        punctuation = '.,!?;:()[]{}"\'`~@#$%^&*+-=<>/\\|'
        
        # Replace punctuation with spaces
        cleaned = []
        for char in text:
            if char in punctuation:
                cleaned.append(' ')
            else:
                cleaned.append(char)
        
        cleaned_text = ''.join(cleaned)
        # Split into tokens
        tokens = []
        current = []
        for char in cleaned_text:
            if char == ' ' or char == '\t' or char == '\n':
                if current:
                    word = ''.join(current)
                    if len(word) > 1:  # Ignore single characters
                        tokens.append(word)
                    current = []
            else:
                current.append(char)
        
        if current:
            word = ''.join(current)
            if len(word) > 1:
                tokens.append(word)
        
        return tokens
    
    def train(self, training_data):
        """Build vocabulary and calculate probabilities"""
        
        # Initialize
        for item in training_data:
            label = item["label"]
            
            if label not in self.word_counts:
                self.word_counts[label] = {}
                self.total_words[label] = 0
                self.document_counts[label] = 0
            
            self.document_counts[label] += 1
            tokens = self.tokenize(item["text"])
            
            # Count each word once per document (multinomial NB)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.vocabulary.add(token)
                self.word_counts[label][token] = self.word_counts[label].get(token, 0) + 1
                self.total_words[label] += 1
        
        # Calculate priors
        total_docs = sum(self.document_counts.values())
        for label in self.word_counts:
            self.priors[label] = self.document_counts[label] / total_docs
        
        # Pre-compute denominators with smoothing
        vocab_size = len(self.vocabulary)
        for label in self.word_counts:
            self.denominator[label] = self.total_words[label] + self.smoothing_alpha * vocab_size
    
    def predict(self, text):
        """Return predicted class and raw probability scores"""
        tokens = self.tokenize(text)
        vocab_size = len(self.vocabulary)
        
        scores = {}
        
        for label in self.word_counts:
            # Start with prior
            prob = self.priors[label]
            
            # Multiply by conditional probability for each token
            for token in tokens:
                count = self.word_counts[label].get(token, 0)
                token_prob = (count + self.smoothing_alpha) / self.denominator[label]
                prob *= token_prob
            
            scores[label] = prob
        
        # Normalize to get valid probabilities
        total = sum(scores.values())
        if total > 0:
            for label in scores:
                scores[label] /= total
        else:
            # If all probabilities underflowed, use priors
            for label in scores:
                scores[label] = self.priors[label]
        
        predicted = max(scores, key=scores.get)
        confidence = scores[predicted]
        
        return predicted, scores, confidence


# Test
if __name__ == "__main__":
    from dataset import TRAINING_DATA, TEST_DATA
    
    nb = NaiveBayesClassifier(smoothing_alpha=1.0)
    nb.train(TRAINING_DATA)
    
    print("="*60)
    print("NAIVE BAYES CLASSIFIER (As Required by Spec)")
    print("="*60)
    
    for text in TEST_DATA:
        pred, scores, conf = nb.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted: {pred}")
        print(f"P(Happy) = {scores.get('Happy', 0):.6f}")
        print(f"P(Frustrated) = {scores.get('Frustrated', 0):.6f}")

