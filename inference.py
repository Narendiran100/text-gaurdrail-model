import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextGuardrailSST:
    """Text classification model for detecting toxic content using SST-2 fine-tuned DistilBERT."""
    
    def __init__(self, model_path=None):
        """Initialize the model.
        
        Args:
            model_path: Path to the saved model. If None, loads the base SST-2 model.
            Download the model from google drive link in local system: https://drive.google.com/file/d/1ACeFGelgA5T8kMlFf4USnVzdbTni-Jw_/view?usp=drive_link
            pass this path to the model in the model_path argument of TextGuardrailSST class.
        """
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # Use the base SST-2 model if no fine-tuned model is provided
            model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        self.model.eval()
        
    def check_text(self, text, threshold=0.5):
        """Check if input text is toxic.
        
        Args:
            text: Input text to classify
            threshold: Confidence threshold for unsafe classification
            
        Returns:
            dict: Classification results with text, label, confidence, and probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get prediction and confidence
        unsafe_prob = float(probabilities[0][1])
        label = "unsafe" if unsafe_prob > threshold else "safe"
        confidence = unsafe_prob if label == "unsafe" else float(probabilities[0][0])
        
        return {
            "text": text,
            "label": label,
            "confidence": f"{confidence:.2%}",
            "probabilities": {
                "safe": f"{float(probabilities[0][0]):.4f}",
                "unsafe": f"{float(probabilities[0][1]):.4f}"
            }
        }
    
    def batch_check(self, texts, threshold=0.5):
        """Check multiple texts at once.
        
        Args:
            texts: List of strings to classify
            threshold: Confidence threshold for unsafe classification
            
        Returns:
            list: List of dictionaries containing results for each text
        """
        results = []
        for text in texts:
            results.append(self.check_text(text, threshold))
        return results


def main():
    """Example usage of the TextGuardrailSST class."""
    # Initialize the guardrail
    guardrail = TextGuardrailSST()
    
    # Example texts
    test_texts = [
        "Hello, how are you today?",
        "I really enjoyed the movie!",
        "You're so stupid and worthless",
        "The weather is nice today",
        "I hate everyone here",
        "Let's work together to solve this problem",
        "You are a complete idiot"
    ]
    
    print("Testing individual texts:")
    for text in test_texts:
        result = guardrail.check_text(text)
        print(f"\nText: {result['text']}")
        print(f"Classification: {result['label']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Probabilities: {result['probabilities']}")
    
    print("\nBatch testing:")
    results = guardrail.batch_check(test_texts)
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Classification: {result['label']}")
        print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    main()
