def get_sample_reviews():
    """Return a list of sample reviews with gold sentiment and difficulty tags."""
    samples = [
        {
            "text": "The movie was short, simple, and absolutely wonderful.",
            "gold": 1,
            "difficulty": "Simple positive",
            "why": "Straightforward positive sentiment, no ambiguity."
        },
        {
            "text": "The first half was boring and predictable, but the ending completely blew me away.",
            "gold": 1,
            "difficulty": "Shifting sentiment",
            "why": "Starts negative, ends positive — requires weighing overall tone."
        },
        {
            "text": "Yeah, sure, this was the 'best' film ever... if you enjoy watching paint dry.",
            "gold": 0,
            "difficulty": "Sarcasm",
            "why": "Positive words used sarcastically — model must detect irony."
        },
        {
            "text": "The acting was decent, but the script was weak and the pacing dragged.",
            "gold": 0,
            "difficulty": "Mixed but overall negative",
            "why": "Contains both positive and negative cues — requires nuance."
        },
        {
            "text": "I didn’t expect much, yet it turned out surprisingly good.",
            "gold": 1,
            "difficulty": "Low expectations flipped",
            "why": "Negation and contrast — requires handling subtle shift."
        }
    ]
    return samples


def evaluate_samples(pred_probs, threshold=0.5):
    """
    Pretty-print evaluation of sample reviews.
    
    Args:
        pred_probs: list of floats (probability of class 1 = positive sentiment)
        threshold: decision threshold for classification
    """
    samples = get_sample_reviews()
    for i, (sample, prob) in enumerate(zip(samples, pred_probs)):
        pred_label = int(prob >= threshold)
        confidence = prob if pred_label == 1 else 1 - prob
        print(f"\nSample {i+1}")
        print(f"Text       : {sample['text']}")
        print(f"Gold       : {sample['gold']} ({'positive' if sample['gold']==1 else 'negative'})")
        print(f"Difficulty : {sample['difficulty']} — {sample['why']}")
        print(f"Predicted  : {pred_label} ({'positive' if pred_label==1 else 'negative'}) "
              f"with confidence {confidence:.2f}")
