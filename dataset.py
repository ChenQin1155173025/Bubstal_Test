# dataset.py
# E-Commerce Sentiment Dataset for AI Agent Training

# ---------------------------------------------------------
# TRAINING DATA: Use this to build vocabulary and probabilities
# ---------------------------------------------------------
TRAINING_DATA = [
    # --- Happy Class ---
    {"text": "Absolutely love this product, it works perfectly!", "label": "Happy"},
    {"text": "Fast delivery, very satisfied with my purchase.", "label": "Happy"},
    {"text": "Great value for the price, highly recommend.", "label": "Happy"},
    {"text": "The quality is outstanding and it fits perfectly.", "label": "Happy"},
    {"text": "Excellent customer support, they resolved my issue immediately.", "label": "Happy"},
    {"text": "Five stars, exactly what I was looking for.", "label": "Happy"},
    {"text": "Beautiful design and very sturdy materials.", "label": "Happy"},
    {"text": "Arrived earlier than expected, very happy.", "label": "Happy"},
    {"text": "Best purchase I have made this year.", "label": "Happy"},
    {"text": "So glad I bought this, it has been a game changer.", "label": "Happy"},
    
    # --- Frustrated Class ---
    {"text": "Terrible experience, the item broke on the first day.", "label": "Frustrated"},
    {"text": "Shipping took way too long and the box was damaged.", "label": "Frustrated"},
    {"text": "Customer service was incredibly rude and unhelpful.", "label": "Frustrated"},
    {"text": "Completely useless, does not match the description at all.", "label": "Frustrated"},
    {"text": "Waste of money, poor quality materials.", "label": "Frustrated"},
    {"text": "I am still waiting for my refund after three weeks.", "label": "Frustrated"},
    {"text": "Do not buy this, it is a complete scam.", "label": "Frustrated"},
    {"text": "The sizing is completely wrong and they refuse returns.", "label": "Frustrated"},
    {"text": "Extremely disappointed with the lack of communication.", "label": "Frustrated"},
    {"text": "Defective right out of the box, frustrating.", "label": "Frustrated"}
]

# ---------------------------------------------------------
# INFERENCE / TEST DATA: Use this to test your final classifier
# NOTE: These strings contain words that DO NOT appear in the training data.
# ---------------------------------------------------------
TEST_DATA = [
    "The delivery was incredibly quick and the item is brilliant!", # Expected: Happy
    "Horrible scam, totally destroyed my trust in this site.",      # Expected: Frustrated
    "Customer support was fast but the product is useless.",        # Expected: Frustrated (Mixed signals)
    "Outstanding experience, beautiful and sturdy packaging."       # Expected: Happy
]