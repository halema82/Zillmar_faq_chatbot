
from ast import main

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq = {
      "How long does shipping take?":
            "Delivery time depends on your city - orders within Lahore usually arrive "
            "in 2-3 days, while other cities take 5-6 working days.",
    
        "Do you ship internationally?":
            "Right now we only ship within Pakistan.",
    
        "How can I track my order?":
            "Just message us directly and we'll track your order for you - "
            "we monitor each order ourselves.",
    
        "What are your shipping charges?":
            "Shipping is free on most orders. For lower-priced products, "
            "a small Rs. 200 charge may apply.",
    
        "What materials are your pieces made of?":
            "Our products are usually stainless steel, which means they resist "
            "tarnishing and discoloration - even after weeks of exposure to water "
            "or everyday wear.",
    
        "Are your products safe for sensitive skin?":
            "Yes, they're safe for sensitive skin.",
    
        "Will the color or plating fade or tarnish over time?":
            "No, as long as basic care is taken. The plating won't fade or "
            "tarnish with proper care.",
    
        "How should I care for my jewelry to make it last?":
            "Just follow basic jewelry care - store your piece in its original "
            "packaging after use to keep it in top condition.",
    
        "How do I order a custom name piece?":
            "You can message us on Instagram, TikTok, or WhatsApp Business "
            "to place your order.",
    
        "How long does a custom order take to make?":
            "Custom orders usually take 3-4 days to make.",
    
        "Can I choose the font or style for my custom piece?":
            "Yes! We provide a selection of fonts to choose from when placing "
            "your order, available in both English and Urdu/Arabic.",
    
        "Can custom orders be returned or exchanged?":
            "No, custom orders cannot be returned or exchanged since they're "
            "made specifically for you.",
    
        "What payment methods do you accept?":
            "We accept JazzCash and Cash on Delivery.",
    
        "Do you offer discounts on bulk or first orders?":
            "Yes! We offer discounts on bulk orders and on your first order, "
            "plus free gifts.",
    
        "What is your return or exchange policy?":
            "We currently don't offer a return policy.",
    
        "What if I receive a damaged or wrong item?":
            "We'll make it right - either by sending a replacement or refunding "
            "the amount for that item.",
    
        "Can I cancel my order after placing it?":
            "No, orders can't be cancelled once placed.",
    
        "How can I contact customer support?":
            "You can reach us anytime on Instagram, TikTok, or WhatsApp Business - "
            "we're happy to help with anything.",
    
        "Do you have a physical store?":
            "We're currently online-only, operating through Instagram and Shopify.",
    
        "Where do you ship from?":
            "We're based in Pakistan and ship orders from there.",
}

questions = list(faq.keys())
answers = list(faq.values())

vectorizer = TfidfVectorizer(stop_words='english')
faq_vectors = vectorizer.fit_transform(questions)

def get_best_answer(user_question, threshold=0.3):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    
    best_index = similarities.argmax()
    best_score = similarities[best_index]    
    
    if best_score >= threshold:
        return answers[best_index], best_score

    return (
        "Sorry, I couldn't find an exact answer to that. "
        "Please message us directly on Instagram, TikTok, or WhatsApp "
        "Business and our team will help you out!"
    ), best_score


def main():
    print("=" * 50)
    print(" Zillmar Jewels - FAQ Chatbot")
    print(" Ask me anything about orders, shipping, or products!")
    print(" (type 'exit' to quit)")
    print("=" * 55)
    
    while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit", "bye"):
                print("Bot: Thanks for chatting with Zillmar Jewels! ✨")
                break
            if not user_input:
                continue
    
            answer, score = get_best_answer(user_input)
            print(f"Bot: {answer}")
            print(f"(Similarity score: {score:.2f})")
            
if __name__ == "__main__":
    main()
