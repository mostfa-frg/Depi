from src.infrence.model import load_model
from prompts.classifier import build_ticket_classifier
from structured.generator import StructuredGeneration
from structured.schemas import TicketOutput


def main():
    tokenizer, model = load_model()
    generator = StructuredGeneration(model, tokenizer)
    prompt = build_ticket_classifier(
        "Your support team was very helpful, thank you!"
    )
    print(generator.generate(prompt, TicketOutput))


if __name__ == "__main__":
    main()
