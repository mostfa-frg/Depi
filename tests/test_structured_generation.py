import sys
from pathlib import Path

# Make imports work whether this file is run from the project root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.infrence.model import load_model
from structured.generator import StructuredGeneration
from structured.schemas   import TicketOutput
from prompts.base   import build_ticket_classifier_prompt



def main() : 
    tokenizer, model = load_model()

    generator = StructuredGeneration(model, tokenizer)

    user_query = """Your support team was very helpful, thank you!"""

    prompt = build_ticket_classifier_prompt(user_query)

    result = generator.generate(prompt, TicketOutput)

    print(result)


if __name__ == '__main__' :
    main()
