import json

from src.taxonomy import TaxonomyNode, Taxonomy
from llms.client_app import ask_model_plus
import re

system_prompt = """
SYSTEM:
You are an expert intent-tagger. You will be given a document and a fixed list of allowed categories.

TASK:
1. Read the document carefully.
2. For each of your top at max 3 choices, provide:
   • label
   • confidence
   • a few sentence rationale **specific to that label only** clearly justifying the choice.
3. If none of the options fit then only, Give “None” with a rationale. 
FORMAT:
```json
{
  "candidates": [
    {
      "label":   "<Category1>",
      "confidence": 0.72,
      "rationale": "…why Category1 matches this abstract…"
    }
    ...
  ]
}
```"""

user_prompt = """
Document:
{document}


Allowed categories.
{categories}
"""


async def choose_intents(
    document: str,
    options: list[TaxonomyNode],
    model: str = "local-qwen3:0.6b"
):
    formatted_nodes = [f"{opt.name} : {opt.description}" for opt in options]
    options_str = "\n".join(formatted_nodes)
    prompt = user_prompt.format(document=document, categories=options_str)
    response = await ask_model_plus(model, prompt, system_prompt, track=True)
    return parse_response(response)

async def tag_n_level(
    document: str,
    taxonomy: Taxonomy,
    model: str = "local-qwen3:0.6b"
) -> dict:
    """
    Recursively tags an N-level taxonomy for the given document.

    Args:
        document: The text (title + abstract) to classify.
        taxonomy: The Taxonomy object defining the hierarchy (root.children are level-1 options).
        model: The LLM model identifier for choose_intents.

    Returns:
        Nested dict:
          - prediction: top label
          - candidates: list of {label, confidence, rationale}
          - children: nested dict for the chosen label's subtree (always present, empty if leaf)
    """
    async def _tag_options(options: list[TaxonomyNode]) -> list[dict]:
        final, _ = await choose_intents(document, options, model)
        return final.get("candidates", [])

    async def _recurse(options: list[TaxonomyNode]) -> dict:
        candidates = await _tag_options(options)
        if not options or not candidates:
            return {}

        top_label = candidates[0]["label"]
        result = {
            "prediction": top_label,
            "candidates": candidates,
            "children": {}
        }

        # descend into the chosen child's subtree if available
        chosen_node = next((opt for opt in options if opt.name == top_label), None)
        if chosen_node and chosen_node.children:
            result["children"] = await _recurse(
                list(chosen_node.children.values())
            )
        return result

    return await _recurse(list(taxonomy.root.children.values()))


def parse_response(response):
    response_think = extract_text_between('<think>', '</think>', response)
    response_json = extract_text_between('```json', '```', response)
    final = {}
    try:
        final = json.loads(response_json[0]) if response_json else {}
    except json.decoder.JSONDecodeError:
        print("Could not parse response as JSON")
    return final, (response_think[0] if response_think else None)

def extract_text_between(start_delimiter, end_delimiter, text):
    pattern = re.escape(start_delimiter) + r'(.*?)' + re.escape(end_delimiter)
    return re.findall(pattern, text, re.DOTALL)

if __name__ == "__main__":
    print(system_prompt)
    print(user_prompt)
