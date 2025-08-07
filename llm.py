from groq import Groq


class QwenChatbot:
    def __init__(self, api_key):
        self.client = Groq(
            api_key=api_key,
        )

    def summarize_prompt(self, claims):
        completion = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking assistant. When the user makes a claim or shares news, "
                        "summarize it for the user and rephrase it to a search prompt to be used in the search engine."
                        "You only need to return the search prompt, do not reply anything unrelevant"
                    ),
                },
                {
                    "role": "user",
                    "content": claims,
                },
            ],
            temperature=0.2,
            max_completion_tokens=4096,
            top_p=0.95,
            reasoning_format="hidden",
        )
        return completion.choices[0].message.content

    def check_truthiness(self, search_results, claim):
        # system prompt
        grading_prompt = """
    You are tasked with evaluating the truthfulness of a given input statement.
    Assign a numerical score from 0 to 5 based on the following grading scale:

    5 – Completely True: All claims are verifiable, accurate, and supported by reliable evidence.
    4 – Mostly True: Minor inaccuracies may exist, but the core facts are accurate and not misleading.
    3 – Half True: Roughly an equal mix of accurate and inaccurate or misleading information.
    2 – Mostly False: A small element of truth exists, but the claim is mostly inaccurate or misrepresented.
    1 – Completely False: The statement is entirely inaccurate, fabricated, or contradicted by reliable sources.
    0 – Not Evaluated: There is insufficient information to determine the truthfulness of the statement.

    Your task:
    1. Assign a score (0–5).
    2. Provide a concise explanation.
    3. Clearly state which parts of the claim are TRUE and which are FALSE.
    4. For each true/false part, cite supporting links from the following search results.
    """

        completion = self.client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": grading_prompt},
                {
                    "role": "user",
                    "content": f"Here is the claim from the user: {claim} and here is the search results: {search_results}. Cite supporting links from the following search results only",
                },
            ],
            temperature=0.5,
            reasoning_format="hidden",
        )
        return completion.choices[0].message.content
