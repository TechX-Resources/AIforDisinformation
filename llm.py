from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN8B = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
QWEN32B = "Qwen/Qwen3-32B"


class QwenChatbot:
    def __init__(self, model_name=QWEN8B):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def summarize_prompt(self, user_input):
        # system prompt
        self.history = [
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. When the user makes a claim or shares news, "
                    "summarize it for the user and make a search prompt to be used in the search engine."
                ),
            }
        ]

        messages = self.history + [{"role": "user", "content": user_input}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id,  # Add this argument
        )[len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    def check_truthiness(self, user_input):
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
        2. Briefly explain your reasoning.
        """

        self.history = [
            {
                "role": "system",
                "content": (grading_prompt),
            }
        ]

        messages = self.history + [{"role": "user", "content": user_input}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id,  # Add this argument
        )[len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id,  # Add this argument
        )[len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response


# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
