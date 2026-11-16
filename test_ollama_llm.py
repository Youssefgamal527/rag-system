from src.generation.ollama_llm import OllamaLLM


llm = OllamaLLM()

response = llm.generate(
    "Explain what Retrieval-Augmented Generation is in simple terms."
)

print("\nGenerated response:\n")
print(response)