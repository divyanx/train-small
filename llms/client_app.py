import asyncio
from openai import AsyncOpenAI  # 1. Import the Async client
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

# Point the an asynchronous OpenAI client to your LiteLLM proxy
client = AsyncOpenAI(  # 2. Instantiate the Async client
    base_url="http://localhost:4000",
    # api_key is not needed here, as the proxy manages it
)

# 3. Define the function as an 'async' coroutine
async def ask_model(model: str, prompt: str, system: str = "You are an assistant"):
    """
    Makes an asynchronous request to a model via the LiteLLM proxy.
    """
    print(f"--- Firing request to Model: {model} ---")
    try:
        # 4. Use 'await' for the non-blocking API call (method is 'acreate')
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                ChatCompletionSystemMessageParam(role="system", content=system),
                ChatCompletionUserMessageParam(role="user", content=prompt)
            ],
        )

        print(f"--- Response from {model} ---")
        print(completion.choices[0].message.content)
        print("-" * 25 + "\n")
    except Exception as e:
        print(f"--- An error occurred for {model} ---")
        print(f"{e}\n")
        print("-" * 25 + "\n")


# 5. Create a main async function to run our concurrent tasks
async def main():
    user_prompt = "Explain the concept of zero-shot learning in 2-3 sentences."

    # 6. Use asyncio.gather to run all model calls concurrently
    await asyncio.gather(
        ask_model(model="gpt-4o-mini", prompt=user_prompt),
        ask_model(model="gemini-2.5-flash", prompt=user_prompt), # Corrected your model name
        ask_model(model="local-qwen3:0.6b", prompt=user_prompt)
    )


# 7. Run the main async function using asyncio.run()
if __name__ == "__main__":
    # Note: You might need to install 'nest_asyncio' if you run into issues
    # in environments like Jupyter notebooks. For scripts, this is fine.
    # import nest_asyncio
    # nest_asyncio.apply()
    asyncio.run(main())