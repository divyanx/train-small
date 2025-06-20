import asyncio
import json
import uuid

from openai import AsyncOpenAI  # 1. Import the Async client
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
import time

client = AsyncOpenAI(
    base_url="http://localhost:4000",
)
token_counter = []
run_id = uuid.uuid4()


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

async def ask_model_plus(model: str, prompt: str, system: str = "You are an assistant", verbose: bool = False, track: bool = True):
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                ChatCompletionSystemMessageParam(role="system", content=system),
                ChatCompletionUserMessageParam(role="user", content=prompt)
            ],
        )

        if track:
            try:
                await save_num_tokens(model, completion.usage.prompt_tokens, completion.usage.completion_tokens)
            except Exception as e:
                print(f"--- An error occurred while saving token {model} --- \n {e}")

            try:
                await save_llm_interactions(model, prompt, system, completion)
            except Exception as e:
                print(f"--- An error occurred while saving interactions {model} --- \n {e}")


        if verbose:
            print(f"--- Response from {model} ---")
            print(completion.choices[0].message.content)
            print("-" * 25 + "\n")

        return completion.choices[0].message.content
    except Exception as e:
        print(f"--- An error occurred for {model} ---")
        print(f"{e}\n")
        print("-" * 25 + "\n")


async def save_num_tokens(model, input_token, output_token, save_at = 100):
    time_stamp = int(time.time())
    token_counter.append((model, input_token, output_token, time_stamp))
    if len(token_counter) % save_at == 0:
        file_name = str(run_id) + ".tokens"
        with open(file_name, "w") as f:
            f.write(str(token_counter))


async def save_llm_interactions(model, prompt, system, completion):
    file_name = './llm_logs/' + str(run_id) + "_llm" + ".json"
    with open(file_name, "a") as f:
        interaction = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "messages": completion.choices[0].message.content,
            "timestamp": time.time(),
        }
        f.write(json.dumps(interaction))


# 5. Create a main async function to run our concurrent tasks
async def main():
    user_prompt = "Explain the concept of zero-shot learning in 2-3 sentences."

    # 6. Use asyncio.gather to run all model calls concurrently
    await asyncio.gather(
        # ask_model(model="gpt-4o-mini", prompt=user_prompt),
        # ask_model(model="gemini-2.5-flash", prompt=user_prompt), # Corrected your model name
        ask_model(model="local-qwen3:0.6b", prompt=user_prompt)
    )


# 7. Run the main async function using asyncio.run()
if __name__ == "__main__":
    # Note: You might need to install 'nest_asyncio' if you run into issues
    # in environments like Jupyter notebooks. For scripts, this is fine.
    # import nest_asyncio
    # nest_asyncio.apply()
    asyncio.run(main())