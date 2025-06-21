# proxy.py
from fastapi import FastAPI, Request
from starlette.responses import Response
import httpx

TARGET = "http://prod0-intuitionx-llm-router-v2.sprinklr.com"
app = FastAPI()

import json
import httpx
from fastapi import Request, Response # Assuming you are using FastAPI

# Assume TARGET is defined elsewhere, for example:
# TARGET = "http://your-target-service.com"

async def forward(path: str, request: Request):
    default_body_params = {
        "client_identifier": "ml-ca-dev",
        "temperature": 0,
        "max_tokens": 300,
        "tracking_params": {
            "release": "ml_ca_divyansh"
        }
    }

    # 1. Get and decode the request body
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")

    # 2. Parse the body string into a dictionary and merge with defaults
    final_body_params = default_body_params.copy()
    if body_str:
        try:
            incoming_params = json.loads(body_str)
            final_body_params.update(incoming_params)
        except json.JSONDecodeError:
            # Handle cases where the body is not valid JSON, if necessary.
            # For now, we'll proceed with only the default params.
            # You could also return a 400 Bad Request error here.
            pass

    # 3. Convert the merged dictionary back to a JSON string for the request
    #    This is the new body that will be forwarded.
    updated_body_content = json.dumps(final_body_params)

    # Prepare headers and URL for the forwarded request
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
    # It's good practice to let httpx set the correct content-length
    headers["content-type"] = "application/json" # Ensure the target service knows it's JSON

    url = f"{TARGET}{path}"

    async with httpx.AsyncClient() as client:
        # 4. Use the 'updated_body_content' in the outgoing request
        resp = await client.request(
            request.method,
            url,
            content=updated_body_content,
            headers=headers,
            params=request.query_params
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp.headers
    )
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # rewrite to /chat-completion
    return await forward("/chat-completion", request)


@app.post("/chat/completions")
async def chat_completions(request: Request):
    # rewrite to /chat-completion
    return await forward("/chat-completion", request)

@app.get("/v1/models")
async def models(request: Request):
    # rewrite to /models
    return await forward("/models", request)

@app.post("/v1/completions")
async def completions(request: Request):
    # rewrite to /completion
    return await forward("/completion", request)

# add any other routes you need...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4001)
