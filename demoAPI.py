from fastapi import FastAPI
from pydantic import BaseModel
from train import predict_intent, get_response
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
# Model
class ChatInput(BaseModel):
    message: str
@app.get("/")
async def root():
    return {"message":"Zoo đc roi ne hihi"}
@app.post("/chat")
async def chat_api(input_data: ChatInput):
    user_input = input_data.message

    if user_input.lower() == 'exit':
        return {"response": "Chatbot đã tắt"}

    intent_tag = predict_intent(user_input)
    response = get_response(intent_tag)
    return {"response": response}

# Chạy API với uvicorn
if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
