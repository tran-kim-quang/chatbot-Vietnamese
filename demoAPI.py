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
# Định nghĩa model cho dữ liệu đầu vào
class ChatInput(BaseModel):
    message: str
@app.get("/")
async def root():
    return {"message":"Zoo đc roi ne hihi"}
@app.post("/chat")
async def chat_api(input_data: ChatInput):
    user_input = input_data.message
    
    # Nếu nhận được 'exit', trả về thông báo tắt chatbot
    if user_input.lower() == 'exit':
        return {"response": "Chatbot đã tắt"}
    
    # Gọi các hàm có sẵn để xử lý
    intent_tag = predict_intent(user_input)
    response = get_response(intent_tag)
    # thêm message là success hay fail gì đó, thêm đống try catch xử lý, sau dễ debug hơn, ví dụ nếu k phản hồi từ AI đc thì message là fail, response = ""
    return {"response": response}

# Chạy API với uvicorn
if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)