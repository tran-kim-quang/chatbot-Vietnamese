from train import predict_intent, get_response
def chat():
    print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'exit':
            break
        
        # Dự đoán intent
        intent_tag = predict_intent(user_input)
        # Lấy phản hồi
        response = get_response(intent_tag)
        print("Chatbot:", response)
