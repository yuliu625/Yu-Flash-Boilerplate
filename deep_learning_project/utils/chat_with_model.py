"""
使用结构化的json格式数据与LLM持续对话。

这里记录了简易的版本，实现的是基础的处理逻辑。
即：维持json的添加，模型输入字符串需要转换，模型输出字符串需要提取。
"""


# 这里是全局变量，之后需要封装在类中，或做其他操作。
message_history: list[dict[str, str]] = [
    {
        'role': 'system',
        'content': 'some instructions.'
    },
]


def get_response_from_ai(message_history: list[dict[str, str]]):
    """
    每个模型都会根据具体的输入进行生成。
    """
    # 这里模拟进行了生成。
    message_history.append({'role': 'ai', 'content': 'torch_models generate some text.'})
    return message_history


def chat_with_model(user_message: str):
    # message_history在这里是全局变量，到处都会被修改。
    # 将用户的输入加入当前的所有历史记录。
    message_history.append({'role': 'user', 'content': user_message})
    # 模型进行生成。
    model_output = get_response_from_ai(message_history)
    # 获取模型新生成的部分，即为模型的推理。
    # ai_message: str = model_output.choice[0].message.content
    ai_message = model_output[-1]['content']  # 这个是模拟的提取输出。
    # 更新历史记录。
    message_history.append({'role': 'ai', 'content': ai_message})
    # 返回输出。
    return ai_message


if __name__ == '__main__':
    while True:
        user_message: str = input('User >_ ')
        ai_message = chat_with_model(user_message)
        print(f"AI >_ {ai_message}")
