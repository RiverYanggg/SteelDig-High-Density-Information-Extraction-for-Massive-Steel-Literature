import os
from openai import OpenAI
import base64

client = OpenAI(
    api_key="sk-EB6KoAilA7WwElAC3ZeHui4zVavcoraWXNbDRAEnFaRMv2o2",
    base_url="https://api.moonshot.cn/v1"
)

def encode_image(image_path):
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, prompt="请详细描述这张图片的内容"):
    # 获取图片的 base64 编码
    base64_image = encode_image(image_path)
    
    # 构建消息，包含图片和文本
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # 调用 Kimi 多模态模型
    response = client.chat.completions.create(
        model="kimi-k2.5",  # 或指定具体版本如 "kimi-k2.5"
        messages=messages,
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    image_path = "/Users/mac/Desktop/paper_extractor/extractor_ref/multimodal-test/ScreenShot_2026-03-16_153055_816.png"
    prompt = "请详细描述这张图片的内容，注意描述位错密度"
    response = analyze_image(image_path, prompt)
    print(response)