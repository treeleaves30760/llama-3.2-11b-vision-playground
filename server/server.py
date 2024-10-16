import io
import base64
import torch
from PIL import Image
from flask import Flask, request, jsonify
from transformers import MllamaForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
import os
import re

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")

app = Flask(__name__)

# 加載模型和處理器
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HUGGINGFACE_API_KEY
)
processor = AutoProcessor.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)

def extract_messages(response):
    pattern = r'<\|start_header_id\|>(user|assistant)<\|end_header_id\|>(.*?)(?=<\|start_header_id\||$)'
    matches = re.findall(pattern, response, re.DOTALL)
    return [{"role": role, "content": content.strip()} for role, content in matches]

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    image_data = data.get('image', '')
    max_new_tokens = data.get('max_new_tokens', 3000)

    # 準備輸入
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": text}
        ]}
    ]

    # 如果提供了圖片，則添加到消息中
    if image_data:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        messages[0]["content"].insert(0, {"type": "image"})
    else:
        image = None

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    if image:
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

    # 生成輸出
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(output[0])

    # 提取並格式化消息
    formatted_messages = extract_messages(response)

    return jsonify({'messages': formatted_messages})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)