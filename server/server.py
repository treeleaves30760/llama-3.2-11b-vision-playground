from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 加載模型
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)
processor = AutoProcessor.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)

@app.route('/generation', methods=['POST'])
def generation():
    # 獲取參數
    data = request.json or {}
    prompt = data.get('prompt', '')
    max_new_tokens = int(data.get('max_new_tokens', 30))
    temperature = float(data.get('temperature', 1.0))
    top_p = float(data.get('top_p', 1.0))

    inputs = processor(
        text=prompt,
        images=None,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # 生成输出
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    response_text = processor.decode(output[0], skip_special_tokens=True)
    return jsonify({'response': response_text})

@app.route('/chat', methods=['POST'])
def chat():
    # 獲取參數
    max_new_tokens = int(request.form.get('max_new_tokens', 30))
    question = request.form.get('question', '')
    do_sample = request.form.get('do_sample', 'False').lower() == 'true'

    if do_sample:
        temperature = float(request.form.get('temperature', 1.0))
        top_p = float(request.form.get('top_p', 1.0))
    else:
        temperature = None
        top_p = None

    # 獲取圖片
    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
    else:
        image = None

    # 創建對話模板
    messages = [
        {"role": "user", "content": []}
    ]
    if image is not None:
        messages[0]['content'].append({"type": "image"})
    if question:
        messages[0]['content'].append({"type": "text", "text": question})
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # 生成輸出
    generate_kwargs = {
        'inputs': inputs,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample
    }

    if do_sample:
        generate_kwargs['temperature'] = temperature
        generate_kwargs['top_p'] = top_p

    output = model.generate(**generate_kwargs)
    response_text = processor.decode(output[0], skip_special_tokens=True)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
