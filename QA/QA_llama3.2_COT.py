import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import dotenv
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Load model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)
processor = AutoProcessor.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)

def analyze_image(image_path, question):
    image = Image.open(image_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=800)
    return processor.decode(output[0])

# Folder containing images
image_folder = os.path.join("images", "v1")

# Questions to ask
questions = [
    "請先描述圖片有哪些特徵，並列出幾個具有類似特徵的台灣景點，再搭配圖片細節推測圖片中是台灣哪一個景點",
    "請先描述圖片有哪些特徵，並列出幾個具有類似特徵的景點，再搭配圖片細節推測圖片中是哪一個景點",
]

# Output file
output_file = "output_llama3.2_COT.md"

# Process all images
with open(output_file, "w", encoding="utf-8") as f:
    for image_file in tqdm(os.listdir(image_folder)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(image_folder, image_file)
            
            f.write(f"## Image: {image_file}\n\n")
            
            f.write(f'![{image_file}](images/v1/{image_file})\n\n')
            
            for question in questions:
                answer = analyze_image(image_path, question)
                f.write(f"### Question: {question}\n\n")
                f.write(f"Answer: {answer}\n\n")
            
            f.write("---\n\n")

print(f"Analysis complete. Results written to {output_file}")