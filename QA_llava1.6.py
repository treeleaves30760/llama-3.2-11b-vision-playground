import os
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import dotenv
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Load model and processor
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)
processor = LlavaNextProcessor.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)

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
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0])

# Folder containing images
image_folder = os.path.join("images", "v1")

# Questions to ask
questions = [
    "請問圖片中是台灣哪一個景點",
    "請問圖片中是哪一個景點"
]

# Output file
output_file = "output_llava1.6.md"

# Process all images
with open(output_file, "w", encoding="utf-8") as f:
    for image_file in tqdm(os.listdir(image_folder)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(image_folder, image_file)
            
            f.write(f"## Image: {image_file}\n\n")
            
            f.write(f'![{image_file}]({image_path})\n\n')
            
            for question in questions:
                answer = analyze_image(image_path, question)
                f.write(f"### Question: {question}\n\n")
                f.write(f"Answer: {answer}\n\n")
            
            f.write("---\n\n")

print(f"Analysis complete. Results written to {output_file}")