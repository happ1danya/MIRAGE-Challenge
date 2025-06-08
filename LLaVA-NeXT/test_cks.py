from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
import torch, pathlib
from pathlib import Path

# 1) load
ckpt_dir = Path("/home/karen/Documents/Projects/mirage/LLaVA-NeXT/checkpoint/llava-interleave-qwen-7b-hf").resolve()
#ckpt = "/checkpoint"
tok, model, proc, _ = load_pretrained_model(
        ckpt_dir,  model_base=None, model_name="llava-interleave-qwen-7b-hf")
model.to(torch.float16)
#load_4bit=True,
# 2) two sample images (replace with real files)
img1 = Image.new("RGB", (224, 224), (255, 0, 0))
img2 = Image.new("RGB", (224, 224), (0, 0, 255))

prompt = "Return 'yes' if the two squares are the same colour, else 'no'."

conv = conv_templates["one_v1"].copy()
conv.messages = []
conv.append_message(conv.roles[0],
    "You are an expert visual assistant. Answer ONLY with 'yes' or 'no'.")
conv.append_message(conv.roles[1], prompt)
conv.append_message(conv.roles[1], "<image>")
conv.append_message(conv.roles[1], "<image>")
conv.append_message(conv.roles[2], None)
chat = conv.get_prompt()

input_ids = tok([chat]).input_ids[0].unsqueeze(0).to(model.device)
pixels = torch.stack([proc.preprocess(i)["pixel_values"][0] for i in (img1, img2)]
                     ).to(model.device)

out = model.generate(input_ids=input_ids,
                     images=pixels,
                     max_new_tokens=3,
                     temperature=0.0,
                     do_sample=False)

print(tok.decode(out[0][input_ids.size(-1):], skip_special_tokens=True))
