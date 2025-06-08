import json
import os
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import difflib


#################### Load model ####################


#if u need 4-bit quantization 
#   uncomment the bnb_config, 
#   move quantization_config into model parameter,
#   delete .to(0)

#bnb_config = BitsAndBytesConfig(load_in_4bit=True)
#quantization_config=bnb_config

model_id = "llava-hf/llava-interleave-qwen-7b-hf"


model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",   
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


# Base datasets folder
base_root = "inova_test/Comprehension"
 

def map_response_to_choice(response: str, choice_list: list[str]):
    """
    Further address the multi-choices problem
    1. if the response is inside the choice-list return
    2. if is A,B,C,D map to choice-list[0-3] 
    3. use a difflib choose the closest answer in choice-list
    """
    abcd_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    resp = response.strip()

    # 1) Exact match
    if resp in choice_list:
        return resp
    
    # 2) Letter mapping
    if resp in abcd_map and abcd_map[resp] < len(choice_list):
        return choice_list[abcd_map[resp]]

    # 3) Closest match (cutoff=0 guarantees ≥1 result when list not empty)
    matches = difflib.get_close_matches(resp, choice_list, n=1, cutoff=0.0)
    if matches:
        return matches[0]


# Go through each dataset folder
for dataset_name in os.listdir(base_root):

    dataset_path = os.path.join(base_root, dataset_name, "test")
    json_path = os.path.join(dataset_path, "test.json")
    images_dir = os.path.join(dataset_path, "images")

    if not os.path.exists(json_path) or not os.path.isdir(images_dir):
        print(f"[!] Skipping {dataset_name} — test.json or images/ missing.")
        continue

    print(f"[+] Processing dataset: {dataset_name}")
    with open(json_path, "r") as f:
        dataset_inf = json.load(f)

    instructions_list = dataset_inf["metadata"]["task_instruction"]
    data_list = dataset_inf["data"]
    question_type = dataset_inf["metadata"]["question_type"]


    for i in tqdm(range(len(data_list)), desc=f"Inferencing {dataset_name}"):
        sample = data_list[i]
        instruction = instructions_list[sample["task_instruction_id"]]
        context = sample["task_instance"]["context"]
        images_path = [os.path.join(images_dir, name) for name in sample["task_instance"]["images_path"]]

        if question_type == "open-ended":

            messages = [{
                "role": "user",
                "content": (
                    [{"type": "text", "text": instruction}]
                    + [{"type": "image", "url": path} for path in images_path]
                    + [{"type": "text", "text": context}]
                )
            }]
        
        elif question_type == "multi-choice":

            choice_list = sample["task_instance"]["choice_list"]
            
            choice_text = (
            "Please choose the best answer **by replying with the exact same words** as one of the choices below:\n"
            + "\n".join([f"- {c}" for c in choice_list])
            )

            if dataset_name == "MIT-States_StateCoherence":

                messages = [{
                "role": "user",
                "content": (
                    [{"type": "text", "text": instruction}]
                    + [{"type": "text", "text": "image set 1:"}]
                    + [{"type": "image", "url": p} for p in images_path[:2]]   # 0.jpg, 1.jpg
                    + [{"type": "text", "text": "image set 2:"}]
                    + [{"type": "image", "url": p} for p in images_path[2:4]]  # 2.jpg, 3.jpg
                    + [{"type": "text", "text": context}]
                    + [{"type": "text", "text": choice_text}]
                    )
                }]

            else:

                messages = [{
                    "role": "user",
                    "content": (
                        [{"type": "text", "text": instruction}]
                        + [{"type": "image", "url": path} for path in images_path]
                        + [{"type": "text", "text": context}]
                        + [{"type": "text", "text": choice_text}]
                    )
                }]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to("cuda")

        output = model.generate(**inputs, max_new_tokens=50)
        input_len = inputs["input_ids"].shape[1]
        generated = output[0][input_len:]
        decoded_response = processor.tokenizer.decode(generated, skip_special_tokens=True)

        if question_type == "multi-choice":

            mapped_response = map_response_to_choice(decoded_response, choice_list)
            sample["response"] = mapped_response
            print(mapped_response)

        else:
            
            sample["response"] = decoded_response.strip()
            print(decoded_response)

    output_filename = f"{dataset_name}_with_response.json"
    save_path = os.path.join(dataset_path, output_filename)
    with open(save_path, "w") as f:
        json.dump(dataset_inf, f, indent=2)

    print(f"[✓] Saved {save_path}")

