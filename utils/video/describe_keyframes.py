
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import requests
import pandas as pd
import cv2
import os

def init_model(model_id, revision, cache_dir="./model_cache"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_type = torch.float16 if device == "cuda" else torch.float32

    model_dir = os.path.join(cache_dir, model_id.replace('/', '_'), revision)

    # Create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        # Model is not saved locally, download and save it
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision,
            torch_dtype=torch_type
        )
        # model.save_pretrained(model_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # tokenizer.save_pretrained(model_dir)
    else:
        # Load the model and tokenizer from the local disk
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_type
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = model.to(device)
    
    return model, tokenizer
    

def encode_image(model, image):
    enc_image = model.encode_image(image)
    return enc_image


def answer_question(model, tokenizer, enc_image, prompt):
    prompt_response = model.answer_question(enc_image, prompt, tokenizer)
    return prompt_response













def get_column_values(csv_file, column_name):
    df = pd.read_csv(csv_file)
    return df[column_name].values.tolist()


model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)

prompt = "Describe the image with as much detail as possible. Generate as much information that can be turned into meta data as possible"

model, tokenizer = init_model(model_id, revision)

enc_image = encode_image(model, image)

prompt_response = answer_question(model,tokenizer, enc_image, prompt)

print(prompt_response)





csv_filepath = "videos/keyframes/extracted_keyframes.csv"
filename_column = 'Filename'

filenames = get_column_values(csv_filepath, filename_column)

# print(filenames)

full_filepaths = [f"videos/keyframes/{''.join(filename.split('_')[:-1])}/{filename}" for filename in filenames]

# print(full_filepaths)

# for filepath in full_filepaths:
#     image = cv2.imread(filepath)
#     # cv2.imshow("Image", image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     enc_image = encode_image(model, image)

#     prompt_response = answer_question(model,tokenizer, enc_image, prompt)

#     print(prompt_response)
