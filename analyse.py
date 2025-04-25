import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 启用 cuDNN 自动优化
torch.backends.cudnn.benchmark = True

# 加载优化后的 Phi-3.5-mini 模型
model_path = "/share/home/liuxingy/real3dportrait/models/Phi-3.5-mini"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:3",
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 提高 V100 计算效率
    trust_remote_code=True
)

# 使用 torch.compile() 进行模型优化
# model = torch.compile(model)
model = torch.compile(model).to("cuda:3")

tokenizer = AutoTokenizer.from_pretrained(model_path)


def analyze_task(description):
    """调用模型进行推理，返回结构化的 Photo, TTS Prompt, Speaker"""
    # 记录开始时间
    start_time = time.time()
    # **恢复原始 `messages` 格式**
    messages = [
        {"role": "system",
         "content": "You are an advanced AI assistant specializing in media content generation. Given a description, generate structured outputs for photo, TTS Prompt, and Speaker Description."},
        {"role": "user", "content": "Description: I need a video of a woman reading the news"},
        {"role": "assistant",
         "content": "Photo: A high-resolution, well-lit **frontal portrait** of a professional woman in formal attire, seated behind a news desk with a microphone. She has a confident expression, with even lighting and a neutral studio background.\nTTS Prompt: Tonight’s top story covers the latest developments in global politics.\nSpeaker: A female speaker with a clear, authoritative tone, moderate pitch, and steady pacing, conveying professionalism."},
        {"role": "user", "content": "Description: A young boy talking about his favorite superhero"},
        {"role": "assistant",
         "content": "Photo: A **frontal portrait** of a cheerful young boy wearing a superhero T-shirt, smiling excitedly. The background is softly blurred, with bright and even lighting.\nTTS Prompt: I love Spider-Man! He swings between buildings and fights bad guys!\nSpeaker: A young male speaker with a high-pitched and enthusiastic voice, speaking quickly with excitement."},
        {"role": "user", "content": f"Description: {description}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")

    generation_args = {
        "max_new_tokens": 150,
        "temperature": 0.6,
        "do_sample": True,
        "use_cache": True,  # 启用 KV 缓存优化
    }

    output_ids = model.generate(**inputs, **generation_args)

    # **修复 `return_full_text=False` 失效问题**
    generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    end_time = time.time()  # 记录结束时间
    return parse_result(generated_text),end_time - start_time


def parse_result(result):
    """ 改进解析逻辑，确保每个部分准确拆分"""
    sections = {"Photo": "", "TTS Prompt": "", "Speaker": ""}

    # **使用正则匹配 `Photo:` `TTS Prompt:` `Speaker:` 并拆分**
    matches = re.split(r"(?=Photo:|TTS Prompt:|Speaker:)", result)

    for match in matches:
        match = match.strip()
        if match.startswith("Photo:"):
            sections["Photo"] = match.replace("Photo:", "").strip()
        elif match.startswith("TTS Prompt:"):
            sections["TTS Prompt"] = match.replace("TTS Prompt:", "").strip()
        elif match.startswith("Speaker:"):
            sections["Speaker"] = match.replace("Speaker:", "").strip()

    return sections


def extract_tts_prompt(text):
    """Extract only the TTS Prompt from the generated output, removing explanations."""
    match = re.search(r'TTS Prompt:\s*(.+)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()  # 如果找不到，就返回原始文本（以防万一）


def analyze_speech_edit(new_description, original_speech):
    """Modify an existing speech script based on user instructions (English only)."""

    # **Ensure input is English**
    if not re.match(r'^[\x00-\x7F]+$', new_description):
        raise ValueError("The new_description should be in English.")
    if not re.match(r'^[\x00-\x7F]+$', original_speech):
        raise ValueError("The original_speech should be in English.")

    start_time = time.time()

    # **Optimized English-only prompt with original speech content**
    messages = [
        {"role": "system",
         "content": "You are an AI specializing in modifying short speech scripts, you do not need to explain. "
                    "Given an original speech and a modification request, generate a new version of the speech accordingly."
                    "The new speech should have a similar length to the original speech. "
                    "**Only output the modified TTS Prompt. Do not provide any explanations.**"
         },
        # **Example 1**
        {"role": "user",
         "content": "Original Speech: 'Science fiction novels take us into unknown worlds, exploring futuristic technology and the mysteries of the universe.'\n"
                    "Modify speech: Focus more on space travel."},
        {"role": "assistant",
         "content": "TTS Prompt: 'Science fiction novels often transport us into the vastness of space, where interstellar journeys and cosmic wonders await exploration.'"},

        # **User's actual request**
        {"role": "user",
         "content": f"Original Speech: {original_speech}\nModify speech: {new_description}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")

    generation_args = {
        "max_new_tokens": 100,  # Allowing a bit more length for natural speech
        "temperature": 0.5,  # Keeping it balanced for structured output
        "do_sample": True,
        "use_cache": True,
    }

    output_ids = model.generate(**inputs, **generation_args)
    generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    cleaned_text = extract_tts_prompt(generated_text)
    end_time = time.time()
    return cleaned_text, end_time - start_time


# 测试调用
# description = "A young girl talking about her favorite game."
# analyze_start = time.time()
# result = analyze_task(description)
# analyze_end = time.time()
#
# print(result)
# print(f"Execution Time: {analyze_end - analyze_start:.2f}s")
# 测试案例 1：修改科学小说的内容
# original_speech_1 = "Science fiction novels take us into unknown worlds, exploring futuristic technology and the mysteries of the universe."
# new_description_1 = "Focus more on space travel."
#
# # 测试案例 2：修改儿童故事的内容
# original_speech_2 = "Once upon a time, a brave little rabbit set off on an adventure to find the hidden treasure."
# new_description_2 = "Make the story about a young girl instead of a rabbit."
#
# # 执行测试
# result_1, time_1 = analyze_speech_edit(new_description_1, original_speech_1)
# result_2, time_2 = analyze_speech_edit(new_description_2, original_speech_2)
#
# # 输出结果
# print("Test Case 1 Output:", result_1)
# print("Execution Time:", round(time_1, 2), "seconds\n")
#
# print("Test Case 2 Output:", result_2)
# print("Execution Time:", round(time_2, 2), "seconds\n")
