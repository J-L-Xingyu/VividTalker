import torch
import soundfile as sf
import asyncio
import time
from transformers import AutoTokenizer
from diffusers import StableDiffusion3Pipeline
from parler_tts import ParlerTTSForConditionalGeneration

# 设置随机种子
torch.random.manual_seed(0)

# 模型路径
tts_model_path = "/share/home/liuxingy/real3dportrait/models/parler_tts_mini_v0.1"
image_model_path = "/share/home/liuxingy/real3dportrait/models/stable-diffusion-3-medium-diffusers"

# 1️⃣ **加载 TTS 模型**
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_path)
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_path).to("cuda:2")

# 2️⃣ **加载 Stable Diffusion 3 图像生成模型**
pipe = StableDiffusion3Pipeline.from_pretrained(image_model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda:1")
pipe.enable_attention_slicing("auto")  # 优化显存占用

def generate_image(prompt):
    image = pipe(prompt).images[0]
    image_path = "/share/home/liuxingy/2/data/picture/image_5.png"
    image.save(image_path)
    return image_path

# 3️⃣ **异步音频生成**
async def generate_audio(prompt, speaker_description, idx):
    """异步生成音频，并记录时间"""
    start_time = time.time()  # 记录开始时间

    input_ids = tts_tokenizer(speaker_description, return_tensors="pt").input_ids.to("cuda:2")
    prompt_input_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:2")

    loop = asyncio.get_running_loop()
    generation = await loop.run_in_executor(None, lambda: tts_model.generate(input_ids=input_ids,
                                                                             prompt_input_ids=prompt_input_ids))

    audio_arr = generation.cpu().numpy().squeeze()

    # 保存音频文件
    audio_path = f"/share/home/liuxingy/2/data/audio/audio_{idx + 1}.wav"
    sf.write(audio_path, audio_arr, tts_model.config.sampling_rate)

    end_time = time.time()  # 记录结束时间
    print(f"[TTS] 音频生成时间: {end_time - start_time:.2f} 秒")

    return audio_path, end_time - start_time


# 4️⃣ **异步图像生成**
# async def generate_image(prompt, idx):
#     """异步生成图像，并记录时间"""
#     start_time = time.time()  # 记录开始时间
#
#     new_prompt = prompt
#
#     loop = asyncio.get_running_loop()
#     image = await loop.run_in_executor(None,
#                                        lambda: pipe(new_prompt, num_inference_steps=20, guidance_scale=7.0).images[0])
#
#     image_path = f"/share/home/liuxingy/2/data/picture/image_{idx + 1}.png"
#     image.save(image_path)
#
#     end_time = time.time()  # 记录结束时间
#     print(f"[SD] 图像生成时间: {end_time - start_time:.2f} 秒")
#
#     return image_path, end_time - start_time


# 5️⃣ **并行执行 TTS & SD**
async def generate_media(photo_prompt, tts_prompt, speaker, idx):
    """
    并行生成音频和图像，并记录总时间
    :return: 生成的图像和音频文件路径, 以及执行时间
    """
    total_start_time = time.time()  # 记录总开始时间

    image_task = asyncio.create_task(generate_image(photo_prompt, idx))
    audio_task = asyncio.create_task(generate_audio(tts_prompt, speaker, idx))

    image_path, image_time = await image_task
    audio_path, audio_time = await audio_task

    total_end_time = time.time()  # 记录总结束时间
    total_time = total_end_time - total_start_time

    print(f"🌟 [总计] TTS + SD 并行运行时间: {total_time:.2f} 秒")
    print(f"⚡ [串行预估] TTS + SD 预计时间: {image_time + audio_time:.2f} 秒")

    return image_path, audio_path, total_time

photo_description = "A high-resolution, well-lit frontal portrait of a professional woman in formal attire, seated behind a news desk with a microphone. She has a confident expression, with even lighting and a neutral studio background."
generate_image(photo_description)