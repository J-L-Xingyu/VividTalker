import time
import json
from datetime import datetime
import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
from diffusers import StableDiffusion3Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import subprocess
import requests
import glob

# 定义模型路径
# analyst_model_path = "/share/home/liuxingy/real3dportrait/models/vicuna-7b-v1.5"
tts_model_path = "models/parler_tts_mini_v0.1"
image_model_path = "models/stable-diffusion-3-medium-diffusers"

torch.random.manual_seed(0)

# 加载TTS模型
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_path)
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_path).to("cuda:2")
#
# # 加载图像生成模型
pipe = StableDiffusion3Pipeline.from_pretrained(image_model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda:1")
pipe.enable_attention_slicing("auto")  # 如果已有较大显存


# 批量描述列表
descriptions = [
    "A female teacher explaining how photosynthesis works in plants.",
    "A woman sharing her tips for cooking pasta.",
    "A male artist talking about his latest painting.",
    "A male scientist explaining the water cycle.",
    "An elderly woman recounting her childhood memories.",
    "A man discussing his recent travel experiences.",
    # "A female student practicing Spanish vocabulary.",这个不合适，不能用其他语言
    "A male chef demonstrating how to make sushi.",
    "A mother talking about her experience raising twins.",
    "A male athlete sharing insights on daily workout routines.",
    "A female nurse explaining first aid basics.",
    "A male historian discussing ancient civilizations.",
    "A male musician playing an acoustic guitar.",
    "A woman narrating a traditional folk tale.",
    "A female fitness trainer describing the benefits of yoga.",
    "A male photographer sharing tips on landscape photography.",
    "A female environmentalist talking about climate change.",
    "An elderly man talking his childhood memories.",
    "A little girl explaining how to play a card game.",
    "A young boy describing his favorite animal.",
    "A young girl explaining the rules of a board game."
]



# 定义记录日志的函数
def log_info(log_file, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a") as file:
        file.write(f"[{timestamp}] {message}\n")


# 创建一个日志文件
log_file = "/output/gemini_op_log.txt"

# 定义分析任务的函数
def analyst_agent(description):
    # 生成消息格式的对话
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant.Please design the character appearance in the video according to user requirements."},
        {"role": "user", "content": "Description: I need a video of a man giving a weather forecast."},
        {"role": "assistant", "content": "1. A frontal photo of a man in a suit, possibly with a tie, looking professional and friendly.\n2. An audio clip of a weather forecast by a man, mentioning today's weather and upcoming week."},
        {"role": "user", "content": "Description: Create a video of a woman reading the news."},
        {"role": "assistant", "content": "1. A frontal photo of a woman in formal attire, looking confident and serious.\n2. An audio clip of a news report by a woman, covering recent events and headlines."},
        {"role": "user", "content": f'Description: {description}, Remember, there should be only one person in the picture. You should follow the reply format above.'}
    ]

    # Make a request to the Flask API
    api_url = "http://localhost:6000/generate_text"
    payload = {
        "messages": messages,
        # "max_new_tokens": 200,
        # "return_full_text": False,
        # "temperature": 0.8,
        # "do_sample": False
    }

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        # response_text = response_data['generated_text']
        response_text = response_data.get("content")
        print(response_text)

        # Extract descriptions
        photo_description, audio_description = extract_descriptions(response_text)

        # Print results
        print(f"Photo Description: {photo_description}")
        print(f"Audio Description: {audio_description}")

        return photo_description, audio_description
    else:
        log_info(log_file, f"API call failed with status {response.status_code}: {response.text}")
        raise RuntimeError(f"Failed to generate text: {response.status_code} {response.text}")

# 提取描述的辅助函数
def extract_descriptions(response):
    lines = response.split('\n')
    photo_description = None
    audio_description = None
    for line in lines:
        line = line.strip()
        if line.startswith("1.") and photo_description is None:
            photo_description = line[2:].strip()
        elif line.startswith("2.") and audio_description is None:
            audio_description = line[2:].strip()
    return photo_description, audio_description

# 定义生成音频脚本的函数
def generate_audio(description,idx):
    # 生成消息格式的对话
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specializing in generating detailed and accurate descriptions for audio, text. You provide specific instructions about the content style, tone, "},
        {"role": "user", "content": "Description: An audio clip of a weather forecast by a man, mentioning today's weather and upcoming week."},
        {"role": "assistant", "content": "Words: Over the next few days, the forecast predicts partly cloudy skies with scattered showers.\nSpeaker: A male speaker with a calm and clear voice, speaking at a moderate pace with a neutral pitch."},
        {"role": "user", "content": "Description: An audio clip of a little girl talking about her favorite game."},
        {"role": "assistant", "content": "Words: I love playing hide and seek with my friends. It's so much fun!\nSpeaker: A little girl with a high-pitched, fast voice, speaking quickly with excitement and energy."},
        {"role": "user", "content": "Description: An audio clip of an elderly woman sharing a story from her youth."},
        {"role": "assistant", "content": "Words: When I was young, we used to spend our summers by the lake, fishing and playing all day long.\nSpeaker: An elderly female speaker with a soft, slow voice, speaking at a gentle pace with a warm and nostalgic tone."},
        {"role": "user", "content": f'Description: {description}, remember,The script for the words spoken should be no more than two sentence.You should follow the reply format above.'}
    ]

    api_url = "http://localhost:6000/generate_text"
    payload = {
        "messages": messages,
        # "max_new_tokens": 150,
        # "return_full_text": False,
        # "temperature": 0.8,
        # "do_sample": False
    }

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        # response_text = response_data['generated_text']
        response_text = response_data.get("content")
        log_info(log_file, response_text)

        # Extract descriptions
        prompt, description = extract_prompt_and_speaker(response_text)

        # Print results
        log_info(log_file, f"TTS Prompt: {prompt}")
        log_info(log_file, f"Speaker Description: {description}")

        if is_child_voice(description):
            print("Detected child voice, using openvoice API.")
            log_info(log_file,"Detected child voice, using openvoice API.")
            # 调用openvoice API生成音频
            audio_path = call_openvoice_api(prompt, description)
        else:
            print("Detected adult voice, using the local TTS model.")
            # 使用本地TTS模型生成音频
            audio_path = generate_local_audio(prompt, description,idx)

        return audio_path
    else:
        log_info(log_file, f"API call failed with status {response.status_code}: {response.text}")
        raise RuntimeError(f"Failed to generate text: {response.status_code}")

# 辅助函数：判断是否为小孩子的声音
def is_child_voice(speaker_description):
    keywords = ['little girl', 'little boy', 'child', 'kid','young boy','young girl']
    for keyword in keywords:
        if keyword in speaker_description.lower():
            return True
    return False

    # 辅助函数：调用openvoice API生成音频
def call_openvoice_api(prompt, speaker_description):
    voice_type = 'boy' if 'boy' in speaker_description.lower() else 'girl'
    api_url = "http://localhost:5000/generate_audio"
    payload = {
        "text": prompt,
        "voice_type": voice_type,
        "style": "default",
        "speed": 1.0
    }

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data['file_path']
    else:
        raise RuntimeError(f"OpenVoice API call failed: {response.status_code} {response.text}")

# 辅助函数：调用本地TTS模型生成音频
def generate_local_audio(prompt, speaker_description, idx):
    input_ids = tts_tokenizer(speaker_description, return_tensors="pt").input_ids.to("cuda:2")
    prompt_input_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:2")
    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # 保存生成的音频文件
    audio_path = f"/data/audio/audio_{idx + 1}.wav"
    sf.write(audio_path, audio_arr, tts_model.config.sampling_rate)
    return audio_path

# 提取生成的prompt和speaker description的辅助函数
def extract_prompt_and_speaker(response):
    lines = response.split('\n')
    prompt = None
    speaker_description = None
    for line in lines:
        line = line.strip()
        if line.startswith("Words:"):
            prompt = line[6:].strip()
        elif line.startswith("Speaker:"):
            speaker_description = line[8:].strip()
    return prompt, speaker_description


def generate_image(prompt, idx):
    new_prompt = prompt + " A simple background. warm lighting"
    image = pipe(new_prompt,num_inference_steps=20, guidance_scale=7.0).images[0]
    image_path = f"/data/main-character/image_{idx + 1}.png"
    image.save(image_path)
    return image_path


def generate_final_video(src_img_path, drv_aud_path, output_video_path):
    # 构建命令
    command = [
        "python", "inference/real3d_infer.py",
        "--src_img", src_img_path,
        "--drv_aud", drv_aud_path,
        "--out_name", output_video_path,
        "--drv_pose", "static",  # 使用静态姿势
        # "--mouth_amp", "0.4",
        # "--map_to_init_pose", "True",
        # "--temperature", "0.6",
        "--out_mode", "final"
    ]

    # 运行命令
    subprocess.run(command)

def optimize_video(input_video_path):
    command = [
        "python", "inference_realesrgan_video.py",
        "-n", "RealESRGAN_x4plus",
        "-i", input_video_path,
        "-s", "1",  #
        "--face_enhance",  # 启用人脸增强
        "--tile", "256",  # 大图像处理优化
        "--tile_pad", "10",
        #"--num_process_per_gpu", "2"
    ]

    # 运行命令
    subprocess.run(command)


# 综合框架（支持批量）
def create_video_batch(descriptions):
    for idx, description in enumerate(descriptions, start=0):
        try:
            start_time = time.time()
            log_info(log_file, f"Starting task {idx + 1}: {description}")

            # 分析任务
            log_info(log_file, "Analyzing task...")
            analysis_start = time.time()
            photo_description, audio_description = analyst_agent(description)
            analysis_end = time.time()
            log_info(log_file, f"Analysis completed in {analysis_end - analysis_start:.2f} seconds.")
            log_info(log_file, f"Photo Description: {photo_description}")
            log_info(log_file, f"Audio Description: {audio_description}")

            # 生成图片
            log_info(log_file, "Generating image...")
            image_start = time.time()
            image_path = generate_image(photo_description,idx)
            image_end = time.time()
            log_info(log_file,
                     f"Image generation completed in {image_end - image_start:.2f} seconds. Image saved at {image_path}")

            # 生成音频
            log_info(log_file, "Generating audio...")
            audio_start = time.time()
            audio_path = generate_audio(audio_description,idx)
            audio_end = time.time()
            log_info(log_file,
                     f"Audio generation completed in {audio_end - audio_start:.2f} seconds. Audio saved at {audio_path}")

            # 生成视频
            log_info(log_file, "Generating video...")
            video_start = time.time()
            output_video_path = f"/new_output/video_{idx + 1}.mp4"
            generate_final_video(image_path, audio_path, output_video_path)
            video_end = time.time()
            log_info(log_file,
                     f"Video generation completed in {video_end - video_start:.2f} seconds. Video saved at {output_video_path}")

            # 优化视频
            # log_info(log_file, "Optimizing video...")
            # optimize_start = time.time()
            # optimize_video(output_video_path)
            # optimize_end = time.time()
            # log_info(log_file, f"Video optimization completed in {optimize_end - optimize_start:.2f} seconds.")

            end_time = time.time()
            log_info(log_file,
                     f"Task {idx + 1} completed in {end_time - start_time:.2f} seconds. Final video saved at {output_video_path}\n")
        except Exception as e:
            log_info(log_file, f"Task {idx + 1} failed with error: {str(e)}\n")


# 运行批量任务
# create_video_batch(descriptions)

def process_videos_in_folder(folder_path):
    # 获取指定文件夹下的所有视频文件路径
    video_files = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))  # 根据实际视频格式修改扩展名

    # 遍历所有视频文件，依次调用 optimize_video 进行优化
    for idx, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)  # 获取文件名
        log_info(log_file, f"Optimizing video: {video_name} ")
        optimize_start = time.time()
        optimize_video(video_path)
        optimize_end = time.time()
        log_info(log_file, f"Video optimization completed in {optimize_end - optimize_start:.2f} seconds.")


# 调用函数，指定视频文件夹路径
process_videos_in_folder("/output")