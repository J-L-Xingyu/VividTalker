import asyncio
from analyse import analyze_task, analyze_speech_edit  # 调用 LM 解析任务
from vision_audio import generate_media  # 调用 SD & TTS 生成图像 & 音频
from video_generation import generate_final_video
from edit_analyze import classify_edit_request
from edit_handler import handle_incremental_edit


async def main():
    # 1️⃣ **用户输入**
    user_description = "a bride reciting her wedding vows"

    # 2️⃣ **解析用户输入，获取 Photo & TTS 语音描述**
    print("🚀 [1] 解析用户输入...")
    task_info, analyse_time = analyze_task(user_description)  # 解析任务
    original_task_info = task_info  # ✅ 确保 task_info 传递到 `handle_incremental_edit()`

    photo_description = task_info["Photo"]
    tts_prompt = task_info["TTS Prompt"]
    speaker_description = task_info["Speaker"]

    print("✅ 解析完成，生成的任务信息：")
    print(f"🖼️ [Photo] {photo_description}")
    print(f"🔊 [TTS Prompt] {tts_prompt}")
    print(f"🗣️ [Speaker] {speaker_description}")

    # 3️⃣ **并行生成图像 & 音频**
    print("🚀 [2] 并行生成图像 & 音频...")
    image_path, audio_path, audio_time = await generate_media(photo_description, tts_prompt, speaker_description, idx=1)

    # 4️⃣ **生成最终视频**
    print("🚀 [3] 生成最终视频...")
    video_path, video_time = generate_final_video(image_path, audio_path, idx=3)

    total_time = analyse_time + video_time + audio_time

    print(f"✅ [4] 视频生成完成，文件路径: {video_path}")
    print(f"⚡ 总耗时: {total_time:.2f} 秒")



# 运行 `main.py`
if __name__ == "__main__":
    asyncio.run(main())

#Make the voice deeper