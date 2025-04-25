# import subprocess
# import time
#
#
# def generate_final_video(src_img_path, drv_aud_path, idx):
#     """
#     调用 Real3D-Portrait 生成最终视频，并记录时间
#     :param src_img_path: 输入人物图像路径
#     :param drv_aud_path: 输入音频路径
#     :param idx: 任务编号
#     :return: 生成的视频文件路径 & 生成时间
#     """
#     output_video_path = f"/share/home/liuxingy/2/data/video/final_video_{idx + 1}.mp4"
#
#     # 记录开始时间
#     start_time = time.time()
#
#     # 构建命令
#     command = [
#         "python", "real3d/inference/real3d_infer.py",
#         "--src_img", src_img_path,
#         "--drv_aud", drv_aud_path,
#         "--out_name", output_video_path,
#         "--drv_pose", "static",  # 使用静态姿势
#         "--out_mode", "final"
#     ]
#
#     print(f"🚀 [Real3D] 开始生成视频: {output_video_path}")
#
#     # 运行命令
#     subprocess.run(command, check=True)
#
#     # 记录结束时间
#     end_time = time.time()
#     generation_time = end_time - start_time
#
#     print(f"✅ [Real3D] 视频生成完成: {output_video_path}，耗时 {generation_time:.2f} 秒")
#
#     return output_video_path, generation_time


import subprocess
import time


def generate_final_video(src_img_path, drv_aud_path, idx=0):
    """
    调用 Real3D-Portrait 生成最终视频，并记录时间
    """
    output_video_path = f"/share/home/liuxingy/2/data/video/final_video_{idx + 1}.mp4"

    # 记录开始时间
    start_time = time.time()

    # 构建命令
    command = [
        "python", "real3d/inference/real3d_infer.py",
        "--src_img", src_img_path,
        "--drv_aud", drv_aud_path,
        "--out_name", output_video_path,
        "--drv_pose", "static",  # 使用静态姿势
        "--out_mode", "final"
    ]

    print(f"🚀 [Real3D] 开始生成视频: {output_video_path}")

    try:
        # 运行命令
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ [Real3D] 运行失败！错误信息: {e}")
        return None, None

    # 记录结束时间
    end_time = time.time()
    generation_time = end_time - start_time

    print(f"✅ [Real3D] 视频生成完成: {output_video_path}，耗时 {generation_time:.2f} 秒")
    return output_video_path, generation_time


# **测试代码**
if __name__ == "__main__":
    test_image = "/share/home/liuxingy/2/data/picture/image_1.png"
    test_audio = "/share/home/liuxingy/2/data/audio/audio_1.wav"

    generate_final_video(test_image, test_audio, idx=5)
