import pyaudio
import wave
import os
from datetime import datetime
from funasr import AutoModel

# 初始化情绪识别模型
model = AutoModel(model="iic/emotion2vec_plus_large")

# 设置录音参数
chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 16000
record_seconds = 5  # 设置录音时长为5秒

# 创建PyAudio对象
audio = pyaudio.PyAudio()

# 打开音频流
stream = audio.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

print("录音开始，请说话...")

# 开始录音
frames = []
for _ in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("录音结束.")

# 停止和关闭音频流
stream.stop_stream()
stream.close()
audio.terminate()

# 创建 emotion 文件夹
emotion_folder = "emotion"
os.makedirs(emotion_folder, exist_ok=True)

# 获取当前时间并生成文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"{current_time}.wav"
file_path = os.path.join(emotion_folder, file_name)

# 保存录音文件
with wave.open(file_path, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))

print(f"录音文件已保存到: {file_path}")

# 对当前保存的音频文件进行情绪识别
print(f"正在处理音频文件: {file_path}")

# 对音频文件进行情绪识别
res = model.generate(file_path, granularity="utterance", extract_embedding=False)

# 提取最有可能的情绪类别和分数
if res:
    emotions = res[0]['labels']
    scores = res[0]['scores']

    # 打印调试信息
    print("所有情绪及其分数：")
    for emotion, score in zip(emotions, scores):
        print(f"{emotion}: {score}")

    best_index = scores.index(max(scores))  # 选择分数最高的索引
    best_label = emotions[best_index]
    best_score = scores[best_index]

    print(f"最有可能的情绪类别: {best_label}, 分数: {best_score}")
else:
    print("无法识别情绪。")
