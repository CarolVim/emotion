import os
import ssl
import asyncio
import argparse
import json
import pyaudio
import websockets
import logging
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.chat_models import ChatOllama
from llama_index.llms.ollama import Ollama
import edge_tts
import webrtcvad
from funasr import AutoModel
import tempfile
import wave

# 设置 Tavily API 密钥
os.environ["TAVILY_API_KEY"] = ""

# 创建 Tavily 搜索 API 检索器
retriever = TavilySearchAPIRetriever(k=5)

# 创建 ChatOllama 对象
llm = ChatOllama(model="qwen2:7b")

# 获取当前日期
current_date = datetime.now().strftime("%Y-%m-%d")

# 录音设置
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 录音通道数
RATE = 16000  # 采样率
CHUNK = 1024  # 每个数据块的帧数

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    required=False,
                    help="主机 IP，localhost，0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10096,
                    required=False,
                    help="grpc 服务器端口")
parser.add_argument("--chunk_size",
                    type=str,
                    default="5, 10, 5",
                    help="数据块大小")
parser.add_argument("--chunk_interval",
                    type=int,
                    default=10,
                    help="数据块间隔（毫秒）")
parser.add_argument("--hotword",
                    type=str,
                    default="",
                    help="热词文件路径，每行一个热词（例如：阿里巴巴 20）")
parser.add_argument("--audio_fs",
                    type=int,
                    default=16000,
                    help="音频采样率")
parser.add_argument("--use_itn",
                    type=int,
                    default=1,
                    help="1 表示使用 ITN，0 表示不使用 ITN")
parser.add_argument("--mode",
                    type=str,
                    default="2pass",
                    help="离线，在线，2pass")
parser.add_argument("--ssl",
                    type=int,
                    default=1,
                    help="1 表示 SSL 连接，0 表示不使用 SSL")
parser.add_argument("--record_time",
                    type=int,
                    default=10,
                    help="录音时间（秒）")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
recognized_text = ""
stop_signal = False
stream = None
p = None
best_emotion_label = ""
best_emotion_score = 0.0
model_response_text = ""

# 初始化情绪识别模型
emotion_model = AutoModel(model="iic/emotion2vec_plus_base")

# 固定的 prompt 模板
PROMPT_TEMPLATE = """你是一个情绪识别和语音分析助手。根据以下识别到的情绪和文本，请生成适当的回复。
情绪: {emotion}
文本: {text}
回复:"""

# TTS 合成函数
async def synthesize_and_save_speech(text, filename):
    voice = "zh-HK-HiuGaaiNeural"  # HiuGaai 是粤语的女声
    tts = edge_tts.Communicate(text, voice)
    await tts.save(filename)
    logging.info(f"生成的语音文件已保存到 {filename}")

# 录音并发送音频数据到 WebSocket 服务器
async def record_microphone(websocket):
    global recognized_text, stop_signal, stream, p, best_emotion_label, best_emotion_score
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = args.audio_fs
    CHUNK = int(RATE / 1000 * args.chunk_interval)

    vad = webrtcvad.Vad()
    vad.set_mode(1)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # 读取热词文件并构建热词字典
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        with open(args.hotword) as f_scp:
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please check format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please check format of hotwords")
            hotword_msg = json.dumps(fst_dict)

    use_itn = True
    if args.use_itn == 0:
        use_itn = False

    # 发送初始配置信息
    message = json.dumps({"mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval,
                          "wav_name": "microphone", "is_speaking": True, "hotwords": hotword_msg, "itn": use_itn})
    await websocket.send(message)

    silence_threshold = 20  # 阈值为连续20个静音帧（约0.4秒）
    silence_count = 0

    try:
        frames = []
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            is_speech = vad.is_speech(data, RATE)
            if is_speech:
                silence_count = 0
                await websocket.send(data)
            else:
                silence_count += 1
                if silence_count > silence_threshold:
                    break
            await asyncio.sleep(0.005)

        # 确保 input 文件夹存在
        os.makedirs("input", exist_ok=True)

        # 保存录音文件
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", dir="input", delete=False)
        temp_wav.close()

        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        # 进行情绪识别
        emotion_res = emotion_model.generate(temp_wav.name, granularity="utterance", extract_embedding=False)
        best_emotion_label = emotion_res[0]['labels'][0]
        best_emotion_score = emotion_res[0]['scores'][0]
        print(f"情绪识别结果: {best_emotion_label}，分数: {best_emotion_score}")
        logging.info(f"最有可能的情绪类别: {best_emotion_label}, 分数: {best_emotion_score}")

        # 录音结束后，发送结束标志
        await websocket.send(json.dumps({"is_speaking": False}))
        stop_signal = True

    except asyncio.CancelledError:
        logging.info("录音任务取消。")
    except Exception as e:
        logging.error(f"录音过程中出现错误: {e}")
    finally:
        if stream is not None and stream.is_active():
            stream.stop_stream()
            stream.close()
        if p is not None:
            p.terminate()

# 接收服务器返回的消息并更新识别文本
async def message(websocket):
    global recognized_text, stop_signal
    try:
        while True:
            msg = await websocket.recv()
            msg = json.loads(msg)
            text = msg.get("text", "")
            recognized_text = text[-10000:]  # 只保留最新的 10000 字符
            if stop_signal:
                break
    except Exception as e:
        logging.error(f"接收消息过程中出现错误: {e}")

    # 完整识别后输出结果
    print("Recognized Text: recognized_text")
    logging.info(f"Recognized Text: {recognized_text}")
    print(("识别完成，输出结果..."))
    logging.info("语音识别完成")

# WebSocket 客户端
async def ws_client():
    global recognized_text, stop_signal, stream, p, best_emotion_label, best_emotion_score, model_response_text
    ssl_context = None
    if args.ssl == 1:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    uri = f"wss://{args.host}:{args.port}" if args.ssl == 1 else f"ws://{args.host}:{args.port}"

    try:
        async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
            while True:
                task1 = asyncio.create_task(record_microphone(websocket))
                task2 = asyncio.create_task(message(websocket))
                await asyncio.gather(task1, task2)

                # 检查是否包含停止指令
                if "停止" in recognized_text:
                    logging.info("检测到停止指令，退出程序。")
                    break

                # 在完整识别后执行模型处理
                print(("与大模型交互，生成回复..."))
                logging.info("与大模型交互，生成回复...")
                try:
                    prompt = PROMPT_TEMPLATE.format(emotion=best_emotion_label, text=recognized_text)
                    model_response = await asyncio.wait_for(asyncio.to_thread(llm.invoke, prompt), timeout=600)
                    model_response_text = model_response.content if hasattr(model_response, 'content') else str(model_response)
                    logging.info(f"大模型回复: {model_response_text}")

                    # 打印情绪识别结果、语音识别文本和大模型回复文本
                    logging.info(f"情绪识别结果: {best_emotion_label}, 分数: {best_emotion_score}")
                    logging.info(f"语音识别内容: {recognized_text}")
                    logging.info(f"大模型回复: {model_response_text}")
                    print("大模型回复: model_response_text")
                    # 生成回复的语音文件并播放
                    await synthesize_and_save_speech(model_response_text, f"output/response_{current_date}.mp3")
                    play(AudioSegment.from_file(f"output/response_{current_date}.mp3"))
                except asyncio.TimeoutError:
                    logging.error("大模型回复超时。")
                except Exception as e:
                    logging.error(f"生成回复时出现错误: {e}")

                # 清除标志以继续下一次录音
                stop_signal = False

    except websockets.exceptions.ConnectionClosedError:
        logging.error("WebSocket 连接关闭。")
    except websockets.exceptions.WebSocketException as e:
        logging.error(f"WebSocket 连接错误: {e}")
    except Exception as e:
        logging.error(f"发生未知错误: {e}")
    finally:
        if stream is not None and stream.is_active():
            stream.stop_stream()
            stream.close()
        if p is not None:
            p.terminate()
        logging.info("程序手动终止。")

# 主函数
if __name__ == "__main__":
    asyncio.run(ws_client())
