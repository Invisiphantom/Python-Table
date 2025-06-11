import os
import torch
import gradio as gr
import numpy as np
from scipy.io import wavfile
from scipy import signal
from datetime import datetime
from util import ManualMFCC, SpeechRecognizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和参数
mfcc_extractor = ManualMFCC(sample_rate=8000)
model = SpeechRecognizer().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

VOCAB = {"00": "数字", "01": "语音", "02": "语言", "03": "处理", "04": "中国", "05": "忠告", "06": "北京", "07": "背景", "08": "上海", "09": "商行", "10": "Speech", "11": "Speaker", "12": "Signal", "13": "Sequence", "14": "Processing", "15": "Print", "16": "Project", "17": "File", "18": "Open", "19": "Close"}


def resample_audio(audio_data, original_rate, target_rate):
    """将音频从原始采样率转换为目标采样率"""
    ratio = target_rate / original_rate
    n_samples = int(len(audio_data) * ratio)
    resampled = signal.resample(audio_data, n_samples)
    return resampled.astype(np.int16)


def predict_audio(audio):
    """处理音频并进行预测"""
    try:
        sample_rate, audio_data = audio
        if sample_rate != 8000:
            audio_data = resample_audio(audio_data, sample_rate, 8000)
            sample_rate = 8000

        # 生成带时间戳的文件名
        wav_filename = f"audio_{datetime.now().strftime("%H-%M-%S")}.wav"
        wavfile.write(wav_filename, sample_rate, audio_data)

        # 提取MFCC特征
        mfcc = mfcc_extractor.compute_mfcc(wav_filename)

        # 转换为模型输入格式
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).permute(0, 2, 1).to(device)

        # 进行预测
        with torch.no_grad():
            outputs = model(mfcc_tensor)
            _, predicted = torch.max(outputs, 1)

        # 获取预测结果
        predicted_idx = str(predicted.item()).zfill(2)
        result = VOCAB.get(predicted_idx, "未知单词")

        return f"识别结果: {result}"
    except Exception as e:
        return f"识别出错: {str(e)}"


# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 语音识别系统")
    gr.Markdown("请点击下方录音按钮，说出一个单词，系统将自动识别并保存8kHz WAV文件到当前目录")

    with gr.Row():
        audio_input = gr.Audio(type="numpy", label="录音")
        output_text = gr.Textbox(label="识别结果")

    submit_btn = gr.Button("识别")
    submit_btn.click(fn=predict_audio, inputs=audio_input, outputs=output_text)

# 启动应用
if __name__ == "__main__":
    demo.launch()
