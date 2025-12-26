import df
import os
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal
import torch
import soundfile as sf
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def plot_spectrogram(audio_data, sample_rate, title = None):
    """绘制时间频谱图"""
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate)
    # 添加一个小的常数避免零值
    Sxx = Sxx + 1e-10
    # 使用 vmin 和 vmax 限制颜色映射范围
    pcm = plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    # pcm = plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', vmin=-100, vmax=-55)
    plt.ylabel('频率(Hz)', fontsize=11)
    plt.xlabel('时间(s)', fontsize=11)
    plt.ylim(0, 10000)
    plt.xlim(0, 4)
    if title: plt.title(title)
    # 添加带范围限制的 colorbar
    cbar = plt.colorbar(pcm, label='功率谱密度(dB)')
    cbar.set_label('功率谱密度(dB)', fontsize=11)  # 设置标签字号
    # cbar.ax.tick_params(labelsize=10)  # 设置刻度数字字号

def main():
    # 读取输入音频文件
    input_file = "mydata/pair2_att_01.wav"
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在")
        return
    
    # 保存去噪后的音频
    output_file = input_file.replace(".wav", "_denoised.wav")

    # 读取音频数据
    audio_data, sample_rate = sf.read(input_file)
    audio_data= audio_data / np.max(np.abs(audio_data)) # 读进来的音频太小了，所以放大
    # 绘制曲线图
    plt.figure(figsize=(12, 4))
    plt.plot(audio_data)
    plt.title("Audio Signal")
    plt.show()


    # 确保 audio_data 是 float32（DeepFilterNet 要求）
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # 转为 PyTorch Tensor，并确保 shape 是 (channels, time)
    # 关键修改：确保是二维张量 [1, time] 对于单声道
    if audio_data.ndim == 1:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # 变成 [1, time]
    else:
        audio_tensor = torch.from_numpy(audio_data.T)  # 转置确保通道在第0维

    model, df_state, _ = df.init_df()  # Load default model
    enhanced_tensor = df.enhance(model, df_state, audio_tensor)
    

    # 转回 numpy
    enhanced_audio = enhanced_tensor.numpy()
    print("enhanced_audio.shape: ", enhanced_audio.shape)
    # 绘制曲线图
    plt.figure(figsize=(12, 4))
    plt.plot(enhanced_audio.T)
    plt.title("Audio Signal")
    plt.show()

    
    
    

    # 绘制时间频谱图进行对比
    plt.figure(figsize=(9, 2.2))
    
    # 原始音频时间频谱图
    plt.subplot(1, 2, 1)
    audio_data = audio_data.T[0]
    print("audio_data.shape: ", audio_data.shape)
    plot_spectrogram(audio_data, sample_rate)
    
    # 去噪后音频时间频谱图
    plt.subplot(1, 2, 2)
    enhanced_audio = enhanced_audio[0]
    print("enhanced_audio.shape: ", enhanced_audio.shape)
    plot_spectrogram(enhanced_audio, sample_rate)
    
    plt.tight_layout()
    plt.savefig("spectrogram_comparison.pdf")
    plt.show()
    
    print("时间频谱图已保存到 spectrogram_comparison.pdf")

    # 关键修改：规范化并转换为 int16 格式
    # 将浮点数归一化到 [-1, 1] 范围（如果尚未归一化）
    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) if np.max(np.abs(enhanced_audio)) > 0 else enhanced_audio
    # 转换为 16-bit 整数
    enhanced_audio = (enhanced_audio * 32767).astype(np.int16)

    # 保存去噪后的音频
    sf.write(output_file, enhanced_audio, sample_rate)

if __name__ == "__main__":
    main()