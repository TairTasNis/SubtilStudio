import customtkinter as ctk
from tkinter import filedialog
import stable_whisper
import os
import subprocess
import threading
import shutil
import numpy as np
import wave
import colorsys
import yt_dlp
import torch
import torchaudio
import soundfile
from PIL import Image, ImageDraw, ImageFont


# =======================================================
# === ПАТЧ ДЛЯ RTX 5060 Ti / PyTorch Nightly (CUDA 13) ===
def patched_save(filepath, src, sample_rate, **kwargs):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy()
    if src.ndim == 2:
        src = src.transpose()
    soundfile.write(filepath, src, sample_rate)


torchaudio.save = patched_save
# =======================================================

from demucs import separate

# === СПИСОК ШРИФТОВ ===
FONTS_LIST = [
    "Arial", "Arial Black", "Calibri", "Comic Sans MS", "Consolas",
    "Constantia", "Corbel", "Courier New", "Ebrima", "Franklin Gothic Medium",
    "Gabriola", "Gadugi", "Georgia", "Impact", "Leelawadee UI",
    "Lucida Console", "Malgun Gothic", "Microsoft Himalaya", "Microsoft JhengHei",
    "Microsoft Sans Serif", "Microsoft YaHei", "Nirmala UI", "Palatino Linotype",
    "Segoe UI", "SimSun", "Sylfaen", "Tahoma", "Times New Roman",
    "Trebuchet MS", "Verdana"
]

# === НАСТРОЙКИ UI ===
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AudioVisualizer:
    def __init__(self, audio_path, output_path, text_title, fps=30):
        self.audio_path = audio_path
        self.output_path = output_path
        self.text_title = text_title
        self.fps = fps
        self.width = 1920
        self.height = 1080

    def _get_audio_data(self):
        temp_wav = "temp_analysis.wav"
        subprocess.run([
            'ffmpeg', '-i', self.audio_path,
            '-ar', '44100', '-ac', '1', '-f', 'wav',
            temp_wav, '-y', '-v', 'quiet'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            wf = wave.open(temp_wav, 'rb')
            params = wf.getparams()
            frames = wf.readframes(params.nframes)
            wf.close()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio /= 32768.0
            return audio, params.framerate
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

    def get_gradient_color(self, position):
        hue = 0.6 + (position * 0.35)
        saturation = 0.9
        value = 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r * 255), int(g * 255), int(b * 255))

    def render(self, progress_callback=None):
        audio_data, rate = self._get_audio_data()
        total_frames = int(len(audio_data) / rate * self.fps)
        chunk_size = int(rate / self.fps)

        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}', '-pix_fmt', 'rgb24',
            '-r', str(self.fps), '-i', '-', '-i', self.audio_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
            '-c:a', 'aac', '-b:a', '192k', '-shortest', self.output_path
        ]

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(
            command, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )

        try:
            font_title = ImageFont.truetype("arial.ttf", 60)
            font_sub = ImageFont.truetype("arial.ttf", 28)
        except:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()

        cx, cy = self.width // 2, self.height // 2
        num_bars = 120
        base_radius = 240
        max_bar_height = 350
        prev_bars = np.zeros(num_bars)
        smooth_radius = base_radius
        bg_flash = 0.0

        for i in range(total_frames):
            start = i * chunk_size
            end = start + chunk_size
            if end >= len(audio_data): break
            chunk = audio_data[start:end]
            if len(chunk) == 0: break

            window = np.hanning(len(chunk))
            fft = np.abs(np.fft.rfft(chunk * window))
            fft = fft[:len(fft) // 2]

            indices = np.linspace(0, len(fft), num_bars + 1, dtype=int)
            bars = np.array([np.mean(fft[indices[j]:indices[j + 1]]) for j in range(num_bars)])
            bars = np.nan_to_num(bars) * 20
            equalizer = np.logspace(0, 0.5, num_bars)
            bars = bars * equalizer
            bars = np.clip(bars, 0, max_bar_height)

            for j in range(num_bars):
                if bars[j] > prev_bars[j]:
                    prev_bars[j] = prev_bars[j] * 0.4 + bars[j] * 0.6
                else:
                    prev_bars[j] = prev_bars[j] * 0.85 + bars[j] * 0.15

            current_bars = prev_bars
            rms = np.sqrt(np.mean(chunk ** 2))
            target_radius = base_radius + (rms * 400)
            smooth_radius = smooth_radius * 0.8 + target_radius * 0.2
            if rms > 0.2:
                bg_flash = min(bg_flash + 5, 40)
            else:
                bg_flash = max(bg_flash - 1, 0)
            bg_color = (int(10 + bg_flash * 0.5), int(10 + bg_flash * 0.2), int(15 + bg_flash))

            img = Image.new('RGB', (self.width, self.height), bg_color)
            draw = ImageDraw.Draw(img)

            for idx, bar_len in enumerate(current_bars):
                if bar_len < 3: continue
                angle = (idx / num_bars) * 360
                rad = np.radians(angle)
                bar_color = self.get_gradient_color(idx / num_bars)
                offset = 5
                x1 = cx + (smooth_radius + offset) * np.cos(rad)
                y1 = cy + (smooth_radius + offset) * np.sin(rad)
                x2 = cx + (smooth_radius + offset + bar_len) * np.cos(rad)
                y2 = cy + (smooth_radius + offset + bar_len) * np.sin(rad)
                draw.line([(x1, y1), (x2, y2)], fill=bar_color, width=6)

                rad_mirror = np.radians(-angle)
                xm1 = cx + (smooth_radius + offset) * np.cos(rad_mirror)
                ym1 = cy + (smooth_radius + offset) * np.sin(rad_mirror)
                xm2 = cx + (smooth_radius + offset + bar_len) * np.cos(rad_mirror)
                ym2 = cy + (smooth_radius + offset + bar_len) * np.sin(rad_mirror)
                draw.line([(xm1, ym1), (xm2, ym2)], fill=bar_color, width=6)

            draw.ellipse((cx - smooth_radius - 2, cy - smooth_radius - 2,
                          cx + smooth_radius + 2, cy + smooth_radius + 2),
                         outline=(0, 200, 255), width=3)
            draw.ellipse((cx - smooth_radius, cy - smooth_radius,
                          cx + smooth_radius, cy + smooth_radius),
                         fill=(10, 10, 15), outline=None)

            try:
                process.stdin.write(img.tobytes())
            except:
                break
            if progress_callback and i % 30 == 0:
                percent = int(i / total_frames * 100)
                progress_callback(percent)

        process.stdin.close()
        process.wait()


class SubtitleGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neon Pro: Classic Edition")
        self.geometry("600x850")
        self.resizable(False, False)
        self.selected_file_path = ""
        self.model_var = ctk.StringVar(value="large-v2")
        self.lang_var = ctk.StringVar(value="Russian")

        self.header = ctk.CTkLabel(self, text="AI Visualizer (Classic DL)", font=("Arial", 22, "bold"),
                                   text_color="#00E5FF")
        self.header.pack(pady=10)

        self.tab_view = ctk.CTkTabview(self, width=540, height=130)
        self.tab_view.pack(pady=5)

        self.tab_file = self.tab_view.add("Файл")
        self.tab_yt = self.tab_view.add("YouTube")

        # --- ФАЙЛ ---
        self.btn_select = ctk.CTkButton(self.tab_file, text="ВЫБРАТЬ ФАЙЛ", command=self.select_file, height=30,
                                        fg_color="#444")
        self.btn_select.pack(pady=15, fill="x", padx=20)
        self.lbl_file = ctk.CTkLabel(self.tab_file, text="...", text_color="gray")
        self.lbl_file.pack()

        # --- YOUTUBE ---
        self.entry_url = ctk.CTkEntry(self.tab_yt, placeholder_text="Ссылка...", height=30)
        self.entry_url.pack(pady=(15, 5), padx=20, fill="x")
        self.btn_download = ctk.CTkButton(self.tab_yt, text="СКАЧАТЬ (CLASSIC)", command=self.start_download_thread,
                                          height=30, fg_color="#D32F2F", hover_color="#B71C1C")
        self.btn_download.pack(pady=5, padx=20, fill="x")

        # === СТИЛИ ===
        self.style_frame = ctk.CTkFrame(self)
        self.style_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(self.style_frame, text="Настройки субтитров", font=("Arial", 14, "bold")).pack(pady=5)

        self.font_var = ctk.StringVar(value="Arial")
        ctk.CTkLabel(self.style_frame, text="Шрифт:").pack(anchor="w", padx=10)
        self.opt_font = ctk.CTkOptionMenu(self.style_frame, variable=self.font_var, values=FONTS_LIST)
        self.opt_font.pack(fill="x", padx=10, pady=2)

        self.bg_var = ctk.BooleanVar(value=True)
        self.chk_bg = ctk.CTkCheckBox(self.style_frame, text="Включить фон (плашка)", variable=self.bg_var,
                                      command=self.toggle_bg_slider)
        self.chk_bg.pack(anchor="w", padx=10, pady=5)

        self.lbl_opacity = ctk.CTkLabel(self.style_frame, text="Прозрачность фона (60%):")
        self.lbl_opacity.pack(anchor="w", padx=10)
        self.slider_bg_opacity = ctk.CTkSlider(self.style_frame, from_=0, to=100, number_of_steps=100,
                                               command=self.update_opacity_label)
        self.slider_bg_opacity.set(60)
        self.slider_bg_opacity.pack(fill="x", padx=10)

        fade_frame = ctk.CTkFrame(self.style_frame, fg_color="transparent")
        fade_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(fade_frame, text="Появление (сек):").pack(side="left", padx=5)
        self.fade_in_var = ctk.CTkOptionMenu(fade_frame, values=["0", "0.1", "0.2", "0.3", "0.5", "1.0"], width=70)
        self.fade_in_var.set("0.2")
        self.fade_in_var.pack(side="left")

        ctk.CTkLabel(fade_frame, text="Исчезание (сек):").pack(side="left", padx=5)
        self.fade_out_var = ctk.CTkOptionMenu(fade_frame, values=["0", "0.1", "0.2", "0.3", "0.5", "1.0"], width=70)
        self.fade_out_var.set("0.2")
        self.fade_out_var.pack(side="left")

        # === AI ===
        self.frame_opts = ctk.CTkFrame(self)
        self.frame_opts.pack(pady=10)
        ctk.CTkOptionMenu(self.frame_opts, variable=self.model_var, values=["large-v2", "large-v3", "medium"]).pack(
            side="left", padx=10)
        ctk.CTkOptionMenu(self.frame_opts, variable=self.lang_var, values=["Russian", "English"]).pack(side="left",
                                                                                                       padx=10)

        self.lbl_status = ctk.CTkLabel(self, text="Ожидание...", font=("Arial", 14))
        self.lbl_status.pack(pady=(5, 5))
        self.progress_bar = ctk.CTkProgressBar(self, width=450, progress_color="#00E5FF")
        self.progress_bar.pack()
        self.progress_bar.set(0)

        self.btn_start = ctk.CTkButton(self, text="РЕНДЕР", fg_color="#00C853", height=40, font=("Arial", 15, "bold"),
                                       state="disabled", command=self.start_render_thread)
        self.btn_start.pack(pady=20, padx=40, fill="x")

    def toggle_bg_slider(self):
        if self.bg_var.get():
            self.slider_bg_opacity.configure(state="normal")
        else:
            self.slider_bg_opacity.configure(state="disabled")

    def update_opacity_label(self, value):
        self.lbl_opacity.configure(text=f"Прозрачность фона ({int(value)}%):")

    def select_file(self):
        path = filedialog.askopenfilename()
        if path: self.set_active_file(path)

    def set_active_file(self, path):
        self.selected_file_path = path
        filename = os.path.basename(path)
        self.lbl_file.configure(text=filename, text_color="white")
        self.btn_start.configure(state="normal")
        self.update_ui(f"Готов: {filename}", 0)

    def start_download_thread(self):
        url = self.entry_url.get()
        if not url: return
        self.btn_download.configure(state="disabled")
        threading.Thread(target=self.download_youtube, args=(url,), daemon=True).start()

    def download_youtube(self, url):
        try:
            output_folder = os.path.join(os.getcwd(), "DOWNLOADS")
            os.makedirs(output_folder, exist_ok=True)

            # === САМЫЕ ПЕРВЫЕ НАСТРОЙКИ (CLASSIC) ===
            # Возвращаем ту конфигурацию, которая работала у вас быстро.
            # Без маскировки под Android/iOS, просто честный запрос лучшего MP4.
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
                'merge_output_format': 'mp4',
                'progress_hooks': [self.yt_progress_hook],
                'noplaylist': True,
                # Убрали все extractor_args и chunk_size
            }
            # ========================================

            self.update_ui("Скачивание (Classic)...", 0)

            if shutil.which("node") is None:
                print("WARNING: Node.js не найден!")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                base, _ = os.path.splitext(filename)

                final_path = base + ".mp4"
                if not os.path.exists(final_path):
                    if os.path.exists(filename):
                        final_path = filename
                    elif os.path.exists(base + ".mkv"):
                        final_path = base + ".mkv"

            self.update_ui("Скачано! 100%", 100)
            self.set_active_file(final_path)

        except Exception as e:
            self.update_ui(f"Ошибка: {str(e)[:40]}", 0)
            print(f"FULL ERROR: {e}")
        finally:
            self.btn_download.configure(state="normal")

    def yt_progress_hook(self, d):
        if d['status'] == 'downloading':
            try:
                p = d.get('_percent_str', '0%').replace('%', '')
                self.update_ui(f"Загрузка: {d['_percent_str']}", float(p))
            except:
                pass

    def update_ui(self, text, percent):
        self.lbl_status.configure(text=text)
        self.progress_bar.set(percent / 100)
        self.update_idletasks()

    def start_render_thread(self):
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self.process_render).start()

    def save_as_ass(self, segments, output_path):
        opacity_percent = self.slider_bg_opacity.get()
        alpha_val = int(255 - (opacity_percent * 2.55))
        alpha_hex = f"{alpha_val:02X}"
        border_style = 3 if self.bg_var.get() else 1
        font_name = self.font_var.get()
        fade_in = int(float(self.fade_in_var.get()) * 1000)
        fade_out = int(float(self.fade_out_var.get()) * 1000)

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},60,&H00FFFFFF,&H000000FF,&H00000000,&H{alpha_hex}000000,-1,0,0,0,100,100,0,0,{border_style},2,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            for seg in segments:
                start = self.format_ass_time(seg.start)
                end = self.format_ass_time(seg.end)
                text = seg.text.strip()
                line = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\fad({fade_in},{fade_out})}}{text}\n"
                f.write(line)

    def format_ass_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        cs = int((s - int(s)) * 100)
        return f"{int(h)}:{int(m):02}:{int(s):02}.{cs:02}"

    def process_render(self):
        temp_audio_path = "temp_audio_original.wav"
        temp_vocals_path = "temp_audio_vocals.wav"
        temp_boosted_path = "temp_audio_boosted.wav"

        try:
            input_path = os.path.abspath(self.selected_file_path)
            output_dir = os.path.join(os.path.dirname(input_path), "RESULT")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_path))[0]

            video_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv')
            is_video = input_path.lower().endswith(video_extensions)

            self.update_ui("1/6 Извлечение аудио...", 5)
            subprocess.run([
                'ffmpeg', '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                temp_audio_path, '-y', '-v', 'quiet'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.update_ui("2/6 Demucs...", 15)
            demucs_args = ["--two-stems=vocals", "-n", "htdemucs", "--shifts", "2", temp_audio_path, "-o",
                           "temp_demucs_out"]

            try:
                separate.main(demucs_args)
                demucs_result = os.path.join("temp_demucs_out", "htdemucs", "temp_audio_original", "vocals.wav")
                if os.path.exists(demucs_result):
                    shutil.move(demucs_result, temp_vocals_path)
                else:
                    raise Exception("Demucs fail")
            except Exception as e:
                print(e)
                shutil.copy(temp_audio_path, temp_vocals_path)

            self.update_ui("3/6 Boost Volume...", 40)
            subprocess.run([
                'ffmpeg', '-i', temp_vocals_path,
                '-filter:a', 'loudnorm=I=-16:TP=-1.5:LRA=11',
                '-ar', '44100', '-y', temp_boosted_path, '-v', 'quiet'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # MP3
            mp3_output = os.path.join(output_dir, f"{base_name}_VOCALS.mp3")
            subprocess.run(['ffmpeg', '-i', temp_boosted_path, '-b:a', '192k', '-y', mp3_output, '-v', 'quiet'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.update_ui("4/6 Whisper...", 50)
            model = stable_whisper.load_model(self.model_var.get())

            initial_prompt_str = "Lyrics: Helltaker, Хеллтейкер, Люцифер, Азазель, Джаджмент, Пандемоника, Цербер, Малина, Здрада, Юстиция, Вельзевул, смертных, пытки, ад, госпожа, царица ада."

            res = model.transcribe(
                temp_boosted_path, vad=False, regroup=False, beam_size=5,
                language="ru" if self.lang_var.get() == "Russian" else "en",
                initial_prompt=initial_prompt_str, temperature=0.0,
                condition_on_previous_text=False, no_speech_threshold=None,
                logprob_threshold=None, compression_ratio_threshold=2.4
            )

            filtered_segments = []
            bad_words = ["субтитры", "subtitle", "создавал", "torzok", "автор", "перевод", "subtitles"]
            for seg in res.segments:
                if any(w in seg.text.lower() for w in bad_words): continue
                if len(seg.text.strip()) < 2: continue
                filtered_segments.append(seg)
            res.segments = filtered_segments

            res.split_by_length(max_chars=40)
            res.merge_by_gap(0.1, max_words=3)

            # ASS
            ass_path = os.path.join(output_dir, f"{base_name}.ass")
            self.save_as_ass(res.segments, ass_path)

            if is_video:
                self.update_ui("6/6 Видео + ASS...", 80)
                video_source = input_path
            else:
                self.update_ui("6/6 Визуал...", 60)
                visual_path = os.path.join(output_dir, f"{base_name}_VISUAL.mp4")
                viz = AudioVisualizer(input_path, visual_path, base_name)
                viz.render(lambda p: self.update_ui(f"Рендер ({p}%)...", 60 + p * 0.3))
                video_source = visual_path

            final_path = os.path.join(output_dir, f"{base_name}_FINAL.mp4")
            cwd = os.getcwd()
            shutil.copy(ass_path, os.path.join(cwd, "temp.ass"))

            cmd = [
                'ffmpeg', '-i', video_source,
                '-vf', "subtitles=temp.ass",
                '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k',
                '-y', final_path
            ]
            if not is_video: cmd[cmd.index('-c:a') + 1] = 'copy'

            subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.update_ui(f"Готово! RESULT", 100)

        except Exception as e:
            self.update_ui(f"Ошибка: {e}", 0)
            print(f"CRITICAL ERROR: {e}")
        finally:
            self.btn_start.configure(state="normal")
            if os.path.exists("temp.ass"): os.remove("temp.ass")
            if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            if os.path.exists(temp_vocals_path): os.remove(temp_vocals_path)
            if os.path.exists(temp_boosted_path): os.remove(temp_boosted_path)
            if os.path.exists("temp_demucs_out"): shutil.rmtree("temp_demucs_out", ignore_errors=True)


if __name__ == "__main__":
    app = SubtitleGeneratorApp()
    app.mainloop()
