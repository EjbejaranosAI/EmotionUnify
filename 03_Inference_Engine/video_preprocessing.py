import os
import moviepy.editor as mp
import config
from audio_preprocessing import AudioPreprocessing
from audio import Audio
from video import Video


def remove_too_small_intervals(intervals):
    filtered_intervals = []

    for start, end in intervals:
        duration = end - start
        if duration >= config.MIN_VIDEO_CHUNK_TIME:
            filtered_intervals.append([start, end])

    return filtered_intervals


class VideoPreprocessing:
    def __init__(self, video_path):
        self.downsampled_video_path = None
        self.downsampled_audio_path = None
        self.video_path = video_path

    def split_video(self, intervals, output_dir):
        try:
            clip = mp.VideoFileClip(self.downsampled_video_path)

            # Crea el directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)

            for i, interval in enumerate(intervals):
                start_time = interval[0]
                end_time = interval[1]

                # Genera el nombre de archivo de salida para el subclip
                base_filename = os.path.splitext(os.path.basename(self.video_path))[0]
                subclip_filename = f"preprocessed_video_{i}.mp4"
                subclip_path = os.path.join(output_dir, subclip_filename)

                if config.REMOVE_AUDIO_FROM_PREPROCESSED_VIDEOS:
                    subclip = clip.subclip(start_time, end_time).set_audio(None)
                else:
                    subclip = clip.subclip(start_time, end_time)

                subclip.write_videofile(subclip_path, codec="libx264")
        except Exception as e:
            raise Exception(f"Error al dividir el video: {str(e)}")

    def split_video_in_talking_intervals(self, output_path):
        try:
            audio = Audio(self.downsampled_audio_path)
            if audio.file_source != "meld":
                audio_preprocessing = AudioPreprocessing(self.downsampled_audio_path)
                intervals = audio_preprocessing.split_audio_by_silence()

                output_dir = os.path.dirname(output_path)
                audio_preprocessing.save_audio_intervals(intervals, output_dir)
                intervals_in_seconds = remove_too_small_intervals(intervals / 22000)
                self.split_video(intervals_in_seconds, output_dir)
            else:
                print(f"Saltando la división de audio silencioso porque proviene de {audio.file_source}")
        except Exception as e:
            raise Exception(f"Error al dividir el video en intervalos hablados: {str(e)}")

    def downsample_video(self, output_path, target_fps=config.TARGET_VIDEO_FPS,
                         resolution_scale=config.TARGET_VIDEO_RESOLUTION_SCALE,
                         audio_samplerate=config.TARGET_AUDIO_SAMPLERATE,
                         audio_bitrate=config.TARGET_AUDIO_BITRATE):
        try:
            video = mp.VideoFileClip(self.video_path)

            # Extraer el audio y guardar el archivo mp3
            audio = video.audio
            audio_path = output_path.replace(".mp4", ".mp3")

            self.downsampled_audio_path = audio_path
            audio.write_audiofile(audio_path, fps=audio_samplerate, bitrate=audio_bitrate, logger=None)

            # Reducción proporcional del ancho y alto
            new_width = int(video.size[0] * resolution_scale)
            new_height = int(video.size[1] * resolution_scale)
            target_resolution = (new_width, new_height)

            video = video.resize(target_resolution)

            if config.REMOVE_AUDIO_FROM_PREPROCESSED_VIDEOS:
                audio = None  # Desactivar el audio
            video = video.set_audio(None)  # Desactivar el audio
            video = video.set_fps(target_fps)
            video.write_videofile(output_path, codec="libx264", audio=False, logger=None)

            self.downsampled_video_path = output_path
        except Exception as e:
            raise Exception(f"Error al reducir el video: {str(e)}")

    def delete_temp_files(self):
        try:
            os.remove(self.downsampled_video_path)
            os.remove(self.downsampled_audio_path)
        except Exception as e:
            raise Exception(f"Error al eliminar archivos temporales: {str(e)}")

    def execute(self, destination_file_path):
        try:
            self.downsample_video(destination_file_path)
            self.split_video_in_talking_intervals(destination_file_path)

        except Exception as e:
            raise Exception(f"Error al ejecutar el procesamiento de video: {str(e)}")
