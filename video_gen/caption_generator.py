import whisper
import os
import shutil
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm
import numpy as np  # For color interpolation


class VideoTranscriber:
    def __init__(self, model_path, video_path,
                 font=cv2.FONT_HERSHEY_DUPLEX,
                 font_scale=0.8,
                 font_thickness=2,
                 text_color=(255, 255, 255),  # Color for the current word
                 spoken_text_color=(255, 255, 255),  # Color for already spoken words
                 add_text_background=False,
                 bg_color=(0, 0, 0),
                 text_y_position_ratio=0.9,
                 text_padding=10,
                 line_type=cv2.LINE_AA,
                 text_wrap_margin_ratio=0.05,
                 add_text_outline=True,
                 outline_color=(0, 0, 0),  # Outline for the current word
                 spoken_outline_color=(50, 50, 50),  # Outline for spoken words (can be same as outline_color)
                 outline_thickness_factor=1.5,
                 caption_transition_duration_seconds=0.3,
                 word_level_highlight=True
                 ):
        """
        Initializes the VideoTranscriber.
        Args:
            spoken_text_color (tuple): BGR color for words already spoken.
            outline_color (tuple): BGR color for the current word's outline.
            spoken_outline_color (tuple): BGR color for spoken words' outline.
            word_level_highlight (bool): Enable karaoke-style word highlighting.
            (Other args same as before)
        """
        print(f"Loading Whisper model: {model_path}")
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        base_video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.audio_path = os.path.join(os.path.dirname(self.video_path), f"temp_audio_{base_video_name}.mp3")

        # text_array will store line objects
        # Each line object: { "overall_start_frame": S, "overall_end_frame": E, "words": [word_obj_1, ...] }
        # Each word_obj: { "text": "Hi", "start_frame": S, "end_frame": E, "x_offset_on_line": X, "width": W }
        self.text_array = []
        self.fps = 0
        self.char_width = 0
        self.has_audio = False

        # Styling attributes
        self.font = font
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text_color = tuple(text_color)
        self.spoken_text_color = tuple(spoken_text_color)
        self.add_text_background = add_text_background
        self.bg_color = tuple(bg_color)
        self.text_y_position_ratio = text_y_position_ratio
        self.text_padding = text_padding
        self.line_type = line_type
        self.text_wrap_margin_ratio = text_wrap_margin_ratio
        self.add_text_outline = add_text_outline
        self.outline_color = tuple(outline_color)
        self.spoken_outline_color = tuple(spoken_outline_color)
        self.outline_thickness = max(1, int(font_thickness * outline_thickness_factor))

        self.caption_transition_duration_seconds = caption_transition_duration_seconds
        self.caption_transition_duration_frames = 0
        self.word_level_highlight = word_level_highlight

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def _interpolate_color(self, color_from_tuple, color_to_tuple, progress):
        """Helper to interpolate between two BGR colors (inputs and output are tuples)."""
        color_from_np = np.array(color_from_tuple, dtype=np.float32)
        color_to_np = np.array(color_to_tuple, dtype=np.float32)
        interpolated_np = color_from_np * (1 - progress) + color_to_np * progress
        return tuple(interpolated_np.astype(np.uint8).tolist())

    def extract_audio(self):
        """Extracts audio from the video file."""
        print('Extracting audio...')
        try:
            video_clip = VideoFileClip(self.video_path)
            if video_clip.audio:
                audio_clip = video_clip.audio
                audio_clip.write_audiofile(self.audio_path, codec='mp3', logger=None)
                audio_clip.close()
                self.has_audio = True
                print(f'Audio extracted successfully to {self.audio_path}')
            else:
                self.has_audio = False;
                self.audio_path = None
                print('No audio track found in the video.')
            video_clip.close()
        except Exception as e:
            print(f"Error extracting audio: {e}");
            self.has_audio = False;
            self.audio_path = None

    def transcribe_video(self):
        """Transcribes audio and prepares text segments with word-level timings if enabled."""
        if not self.has_audio or not self.audio_path:
            print('Transcription skipped: No audio available.');
            return

        print('Transcribing audio (word timestamps enabled for karaoke effect)...')
        try:
            # Request word_timestamps for karaoke style
            result = self.model.transcribe(self.audio_path, fp16=False, word_timestamps=True)
        except Exception as e:
            print(f"Error during transcription: {e}");
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video for properties.");
            cap.release();
            return

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not (self.fps and self.fps > 0):
            print("Warning: Could not determine FPS. Defaulting to 25 FPS.");
            self.fps = 25.0

        self.caption_transition_duration_frames = int(self.caption_transition_duration_seconds * self.fps)

        total_frames_from_opencv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"OpenCV reports video has {total_frames_from_opencv} total frames (CAP_PROP_FRAME_COUNT).")

        text_area_margin = int(original_width * self.text_wrap_margin_ratio)
        text_wrap_width = original_width - (2 * text_area_margin)
        if text_wrap_width <= 0:
            text_wrap_width = original_width / 2 if original_width > 0 else 200
            print(f"Warning: text_wrap_width too small. Using {text_wrap_width}.")

        print("Processing transcription segments for karaoke layout...")
        self.text_array = []

        for segment in tqdm(result.get("segments", []), desc="Processing Segments"):
            if not segment.get("words"):
                print(f"Segment skipped, no word timestamps: {segment.get('text', '')[:30]}")
                continue

            segment_words_data = segment["words"]

            current_line_words_info = []
            current_x_offset = 0
            line_overall_start_frame = -1
            line_overall_end_frame = -1

            for i, word_data in enumerate(segment_words_data):
                word_text = word_data["word"].strip()
                if not word_text: continue

                word_start_time = word_data["start"]
                word_end_time = word_data["end"]
                word_start_frame = int(word_start_time * self.fps)
                word_end_frame = int(word_end_time * self.fps)
                if word_end_frame < word_start_frame: word_end_frame = word_start_frame + 1

                display_word = (" " if current_line_words_info else "") + word_text
                word_size_dims, _ = cv2.getTextSize(display_word, self.font, self.font_scale, self.font_thickness)
                word_pixel_width = word_size_dims[0]

                if current_x_offset + word_pixel_width > text_wrap_width and current_line_words_info:
                    self.text_array.append({
                        "overall_start_frame": line_overall_start_frame,
                        "overall_end_frame": line_overall_end_frame,
                        "words": current_line_words_info
                    })
                    current_line_words_info = []
                    current_x_offset = 0
                    line_overall_start_frame = -1
                    line_overall_end_frame = -1
                    display_word = word_text
                    word_size_dims, _ = cv2.getTextSize(display_word, self.font, self.font_scale, self.font_thickness)
                    word_pixel_width = word_size_dims[0]

                word_info = {
                    "text": display_word,
                    "raw_text": word_text,
                    "start_frame": word_start_frame,
                    "end_frame": word_end_frame,
                    "x_offset_on_line": current_x_offset,
                    "width": word_pixel_width
                }
                current_line_words_info.append(word_info)
                current_x_offset += word_pixel_width

                if line_overall_start_frame == -1:
                    line_overall_start_frame = word_start_frame
                line_overall_end_frame = max(line_overall_end_frame, word_end_frame)

            if current_line_words_info:
                self.text_array.append({
                    "overall_start_frame": line_overall_start_frame,
                    "overall_end_frame": line_overall_end_frame,
                    "words": current_line_words_info
                })

        cap.release()
        print('Transcription and karaoke layout complete.')

    def extract_frames_and_add_text(self, output_folder_frames):
        """Extracts frames, adds timed text (karaoke/line), and saves them."""
        print('Extracting frames and adding text...')
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video for frame extraction.");
            cap.release();
            return

        frame_count = 0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_video == 0: print("Warning: Video frame count is 0 from OpenCV.")

        pbar_frames = tqdm(total=total_frames_video if total_frames_video > 0 else None, desc="Adding Text to Frames")
        fade_target_color = (255, 255, 255)

        while True:
            ret, frame = cap.read()
            if not ret: break

            active_line_object = None
            line_base_text_color = self.text_color
            line_base_outline_color = self.outline_color
            line_base_spoken_text_color = self.spoken_text_color
            line_base_spoken_outline_color = self.spoken_outline_color

            for line_obj in self.text_array:
                line_start_frame = line_obj["overall_start_frame"]
                line_end_frame = line_obj["overall_end_frame"]

                if frame_count >= line_start_frame and frame_count <= line_end_frame:
                    active_line_object = line_obj

                    if self.caption_transition_duration_frames > 0:
                        frames_into_line = frame_count - line_start_frame
                        frames_from_line_end = line_end_frame - frame_count
                        line_duration = line_end_frame - line_start_frame
                        actual_transition = min(self.caption_transition_duration_frames,
                                                line_duration // 2 if line_duration > 0 else 1)
                        if actual_transition <= 0: actual_transition = 1

                        if frames_into_line < actual_transition:
                            progress = frames_into_line / float(actual_transition)
                            line_base_text_color = self._interpolate_color(fade_target_color, self.text_color, progress)
                            line_base_outline_color = self._interpolate_color(fade_target_color, self.outline_color,
                                                                              progress)
                            line_base_spoken_text_color = self._interpolate_color(fade_target_color,
                                                                                  self.spoken_text_color, progress)
                            line_base_spoken_outline_color = self._interpolate_color(fade_target_color,
                                                                                     self.spoken_outline_color,
                                                                                     progress)
                        elif frames_from_line_end < actual_transition:
                            progress = frames_from_line_end / float(actual_transition)
                            line_base_text_color = self._interpolate_color(fade_target_color, self.text_color, progress)
                            line_base_outline_color = self._interpolate_color(fade_target_color, self.outline_color,
                                                                              progress)
                            line_base_spoken_text_color = self._interpolate_color(fade_target_color,
                                                                                  self.spoken_text_color, progress)
                            line_base_spoken_outline_color = self._interpolate_color(fade_target_color,
                                                                                     self.spoken_outline_color,
                                                                                     progress)
                    break

            if active_line_object:
                total_line_width = 0
                if active_line_object["words"]:
                    last_word = active_line_object["words"][-1]
                    total_line_width = last_word["x_offset_on_line"] + last_word["width"]

                line_start_x = int((frame.shape[1] - total_line_width) / 2)
                if line_start_x < 0: line_start_x = 5
                line_y_baseline = int(frame.shape[0] * self.text_y_position_ratio)

                if self.add_text_background:
                    # Simplified background for karaoke: draw a single rectangle for the whole line
                    # More complex background would involve per-word or per-highlighted-segment boxes
                    if total_line_width > 0:
                        text_height_approx, _ = cv2.getTextSize("Tg", self.font, self.font_scale,
                                                                self.font_thickness)  # Approx height
                        bg_x1 = line_start_x - self.text_padding
                        bg_y1 = line_y_baseline - text_height_approx[1] - self.text_padding
                        bg_x2 = line_start_x + total_line_width + self.text_padding
                        bg_y2 = line_y_baseline + self.text_padding + (
                                    text_height_approx[1] // 4)  # A bit below baseline

                        bg_x1 = max(0, bg_x1);
                        bg_y1 = max(0, bg_y1)
                        bg_x2 = min(frame.shape[1], bg_x2);
                        bg_y2 = min(frame.shape[0], bg_y2)
                        if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.bg_color, cv2.FILLED)

                for word_info in active_line_object["words"]:
                    word_text_to_draw = word_info["text"]
                    word_x = line_start_x + word_info["x_offset_on_line"]
                    word_y = line_y_baseline

                    current_word_text_color = line_base_spoken_text_color
                    current_word_outline_color = line_base_spoken_outline_color

                    if self.word_level_highlight:
                        if frame_count >= word_info["start_frame"] and frame_count <= word_info["end_frame"]:
                            current_word_text_color = line_base_text_color
                            current_word_outline_color = line_base_outline_color
                        # No explicit elif for upcoming words, they default to spoken/dim color
                        # Words already past their end_frame also use spoken/dim color by default

                    if self.add_text_outline and self.outline_thickness > 0:
                        cv2.putText(frame, word_text_to_draw, (word_x, word_y),
                                    self.font, self.font_scale, current_word_outline_color,
                                    self.outline_thickness, self.line_type)

                    cv2.putText(frame, word_text_to_draw, (word_x, word_y),
                                self.font, self.font_scale, current_word_text_color,
                                self.font_thickness, self.line_type)

            cv2.imwrite(os.path.join(output_folder_frames, f"{frame_count}.jpg"), frame)
            frame_count += 1
            if total_frames_video > 0: pbar_frames.update(1)

        cap.release()
        if total_frames_video > 0: pbar_frames.close()
        print(f'Frames extracted and text added. Total frames processed: {frame_count}')

    def create_video(self, output_video_path):
        """Creates the final video with captions and audio."""
        print('Creating final video...')
        base_video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        image_folder = os.path.join(os.path.dirname(self.video_path), f"temp_frames_{base_video_name}")

        if os.path.exists(image_folder): shutil.rmtree(image_folder)
        os.makedirs(image_folder, exist_ok=True)

        self.extract_frames_and_add_text(image_folder)

        image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        if not image_files:
            print("Error: No frames found.")
            if os.path.exists(image_folder): shutil.rmtree(image_folder)
            if self.has_audio and self.audio_path and os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                except Exception:
                    pass
            return

        image_files.sort(key=lambda x: int(x.split(".")[0]))
        full_image_paths = [os.path.join(image_folder, img_file) for img_file in image_files]

        if not (self.fps and self.fps > 0):
            video_fps = 25.0
        else:
            video_fps = self.fps

        final_clip = None;
        final_audio_clip = None
        try:
            video_clip_from_frames = ImageSequenceClip(full_image_paths, fps=video_fps)
            final_clip = video_clip_from_frames

            if self.has_audio and self.audio_path and os.path.exists(self.audio_path):
                try:
                    final_audio_clip = AudioFileClip(self.audio_path)
                    if final_audio_clip.duration < final_clip.duration:
                        final_clip = final_clip.set_duration(final_audio_clip.duration)
                    elif final_audio_clip.duration > final_clip.duration and final_clip.duration > 0:  # Check final_clip.duration > 0
                        # If audio is longer, and video has some duration, set audio to video's duration
                        final_audio_clip = final_audio_clip.set_duration(final_clip.duration)

                    final_clip = final_clip.set_audio(final_audio_clip)
                    print("Audio attached to video.")
                except Exception as e:
                    print(f"Warning: Could not load/attach audio: {e}. Video silent.")
            else:
                print("No audio track to attach or audio extraction failed.")

            final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", preset='medium',
                                       logger=None)

        except Exception as e:
            print(f"Error during final video creation: {e}")
        finally:
            print("Cleaning up temporary files...")
            if final_clip: final_clip.close()
            if final_audio_clip: final_audio_clip.close()
            if os.path.exists(image_folder): shutil.rmtree(image_folder); print(f"Removed temp frame folder.")
            if self.has_audio and self.audio_path and os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path); print(f"Removed temp audio file.")
                except Exception as e:
                    print(f"Warning: Could not remove temp audio: {e}")


def generate_captions(input_path,output_path,model_path):
    if not os.path.exists(input_path):
        print(f"'{input_path}' not found. Attempting to create a dummy video...")
        return -1
    else:
        model_size = "base"

        transcriber = VideoTranscriber(
            model_path=model_path,
            video_path=input_path,
            font=cv2.FONT_HERSHEY_DUPLEX,
            font_scale=1,
            font_thickness=2,
            text_color=(255, 255, 0),  # Cyan for current word
            spoken_text_color=(255, 255, 255),  # White for spoken words
            add_text_background=False,  # Set to True to test background with karaoke
            # bg_color=(30,30,30, 180), # Example: semi-transparent dark background if using alpha blending
            text_y_position_ratio=0.9,
            line_type=cv2.LINE_AA,
            text_wrap_margin_ratio=0.08,
            add_text_outline=True,
            outline_color=(255, 255, 255),  # White outline for current word
            spoken_outline_color=(40, 40, 40),  # Dark gray outline for spoken
            outline_thickness_factor=1.15,
            caption_transition_duration_seconds=0.2,
            word_level_highlight=True
        )

        transcriber.extract_audio()
        transcriber.transcribe_video()

        if transcriber.text_array:
            transcriber.create_video(output_path)
            print(f"Process complete. Output video: {output_path}")
        else:
            print("Process aborted: No text was transcribed or prepared for captions.")

if __name__ == "__main__":
    input_path = "/home/amal/amal/side/youtube-automation/data/Canada:_51st_State?!/avatar_video.mp4"
    output_path = "/home/amal/amal/side/youtube-automation/data/Canada:_51st_State?!/final_video.mp4"
    model_path = "/home/amal/amal/side/youtube-automation/models/base.pt"

    generate_captions(input_path,output_path,model_path)