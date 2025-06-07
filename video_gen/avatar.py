import os
import tempfile
import subprocess
from pytoon.animator import animate
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
import math
import gc

def extract_audio_with_ffmpeg(video_file, start_time, duration, output_path):
    """
    Extract audio using FFmpeg directly to avoid MoviePy audio issues
    """
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output files
        '-i', video_file,
        '-ss', str(start_time),
        '-t', str(duration),
        '-vn',  # no video
        '-acodec', 'mp3',
        '-ar', '44100',  # sample rate
        '-ab', '192k',   # bitrate
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg audio extraction failed: {e}")
        return False

def combine_video_audio_with_ffmpeg(video_file, audio_file, output_file):
    """
    Combine video and audio using FFmpeg directly
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-i', audio_file,
        '-c:v', 'copy',  # copy video stream without re-encoding
        '-c:a', 'aac',   # encode audio as AAC
        '-map', '0:v:0',  # map video from first input
        '-map', '1:a:0',  # map audio from second input
        '-shortest',      # end output when shortest input ends
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg combine failed: {e}")
        return False

def get_precise_duration(file_path):
    """
    Get precise duration using FFmpeg
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return None

def add_avatar_chunked(audio_file, video_file, output_file, scale=0.5, chunk_duration=10):
    """
    Process video in chunks to avoid RAM overflow - FIXED VERSION
    
    Args:
        audio_file: Path to audio file
        video_file: Path to input video file
        output_file: Path to output video file
        scale: Scale factor for avatar
        chunk_duration: Duration of each chunk in seconds (default: 10)
    """
    
    # Get precise durations
    video_duration = get_precise_duration(video_file)
    audio_duration = get_precise_duration(audio_file)
    
    if video_duration is None or audio_duration is None:
        # Fallback to MoviePy
        background_video = VideoFileClip(video_file)
        video_duration = background_video.duration
        background_video.close()
        
        audio_clip = AudioFileClip(audio_file)
        audio_duration = audio_clip.duration
        audio_clip.close()
    
    # Use the shorter duration to avoid extending the video
    total_duration = min(video_duration, audio_duration)
    
    # Calculate number of chunks needed
    num_chunks = math.ceil(total_duration / chunk_duration)
    
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Using duration: {total_duration:.2f} seconds")
    print(f"Processing in {num_chunks} chunks of {chunk_duration} seconds each")
    
    # Create temporary directory for chunk files
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    
    try:
        # Load background video once
        background_video = VideoFileClip(video_file)
        
        # Process each chunk
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, total_duration)
            actual_duration = end_time - start_time
            
            # Skip if duration is too small
            if actual_duration < 0.1:
                print(f"Skipping chunk {i+1} - duration too small: {actual_duration:.3f}s")
                continue
            
            print(f"Processing chunk {i+1}/{num_chunks}: {start_time:.2f}s - {end_time:.2f}s (duration: {actual_duration:.2f}s)")
            
            # Extract audio chunk for animation generation using the SAME source
            audio_chunk_path = os.path.join(temp_dir, f"audio_chunk_{i:03d}.mp3")
            
            # Use FFmpeg for consistent audio extraction
            if not extract_audio_with_ffmpeg(audio_file, start_time, actual_duration, audio_chunk_path):
                print(f"Failed to extract audio chunk {i+1}, skipping...")
                continue
            
            # Verify the extracted audio chunk duration
            chunk_audio_duration = get_precise_duration(audio_chunk_path)
            if chunk_audio_duration and abs(chunk_audio_duration - actual_duration) > 0.1:
                print(f"Warning: Audio chunk duration mismatch - expected: {actual_duration:.2f}s, got: {chunk_audio_duration:.2f}s")
            
            # Extract video chunk with precise timing
            video_chunk = background_video.subclip(start_time, end_time)
            
            # Export paths
            chunk_output_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp4")
            temp_avatar_path = os.path.join(temp_dir, f"temp_avatar_{i:03d}.mp4")
            temp_video_no_audio_path = os.path.join(temp_dir, f"temp_video_no_audio_{i:03d}.mp4")
            temp_audio_final_path = os.path.join(temp_dir, f"temp_audio_final_{i:03d}.mp3")
            
            try:
                # Create animation for this chunk
                print(f"Creating animation for chunk {i+1}...")
                animation = animate(audio_file=audio_chunk_path)
                
                # Export avatar animation with background but without audio
                animation.export(
                    path=temp_avatar_path, 
                    background=video_chunk, 
                    scale=scale,
                )
                
                # Load the avatar clip and ensure it matches the expected duration
                avatar_clip = VideoFileClip(temp_avatar_path)
                avatar_duration = avatar_clip.duration
                
                # Trim avatar clip to match expected duration if necessary
                if abs(avatar_duration - actual_duration) > 0.1:
                    print(f"Trimming avatar clip from {avatar_duration:.2f}s to {actual_duration:.2f}s")
                    avatar_clip = avatar_clip.subclip(0, actual_duration)
                
                # Create a fresh video chunk for compositing with exact timing
                video_chunk_fresh = background_video.subclip(start_time, end_time)
                if abs(video_chunk_fresh.duration - actual_duration) > 0.1:
                    video_chunk_fresh = video_chunk_fresh.subclip(0, actual_duration)
                
                # Create composite video WITHOUT audio first
                composite_video = CompositeVideoClip([
                    video_chunk_fresh, 
                    avatar_clip.set_position('center')
                ], size=video_chunk_fresh.size)
                
                # Ensure composite video has the correct duration
                composite_video = composite_video.set_duration(actual_duration)
                
                # Write video without audio first
                composite_video.write_videofile(
                    temp_video_no_audio_path,
                    codec='libx264',
                    audio=False,  # No audio
                    verbose=False,
                    logger=None
                )
                
                # Clean up video resources
                composite_video.close()
                avatar_clip.close()
                video_chunk_fresh.close()
                del composite_video, avatar_clip, video_chunk_fresh
                gc.collect()
                
                # Extract the ORIGINAL audio segment that matches this video chunk exactly
                print(f"Extracting original audio for chunk {i+1}...")
                if not extract_audio_with_ffmpeg(audio_file, start_time, actual_duration, temp_audio_final_path):
                    print(f"Failed to extract final audio for chunk {i+1}")
                    continue
                
                # Verify final audio duration
                final_audio_duration = get_precise_duration(temp_audio_final_path)
                if final_audio_duration and abs(final_audio_duration - actual_duration) > 0.1:
                    print(f"Warning: Final audio duration mismatch - expected: {actual_duration:.2f}s, got: {final_audio_duration:.2f}s")
                
                gc.collect()
                
                # Combine video and audio using FFmpeg with exact timing
                print(f"Combining video and audio for chunk {i+1}...")
                if not combine_video_audio_with_ffmpeg(temp_video_no_audio_path, temp_audio_final_path, chunk_output_path):
                    print(f"FFmpeg combine failed for chunk {i+1}, trying MoviePy...")
                    
                    # Fallback to MoviePy with precise timing
                    final_video_clip = VideoFileClip(temp_video_no_audio_path)
                    final_audio_clip = AudioFileClip(temp_audio_final_path)
                    
                    # Ensure both have the same duration
                    min_duration = min(final_video_clip.duration, final_audio_clip.duration, actual_duration)
                    final_video_clip = final_video_clip.subclip(0, min_duration)
                    final_audio_clip = final_audio_clip.subclip(0, min_duration)
                    
                    final_composite = final_video_clip.set_audio(final_audio_clip)
                    
                    final_composite.write_videofile(
                        chunk_output_path,
                        codec='libx264',
                        audio_codec='aac',
                        verbose=False,
                        logger=None
                    )
                    
                    # Clean up final resources
                    final_composite.close()
                    final_video_clip.close()
                    final_audio_clip.close()
                    del final_composite, final_video_clip, final_audio_clip
                
                # Verify the output chunk duration
                chunk_output_duration = get_precise_duration(chunk_output_path)
                if chunk_output_duration:
                    print(f"Chunk {i+1} output duration: {chunk_output_duration:.2f}s (expected: {actual_duration:.2f}s)")
                    if abs(chunk_output_duration - actual_duration) > 0.1:
                        print(f"Warning: Output chunk duration mismatch!")
                
                gc.collect()
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                # Clean up partial files
                for temp_file in [temp_avatar_path, temp_video_no_audio_path, temp_audio_final_path]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                continue  # Skip this chunk instead of failing completely
            finally:
                # Always clean up temporary files for this chunk
                for temp_file in [temp_avatar_path, temp_video_no_audio_path, temp_audio_final_path]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            
            chunk_files.append(chunk_output_path)
            
            # Clean up chunk resources
            video_chunk.close()
            del video_chunk
            if os.path.exists(audio_chunk_path):
                os.remove(audio_chunk_path)
            
            # Force garbage collection
            gc.collect()
            
            print(f"Completed chunk {i+1}/{num_chunks}")
        
        # Close the original background video
        background_video.close()
        del background_video
        gc.collect()
        
        # Load and concatenate all processed chunks
        print("Merging all chunks...")
        if not chunk_files:
            raise Exception("No valid chunks were created")
        
        # Use FFmpeg for concatenation to avoid timing issues
        concat_file_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file_path, 'w') as f:
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    f.write(f"file '{chunk_file}'\n")
        
        # Use FFmpeg concat demuxer for precise concatenation
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file_path,
            '-c', 'copy',  # Copy streams without re-encoding
            output_file
        ]
        
        try:
            subprocess.run(concat_cmd, check=True, capture_output=True)
            print(f"Successfully created {output_file} using FFmpeg concat")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg concat failed: {e}")
            print("Falling back to MoviePy concatenation...")
            
            # Fallback to MoviePy
            processed_clips = []
            
            try:
                for chunk_file in chunk_files:
                    if os.path.exists(chunk_file):
                        clip = VideoFileClip(chunk_file)
                        processed_clips.append(clip)
                
                if not processed_clips:
                    raise Exception("No valid chunks were created")
                
                # Concatenate all clips
                final_video = concatenate_videoclips(processed_clips, method="compose")
                
                # Write the final output
                final_video.write_videofile(
                    output_file,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                
                # Clean up
                final_video.close()
                del final_video
                
            finally:
                # Clean up processed clips
                for clip in processed_clips:
                    if clip:
                        clip.close()
                del processed_clips
                gc.collect()
        
        # Verify final output duration
        final_duration = get_precise_duration(output_file)
        if final_duration:
            print(f"Final output duration: {final_duration:.2f}s (expected: {total_duration:.2f}s)")
            if abs(final_duration - total_duration) > 0.5:
                print(f"Warning: Final output duration significantly different from input!")
        
        print(f"Successfully created {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except OSError:
                    pass  # File might be in use, skip
        
        # Remove temp directory if possible
        try:
            # Remove any remaining files in temp directory
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            os.rmdir(temp_dir)
        except OSError:
            pass  # Directory not empty or in use, leave it

def add_avatar(audio_file, video_file, output_file, scale=0.5):
    """
    Original function wrapper - now uses chunked processing by default
    """
    add_avatar_chunked(audio_file, video_file, output_file, scale)

if __name__ == "__main__":
    audio_path = "/home/amal/amal/side/youtube-automation/data/Canada:_51st_State?!/audio/audio_output.mp3"
    video_path = "/home/amal/amal/side/youtube-automation/data/Canada:_51st_State?!/merged_video.mp4"
    output_path = "/home/amal/amal/side/youtube-automation/data/Canada:_51st_State?!/avatar_video.mp4"
    scale = 0.5
    
    # Use chunked processing (you can adjust chunk_duration if needed)
    add_avatar_chunked(
        audio_file=audio_path, 
        video_file=video_path, 
        output_file=output_path, 
        scale=scale,
        chunk_duration=10  # Process in 10-second chunks
    )