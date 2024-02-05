import cv2
import streamlink
import streamlit as st
import time
import tempfile
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import speech_recognition as sr

# Load environment variables
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def main():
    st.title("Insightly Live Video Analysis")

    youtube_video_url = st.text_input("Enter YouTube Video URL:")
    duration = st.slider("Select Duration (seconds):", min_value=1, max_value=60, value=10)
    frames_per_second = st.slider("Select Frames per Second:", min_value=1, max_value=10, value=1)

    # Create empty space for frames
    frames_placeholder = st.empty()

    if st.button("Extract Frames"):
        st.info("Extracting frames. Please wait...")
        extracted_frames = extract_recent_frames(youtube_video_url, "temp_frames", duration, frames_per_second)

        if extracted_frames:
            st.success("Frames extracted successfully!")

            # Display frames in a grid format with frame description on click
            display_frame_grid(extracted_frames, frames_placeholder)

    st.write("Actions:")  # Header for the actions/buttons section

    # Create empty spaces for analysis results
    description_placeholder = st.empty()
    frame_description_placeholder = st.empty()
    category_placeholder = st.empty()

    if st.button("Generate Description"):
        st.info("Generating video description. Please wait...")
        description = generate_description(st.session_state.extracted_frames) if st.session_state.extracted_frames else None

        if description:
            st.subheader("Video Description")
            description_placeholder.write(description)
        else:
            st.error("Failed to generate video description.")

    if st.button("Frame Description"):
        st.info("Generating frame description. Please wait...")
        frame_description = generate_frame_description(st.session_state.extracted_frames) if st.session_state.extracted_frames else None

        if frame_description:
            st.subheader("Frame Description")
            frame_description_placeholder.write(frame_description)
        else:
            st.error("Failed to generate frame description.")

    if st.button("Category of Video"):
        st.info("Generating categories. Please wait...")
        categories = generate_category(st.session_state.extracted_frames) if st.session_state.extracted_frames else None

        if categories:
            st.subheader("Categories of live stream:")
            category_placeholder.write(categories)
        else:
            st.error("Failed to generate video description.")


@st.cache_data
def extract_recent_frames(video_url, output_folder, duration=10, frames_per_second=1):
    streams = streamlink.streams(video_url)
 
    if not streams:
        st.error("Error: Unable to retrieve streams. Make sure the YouTube video URL is valid.")
        return
 
    stream_url = streams['best'].url
 
    cap = cv2.VideoCapture(stream_url)
 
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * duration)
    frame_interval = int(fps / frames_per_second)
 
    frame_count = 0
    start_time = time.time()
 
    # Clear previous frames without using st.session_state
    extracted_frames = []
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Couldn't read frame.")
            break
 
        elapsed_time = time.time() - start_time
        if frame_count % frame_interval == 0 and elapsed_time <= duration:
            # Convert frame to base64
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            extracted_frames.append(base64_frame)
 
        frame_count += 1
 
        if elapsed_time > duration:
            break
 
    cap.release()
 
    return extracted_frames
    pass


def generate_description(base64_frames):
    try:
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "1. Generate a description for this sequence of video frames in about 90 words. 2.Return the following: i. List of objects in the video ii. Any restrictive content or sensitive content and if so which frame.",
                    *map(lambda x: {"image": x, "resize": 428}, base64_frames[0::30]),
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt_messages,
            max_tokens=3000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_description: {e}")
        return None
    return "Video description placeholder"


def generate_frame_description(base64_frames):
    try:
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "Describe what is happening in each frame.",
                    *map(lambda x: {"image": x, "resize": 428}, base64_frames[0::30]),
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt_messages,
            max_tokens=3000,
        )
        return response.choices[0].message.content  
 
    except Exception as e:
        print(f"Error in generate_description: {e}")
        return None
    return "Frame description placeholder"


def generate_category(base64_frames):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                "What category can this video be tagged to?",
                *map(lambda x: {"image": x, "resize": 428}, base64_frames[0::30]),
            ],
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=prompt_messages,
        max_tokens=3000,
    )
    return response.choices[0].message.content  
    return "Video category placeholder"


def display_frame_grid(extracted_frames, placeholder):
    cols_per_row = 3
    n_frames = len(extracted_frames)
    for idx in range(0, n_frames, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_index in range(cols_per_row):
            frame_idx = idx + col_index
            if frame_idx < n_frames:
                with cols[col_index]:
                    # Decode base64 and display the frame
                    decoded_frame = base64.b64decode(extracted_frames[frame_idx])
                    st.image(decoded_frame, channels="BGR", caption=f'Frame {frame_idx + 1}', use_column_width=True,
                             output_format="JPEG")

    # Update the placeholder to keep the frames visible
    placeholder.empty()

if __name__ == "__main__":
    main()
