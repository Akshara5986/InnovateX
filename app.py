import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Configuration Class
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_gen_model_id = "runwayml/stable-diffusion-v1-5"
    prompt_gen_model_id = "gpt2"
    audio_sample_rate = 16000
    image_save_path = "assets/generated_image.png"

# Load Speech-to-Text Model
@st.cache_resource
def load_speech_to_text_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(CFG.device)
    return processor, model

# Load Text Generation Model
@st.cache_resource
def load_text_generation_model():
    return pipeline("text-generation", model=CFG.prompt_gen_model_id)

# Load Image Generation Model
@st.cache_resource
def load_image_gen_model():
    model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float16
    )
    model = model.to(CFG.device)
    return model

# Audio to Text Function
def audio_to_text(audio_file):
    processor, model = load_speech_to_text_model()
    speech, rate = torchaudio.load(audio_file)
    resampler = torchaudio.transforms.Resample(rate, CFG.audio_sample_rate)
    speech = resampler(speech[0]).numpy()

    input_values = processor(speech, return_tensors="pt", sampling_rate=CFG.audio_sample_rate).input_values.to(CFG.device)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Enhance Text Prompt
def enhance_prompt(text):
    text_gen = load_text_generation_model()
    enhanced_prompt = text_gen(text, max_length=50, num_return_sequences=1)[0]['generated_text']
    return enhanced_prompt

# Generate Image from Text
def generate_image(prompt):
    image_gen_model = load_image_gen_model()
    image = image_gen_model(prompt).images[0]
    image.save(CFG.image_save_path)
    return image

# Streamlit UI with Enhanced Design
st.set_page_config(page_title="Audio2Art: Voice to Visuals", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f0f2f6; padding: 2rem; border-radius: 10px;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
        .stButton>button:hover {background-color: #45a049;}
    </style>
""", unsafe_allow_html=True)

st.title("🎨 Audio2Art: Transform Your Voice into Visuals")
st.header("Upload Your Audio File Below")

# File Uploader
audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
if audio_file:
    st.audio(audio_file, format='audio/mp3')
    
    # Step 1: Convert Audio to Text
    with st.spinner("Transcribing Audio..."):
        transcription = audio_to_text(audio_file)
    st.subheader("Transcription:")
    st.write(transcription)

    # Step 2: Enhance Text Prompt
    with st.spinner("Enhancing Prompt..."):
        enhanced_prompt = enhance_prompt(transcription)
    st.subheader("Enhanced Prompt:")
    st.write(enhanced_prompt)

    # Step 3: Generate Image
    with st.spinner("Generating Image..."):
        generated_image = generate_image(enhanced_prompt)
    st.subheader("Generated Image:")
    st.image(generated_image, caption='Generated Image',width=True)

    st.success("Image Generation Complete!")
