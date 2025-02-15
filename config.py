import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    audio_file = "assets/audio1.mp3"
