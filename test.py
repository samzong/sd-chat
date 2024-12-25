from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt_str = input("Enter a prompt: ")

if prompt_str == "":
    prompt = "a photo of an astronaut riding a horse on mars"
else:
    prompt = prompt_str

image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
