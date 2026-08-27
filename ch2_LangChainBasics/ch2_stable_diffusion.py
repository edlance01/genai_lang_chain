import replicate

output = replicate.run(
    "stability-ai/stable-diffusion-3.5-medium",
    input={
        "prompt": "A detailed technical diagram of an AI agent",
        "prompt_strength": 0.85,
        "cfg": 4.5,
        "steps": 30,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "output_quality": 90,
    },
)

print(output[0])
