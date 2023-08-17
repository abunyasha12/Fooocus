import threading

from pydantic import BaseModel

buffer = []
outputs = []

class Txt2imgRequest(BaseModel):
    prompt: str = "field of violet flowers with lake and mountains in the back"
    negative_prompt: str = ""
    style_selction: str = "game-minecraft"
    performance_selction: str = "Speed"
    aspect_ratios_selction: str = "1024×1024"
    image_number: int = 1
    image_seed: int = -1
    sharpness: int = 2
    base_model: str = "sd_xl_base_1.0_0.9vae.safetensors"
    refiner_model: str = "sd_xl_refiner_1.0_0.9vae.safetensors"
    lora_ctrls: tuple = ("None", 0, "None", 0, "None", 0, "None", 0, "None", 0)

class Txt2imgResponse(BaseModel):
    images: tuple[bytes]


def worker():
    global buffer, outputs

    import random
    import time

    import modules.default_pipeline as pipeline
    import modules.patch
    import modules.path
    from modules.sdxl_styles import apply_style, aspect_ratios


    def handler(task: Txt2imgRequest):
        prompt = task.prompt
        negative_prompt = task.negative_prompt
        style_selction = task.style_selction
        performance_selction = task.performance_selction
        aspect_ratios_selction = task.aspect_ratios_selction
        image_number = task.image_number
        image_seed = task.image_seed
        sharpness = task.sharpness
        base_model_name = task.base_model
        refiner_model_name = task.refiner_model
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5 = task.lora_ctrls

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        pipeline.clean_prompt_cond_caches()

        p_txt, n_txt = apply_style(style_selction, prompt, negative_prompt)

        if performance_selction == 'Speed':
            steps = 30
            switch = 20
        else:
            steps = 60
            switch = 40

        width, height = aspect_ratios[aspect_ratios_selction]

        results = []
        seed = image_seed
        if not isinstance(seed, int) or seed < 0 or seed > 1024*1024*1024:
            seed = random.randint(1, 1024*1024*1024)

        def callback(step, x0, x, total_steps, y):
            pass

        for i in range(image_number):
            imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, callback=callback)

            seed += 1
            results += imgs

        outputs.append(['results', results])
        return

    while True:
        time.sleep(0.1)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)



threading.Thread(target=worker, daemon=True).start()
