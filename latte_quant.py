from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForLatte
from Latte.models.latte_t2v import LatteT2V
from Latte.download import find_model
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import T5Tokenizer, T5EncoderModel
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from Latte.sample.pipeline_videogen import VideoGenPipeline
from Latte.utils import save_video_grid
import torch
quantized_model_dir = "/data/data1/pretrained_model/t2v_quant"
pretrained_model_path = '/home/LeiFeng/cgj/Latte/hf_files/t2v_required_models'
video_length = 16
ckpt = '/home/LeiFeng/cgj/Latte/hf_files/t2v.pt'

def quant_and_save():
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16)
    
    examples = get_captions()

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForLatte.from_pretrained(pretrained_model_path, quantize_config, video_length, ckpt)
    scheduler = PNDMScheduler.from_pretrained(pretrained_model_path, 
                                                subfolder="scheduler",
                                                beta_start=0.0001, 
                                                beta_end=0.02, 
                                                beta_schedule='linear',
                                                variance_type='learned_range')
    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                        text_encoder=text_encoder, 
                                        tokenizer=tokenizer, 
                                        scheduler=scheduler, 
                                        transformer=model)
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples, videogen_pipeline=videogen_pipeline, text_encoder=text_encoder)

    # save quantized model
    model.save_quantized(quantized_model_dir)

def load_quant_model():
    model = AutoGPTQForLatte.from_quantized(quantized_model_dir, device="cuda:0", pretrained_model_path=pretrained_model_path, video_length=video_length)
    return model

def generate_video(model, device='cuda:0'):
    # pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    # print(pipeline("auto-gptq is")[0]["generated_text"])
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    examples = [
              'Yellow and black tropical fish dart through the sea.',
              'An epic tornado attacking above aglowing city at night.',
              'Slow pan upward of blazing oak fire in an indoor fireplace.',
              'a cat wearing sunglasses and working as a lifeguard at pool.',
              'Sunset over the sea.',
              'A dog in astronaut suit and sunglasses floating in space.',
              ]
    scheduler = PNDMScheduler.from_pretrained(pretrained_model_path, 
                                                subfolder="scheduler",
                                                beta_start=0.0001, 
                                                beta_end=0.02, 
                                                beta_schedule='linear',
                                                variance_type='learned_range')
    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                        text_encoder=text_encoder, 
                                        tokenizer=tokenizer, 
                                        scheduler=scheduler, 
                                        transformer=model)
    import time
    import imageio
    import os
    video_grids = []
    store_dir = 'res_quant'
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    for prompt in examples:
        start_time = time.time()
        print('Processing the ({}) prompt'.format(prompt))
        video = videogen_pipeline(prompt, 
            video_length=video_length, 
            height=512, 
            width=512, 
            num_inference_steps=50,
            guidance_scale=7.5,
            enable_temporal_attentions=True,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=True,
            device = torch.device('cuda:0'),
            generator=generator,
            ).video
        try:
            name = prompt.replace(' ', '_') + '.mp4'
            path = os.path.join(store_dir, name)
            imageio.mimwrite(path, video[0], fps=8, quality=9) # highest quality is 10, lowest is 0
        except Exception as e:
            print(e)
            print('Error when saving {}'.format(prompt))
        video_grids.append(video)
    video_grids = torch.cat(video_grids, dim=0)
    video_grids = save_video_grid(video_grids)
    
    imageio.mimwrite(os.path.join(store_dir, 'grid.mp4'), video_grids, fps=8, quality=5)
    print('save generated videos')


def generate_video_without_quant(device='cuda:0', dtype=torch.float16):
    # pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    # print(pipeline("auto-gptq is")[0]["generated_text"])
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    model = LatteT2V.from_pretrained_2d(pretrained_model_path, subfolder="transformer", video_length=video_length)
    state_dict = find_model(ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device, dtype=dtype)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=dtype).to(device)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
    examples = [
              'Yellow and black tropical fish dart through the sea.',
              'An epic tornado attacking above aglowing city at night.',
              'Slow pan upward of blazing oak fire in an indoor fireplace.',
              'a cat wearing sunglasses and working as a lifeguard at pool.',
              'Sunset over the sea.',
              'A dog in astronaut suit and sunglasses floating in space.',
              ]
    scheduler = PNDMScheduler.from_pretrained(pretrained_model_path, 
                                                subfolder="scheduler",
                                                beta_start=0.0001, 
                                                beta_end=0.02, 
                                                beta_schedule='linear',
                                                variance_type='learned_range')
    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                        text_encoder=text_encoder, 
                                        tokenizer=tokenizer, 
                                        scheduler=scheduler, 
                                        transformer=model)
    import time
    import imageio
    import os
    video_grids = []
    store_dir = 'res_'+str(dtype)
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    for prompt in examples:
        start_time = time.time()
        print('Processing the ({}) prompt'.format(prompt))
        video = videogen_pipeline(prompt, 
            video_length=video_length, 
            height=512, 
            width=512, 
            num_inference_steps=50,
            guidance_scale=7.5,
            enable_temporal_attentions=True,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=True,
            device = torch.device('cuda:0'),
            generator=generator,
            ).video
        try:
            name = prompt.replace(' ', '_') + '.mp4'
            path = os.path.join(store_dir, name)
            imageio.mimwrite(path, video[0], fps=8, quality=9) # highest quality is 10, lowest is 0
        except Exception as e:
            print(e)
            print('Error when saving {}'.format(prompt))
        video_grids.append(video)
    video_grids = torch.cat(video_grids, dim=0)
    video_grids = save_video_grid(video_grids)
    
    imageio.mimwrite(os.path.join(store_dir, 'grid.mp4'), video_grids, fps=8, quality=5)
    print('save generated videos')



def main():
    # quant_and_save()
    model = load_quant_model()
    generate_video(model)
    # generate_video_without_quant(dtype=torch.float32)
    
def get_captions():
    import json
    prompts = []
    with open('/home/LeiFeng/cgj/Latte/webvid/captions.json', 'r') as json_file:
        captions = json.load(json_file)
    cont = 0
    for key in captions.keys():
        if cont > 128:
            break
        cont += 1
        prompt = captions[key]
        prompts.append(prompt)
    return prompts

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()

