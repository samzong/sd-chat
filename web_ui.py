import gradio as gr
import requests
import base64
from PIL import Image
import io
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_models():
    try:
        response = requests.get("http://localhost:8000/models")
        models = response.json()["models"]
        logger.info(f"获取到可用模型: {models}")
        return models
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return ["Stable Diffusion XL"]  # 默认返回最新模型

def get_available_video_models():
    try:
        response = requests.get("http://localhost:8001/video_models")
        models = response.json()["models"]
        logger.info(f"获取到可用视频模型: {models}")
        return models
    except Exception as e:
        logger.error(f"获取视频模型列表失败: {str(e)}")
        return ["Stable Video Diffusion"]  # 默认返回最新模型

def generate_image(model_name, prompt, negative_prompt, steps, cfg_scale, width, height, seed):
    try:
        logger.info(f"开始生成图像请求...")
        logger.info(f"模型: {model_name}")
        logger.info(f"参数: steps={steps}, cfg_scale={cfg_scale}, size={width}x{height}, seed={seed}")
        
        # 调用 API
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "model_name": model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "seed": seed
            }
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("detail", "未知错误")
            logger.error(f"API 错误: {error_msg}")
            raise gr.Error(f"生成失败: {error_msg}")
            
        # 解码返回的图像
        try:
            img_data = base64.b64decode(response.json()["image"])
            image = Image.open(io.BytesIO(img_data))
            logger.info("图像生成成功")
            return image
        except Exception as e:
            logger.error(f"图像处理错误: {str(e)}")
            raise gr.Error(f"图像处理错误: {str(e)}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API 请求错误: {str(e)}")
        raise gr.Error(f"API 连接失败: {str(e)}")
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
        raise gr.Error(str(e))

def generate_video(model_name, image, prompt, negative_prompt, width, height, num_frames, fps, 
                  motion_bucket_id, noise_aug_strength, seed):
    try:
        logger.info(f"开始生成视频请求...")
        logger.info(f"模型: {model_name}")
        logger.info(f"参数: frames={num_frames}, fps={fps}, size={width}x{height}")
        
        # 准备请求数据
        request_data = {
            "model_name": model_name,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "motion_bucket_id": motion_bucket_id,
            "noise_aug_strength": noise_aug_strength,
            "seed": seed
        }
        
        # 如果提供了图片，转换为base64
        if image is not None:
            if isinstance(image, str):  # 图片路径
                with open(image, 'rb') as img_file:
                    img_data = img_file.read()
                    request_data["image_base64"] = base64.b64encode(img_data).decode()
            else:  # PIL Image
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                request_data["image_base64"] = base64.b64encode(img_buffer.getvalue()).decode()
        elif prompt:  # 如果提供了提示词
            request_data["prompt"] = prompt
        else:
            raise gr.Error("必须提供图片或提示词其中之一")
        
        # 调用 API
        response = requests.post(
            "http://localhost:8001/generate_video",
            json=request_data
        )
        
        if response.status_code != 200:
            error_msg = response.json().get("detail", "未知错误")
            logger.error(f"API 错误: {error_msg}")
            raise gr.Error(f"生成失败: {error_msg}")
        
        # 解码返回的视频
        try:
            video_data = base64.b64decode(response.json()["video"])
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(video_data)
            logger.info("视频生成成功")
            return temp_file
        except Exception as e:
            logger.error(f"视频处理错误: {str(e)}")
            raise gr.Error(f"视频处理错误: {str(e)}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API 请求错误: {str(e)}")
        raise gr.Error(f"API 连接失败: {str(e)}")
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
        raise gr.Error(str(e))

# 创建 Gradio 界面
with gr.Blocks(title="AI 创作平台", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI 创作平台
    """)
    
    with gr.Tabs():
        # 图像生成标签页
        with gr.Tab("图像生成"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        value="Stable Diffusion XL",
                        label="选择模型",
                        info="选择要使用的 AI 模型"
                    )
                    prompt = gr.Textbox(
                        label="正面提示词",
                        placeholder="描述你想要生成的图像内容",
                        info="详细描述你想要的图像效果",
                        lines=3
                    )
                    negative_prompt = gr.Textbox(
                        label="负面提示词",
                        placeholder="描述你不想在图像中出现的元素",
                        info="指定你不想要的内容",
                        value="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft",
                        lines=2
                    )
                    
                    with gr.Row():
                        width = gr.Slider(
                            minimum=384,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="宽度",
                            info="图像宽度（像素）"
                        )
                        height = gr.Slider(
                            minimum=384,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="高度",
                            info="图像高度（像素）"
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(
                            minimum=1,
                            maximum=150,
                            step=1,
                            value=20,
                            label="推理步数",
                            info="值越大生成质量越高，但速度更慢"
                        )
                        cfg_scale = gr.Slider(
                            minimum=1,
                            maximum=20,
                            step=0.5,
                            value=7,
                            label="CFG Scale",
                            info="提示词相关性权重，值越大越严格遵循提示词"
                        )
                    
                    seed = gr.Number(
                        value=-1,
                        label="随机种子",
                        info="设置固定值可以重复生成相同的图像，-1 表示随机"
                    )
                    
                    generate_btn = gr.Button("生成图像", variant="primary")
                
                with gr.Column(scale=1):
                    output = gr.Image(label="生成结果", show_label=True)
            
            # 绑定图像生成按钮
            generate_btn.click(
                fn=generate_image,
                inputs=[model_dropdown, prompt, negative_prompt, steps, cfg_scale, width, height, seed],
                outputs=output
            )
            
            # 添加图像生成示例
            gr.Examples(
                examples=[
                    ["Stable Diffusion XL", "a beautiful girl in white dress, high quality, best quality, extremely detailed", 
                     "ugly, bad quality", 30, 7.0, 512, 512, -1],
                    ["Realistic Vision", "ultra realistic portrait of a beautiful young woman, perfect face, detailed eyes, 8k uhd, high quality, photorealistic", 
                     "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured", 30, 7.0, 512, 512, -1],
                    ["Dreamshaper", "magical fantasy landscape with floating islands, ethereal atmosphere, mystical lighting, detailed vegetation, 8k quality", 
                     "ugly, blurry, low quality, distorted", 30, 7.0, 512, 512, -1],
                    ["Stable Diffusion 1.5", "professional studio portrait of a young woman, natural lighting, bokeh background, 8k", 
                     "cartoon, anime, illustration, painting, drawing, low quality", 30, 7.0, 512, 512, 42],
                ],
                inputs=[model_dropdown, prompt, negative_prompt, steps, cfg_scale, width, height, seed],
                outputs=output,
                fn=generate_image,
                cache_examples=True
            )
        
        # 视频生成标签页
        with gr.Tab("视频生成"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_model = gr.Dropdown(
                        choices=get_available_video_models(),
                        value="Stable Video Diffusion",
                        label="选择模型",
                        info="选择要使用的视频生成模型"
                    )
                    
                    with gr.Tab("文本生成"):
                        video_prompt = gr.Textbox(
                            label="场景描述",
                            placeholder="详细描述你想要生成的视频场景",
                            info="描述视频中的动作、场景和氛围",
                            lines=3
                        )
                    
                    with gr.Tab("图片生成"):
                        input_image = gr.Image(
                            label="上传图片（作为视频的起始帧）",
                            type="pil"
                        )
                    
                    video_negative_prompt = gr.Textbox(
                        label="负面提示词",
                        placeholder="描述你不想在视频中出现的元素",
                        info="指定你不想要的内容",
                        value="ugly, blurry, low quality, distorted, shaky camera, bad lighting",
                        lines=2,
                        visible=lambda: video_prompt.value is not None  # 只在文本生成时显示
                    )
                    
                    with gr.Row():
                        video_width = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="视频宽度",
                            info="视频画面宽度（像素）"
                        )
                        video_height = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="视频高度",
                            info="视频画面高度（像素）"
                        )
                    
                    with gr.Row():
                        num_frames = gr.Slider(
                            minimum=8,
                            maximum=128,
                            step=8,
                            value=16,
                            label="帧数",
                            info="生成视频的总帧数"
                        )
                        fps = gr.Slider(
                            minimum=8,
                            maximum=60,
                            step=1,
                            value=24,
                            label="帧率(FPS)",
                            info="视频播放速度"
                        )
                    
                    with gr.Row():
                        motion_bucket_id = gr.Slider(
                            minimum=1,
                            maximum=255,
                            step=1,
                            value=127,
                            label="运动幅度",
                            info="控制视频中的运动强度"
                        )
                        noise_aug_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.4,
                            label="噪声强度",
                            info="控制视频的随机性"
                        )
                    
                    video_seed = gr.Number(
                        value=-1,
                        label="随机种子",
                        info="设置固定值可以重复生成相同的视频，-1 表示随机"
                    )
                    
                    generate_video_btn = gr.Button("生成视频", variant="primary")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="生成结果",
                        format="mp4",
                        show_label=True
                    )
            
            # 绑定视频生成按钮
            generate_video_btn.click(
                fn=generate_video,
                inputs=[video_model, input_image, video_prompt, video_negative_prompt, 
                        video_width, video_height, num_frames, fps,
                        motion_bucket_id, noise_aug_strength, video_seed],
                outputs=video_output
            )
            
            # 添加视频生成示例
            with gr.Tab("文本示例"):
                gr.Examples(
                    examples=[
                        ["Stable Video Diffusion", None, "A beautiful butterfly flying in a garden with flowers", 
                         "shaky, blurry, low quality", 512, 512, 16, 24, 127, 0.4, -1],
                        ["Stable Video Diffusion", None, "A serene waterfall in a lush forest, sunlight filtering through trees", 
                         "distorted, pixelated, low quality", 512, 512, 24, 30, 150, 0.5, 42],
                        ["Stable Video Diffusion", None, "A space ship flying through a colorful nebula with stars twinkling", 
                         "blurry, shaky, poor quality", 512, 512, 32, 24, 180, 0.6, -1],
                    ],
                    inputs=[video_model, input_image, video_prompt, video_negative_prompt, 
                           video_width, video_height, num_frames, fps,
                           motion_bucket_id, noise_aug_strength, video_seed],
                    outputs=video_output,
                    fn=generate_video,
                    cache_examples=True
                )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    ) 