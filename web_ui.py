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
        return ["Stable Diffusion 3.5"]  # 默认返回最新模型

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

# 创建 Gradio 界面
with gr.Blocks(title="Stable Diffusion WebUI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Stable Diffusion 图像生成器
    
    ### 使用说明
    1. 选择想要使用的模型
    2. 输入正面提示词（描述你想要的图像）
    3. 可选：输入负面提示词（描述你不想要的元素）
    4. 调整参数（步数、CFG Scale、尺寸等）
    5. 点击生成按钮
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                value="Stable Diffusion 3.5",
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
            
    # 添加示例
    gr.Examples(
        examples=[
            ["Stable Diffusion 3.5", "a beautiful girl in white dress, high quality, best quality, extremely detailed", 
             "ugly, bad quality", 30, 7.0, 512, 512, -1],
            ["Stable Diffusion 3.5", "professional portrait photo of a young woman, photorealistic, 8k, detailed lighting", 
             "cartoon, anime, illustration", 30, 8.0, 512, 768, 42],
        ],
        inputs=[model_dropdown, prompt, negative_prompt, steps, cfg_scale, width, height, seed],
        outputs=output,
        fn=generate_image,
        cache_examples=True
    )
    
    # 绑定生成按钮
    generate_btn.click(
        fn=generate_image,
        inputs=[model_dropdown, prompt, negative_prompt, steps, cfg_scale, width, height, seed],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    ) 