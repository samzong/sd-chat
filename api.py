import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import base64
import logging
import platform

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查 MPS 可用性
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("使用 MPS 设备")
else:
    DEVICE = "cpu"
    logger.info("使用 CPU 设备")

app = FastAPI()

# 定义可用的模型列表
AVAILABLE_MODELS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5"
}

# 更新请求模型
class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model_name: str = "Stable Diffusion 1.5"
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    width: int = Field(default=512, ge=384, le=1024)
    height: int = Field(default=512, ge=384, le=1024)
    seed: int = Field(default=-1)

# 模型缓存
model_cache = {}

def get_pipe(model_name: str):
    if model_name not in model_cache:
        model_id = AVAILABLE_MODELS[model_name]
        try:
            logger.info(f"开始加载模型: {model_name} ({model_id})")
            
            # 基础配置
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            logger.info("成功创建 pipeline")
            
            # 移动到 MPS 设备
            logger.info(f"正在将模型移动到 {DEVICE} 设备...")
            pipe = pipe.to(DEVICE)
            
            # 内存优化
            pipe.enable_attention_slicing()
            logger.info("已应用内存优化")
            
            model_cache[model_name] = pipe
            logger.info(f"模型 {model_name} 加载完成并已优化")
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            logger.error(error_msg)
            if model_name in model_cache:
                del model_cache[model_name]
            raise HTTPException(status_code=500, detail=error_msg)
    return model_cache[model_name]

@app.get("/models")
async def get_models():
    return {"models": list(AVAILABLE_MODELS.keys())}

@app.post("/generate")
async def generate_image(request: PromptRequest):
    try:
        logger.info(f"收到生成请求，模型: {request.model_name}")
        logger.info(f"提示词: {request.prompt}")
        logger.info(f"负面提示词: {request.negative_prompt}")
        logger.info(f"参数: steps={request.steps}, cfg_scale={request.cfg_scale}, "
                   f"size={request.width}x{request.height}, seed={request.seed}")
        
        # 获取对应的模型
        pipe = get_pipe(request.model_name)
        
        # 设置随机种子
        generator = None
        if request.seed != -1:
            generator = torch.Generator(DEVICE).manual_seed(request.seed)
        
        # 处理 CLIP 输入限制
        max_length = 77
        prompt = request.prompt[:max_length]
        negative_prompt = request.negative_prompt[:max_length]
        
        # 生成图像
        logger.info("开始生成图像...")
        try:
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=request.steps,
                    guidance_scale=request.cfg_scale,
                    width=request.width,
                    height=request.height,
                    generator=generator
                ).images[0]
        except Exception as e:
            logger.error(f"图像生成失败: {str(e)}")
            if request.model_name in model_cache:
                del model_cache[request.model_name]
            raise HTTPException(status_code=500, detail=f"图像生成失败: {str(e)}")
        
        logger.info("图像生成完成，正在转换格式...")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("处理完成，返回结果")
        return {"image": img_str}
    except Exception as e:
        error_msg = f"生成过程出错: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 