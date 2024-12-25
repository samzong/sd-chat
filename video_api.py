import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from diffusers import StableVideoDiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import cv2
import io
import base64
import logging
import tempfile
import os
from PIL import Image

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
    "Stable Video Diffusion": {
        "model_id": "stabilityai/stable-video-diffusion-img2vid",
        "pipeline_class": StableVideoDiffusionPipeline,
        "image_model_id": "stabilityai/stable-diffusion-xl-base-1.0"
    }
}

# 模型特定配置
MODEL_CONFIGS = {
    "Stable Video Diffusion": {
        "torch_dtype": torch.float32,
        "variant": "fp16"
    }
}

class VideoRequest(BaseModel):
    prompt: str | None = None
    image_base64: str | None = None  # 新增：支持base64编码的图片输入
    negative_prompt: str = ""
    model_name: str = "Stable Video Diffusion"
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    num_frames: int = Field(default=16, ge=8, le=128)
    fps: int = Field(default=24, ge=8, le=60)
    motion_bucket_id: int = Field(default=127, ge=1, le=255)
    noise_aug_strength: float = Field(default=0.4, ge=0.0, le=1.0)
    seed: int = Field(default=-1)

# 模型缓存
model_cache = {}
image_model_cache = {}

def get_image_pipe(model_id: str):
    if model_id not in image_model_cache:
        try:
            logger.info(f"开始加载图像模型: {model_id}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_safetensors=True,
                variant="fp16"
            )
            pipe = pipe.to(DEVICE)
            pipe.enable_attention_slicing()
            image_model_cache[model_id] = pipe
            logger.info(f"图像模型加载完成")
        except Exception as e:
            error_msg = f"加载图像模型失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
    return image_model_cache[model_id]

def get_video_pipe(model_name: str):
    if model_name not in model_cache:
        model_info = AVAILABLE_MODELS[model_name]
        config = MODEL_CONFIGS[model_name]
        try:
            logger.info(f"开始加载视频模型: {model_name} ({model_info['model_id']})")
            
            pipe = model_info["pipeline_class"].from_pretrained(
                model_info["model_id"],
                torch_dtype=config["torch_dtype"],
                variant=config["variant"]
            )
            pipe = pipe.to(DEVICE)
            
            # 内存优化
            pipe.enable_attention_slicing(1)
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
            
            model_cache[model_name] = pipe
            logger.info(f"视频模型加载完成")
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            logger.error(error_msg)
            if model_name in model_cache:
                del model_cache[model_name]
            raise HTTPException(status_code=500, detail=error_msg)
    
    return model_cache[model_name]

@app.get("/video_models")
async def get_video_models():
    return {"models": list(AVAILABLE_MODELS.keys())}

@app.post("/generate_video")
async def generate_video(request: VideoRequest):
    try:
        logger.info(f"收到视频生成请求，模型: {request.model_name}")
        logger.info(f"参数: frames={request.num_frames}, fps={request.fps}, "
                   f"size={request.width}x{request.height}")
        
        # 准备初始图像
        if request.image_base64:
            # 如果提供了图片，直接使用
            logger.info("使用提供的图片生成视频...")
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            # 调整图片尺寸
            if image.size != (request.width, request.height):
                image = image.resize((request.width, request.height))
        elif request.prompt:
            # 如果提供了提示词，先生成图片
            logger.info(f"使用提示词生成初始图像: {request.prompt}")
            model_info = AVAILABLE_MODELS[request.model_name]
            image_pipe = get_image_pipe(model_info["image_model_id"])
            
            # 设置随机种子
            generator = None
            if request.seed != -1:
                generator = torch.Generator(DEVICE).manual_seed(request.seed)
            
            # 生成初始图像
            logger.info("生成初始图像...")
            image = image_pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=30,
                generator=generator
            ).images[0]
        else:
            raise HTTPException(status_code=400, detail="必须提供图片或提示词其中之一")
        
        # 获取视频模型
        video_pipe = get_video_pipe(request.model_name)
        
        # 生成视频
        logger.info("开始生成视频...")
        try:
            with torch.inference_mode():
                # 分批处理帧
                batch_size = 8  # 每批处理的帧数
                all_frames = []
                
                for i in range(0, request.num_frames, batch_size):
                    current_batch_size = min(batch_size, request.num_frames - i)
                    logger.info(f"处理帧 {i+1} 到 {i+current_batch_size}")
                    
                    frames = video_pipe(
                        image=image,
                        num_frames=current_batch_size,
                        fps=request.fps,
                        motion_bucket_id=request.motion_bucket_id,
                        noise_aug_strength=request.noise_aug_strength,
                        num_inference_steps=25  # 减少推理步数以节省内存
                    ).frames[0]
                    
                    all_frames.extend(frames)
                    
                    # 清理内存
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    import gc
                    gc.collect()
            
            # 保存视频到临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                # SVD 输出的是 PIL 图像序列，需要转换为视频
                frame_array = []
                for frame in all_frames:
                    frame_array.append(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                
                out = cv2.VideoWriter(
                    temp_file.name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    request.fps,
                    (request.width, request.height)
                )
                
                for frame in frame_array:
                    out.write(frame)
                out.release()
                
                # 读取视频文件并转换为 base64
                with open(temp_file.name, 'rb') as f:
                    video_bytes = f.read()
                
                video_base64 = base64.b64encode(video_bytes).decode()
                
                # 删除临时文件
                os.unlink(temp_file.name)
            
            logger.info("视频生成完成")
            return {"video": video_base64}
            
        except Exception as e:
            logger.error(f"视频生成失败: {str(e)}")
            if request.model_name in model_cache:
                del model_cache[request.model_name]
            raise HTTPException(status_code=500, detail=f"视频生成失败: {str(e)}")
        
    except Exception as e:
        error_msg = f"处理过程出错: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 