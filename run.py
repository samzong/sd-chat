import subprocess
import time
import signal
import sys
import os
import psutil
import logging
import traceback
from typing import List, Dict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('service.log')
    ]
)
logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.service_ports = {
            "api": 7858,
            "video_api": 7859,
            "web_ui": 7860
        }
        self.is_shutting_down = False
        
        # 确保所有脚本文件存在
        self.check_required_files()
        # 检查端口可用性
        self.check_ports_availability()
    
    def check_required_files(self):
        """检查必需的脚本文件是否存在"""
        required_files = ["api.py", "video_api.py", "web_ui.py"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"缺少必需的文件: {', '.join(missing_files)}")

    def check_ports_availability(self):
        """检查所有端口的可用性"""
        for service, port in self.service_ports.items():
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # 尝试绑定端口
                    s.bind(('127.0.0.1', port))
                    logger.info(f"端口 {port} 可用于 {service}")
            except Exception as e:
                raise RuntimeError(f"端口 {port} 不可用，请确保该端口未被占用: {str(e)}")

    def check_port(self, port: int) -> bool:
        """检查端口是否被占用"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # 设置超时时间
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                return result == 0
        except Exception as e:
            logger.error(f"检查端口 {port} 时发生错误: {str(e)}")
            return True  # 如果检查出错，保守起见认为端口被占用

    def kill_process_on_port(self, port: int):
        """结束占用指定端口的进程"""
        try:
            # 使用 lsof 命令查找占用端口的进程
            import subprocess
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            if result.stdout:
                # 解析输出找到 PID
                for line in result.stdout.split('\n')[1:]:  # 跳过标题行
                    if line:
                        pid = int(line.split()[1])
                        try:
                            logger.warning(f"正在终止端口 {port} 的进程 {pid}")
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(1)  # 等待进程终止
                            try:
                                os.kill(pid, 0)  # 检查进程是否还存在
                                logger.warning(f"进程 {pid} 未响应 SIGTERM，使用 SIGKILL")
                                os.kill(pid, signal.SIGKILL)
                            except OSError:
                                pass  # 进程已终止
                        except ProcessLookupError:
                            pass  # 进程已不存在
            
            # 等待端口释放
            max_wait = 10
            start_time = time.time()
            while time.time() - start_time < max_wait:
                if not self.check_port(port):
                    logger.info(f"端口 {port} 已释放")
                    return True
                time.sleep(0.5)
            
            logger.error(f"端口 {port} 在 {max_wait} 秒内未能释放")
            return False
            
        except Exception as e:
            logger.error(f"终止端口 {port} 的进程时发生错误: {str(e)}")
            return False

    def start_service(self, name: str, script: str):
        """启动单个服务"""
        port = self.service_ports[name]
        
        # 检查端口是否被占用
        if self.check_port(port):
            logger.warning(f"端口 {port} 已被占用，尝试释放...")
            self.kill_process_on_port(port)
            time.sleep(2)  # 等待端口释放
        
        try:
            logger.info(f"正在启动 {name}...")
            
            # 检查 Python 环境
            python_path = sys.executable
            if not os.path.exists(python_path):
                raise RuntimeError(f"Python 解释器不存在: {python_path}")
            
            # 检查脚本文件
            script_path = os.path.abspath(script)
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"脚本文件不存在: {script_path}")
            
            # 启动进程
            process = subprocess.Popen(
                [python_path, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()  # 继承当前环境变量
            )
            
            self.processes[name] = process
            
            # 等待服务启动
            time.sleep(5)
            
            # 检查进程是否还在运行
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"{name} 启动失败")
                logger.error(f"标准输出:\n{stdout}")
                logger.error(f"错误输出:\n{stderr}")
                return False
            
            # 检查端口是否正在监听
            if not self.check_port(port):
                logger.error(f"{name} 端口 {port} 未能成功启动")
                self.stop_service(name)
                return False
            
            logger.info(f"{name} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动 {name} 时发生错误: {str(e)}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            return False

    def stop_service(self, name: str):
        """停止单个服务"""
        if name in self.processes:
            try:
                process = self.processes[name]
                port = self.service_ports[name]
                
                if process.poll() is None:  # 进程还在运行
                    logger.info(f"正在停止 {name}...")
                    
                    # 首先尝试优雅终止
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"{name} 未响应，强制终止")
                        process.kill()
                        process.wait(timeout=5)
                
                # 确保进程相关资源被清理
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
                
                # 确保端口被释放
                if self.check_port(port):
                    logger.warning(f"端口 {port} 仍被占用，尝试强制释放")
                    self.kill_process_on_port(port)
                
                del self.processes[name]
                logger.info(f"{name} 已停止")
                
            except Exception as e:
                logger.error(f"停止 {name} 时发生错误: {str(e)}")
                logger.error(f"错误堆栈:\n{traceback.format_exc()}")

    def start_all(self):
        """按顺序启动所有服务"""
        services = [
            ("api", "api.py"),
            ("video_api", "video_api.py"),
            ("web_ui", "web_ui.py")
        ]
        
        for name, script in services:
            if not self.start_service(name, script):
                logger.error(f"{name} 启动失败，开始清理...")
                self.stop_all()
                return False
            time.sleep(2)  # 服务之间的启动间隔
        
        logger.info("所有服务启动成功")
        return True

    def stop_all(self):
        """停止所有服务"""
        if not self.is_shutting_down:
            self.is_shutting_down = True
            logger.info("正在停止所有服务...")
            
            # 按照相反的顺序停止服务
            for name in reversed(list(self.processes.keys())):
                self.stop_service(name)
            
            # 最后检查所有端口
            for service, port in self.service_ports.items():
                if self.check_port(port):
                    logger.warning(f"端口 {port} 仍被占用，尝试强制释放")
                    self.kill_process_on_port(port)
            
            self.is_shutting_down = False
            logger.info("所有服务已停止")

    def monitor_processes(self):
        """监控所有进程的状态"""
        while not self.is_shutting_down:
            try:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        logger.error(f"{name} 意外停止，退出码: {process.poll()}")
                        logger.error(f"标准输出:\n{stdout}")
                        logger.error(f"错误输出:\n{stderr}")
                        self.stop_all()
                        return False
                time.sleep(1)
            except Exception as e:
                logger.error(f"监控进程时发生错误: {str(e)}")
                logger.error(f"错误堆栈:\n{traceback.format_exc()}")
                self.stop_all()
                return False
        return True

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，开始清理...")
    manager.stop_all()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 检查 Python 环境
        logger.info(f"Python 版本: {sys.version}")
        logger.info(f"Python 路径: {sys.executable}")
        logger.info(f"工作目录: {os.getcwd()}")
        
        manager = ServiceManager()
        
        # 启动所有服务
        if manager.start_all():
            logger.info("服务管理器启动成功，按 Ctrl+C 停止...")
            manager.monitor_processes()
        else:
            logger.error("服务启动失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        if 'manager' in locals():
            manager.stop_all()
        sys.exit(1) 