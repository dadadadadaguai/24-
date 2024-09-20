# 配置日志
import logging
from  datetime import  datetime

current_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
log_file_name = f'./log/app_{current_time}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_name,
    filemode='a'
)

logger = logging.getLogger(__name__)

# 添加控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)