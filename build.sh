# 首先将bge-large-zh-v1.5模型文件存到项目根目录下
pip install -U huggingface_hub && export HF_ENDPOINT=https://hf-mirror.com 
huggingface-cli download --resume-download --local-dir-use-symlinks False --local-dir bge-large-zh-v1.5 BAAI/bge-large-zh-v1.5
# pip install -U --pre "weaviate-client==v4.4b2" 安装python客户端
pip install --pre -U "weaviate-client==4.*"
# 构建推理text2vec-transformers推理服务镜像
docker build -f bge-large-v1.5-zh.Dockerfile -t bge-large-v1.5-zh .
# docker compose 启动容器
docker compose up -d 
