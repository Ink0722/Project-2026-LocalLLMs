# Hugging Face 多模型本地推理项目

本项目已从“单模型脚本”重构为“配置驱动的多模型本地推理骨架”，目标是让你通过统一入口运行多个 Hugging Face 主流文本大模型，并在后续通过新增配置文件继续扩展。

## 当前能力
- 统一 CLI 入口：`chat` / `infer` / `download`
- 配置驱动选模型：每个模型一份 `configs/models/*.yaml`
- 统一加载链路：量化、显存限制、生成参数都从配置读取
- 当前支持两类后端：
  `transformers` 文本模型
  `llama-cpp-python` GGUF 文本模型

## 目录说明
```text
configs/
  models/      # 每个模型一份配置
  runtime/     # 运行时默认配置
src/
  cli/         # 命令行入口
  core/        # 配置、注册、加载、生成等核心逻辑
  adapters/    # 按模型族拆分的适配器
  services/    # 推理和下载编排
scripts/       # 薄封装脚本入口
```

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 下载指定模型
```bash
python -m src.cli.main download --model qwen2_5_7b_instruct
```

3. 交互式推理
```bash
python -m src.cli.main chat --model deepseek_chat
```

4. 单条 prompt 推理
```bash
python -m src.cli.main infer --model llama3_1_8b_instruct --prompt "解释一下 Transformer"
```

## 可用模型配置
- `deepseek_chat`
- `qwen2_5_7b_instruct`
- `llama3_1_8b_instruct`
- `mistral_7b_instruct`
- `qwen35_9b_uncensored_hauhaucs_aggressive`

## GGUF 模型说明
像 `qwen35_9b_uncensored_hauhaucs_aggressive` 这类 GGUF 模型会走 `llama-cpp-python` 后端，而不是 `transformers`。

下载：
```bash
python3 -m src.cli.main download --model qwen35_9b_uncensored_hauhaucs_aggressive
```

推理：
```bash
python3 -m src.cli.main chat --model qwen35_9b_uncensored_hauhaucs_aggressive
python3 -m src.cli.main infer --model qwen35_9b_uncensored_hauhaucs_aggressive --prompt "你好"
```

首次安装 `llama-cpp-python` 时如果本机没有现成 wheel，可能会触发本地编译。

## 扩展方式
如果你要新增一个模型，优先这样做：

1. 在 `configs/models/` 下新增一个 yaml
2. 填写 `model_id`、`task_type`、`runtime`、`quantization`、`generation`
3. 如果现有 `chat_causal_lm` 适配器能处理，就无需再写新脚本

## 兼容入口
为了兼容旧用法，以下脚本仍可运行，但它们内部已经改成转调统一入口：

- `python src/inference.py`
- `python src/qwen_inference.py`
- `python src/download.py`
