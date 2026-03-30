import pytest
import torch
from unittest.mock import patch, MagicMock, call
from io import StringIO

from src.inference import run_inference


class TestRunInference:
    """run_inference 方法的单元测试类"""

    @pytest.fixture
    def mock_tokenizer(self):
        """创建 mock tokenizer"""
        mock = MagicMock()
        mock.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock.decode.return_value = "这是一个测试响应"
        return mock

    @pytest.fixture
    def mock_model(self):
        """创建 mock model"""
        mock = MagicMock()
        mock.device = "cuda:0"
        mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        return mock

    @pytest.fixture
    def mock_quantization_config(self):
        """创建 mock BitsAndBytesConfig"""
        with patch("src.inference.BitsAndBytesConfig") as mock_config:
            mock_config.return_value = MagicMock()
            yield mock_config

    def test_normal_user_interaction_single_turn(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试正常用户交互：单轮对话后退出"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["你好", "exit"]) as mock_input, \
             patch("builtins.print") as mock_print:

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证 tokenizer 和 model 加载
            mock_tokenizer_cls.assert_called_once_with(
                "deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True
            )
            mock_model_cls.assert_called_once()

            # 验证 input 被调用两次（一次问题，一次退出）
            assert mock_input.call_count == 2

            # 验证模型生成被调用
            mock_model.generate.assert_called_once()

            # 验证 tokenizer.decode 被调用
            mock_tokenizer.decode.assert_called_once()

    def test_exit_immediately(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试用户立即退出（边界值：第一次输入就是 exit）"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", return_value="exit") as mock_input, \
             patch("builtins.print") as mock_print:

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证模型生成未被调用（因为直接退出）
            mock_model.generate.assert_not_called()

            # 验证 input 只被调用一次
            assert mock_input.call_count == 1

    def test_exit_case_insensitive(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试退出命令不区分大小写（EXIT, Exit, exit）"""
        test_cases = ["EXIT", "Exit", "EXIT", "eXiT"]

        for exit_input in test_cases:
            with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
                 patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
                 patch("builtins.input", return_value=exit_input), \
                 patch("builtins.print"):

                mock_tokenizer_cls.return_value = mock_tokenizer
                mock_model_cls.return_value = mock_model

                run_inference()

                # 验证模型生成未被调用
                mock_model.generate.assert_not_called()

    def test_multiple_turns_conversation(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试多轮对话场景"""
        user_inputs = ["问题1", "问题2", "问题3", "exit"]

        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=user_inputs), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证模型生成被调用 3 次（3 个问题）
            assert mock_model.generate.call_count == 3

    def test_empty_user_input(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试空输入边界值"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["", "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 空输入仍会触发模型生成
            mock_model.generate.assert_called_once()

    def test_quantization_config_parameters(
        self, mock_tokenizer, mock_model
    ):
        """测试 BitsAndBytesConfig 量化配置参数正确性"""
        with patch("src.inference.BitsAndBytesConfig") as mock_bnb_config, \
             patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", return_value="exit"), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证量化配置参数
            mock_bnb_config.assert_called_once_with(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    def test_model_loading_parameters(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试模型加载参数正确性"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", return_value="exit"), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证模型加载参数
            call_kwargs = mock_model_cls.call_args[1]
            assert call_kwargs["quantization_config"] == mock_quantization_config.return_value
            assert call_kwargs["device_map"] == "auto"
            assert call_kwargs["max_memory"] == {0: "4.8GiB", "cpu": "16GiB"}
            assert call_kwargs["trust_remote_code"] is True
            assert call_kwargs["low_cpu_mem_usage"] is True

    def test_generate_parameters(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试模型 generate 方法参数"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["测试问题", "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证 generate 参数
            call_kwargs = mock_model.generate.call_args[1]
            assert call_kwargs["max_new_tokens"] == 512
            assert call_kwargs["do_sample"] is True
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["top_k"] == 50
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["repetition_penalty"] == 1.1

    def test_chat_template_construction(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试对话模板构建"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["测试问题", "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证对话模板构建
            mock_tokenizer.apply_chat_template.assert_called_once()
            call_args = mock_tokenizer.apply_chat_template.call_args

            # 验证 messages 格式
            messages = call_args[0][0]
            assert messages == [{"role": "user", "content": "测试问题"}]
            assert call_args[1]["add_generation_prompt"] is True
            assert call_args[1]["return_tensors"] == "pt"

    def test_tokenizer_decode_parameters(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试 tokenizer.decode 参数"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["测试问题", "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证 decode 参数
            mock_tokenizer.decode.assert_called_once()
            call_kwargs = mock_tokenizer.decode.call_args[1]
            assert call_kwargs["skip_special_tokens"] is True

    def test_print_messages(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试 print 输出消息"""
        print_calls = []

        def capture_print(*args, **kwargs):
            print_calls.append((args, kwargs))

        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["测试", "exit"]), \
             patch("builtins.print", side_effect=capture_print):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证关键打印消息
            printed_messages = [str(call[0][0]) if call[0] else "" for call in print_calls]
            
            # 验证加载消息
            assert any("加载" in msg for msg in printed_messages)
            
            # 验证成功消息
            assert any("模型加载成功" in msg for msg in printed_messages)
            
            # 验证再见消息
            assert any("再见" in msg for msg in printed_messages)

    def test_model_id_constant(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试模型 ID 常量"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", return_value="exit"), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证模型 ID
            assert mock_tokenizer_cls.call_args[0][0] == "deepseek-ai/deepseek-llm-7b-chat"

    def test_special_characters_in_input(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试特殊字符输入"""
        special_inputs = ["你好！@#$%", "测试\n换行", "emoji 🎉", "exit"]

        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=special_inputs), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证特殊字符输入都被处理
            assert mock_model.generate.call_count == 3

    def test_long_user_input(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试长文本输入（边界值）"""
        long_input = "这是一个很长的输入。" * 1000  # 约 10,000 字符

        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=[long_input, "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 验证长文本被正确传递到对话模板
            messages = mock_tokenizer.apply_chat_template.call_args[0][0]
            assert messages[0]["content"] == long_input

    def test_whitespace_only_input(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试纯空白字符输入（边界值）"""
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["   ", "\t", "\n", "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model

            run_inference()

            # 纯空白输入仍会触发模型生成（共 3 次）
            assert mock_model.generate.call_count == 3

    def test_input_tensor_device_placement(
        self, mock_tokenizer, mock_model, mock_quantization_config
    ):
        """测试输入张量设备放置"""
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        
        with patch("src.inference.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("src.inference.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("builtins.input", side_effect=["测试", "exit"]), \
             patch("builtins.print"):

            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model
            mock_tokenizer.apply_chat_template.return_value = mock_tensor

            run_inference()

            # 验证张量被移动到模型设备
            mock_tensor.to.assert_called_once_with(mock_model.device)
