# Checklist

- [x] `build_clone_prompt` 实现了 ICL 按位融合逻辑，文本和音频嵌入按位置相加
- [x] `PromptData` 结构包含 `trailing_text_embd` 字段
- [x] `build_clone_prompt` 添加了 `streaming` 参数区分流式和非流式模式
- [x] `run_inference_stream` 在每步推理时动态融合文本嵌入
- [x] 文本池耗尽时使用 tts_pad 填充
- [ ] `SamplerConfig` 包含 min_p, repeat_penalty, frequency_penalty, presence_penalty 字段（需要修改 FFI 绑定，留待后续）
- [ ] `LlamaSampler` 支持新的采样参数（需要修改 FFI 绑定，留待后续）
- [ ] `LlamaSampler` 支持 allow_tokens 豁免机制（需要修改 FFI 绑定，留待后续）
- [x] `cargo build --release` 编译通过
- [x] `cargo fmt` 和 `cargo clippy` 检查通过
- [ ] 语音克隆功能测试通过
