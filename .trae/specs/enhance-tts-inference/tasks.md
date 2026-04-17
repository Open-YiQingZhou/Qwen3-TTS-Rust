# Tasks

- [x] Task 1: 修改 `build_clone_prompt` 实现 ICL 按位融合逻辑
  - [x] SubTask 1.1: 分析 Python 版本的 ICL 融合算法
  - [x] SubTask 1.2: 修改 `build_clone_prompt` 方法，实现文本池和音频池的按位融合
  - [x] SubTask 1.3: 更新 `PromptData` 结构，添加 `trailing_text_embd` 字段
  - [x] SubTask 1.4: 添加 `streaming` 参数区分流式和非流式模式

- [x] Task 2: 实现 `trailing_text_pool` 文本池动态注入
  - [x] SubTask 2.1: 在 `PromptData` 中添加 `trailing_text_embd` 字段
  - [x] SubTask 2.2: 修改 `run_inference_stream` 方法，在每步推理时动态融合文本嵌入
  - [x] SubTask 2.3: 实现文本池耗尽时使用 tts_pad 填充的逻辑

- [ ] Task 3: 增强采样器配置（需要修改 FFI 绑定，留待后续）
  - [ ] SubTask 3.1: 在 `SamplerConfig` 中添加 `min_p`, `repeat_penalty`, `frequency_penalty`, `presence_penalty` 字段
  - [ ] SubTask 3.2: 修改 `LlamaSampler` 实现，支持新的采样参数
  - [ ] SubTask 3.3: 更新 `llama.cpp` 绑定，支持新的采样器功能

- [ ] Task 4: 添加惩罚项豁免机制（需要修改 FFI 绑定，留待后续）
  - [ ] SubTask 4.1: 在 `LlamaSampler` 中添加 `allow_tokens` 支持
  - [ ] SubTask 4.2: 定义需要豁免的特殊 token 列表

- [x] Task 5: 测试和验证
  - [x] SubTask 5.1: 运行 `cargo build --release` 确保编译通过
  - [x] SubTask 5.2: 运行 `cargo fmt` 和 `cargo clippy` 检查代码质量
  - [ ] SubTask 5.3: 测试语音克隆功能

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 5] depends on [Task 1, Task 2]
- [Task 3, Task 4] 可独立进行，需要修改 FFI 绑定
