# TTS 推理增强 Spec

## Why
当前 Rust 实现与 Python 参考实现 (Qwen3-TTS-GGUF) 存在差异，导致语音克隆效果和流式处理质量不佳。需要借鉴 Python 版本的实现来增强 Rust 项目。

## What Changes
- 修改 `build_clone_prompt` 实现 ICL 按位融合逻辑
- 实现 `trailing_text_pool` 文本池动态注入机制
- 增强采样器配置（min_p, repeat_penalty 等）
- 添加惩罚项豁免机制 (allow_tokens)

## Impact
- Affected code: `src/tts/prompt.rs`, `src/tts/engine.rs`, `src/models/llama/mod.rs`

## ADDED Requirements

### Requirement: ICL 按位融合
系统应支持语音克隆的 ICL (In-Context Learning) 按位融合模式。

#### Scenario: 语音克隆提示词构建
- **WHEN** 用户提供参考音频和参考文本进行语音克隆
- **THEN** 系统应将文本嵌入和音频嵌入按位置相加融合，而非直接拼接

### Requirement: 文本池动态注入
系统应在流式推理过程中动态注入文本嵌入。

#### Scenario: 流式文本注入
- **WHEN** 执行流式推理时
- **THEN** 系统应在每步推理时从文本池中取出对应位置的文本嵌入，与音频嵌入融合
- **AND** 当文本池耗尽时，使用 tts_pad 填充

### Requirement: 增强采样器配置
系统应支持更丰富的采样器配置选项。

#### Scenario: 采样器配置
- **WHEN** 用户配置采样参数
- **THEN** 系统应支持 min_p, repeat_penalty, frequency_penalty, presence_penalty 参数

### Requirement: 惩罚项豁免
系统应支持惩罚项豁免机制，防止特殊 token 被惩罚。

#### Scenario: 特殊 token 豁免
- **WHEN** 应用重复惩罚时
- **THEN** 系统应豁免指定的特殊 token（如 EOS, TTS_EOS 等）

## MODIFIED Requirements

### Requirement: PromptData 结构
`PromptData` 结构应新增 `trailing_text_embd` 字段，用于存储待注入的文本池。

### Requirement: TtsEngine 结构
`TtsEngine` 应支持文本池动态注入机制。

## REMOVED Requirements
无
