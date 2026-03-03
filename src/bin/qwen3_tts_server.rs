use axum::{
    extract::{DefaultBodyLimit, State, ws::{WebSocket, WebSocketUpgrade, Message}},
    response::{Html, IntoResponse},
    Json, Router,
    routing::{get, post},
    http::{header, StatusCode},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use clap::Parser;
use qwen3_tts::{TtsEngine, VoiceFile};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use futures_util::StreamExt;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "models")]
    model_dir: PathBuf,

    #[arg(long, default_value = "q5_k_m")]
    quant: String,

    #[arg(long, default_value = "speakers")]
    speakers_dir: PathBuf,

    #[arg(long, default_value = "3000")]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

struct AppState {
    engine: Arc<Mutex<TtsEngine>>,
    speakers: HashMap<String, VoiceFile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TTSRequest {
    text: String,
    speaker: Option<String>,
    streaming: Option<bool>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    instruction: Option<String>,
    max_steps: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TTSResponse {
    success: bool,
    message: Option<String>,
    audio_base64: Option<String>,
    sample_rate: Option<u32>,
    duration_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpeakersResponse {
    speakers: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Qwen3-TTS Server ===");
    println!("Model Dir: {:?}", args.model_dir);
    println!("Quant:     {}", args.quant);
    println!("Port:      {}", args.port);

    println!("Checking models...");
    TtsEngine::download_models(&args.model_dir, &args.quant)
        .await
        .map_err(|e| format!("Model download failed: {}", e))?;

    println!("Loading engine...");
    let mut engine = TtsEngine::new(&args.model_dir, &args.quant)
        .await
        .map_err(|e| format!("Engine load failed: {}", e))?;

    if args.speakers_dir.exists() {
        engine.load_speakers(&args.speakers_dir)?;
    }

    let speakers = engine.get_speakers_map().clone();
    let engine = Arc::new(Mutex::new(engine));

    let state = AppState {
        engine,
        speakers,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/audio-processor.js", get(audio_processor))
        .route("/api/tts", post(tts_handler))
        .route("/api/tts/stream", get(tts_stream_handler))
        .route("/api/speakers", get(speakers_handler))
        .route("/health", get(health))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(Arc::new(state));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    println!("Server running at http://{}", addr);
    println!("API endpoints:");
    println!("  POST /api/tts        - Generate TTS audio (non-streaming)");
    println!("  GET  /api/tts/stream - WebSocket streaming TTS");
    println!("  GET  /api/speakers   - List available speakers");
    println!("  GET  /health         - Health check");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../../webui/index.html"))
}

async fn audio_processor() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../../webui/audio-processor.js"),
    )
}

async fn health() -> &'static str {
    "OK"
}

async fn speakers_handler(
    State(state): State<Arc<AppState>>,
) -> Json<SpeakersResponse> {
    let names: Vec<String> = state.speakers.keys().cloned().collect();
    Json(SpeakersResponse { speakers: names })
}

async fn tts_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TTSRequest>,
) -> Json<TTSResponse> {
    let speaker_name = req.speaker.unwrap_or_else(|| "vivian".to_string());
    let streaming = req.streaming.unwrap_or(false);

    let voice = state.speakers.get(&speaker_name).cloned();

    let voice = match voice {
        Some(v) => v,
        None => {
            return Json(TTSResponse {
                success: false,
                message: Some(format!("Speaker '{}' not found", speaker_name)),
                audio_base64: None,
                sample_rate: None,
                duration_ms: None,
            });
        }
    };

    let mut engine = state.engine.lock().await;

    // 一次性修改采样器配置
    {
        let config = engine.get_sampler_config().clone();
        let mut new_config = config;
        if let Some(temp) = req.temperature {
            new_config.temperature = temp;
        }
        if let Some(top_k) = req.top_k {
            new_config.top_k = top_k;
        }
        if let Some(top_p) = req.top_p {
            new_config.top_p = top_p;
        }
        if let Some(seed) = req.seed {
            new_config.seed = Some(seed);
        }
        engine.set_sampler_config(new_config);
    }

    // 设置 max_steps
    if let Some(max_steps) = req.max_steps {
        engine.set_max_steps(max_steps);
    }

    let result = engine.generate_with_voice_streaming(
        &req.text,
        &voice,
        req.instruction.as_deref(),
        streaming,
        None,
    );

    match result {
        Ok(audio) => {
            let duration_ms = (audio.samples.len() as f64 / audio.sample_rate as f64) * 1000.0;
            let wav_data = audio.to_wav_bytes();

            Json(TTSResponse {
                success: true,
                message: None,
                audio_base64: Some(BASE64.encode(&wav_data)),
                sample_rate: Some(audio.sample_rate),
                duration_ms: Some(duration_ms),
            })
        }
        Err(e) => Json(TTSResponse {
            success: false,
            message: Some(e),
            audio_base64: None,
            sample_rate: None,
            duration_ms: None,
        }),
    }
}

// WebSocket 流式 TTS 处理器
async fn tts_stream_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_tts_stream(socket, state))
}

async fn handle_tts_stream(mut socket: WebSocket, state: Arc<AppState>) {
    // 文本段通道：前端 -> 生成线程
    let (text_tx, text_rx) = std::sync::mpsc::channel::<(String, Option<String>)>();
    // 音频通道：生成线程 -> 前端
    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    
    // 克隆需要的变量
    let engine = state.engine.clone();
    let speakers = state.speakers.clone();
    
    // 启动生成线程
    let generate_thread = std::thread::spawn(move || {
        let mut engine = engine.blocking_lock();
        let mut voice: Option<VoiceFile> = None;
        
        while let Ok((text, speaker_name)) = text_rx.recv() {
            // 检查是否结束
            if text.is_empty() {
                break;
            }
            
            // 第一次时初始化 speaker
            if voice.is_none() {
                let spk_name = speaker_name.unwrap_or_else(|| "vivian".to_string());
                voice = speakers.get(&spk_name).cloned();
                if voice.is_none() {
                    let _ = audio_tx.send(vec![]); // 发送空表示错误
                    break;
                }
            }
            
            let voice_ref = voice.as_ref().unwrap();
            
            // 生成音频
            let result = engine.generate_with_voice_streaming(
                &text, 
                voice_ref, 
                None, 
                true, 
                Some(audio_tx.clone())
            );
            
            if result.is_err() {
                break;
            }
            
            // 发送段完成信号（通过空 vec）
            let _ = audio_tx.send(vec![]);
        }
        // 线程结束，关闭 audio_tx
        drop(audio_tx);
    });
    
    let mut current_speaker: Option<String> = None;
    let mut end_requested = false;
    
    loop {
        // 先发送所有待处理的音频数据
        loop {
            match audio_rx.try_recv() {
                Ok(samples) => {
                    if samples.is_empty() {
                        let _ = socket.send(Message::Text("segment_done".to_string())).await;
                    } else {
                        let bytes: Vec<u8> = samples
                            .iter()
                            .flat_map(|s| s.to_le_bytes())
                            .collect();
                        if socket.send(Message::Binary(bytes)).await.is_err() {
                            break;
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // 生成线程结束
                    end_requested = true;
                    break;
                }
            }
        }
        
        // 如果生成线程结束，退出
        if end_requested {
            break;
        }
        
        // 等待 WebSocket 消息（带超时，以便定期检查音频数据）
        match tokio::time::timeout(
            std::time::Duration::from_millis(50),
            socket.recv()
        ).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                // 检查是否是结束信号
                if text == "end" {
                    let _ = text_tx.send((String::new(), None));
                    // 标记结束，但继续处理剩余音频
                    end_requested = true;
                    continue;
                }

                // 解析请求
                let req: TTSRequest = match serde_json::from_str(&text) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = socket.send(Message::Text(format!("{{\"error\": \"Invalid request: {}\"}}", e))).await;
                        continue;
                    }
                };

                // 记录 speaker（第一次有效）
                if current_speaker.is_none() {
                    current_speaker = req.speaker.clone();
                }

                let text_to_generate = req.text.trim();
                if text_to_generate.is_empty() {
                    continue;
                }

                // 发送文本到生成线程
                let _ = text_tx.send((text_to_generate.to_string(), current_speaker.clone()));
            }
            Ok(Some(Ok(Message::Close(_)))) => break,
            Ok(Some(Ok(Message::Binary(_)))) => continue,
            Ok(Some(Ok(Message::Ping(_)))) => continue,
            Ok(Some(Ok(Message::Pong(_)))) => continue,
            Ok(Some(Err(_))) => break,
            Ok(None) => break,
            Err(_) => {
                // 超时，继续循环检查音频数据
                continue;
            }
        }
    }

    // 清理
    let _ = text_tx.send((String::new(), None));
    let _ = generate_thread.join();
    
    // 发送结束标记
    let _ = socket.send(Message::Text("done".to_string())).await;
}
