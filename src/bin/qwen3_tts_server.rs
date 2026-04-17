use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        DefaultBodyLimit, State,
    },
    http::header,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use clap::Parser;
use qwen3_tts::{TtsEngine, VoiceFile};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

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

    #[arg(long, default_value_t = 4)]
    threads: i32,
}

struct AppState {
    engine: Arc<Mutex<TtsEngine>>,
    speakers: HashMap<String, VoiceFile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TTSRequest {
    text: String,
    speaker: Option<String>,
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
    println!("Threads:   {}", args.threads);

    println!("Checking models...");
    TtsEngine::download_models(&args.model_dir, &args.quant)
        .await
        .map_err(|e| format!("Model download failed: {}", e))?;

    println!("Loading engine...");
    let mut engine = TtsEngine::new(&args.model_dir, &args.quant, args.threads)
        .await
        .map_err(|e| format!("Engine load failed: {}", e))?;

    if args.speakers_dir.exists() {
        engine.load_speakers(&args.speakers_dir)?;
    }

    let speakers = engine.get_speakers_map().clone();
    let engine = Arc::new(Mutex::new(engine));

    let state = AppState { engine, speakers };

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

async fn speakers_handler(State(state): State<Arc<AppState>>) -> Json<SpeakersResponse> {
    let names: Vec<String> = state.speakers.keys().cloned().collect();
    Json(SpeakersResponse { speakers: names })
}

async fn tts_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TTSRequest>,
) -> Json<TTSResponse> {
    let start = Instant::now();
    let speaker_name = req.speaker.unwrap_or_else(|| "vivian".to_string());

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

    let result =
        engine.generate_with_voice_streaming(&req.text, &voice, req.instruction.as_deref(), None);

    match result {
        Ok(audio) => {
            let duration_ms = (audio.samples.len() as f64 / audio.sample_rate as f64) * 1000.0;
            let audio_duration_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
            let rtf = start.elapsed().as_secs_f64() / audio_duration_secs;
            println!(
                "RTF: {:.3} (gen: {:.2}s, audio: {:.2}s)",
                rtf,
                start.elapsed().as_secs_f64(),
                audio_duration_secs
            );

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
    let (text_tx, text_rx) = std::sync::mpsc::channel::<(String, Option<String>, Option<String>)>();
    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();

    let engine = state.engine.clone();
    let speakers = state.speakers.clone();

    let mut total_samples: usize = 0;
    let stream_start = std::time::Instant::now();

    std::thread::spawn(move || {
        let mut engine = engine.blocking_lock();
        let mut voice: Option<VoiceFile> = None;

        while let Ok((text, speaker_name, instruction)) = text_rx.recv() {
            if text.is_empty() {
                break;
            }

            if voice.is_none() {
                let spk_name = speaker_name.unwrap_or_else(|| "vivian".to_string());
                voice = speakers.get(&spk_name).cloned();
                if voice.is_none() {
                    let _ = audio_tx.send(vec![]);
                    break;
                }
            }

            let voice_ref = voice.as_ref().unwrap();

            let result = engine.generate_with_voice_streaming(
                &text,
                voice_ref,
                instruction.as_deref(),
                Some(audio_tx.clone()),
            );

            if result.is_err() {
                break;
            }

            let _ = audio_tx.send(vec![]);
        }
    });

    let mut current_speaker: Option<String> = None;
    let mut end_requested = false;

    loop {
        if end_requested {
            match audio_rx.recv().await {
                Some(samples) => {
                    if samples.is_empty() {
                        let _ = socket.send(Message::Text("segment_done".to_string())).await;
                    } else {
                        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                        if socket.send(Message::Binary(bytes)).await.is_err() {
                            break;
                        }
                    }
                }
                None => break,
            }
        } else {
            tokio::select! {
                Some(samples) = audio_rx.recv() => {
                    if samples.is_empty() {
                        let gen_time = stream_start.elapsed().as_secs_f64();
                        let audio_duration = total_samples as f64 / 24000.0;
                        let rtf = if audio_duration > 0.0 { gen_time / audio_duration } else { 0.0 };
                        eprintln!("[Stream] RTF: {:.3} (gen: {:.2}s, audio: {:.2}s)", rtf, gen_time, audio_duration);

                        let _ = socket.send(Message::Text("segment_done".to_string())).await;
                    } else {
                        total_samples += samples.len();
                        eprintln!("[Stream] received {} samples, total: {}", samples.len(), total_samples);
                        let bytes: Vec<u8> = samples
                            .iter()
                            .flat_map(|s| s.to_le_bytes())
                            .collect();
                        if socket.send(Message::Binary(bytes)).await.is_err() {
                            break;
                        }
                    }
                }
                msg = socket.recv() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if text == "end" {
                                let _ = text_tx.send((String::new(), None, None));
                                end_requested = true;
                            } else {
                                let req: TTSRequest = match serde_json::from_str(&text) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        let _ = socket.send(Message::Text(format!("{{\"error\": \"Invalid request: {}\"}}", e))).await;
                                        continue;
                                    }
                                };

                                if current_speaker.is_none() {
                                    current_speaker = req.speaker.clone();
                                }

                                let text_to_generate = req.text.trim();
                                if !text_to_generate.is_empty() {
                                    let _ = text_tx.send((text_to_generate.to_string(), current_speaker.clone(), req.instruction.clone()));
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) => break,
                        Some(Ok(Message::Binary(_))) => continue,
                        Some(Ok(Message::Ping(_))) => continue,
                        Some(Ok(Message::Pong(_))) => continue,
                        Some(Err(_)) => break,
                        None => break,
                    }
                }
            }
        }
    }

    let _ = text_tx.send((String::new(), None, None));
    let _ = socket.send(Message::Text("done".to_string())).await;
}
