import os
import json
import pika
import traceback
from dotenv import load_dotenv
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from ultralytics import YOLO

# Загрузка переменных окружения
load_dotenv()

# Константы
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
INPUT_QUEUE = os.getenv("INPUT_QUEUE")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS")

# Загрузка модели один раз
model = YOLO(YOLO_MODEL_PATH)


def buffer_to_video_file(video_buffer):
    path = "temp_input.mp4"
    with open(path, "wb") as f:
        f.write(video_buffer)
    return path


def video_file_to_buffer(path):
    with open(path, "rb") as f:
        return f.read()


def process_video(video_buffer):
    print(f"→ Обработка видео через YOLO по пути: {YOLO_MODEL_PATH}")
    path = buffer_to_video_file(video_buffer)
    clip = VideoFileClip(path)
    fps = clip.fps
    detections, output_frames = [], []

    for frame_id, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
        results = model(frame)
        annotated_frame = results[0].plot()
        output_frames.append(annotated_frame)

        for r in results:
            if r.boxes:
                time = round(frame_id / fps, 2)
                detected = [model.names[int(b.cls)] for b in r.boxes]
                detections.append({
                    "time": time,
                    "objects": list(set(detected))
                })

    annotated_path = "temp_annotated.mp4"
    ImageSequenceClip(output_frames, fps=fps).write_videofile(annotated_path, codec="libx264", logger=None)

    return video_file_to_buffer(annotated_path), json.dumps({"detections": detections}).encode()


def callback(ch, method, properties, body):
    try:
        data = json.loads(body)
        video_id = data.get("videoId")
        video_data = data.get("videoData", {}).get("data")

        if not video_id or not video_data:
            print("‼️ Ошибка: отсутствует videoId или videoData")
            return

        print(f"[>] Видео {video_id} получено, запускается обработка...")
        video_buffer = bytes(video_data)
        processed_video, report = process_video(video_buffer)

        result = {
            "videoId": video_id,
            "processedVideo": {"type": "Buffer", "data": list(processed_video)},
            "report": report.decode()
        }

        # Новое соединение для отправки результата
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        ))
        channel = connection.channel()
        channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
        channel.basic_publish(exchange="", routing_key=OUTPUT_QUEUE, body=json.dumps(result))
        connection.close()

        print(f"[✓] Видео {video_id} обработано и отправлено в очередь {OUTPUT_QUEUE}")
    except Exception as e:
        print("‼️ Ошибка в callback:", e)


def start_consumer():
    try:
        print("🚀 Запуск consumer...")
        print(f"→ Подключение к RabbitMQ: {RABBITMQ_HOST}:{RABBITMQ_PORT} как {RABBITMQ_USER}")

        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        )

        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=INPUT_QUEUE, durable=True)

        print(f"→ Ожидание сообщений из очереди: {INPUT_QUEUE}")
        channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback, auto_ack=True)
        channel.start_consuming()

    except Exception as e:
        print("‼️ Ошибка в start_consumer:", e)
        traceback.print_exc()

if __name__ == "__main__":
    start_consumer()

"""from ultralytics import YOLO
model=YOLO("C:/Users/LEGION/Desktop/Диплом/Test/runs/detect/final_with_negatives/weights/best.pt")
model.predict(source="C:/Users/LEGION/Downloads/South African _Revolver_ Grenade Launcher.mp4",save=True)"""