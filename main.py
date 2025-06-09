import os
import json
import pika
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
INPUT_QUEUE = os.getenv("INPUT_QUEUE")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE")

model = YOLO(os.getenv("YOLO_MODEL_PATH"))

def buffer_to_video_file(video_buffer):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_buffer)
    return temp_path

def video_file_to_buffer(video_path):
    with open(video_path, "rb") as f:
        return f.read()

def process_video(video_buffer):
    video_path = buffer_to_video_file(video_buffer)
    clip = VideoFileClip(video_path)
    fps = clip.fps
    detections = []
    output_frames = []

    for frame_id, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
        results = model(frame)
        annotated_frame = results[0].plot()
        output_frames.append(annotated_frame)

        for r in results:
            if len(r.boxes) > 0:
                detection_time = frame_id / fps
                detected_objects = [model.names[int(box.cls)] for box in r.boxes]
                detections.append({
                    "time": round(detection_time, 2),
                    "objects": list(set(detected_objects))
                })

    json_report = {
        "detections": detections
    }
    json_report_bytes = json.dumps(json_report).encode()
    temp_output_path = "temp_annotated.mp4"
    annotated_clip = ImageSequenceClip(output_frames, fps=fps)
    annotated_clip.write_videofile(temp_output_path, codec="libx264",logger=None)
    processed_video_buffer = video_file_to_buffer(temp_output_path)

    return processed_video_buffer, json_report_bytes

def callback(ch, method, properties, body):
    data = json.loads(body)
    video_id = data.get("videoId")
    video_buffer = bytes(data.get("videoData")["data"])
    if not video_id or not video_buffer:
        print("Ошибка: некорректные данные.")
        return

    print(f"[x] Обработка видео {video_id}")
    processed_video, json_report = process_video(video_buffer)

    result_message = json.dumps({
        "videoId": video_id,
        "processedVideo": {"type": "Buffer", "data": list(processed_video)},
        "report": json_report.decode()
    })

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
    channel.basic_publish(exchange="", routing_key=OUTPUT_QUEUE, body=result_message)
    connection.close()

    print(f"[✓] Обработанное видео {video_id} отправлено в очередь {OUTPUT_QUEUE}")

def start_consumer():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=INPUT_QUEUE, durable=True)
    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback, auto_ack=True)

    print(f"[*] Ожидание сообщений в очереди {INPUT_QUEUE}. Для выхода нажмите CTRL+C")
    channel.start_consuming()

if __name__ == "__main__":
    start_consumer()
"""from ultralytics import YOLO
model=YOLO("C:/Users/LEGION/Desktop/Диплом/Test/runs/detect/final_with_negatives/weights/best.pt")
model.predict(source="C:/Users/LEGION/Downloads/South African _Revolver_ Grenade Launcher.mp4",save=True)"""