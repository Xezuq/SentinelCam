import cv2
import os
import numpy as np
import csv
from datetime import datetime
from insightface.app import FaceAnalysis
from deepface import DeepFace
import json
import requests
import time
import threading
from collections import deque

class FaceRecognizer:
    def __init__(self, dataset_path, config_path="config.json", camera_source=0, 
                 process_every_n_frames=3, save_frames=False, save_dir="captured_frames"):
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.camera_source = camera_source  # 0 для веб-камеры, путь к файлу или RTSP-ссылка
        self.process_every_n_frames = process_every_n_frames  # Пропуск кадров для экономии ресурсов
        self.save_frames = save_frames  # Сохранять ли кадры с детекцией
        self.save_dir = save_dir
        
        if self.save_frames and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.load_config()

        # --- Настройки Telegram ---
        self.TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
        self.TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"
        # --- Конец настроек Telegram ---

        # Инициализация InsightFace для CPU
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))  # Уменьшен размер детекции для скорости

        # Инициализация лога
        self.init_log()

        # Загрузка известных лиц
        self.known_embeddings, self.known_names = self.load_known_faces()

        # --- Для отслеживания отправленных уведомлений ---
        self.notified_unknown_faces = set()
        self.frame_count = 0
        self.reset_interval = 30
        self.last_processed_frame = 0
        
        # --- Для плавной остановки ---
        self.running = True
        
        # --- Очередь задач для Telegram (чтобы не блокировать основной поток) ---
        self.telegram_queue = deque(maxlen=10)
        self.telegram_thread = threading.Thread(target=self._telegram_worker, daemon=True)
        self.telegram_thread.start()

    def _telegram_worker(self):
        """Фоновый обработчик очереди уведомлений Telegram."""
        while self.running:
            if self.telegram_queue:
                task = self.telegram_queue.popleft()
                self._send_telegram_request(*task)
            else:
                time.sleep(0.5)

    def load_config(self):
        """Загружает настройки из JSON файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"❌ Файл конфигурации {self.config_path} не найден. Создаю пример.")
            default_config = {
                "global_settings": {
                    "default_access_level": "denied",
                    "default_time_range": {"start": 9, "end": 22}
                },
                "access_levels": {
                    "admin": {"granted": True, "description": "Полный доступ"},
                    "user": {"granted": True, "description": "Стандартный доступ"},
                    "guest": {"granted": False, "description": "Ограниченный доступ"},
                    "denied": {"granted": False, "description": "Доступ запрещён"}
                },
                "people": {
                    "Xezuq": {
                        "access_level": "admin",
                        "allowed_times": {"start": 0, "end": 24},
                        "send_alert": False
                    }
                }
            }
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            config = default_config

        self.global_settings = config.get('global_settings', {})
        self.access_levels = config.get('access_levels', {})
        self.people_config = config.get('people', {})
        self.known_names_from_dataset = set(self.people_config.keys())

    def init_log(self):
        """Инициализирует CSV-файл для логирования."""
        log_file_exists = os.path.exists("recognition_log.csv")
        with open("recognition_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not log_file_exists:
                writer.writerow(["timestamp", "name", "emotion", "gender", 
                               "access_level", "access_granted", "access_denied_reason", "bbox"])

    def extract_name_from_filename(self, filename):
        """Извлекает имя из имени файла."""
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        if len(parts) > 1:
            name = '_'.join(parts[:-1])
        return name

    def load_known_faces(self):
        """Загружает и вычисляет эмбеддинги для всех лиц в датасете."""
        known_embeddings = []
        known_names = []

        for filename in os.listdir(self.dataset_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.dataset_path, filename)
                name = self.extract_name_from_filename(filename)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ Не удалось загрузить изображение: {filename}")
                    continue

                faces = self.app.get(image)

                if faces:
                    face = sorted(faces, key=lambda x: x.det_score)[-1]
                    embedding = face.normed_embedding
                    known_embeddings.append(embedding)
                    known_names.append(name)
                    print(f"✅ Загружен эмбеддинг для: {name}")
                else:
                    print(f"⚠️ Не найдено лиц на изображении: {filename}")

        print(f"\n✅ Загружено {len(known_embeddings)} эмбеддингов для распознавания.")
        return known_embeddings, known_names

    def recognize_face(self, face_embedding):
        """Сравнивает переданный эмбеддинг с известными."""
        if not self.known_embeddings:
            return "Неизвестный"

        best_match_idx = -1
        best_match_score = -1

        for i, known_emb in enumerate(self.known_embeddings):
            score = np.dot(face_embedding, known_emb) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_emb) + 1e-8
            )
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_score > 0.6:
            return self.known_names[best_match_idx]
        else:
            return "Неизвестный"

    def get_emotion_gender(self, face_image):
        """Определяет эмоцию и пол с помощью DeepFace (CPU-режим)."""
        try:
            # enforce_detection=False и detector_backend='skip' ускоряют работу
            result = DeepFace.analyze(
                face_image,
                actions=['emotion', 'gender'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True  # Отключаем вывод DeepFace
            )
            emotion = result[0]['dominant_emotion']
            gender = result[0]['dominant_gender']
            return emotion, gender
        except Exception as e:
            # print(f"Ошибка при анализе эмоций/пола: {e}")
            return "unknown", "unknown"

    def check_access(self, name):
        """Проверяет доступ для указанного имени на основе config.json."""
        current_hour = datetime.now().hour
        config_entry = self.people_config.get(name)

        if config_entry:
            access_level = config_entry.get("access_level", self.global_settings.get("default_access_level", "denied"))
            allowed_times = config_entry.get("allowed_times", self.global_settings.get("default_time_range", {"start": 0, "end": 24}))
            send_alert_config = config_entry.get("send_alert", True)
        else:
            access_level = self.global_settings.get("default_access_level", "denied")
            allowed_times = self.global_settings.get("default_time_range", {"start": 0, "end": 24})
            send_alert_config = True

        level_granted = self.access_levels.get(access_level, {}).get("granted", False)
        time_ok = allowed_times["start"] <= current_hour < allowed_times["end"]
        access_granted = level_granted and time_ok

        reason = ""
        if not level_granted:
            reason = f"Уровень '{access_level}' запрещён"
        elif not time_ok:
            reason = f"Вне времени доступа ({allowed_times['start']}-{allowed_times['end']})"

        return access_granted, access_level, reason, send_alert_config

    def _send_telegram_request(self, detected_name, emotion, gender, face_image, access_level, reason):
        """Внутренний метод отправки в Telegram (выполняется в фоне)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = f"""
🚨 Обнаружена личность!
Имя: {detected_name}
Уровень доступа: {access_level}
Статус: {'✅ Разрешён' if access_level in self.access_levels and self.access_levels[access_level].get('granted', False) else '❌ Отказано'}
Причина отказа: {reason if reason else 'Нет'}
Эмоция: {emotion}
Пол: {gender}
Время: {timestamp}
        """.strip()

        import io
        try:
            _, encoded_img = cv2.imencode('.jpg', face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            photo_bytes_io = io.BytesIO(encoded_img.tobytes())
            photo_bytes_io.name = 'face.jpg'

            response = requests.post(
                f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={
                    'chat_id': self.TELEGRAM_CHAT_ID,
                    'caption': caption,
                    'parse_mode': 'HTML'
                },
                files={
                    'photo': (photo_bytes_io.name, photo_bytes_io, 'image/jpeg')
                },
                timeout=30
            )
            response.raise_for_status()
            print(f"✅ Уведомление отправлено в Telegram: {detected_name}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка отправки уведомления в Telegram: {e}")

    def send_telegram_alert(self, detected_name, emotion, gender, face_image, access_level, reason):
        """Добавляет задачу в очередь Telegram (не блокирует основной поток)."""
        self.telegram_queue.append((detected_name, emotion, gender, face_image, access_level, reason))

    def log_detection(self, name, emotion, gender, access_level, access_granted, reason, bbox):
        """Записывает результат в CSV-лог."""
        with open("recognition_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                name,
                emotion,
                gender,
                access_level,
                "granted" if access_granted else "denied",
                reason,
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            ])

    def save_frame_if_needed(self, frame, faces_data):
        """Сохраняет кадр с аннотациями если включена опция save_frames."""
        if not self.save_frames:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_frame = frame.copy()
        
        for data in faces_data:
            x1, y1, x2, y2 = data['bbox']
            color = (0, 255, 0) if data['access_granted'] else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{data['name']} [{data['access_level']}]", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        save_path = os.path.join(self.save_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(save_path, annotated_frame)
        print(f"💾 Кадр сохранён: {save_path}")

    def process_frame(self, frame):
        """Обрабатывает один кадр: детекция, распознавание, логирование, уведомления."""
        faces = self.app.get(frame)
        faces_data = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_image = frame[y1:y2, x1:x2]
            if face_image.size == 0:
                continue

            # Распознавание
            name = self.recognize_face(face.normed_embedding)
            emotion, gender = self.get_emotion_gender(face_image)
            access_granted, access_level, reason, send_alert = self.check_access(name)

            # Отправка уведомления (в фоновую очередь)
            if send_alert and (name == "Неизвестный" or not access_granted):
                loc_key = (x1, y1, x2, y2)
                if name == "Неизвестный":
                    if loc_key not in self.notified_unknown_faces:
                        self.notified_unknown_faces.add(loc_key)
                        self.send_telegram_alert(name, emotion, gender, face_image, access_level, reason)
                else:
                    self.send_telegram_alert(name, emotion, gender, face_image, access_level, reason)

            # Логирование
            self.log_detection(name, emotion, gender, access_level, access_granted, reason, bbox)
            
            faces_data.append({
                'bbox': bbox,
                'name': name,
                'access_level': access_level,
                'access_granted': access_granted
            })
        
        # Сохранение кадра если нужно
        if faces_data and self.save_frames:
            self.save_frame_if_needed(frame, faces_data)
            
        return len(faces)

    def run(self):
        """Запускает основной цикл обработки (без графического вывода)."""
        cap = cv2.VideoCapture(self.camera_source)
        
        # Дополнительные настройки для стабильности на сервере
        if isinstance(self.camera_source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимизируем буферизацию
        
        if not cap.isOpened():
            print(f"❌ Не удалось открыть источник видео: {self.camera_source}")
            return

        print(f"✅ Запуск в headless-режиме. Источник: {self.camera_source}")
        print("   Нажмите Ctrl+C для остановки.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Не удалось получить кадр, повторная попытка...")
                    time.sleep(1)
                    # Пробуем переподключиться к камере
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(self.camera_source)
                    continue

                self.frame_count += 1
                
                # Пропускаем кадры для экономии ресурсов CPU
                if self.frame_count % self.process_every_n_frames != 0:
                    continue
                    
                # Сброс отслеживания неизвестных лиц
                if self.frame_count % self.reset_interval == 0:
                    self.notified_unknown_faces.clear()

                # Обработка кадра
                start_time = time.time()
                faces_count = self.process_frame(frame)
                proc_time = time.time() - start_time
                
                if faces_count > 0:
                    print(f"🔍 Кадр #{self.frame_count}: найдено лиц={faces_count}, время обработки={proc_time:.2f}с")

                # Небольшая пауза чтобы не грузить CPU на 100%
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 Получен сигнал остановки, завершаю работу...")
        finally:
            self.running = False
            cap.release()
            # Ждём завершения очереди Telegram
            time.sleep(2)
            print("✅ Работа завершена.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Recognition Server (Headless)')
    parser.add_argument('--dataset', type=str, default='dataset', help='Путь к датасету')
    parser.add_argument('--config', type=str, default='config.json', help='Путь к конфигу')
    parser.add_argument('--camera', type=str, default='0', help='Источник видео: 0, /dev/video0, или RTSP-ссылка')
    parser.add_argument('--skip', type=int, default=3, help='Пропускать N кадров между обработкой')
    parser.add_argument('--save-frames', action='store_true', help='Сохранять кадры с детекцией')
    parser.add_argument('--save-dir', type=str, default='captured_frames', help='Папка для сохранённых кадров')
    
    args = parser.parse_args()
    
    # Преобразуем источник камеры
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    
    recognizer = FaceRecognizer(
        dataset_path=args.dataset,
        config_path=args.config,
        camera_source=camera_source,
        process_every_n_frames=args.skip,
        save_frames=args.save_frames,
        save_dir=args.save_dir
    )
    recognizer.run()


if __name__ == "__main__":
    main()
