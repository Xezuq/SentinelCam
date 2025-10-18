import cv2
import os
import numpy as np
import csv
from datetime import datetime
from insightface.app import FaceAnalysis
from deepface import DeepFace
import json
import requests # Добавлен импорт

class FaceRecognizer:
    def __init__(self, dataset_path, config_path="config.json"):
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.load_config() # Теперь загружает новую структуру

        # --- Настройки Telegram ---
        # Убедитесь, что вы заменили эти значения на свои!
        self.TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN" # Пример: "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
        self.TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID" # Пример: "1234567890"
        # --- Конец настроек Telegram ---

        # Инициализация InsightFace
        # Убедитесь, что 'CUDAExecutionProvider' доступен, если вы хотите использовать GPU
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Инициализация лога
        self.init_log()

        # Загрузка известных лиц
        self.known_embeddings, self.known_names = self.load_known_faces()

        # --- Для отслеживания отправленных уведомлений ---
        # Чтобы не спамить, если одно и то же неизвестное лицо долго находится в кадре
        self.notified_unknown_faces = set()
        # Используем set() для хранения уникальных (top, right, bottom, left) обнаруженных лиц, помеченных как "Неизвестный"
        # Очищаем set каждый N кадров или по таймеру, если нужно.
        self.frame_count = 0
        self.reset_interval = 30 # Скажем, сбрасываем отслеживание каждые 30 кадров
        # --- Конец отслеживания ---

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

        # Сохраняем настройки
        self.global_settings = config.get('global_settings', {})
        self.access_levels = config.get('access_levels', {})
        self.people_config = config.get('people', {})
        # Извлекаем имена из датасета как "известные"
        self.known_names_from_dataset = set(self.people_config.keys())
        # print(f"DEBUG: Загружены настройки для {len(self.people_config)} человек.") # Для отладки

    def init_log(self):
        """Инициализирует CSV-файл для логирования."""
        log_file_exists = os.path.exists("recognition_log.csv")
        with open("recognition_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not log_file_exists:
                writer.writerow(["timestamp", "name", "emotion", "gender", "access_level", "access_granted", "access_denied_reason"])

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
                    # Используем первого (или наиболее достоверного) человека на фото
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
            score = np.dot(face_embedding, known_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_emb))
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_score > 0.6:
            return self.known_names[best_match_idx]
        else:
            return "Неизвестный"

    def get_emotion_gender(self, face_image):
        """Определяет эмоцию и пол с помощью DeepFace."""
        try:
            result = DeepFace.analyze(
                face_image,
                actions=['emotion', 'gender'],
                enforce_detection=False,
                detector_backend='skip'
            )
            emotion = result[0]['dominant_emotion']
            gender = result[0]['dominant_gender']
            return emotion, gender
        except Exception as e:
            # print(f"Ошибка при анализе эмоций/пола: {e}")
            return "unknown", "unknown"

    def check_access(self, name):
        """
        Проверяет доступ для указанного имени на основе config.json.
        Возвращает:
            access_granted (bool): Разрешён ли доступ.
            access_level (str): Уровень доступа.
            reason (str): Причина отказа (если был).
            send_alert (bool): Отправлять ли уведомление.
        """
        current_hour = datetime.now().hour
        config_entry = self.people_config.get(name)

        if config_entry:
            # Лицо найдено в конфиге
            access_level = config_entry.get("access_level", self.global_settings.get("default_access_level", "denied"))
            allowed_times = config_entry.get("allowed_times", self.global_settings.get("default_time_range", {"start": 0, "end": 24}))
            send_alert_config = config_entry.get("send_alert", True)
        else:
            # Лицо НЕ найдено в конфиге (например, "Неизвестный", если не прописан)
            access_level = self.global_settings.get("default_access_level", "denied")
            allowed_times = self.global_settings.get("default_time_range", {"start": 0, "end": 24})
            send_alert_config = True # По умолчанию отправлять уведомление для неизвестных

        # Проверка уровня доступа
        level_granted = self.access_levels.get(access_level, {}).get("granted", False)

        # Проверка времени
        time_ok = allowed_times["start"] <= current_hour < allowed_times["end"]

        access_granted = level_granted and time_ok

        reason = ""
        if not level_granted:
            reason = f"Уровень '{access_level}' запрещён"
        elif not time_ok:
            reason = f"Вне времени доступа ({allowed_times['start']}-{allowed_times['end']})"

        return access_granted, access_level, reason, send_alert_config


    def send_telegram_alert(self, detected_name, emotion, gender, face_image, access_level, reason):
        """
        Отправляет сообщение и фото в Telegram при обнаружении неавторизованной личности.

        Args:
            detected_name (str): Имя, определённое системой (ожидается "Неизвестный").
            emotion (str): Определённая эмоция.
            gender (str): Определённый пол.
            face_image (numpy.ndarray): Изображение лица (BGR), полученное из кадра.
            access_level (str): Уровень доступа.
            reason (str): Причина отказа в доступе.
        """
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

        # --- Отправка фото ---
        import io # Импортируем внутри функции
        _, encoded_img = cv2.imencode('.jpg', face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Сжимаем фото для отправки
        photo_bytes_io = io.BytesIO(encoded_img.tobytes())
        photo_bytes_io.name = 'face.jpg' # Даем имя файлу

        try:
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
                timeout=30 # Увеличим таймаут, так как отправляется файл
            )
            response.raise_for_status()
            print(f"✅ Уведомление (фото и текст) отправлено в Telegram: {detected_name}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка отправки уведомления в Telegram: {e}")
            if 'response' in locals():
                print(f"Тело ответа от API: {response.text}")

    def run(self):
        """Запускает основной цикл камеры."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Не удалось открыть камеру.")
            return

        print("✅ Запуск камеры. Нажмите 'q' для выхода.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Ошибка получения кадра с камеры.")
                break

            self.frame_count += 1
            if self.frame_count % self.reset_interval == 0:
                self.notified_unknown_faces.clear()
                # print("DEBUG: notified_unknown_faces cleared.") # Для отладки

            faces = self.app.get(frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_image = frame[y1:y2, x1:x2] # <-- Это изображение лица
                kps = face.kps  # 2D ключевые точки

                # Распознавание
                name = self.recognize_face(face.normed_embedding)
                emotion, gender = self.get_emotion_gender(face_image)

                # === НОВАЯ СИСТЕМА РАЗРЕШЕНИЙ ===
                access_granted, access_level, reason, send_alert = self.check_access(name)

                # --- Отправка уведомления ---
                # Отправляем, если send_alert == True и (неизвестный ИЛИ доступ запрещён)
                if send_alert and (name == "Неизвестный" or not access_granted):
                    print(f"DEBUG: Отправка уведомления для: {name}, уровень: {access_level}, причина: {reason}") # Отладочный принт
                    # Создаём уникальный идентификатор для текущего лица в кадре
                    loc_key = (x1, y1, x2, y2)
                    # Используем loc_key для неизвестных, чтобы избежать спама
                    # Для известных, но запрещённых, можно также использовать, если нужно
                    if name == "Неизвестный":
                        if loc_key not in self.notified_unknown_faces:
                            self.notified_unknown_faces.add(loc_key)
                            self.send_telegram_alert(name, emotion, gender, face_image, access_level, reason) # <-- Изменено
                    else:
                        # Для известных, но запрещённых, можно отправлять всегда или с другим отслеживанием
                        # В данном случае, отправляем всегда, если send_alert=True и доступ запрещён
                        self.send_telegram_alert(name, emotion, gender, face_image, access_level, reason) # <-- Изменено
                # --- Конец отправки уведомления ---

                # Логирование
                with open("recognition_log.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        name,
                        emotion,
                        gender,
                        access_level,
                        "granted" if access_granted else "denied",
                        reason
                    ])

                # Визуализация: цвет рамки зависит от доступа
                # Зелёный - доступ разрешён, Красный - доступ запрещён
                color = (0, 255, 0) if access_granted else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # --- Отображение уровня доступа СВЕРХУ ---
                # Вычисляем размер текста уровня доступа
                (level_label_width, level_label_height), baseline = cv2.getTextSize(access_level, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)

                # Координаты фона подписи уровня доступа (сверху рамки)
                level_text_origin_x = x1
                level_text_origin_y = y1 - 10 # Немного выше верхней границы

                # Рисуем фон для уровня доступа
                cv2.rectangle(frame, (x1, y1 - level_label_height - 10), (x1 + level_label_width + 10, y1), color, cv2.FILLED)
                cv2.rectangle(frame, (x1, y1 - level_label_height - 10), (x1 + level_label_width + 10, y1), (255, 255, 255), 1) # Белая обводка

                # Отображаем уровень доступа
                cv2.putText(frame, access_level, (level_text_origin_x + 5, level_text_origin_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                # --- Конец отображения уровня доступа ---

                # Формируем основную подпись (имя и эмоция)
                label_parts = [name, f"({emotion})"]
                label = " ".join(label_parts)

                # Отображаем основную подпись СНИЗУ (как было)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(frame, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition + Access Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = FaceRecognizer(dataset_path="dataset", config_path="config.json")
    recognizer.run()

if __name__ == "__main__":
    main()
