# SentinelCam

**SentinelCam** is a Python application for **real-time face recognition** using `insightface` (GPU-accelerated), **emotion and gender analysis** with `deepface`, **access control** based on a flexible permission system (access levels, time schedules), and **Telegram notifications** (including photos) when unauthorized individuals are detected.

## 🚀 Features

*   **Face Recognition:** Accurately identifies faces in real-time video feed using the powerful `insightface` model.
*   **GPU Acceleration:** Utilizes `insightface` with GPU support (via `onnxruntime-gpu`) for high performance.
*   **Attribute Analysis:** Determines **emotion** and **gender** of the recognized face using `deepface`.
*   **Flexible Access Control:**
    *   Checks the **access level** (`admin`, `user`, `guest`, `denied`) for each recognized face from `config.json`.
    *   Verifies **time-based access schedules** defined individually for each user.
    *   Access (`access_granted`) is granted **only if the access level is permitted AND the time is within the allowed range**.
*   **Visual Feedback:**
    *   Displays the **access level** **above** the face rectangle.
    *   Shows the name and emotion **below** the face rectangle.
    *   Changes the rectangle color (green — access granted, red — access denied).
*   **Logging:** Saves all events (name, emotion, gender, access level, status, reason) to `recognition_log.csv`.
*   **Telegram Notifications:**
    *   Sends the **face photo** and **event details** (name, access level, status, reason, emotion, gender) to a Telegram chat when an **unauthorized** face is detected or when a face has `send_alert: true` configured in `config.json`.
*   **Configuration:** Manage users, access levels, time schedules, and notification policies via `config.json`.

## 🛠️ Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SentinelCam.git
    cd SentinelCam
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    source venv/bin/activate.fish # Fish terminal
    # or
    # venv\Scripts\activate # Windows
    ```

3.  **Install dependencies:**
    Ensure you have **Python 3.11 or 3.12** (libraries like `face_recognition`, `insightface`, `tensorflow` might not support Python 3.13).
    ```bash
    pip install insightface onnxruntime-gpu opencv-python numpy deepface requests
    # or install from requirements.txt if available
    # pip install -r requirements.txt
    ```

4.  **Install GUI dependencies for OpenCV (if you get an error displaying the window):**
    ```bash
    pip install opencv-python
    # or, if the above command doesn't help, install system dependencies:
    # sudo apt install libgtk-3-dev pkg-config (for Ubuntu/Debian)
    ```

5.  **Set up a Telegram bot:**
    *   Create a bot via [@BotFather](https://t.me/BotFather) on Telegram.
    *   Obtain the **bot token** (e.g., `123456789:ABCdefGHIjklMNOpqrSTUvwxYZ`).
    *   Send a message to the bot in a chat (e.g., `/start`).
    *   Find your **chat ID** (e.g., via `https://api.telegram.org/bot<TOKEN>/getUpdates` after messaging the bot).

6.  **Configure `config.json`:**
    *   Edit the `config.json` file to add users, assign access levels, set time schedules, and configure notifications.

7.  **Place photos in `dataset/`:**
    *   Put photos of faces the system should recognize into the `dataset/` folder. The filename (without suffix `_1`, etc.) will be used as the person's name (e.g., `Alice.jpg` -> `Alice`).

8.  **Run the application:**
    *   Replace `YOUR_CHAT_ID_HERE` (and `YOUR_BOT_TOKEN_HERE` if not already replaced) in the code (`main.py`) with **your chat ID** (and token).
    *   Run the script:
        ```bash
        python main.py
        ```

## 📁 Project Structure

SentinelCam/

├── dataset/                 # Folder containing photos for recognition

│   ├── Person1.jpg

│   └── ...

├── config.json              # Configuration file (users, levels, time schedules, alerts)

├── recognition_log.csv      # (Created automatically) Recognition log

├── main.py                  # Main script

└── README.md                # This file


## ⚠️ Important

*   **GPU:** Ensure you have CUDA and compatible GPU drivers installed for `onnxruntime-gpu`.
*   **Python Version:** Python 3.11 or 3.12 is recommended.
*   **Telegram API Access:** Ensure `api.telegram.org` is accessible from your network. It might be blocked in some countries (e.g., as noted, Nepal).

## 📄 License

This project is unlicensed. The source code/models for `insightface` and `face_recognition_models` might be under separate licenses (see their repositories).
