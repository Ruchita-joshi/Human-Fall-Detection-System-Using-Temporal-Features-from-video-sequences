import smtplib
import time
from email.mime.text import MIMEText

# -------- EMAIL CONFIG --------
SENDER_EMAIL = "fallalertproject@gmail.com"
SENDER_PASSWORD = "iywghvrqoxgoqfom"
RECEIVER_EMAIL = "ruchitajoshi020@gmail.com"

# -------- RATE LIMIT --------
LAST_ALERT_TIME = 0
COOLDOWN = 60  # seconds


def send_email_alert(frame_no):
    global LAST_ALERT_TIME

    if time.time() - LAST_ALERT_TIME < COOLDOWN:
        print("⏳ Email cooldown active")
        return

    subject = "🚨 Fall Detected Alert"
    body = f"""
ALERT!

A fall has been detected by the system.

Frame No: {frame_no}
Time    : {time.strftime("%Y-%m-%d %H:%M:%S")}

Please take immediate action.
"""

    msg = MIMEText(body)
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        LAST_ALERT_TIME = time.time()
        print("✅ Email alert sent successfully")

    except Exception as e:
        print(f"❌ Email failed: {e}")