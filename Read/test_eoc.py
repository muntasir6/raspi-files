import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.IN)  # EOC pin

print("Reading EOC pin for 5 seconds...")
for i in range(50):
    print(f"EOC state: {GPIO.input(27)}")
    time.sleep(0.1)

GPIO.cleanup()
