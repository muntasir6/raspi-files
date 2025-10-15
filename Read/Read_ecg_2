import RPi.GPIO as GPIO
import time

# Pin assignments
DATA_PINS = [26, 4, 23, 24, 21, 20, 16, 25]  # D0 to D7
CLK_PIN = 13
ALE_PIN = 27
START_PIN = 22
OE_PIN = 5
EOC_PIN = 6

GPIO.setmode(GPIO.BCM)

# Setup pins
for pin in DATA_PINS:
    GPIO.setup(pin, GPIO.IN)

GPIO.setup(CLK_PIN, GPIO.OUT)
GPIO.setup(ALE_PIN, GPIO.OUT)
GPIO.setup(START_PIN, GPIO.OUT)
GPIO.setup(OE_PIN, GPIO.OUT)
GPIO.setup(EOC_PIN, GPIO.IN)

def pulse(pin):
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(0.00001)  # 10us pulse width, adjust if needed
    GPIO.output(pin, GPIO.LOW)
    time.sleep(0.00001)

def read_adc():
    # Start conversion sequence

    # 1. Bring ALE high to latch address bits (here all 0 since A,B,C = GND)
    GPIO.output(ALE_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(ALE_PIN, GPIO.LOW)

    # 2. Start pulse
    GPIO.output(START_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(START_PIN, GPIO.LOW)

    # 3. Wait for EOC (End of Conversion) to go LOW (conversion complete)
    while GPIO.input(EOC_PIN) == GPIO.HIGH:
        time.sleep(0.000001)

    # 4. Enable output (OE)
    GPIO.output(OE_PIN, GPIO.LOW)

    # 5. Read 8 data bits from D0-D7
    value = 0
    for i, pin in enumerate(DATA_PINS):
        bit = GPIO.input(pin)
        value |= (bit << i)

    # 6. Disable output
    GPIO.output(OE_PIN, GPIO.HIGH)

    return value

try:
    GPIO.output(OE_PIN, GPIO.HIGH)  # Disable output initially
    GPIO.output(CLK_PIN, GPIO.LOW)

    print("Starting ADC reading... Press Ctrl+C to stop.")
    while True:
        val = read_adc()
        print(f"ADC Value: {val} (0x{val:02X})")
        time.sleep(0.01)  # 10 ms delay, adjust as needed

except KeyboardInterrupt:
    print("Exiting...")

finally:
    GPIO.cleanup()
