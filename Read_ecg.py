import RPi.GPIO as GPIO
import time
import csv
from datetime import datetime

# GPIO Pin Assignments
START = 17
EOC = 27
OUTEN = 22
ALE = 23
CLOCK = 24
DATA_PINS = [4, 5, 6, 12, 13, 19, 26, 25]  # Adjust based on your wiring (2^0 to 2^7)

# ADC0808 Channel Selection (using IN0 only)
ADDR_A = None  # Pin for address A (ground it if not using)
ADDR_B = None  # Pin for address B (ground it if not using)
ADDR_C = None  # Pin for address C (ground it if not using)

# Sampling parameters
SAMPLING_RATE = 500  # Hz (typical for ECG)
DURATION = 10  # seconds (collect 10 seconds of data)
FILENAME = f"ecg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def setup_gpio():
    """Initialize GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Set output pins
    for pin in [START, OUTEN, ALE, CLOCK]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    # Set input pin
    GPIO.setup(EOC, GPIO.IN)
    
    # Set data pins as inputs
    for pin in DATA_PINS:
        GPIO.setup(pin, GPIO.IN)
    
    print("GPIO setup complete")

def read_adc():
    """Read a single value from ADC0808 IN0"""
    # Set ALE high to latch the channel address
    GPIO.output(ALE, GPIO.HIGH)
    time.sleep(0.000001)  # 1 microsecond
    GPIO.output(ALE, GPIO.LOW)
    
    # Pulse START to begin conversion
    GPIO.output(START, GPIO.HIGH)
    time.sleep(0.000001)
    GPIO.output(START, GPIO.LOW)
    
    # Wait for EOC (End of Conversion) to go high
    timeout = time.time() + 0.01  # 10ms timeout
    while GPIO.input(EOC) == GPIO.LOW:
        if time.time() > timeout:
            print("ADC conversion timeout!")
            return None
    
    # Set OUTEN high to enable output
    GPIO.output(OUTEN, GPIO.HIGH)
    time.sleep(0.000001)
    
    # Read 8 data pins (MSB to LSB)
    value = 0
    for i, pin in enumerate(DATA_PINS):
        bit = GPIO.input(pin)
        value |= (bit << (7 - i))
    
    # Set OUTEN low to disable output
    GPIO.output(OUTEN, GPIO.LOW)
    
    return value

def convert_to_voltage(adc_value, vref=5.0):
    """Convert ADC value to voltage (0-255 maps to 0-Vref)"""
    return (adc_value / 255.0) * vref

def collect_ecg_data():
    """Collect ECG data for specified duration"""
    setup_gpio()
    
    print(f"Starting ECG collection for {DURATION} seconds at {SAMPLING_RATE} Hz")
    print(f"Data will be saved to {FILENAME}")
    
    try:
        ecg_data = []
        sample_interval = 1.0 / SAMPLING_RATE
        start_time = time.time()
        sample_count = 0
        
        # Open CSV file for writing
        with open(FILENAME, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sample', 'Time (s)', 'ADC Value', 'Voltage (V)'])
            
            while time.time() - start_time < DURATION:
                loop_start = time.time()
                
                # Read ADC value
                adc_value = read_adc()
                
                if adc_value is not None:
                    voltage = convert_to_voltage(adc_value)
                    elapsed_time = time.time() - start_time
                    
                    ecg_data.append({
                        'sample': sample_count,
                        'time': elapsed_time,
                        'adc': adc_value,
                        'voltage': voltage
                    })
                    
                    # Write to CSV
                    writer.writerow([sample_count, f"{elapsed_time:.4f}", adc_value, f"{voltage:.4f}"])
                    
                    sample_count += 1
                    
                    if sample_count % 50 == 0:
                        print(f"Sample {sample_count}: {adc_value} (ADC) -> {voltage:.4f}V")
                
                # Maintain sampling rate
                elapsed = time.time() - loop_start
                if elapsed < sample_interval:
                    time.sleep(sample_interval - elapsed)
        
        print(f"\nCollected {sample_count} samples")
        print(f"Data saved to {FILENAME}")
        
        return ecg_data
    
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up")

if __name__ == "__main__":
    collect_ecg_data()
