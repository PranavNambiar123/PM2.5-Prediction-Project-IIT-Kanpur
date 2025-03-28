import os
import sys
import subprocess
import time
import datetime

# Create log file
log_file_path = "ev_loc_training.log"
with open(log_file_path, "w") as f:
    f.write(f"=== PM2.5 Prediction Evaluation Started at {datetime.datetime.now()} ===\n")
    f.write("This log file will track the execution of evaluate_all_localities.py\n\n")

# Function to append to log file
def log_message(message):
    with open(log_file_path, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - {message}\n")
        f.flush()  # Force write to disk
        os.fsync(f.fileno())  # Ensure it's written to the physical disk

# Log start of evaluation
log_message("Starting evaluation of all localities")

# Start the evaluation process
process = subprocess.Popen(
    ["python", "evaluate_all_localities.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1  # Line buffered
)

# Log and print output in real-time
log_message("Evaluation process started with PID: " + str(process.pid))
print(f"Evaluation process started with PID: {process.pid}")
print(f"Logging output to {log_file_path}")

try:
    # Read and log output in real-time
    for line in process.stdout:
        line = line.strip()
        print(line)
        log_message(f"OUTPUT: {line}")
        
    # Wait for process to complete
    return_code = process.wait()
    log_message(f"Evaluation completed with return code: {return_code}")
    print(f"Evaluation completed with return code: {return_code}")
    
except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    process.terminate()
    log_message("Evaluation was interrupted by user")
    print("Evaluation was interrupted by user")
except Exception as e:
    # Log any errors
    error_message = f"Error during evaluation: {str(e)}"
    log_message(error_message)
    print(error_message)
    
print(f"Log file is available at: {os.path.abspath(log_file_path)}")
