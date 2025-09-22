import pandas as pd
import schedule
import time

def read_and_process_data():
    try:
        df = pd.read_csv('C:/temp/Pegatron/test/smart-factory-agent/data/template.csv')
        print(f"Dataset read at {time.ctime()}. First 5 rows:\n{df.head()}")
        # Add your data processing/analysis logic here
    except FileNotFoundError:
        print("Error: your_dataset.csv not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Schedule the function to run every minute
schedule.every(0.1).minutes.do(read_and_process_data)

print("Agent started. Reading dataset every minute...")

while True:
    schedule.run_pending()
    time.sleep(0.1)