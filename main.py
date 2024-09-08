import psutil
import os
import platform
import pandas as pd
from datetime import datetime
from openai import OpenAI, OpenAIError, RateLimitError, APIError, Timeout
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the API key is loaded
# Function to save the API key to a .env file
def save_api_key_to_env(api_key):
    # Remove existing .env file if it exists
    if os.path.exists(".env"):
        os.remove(".env")
    
    # Create a new .env file and save the API key
    with open(".env", "w") as env_file:
        env_file.write(f"OPENAI_API_KEY={api_key}\n")
    print(".env file created with the API key.")


try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    
    # Prompt the user to enter the API key if it's not present or failed
    manual_api_key = input("Enter your OpenAI API key: ")
    
    # Save the entered API key to a new .env file
    try:
        save_api_key_to_env(manual_api_key)
        # Reload the environment with the new API key
        load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print("OpenAI client initialized successfully.")
    except Exception as save_error:
        print(f"Error saving API key: {save_error}")



def gpt_response(scan, history):
    """
    Sends a request to the GPT API with the system scan results
    and conversation history. Updates the conversation history with the new response.
    """
    try:
        history_text = f"Previous system scan: {history['scan']}\nPrevious response: {history['response']}\n"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful system monitoring assistant. Provide a brief summary of the system data."},
                {"role": "user", "content": f"SYSTEM SCAN RESULTS: {scan}\nCONVERSATION HISTORY: {history_text}"}
            ]
        )

        history['scan'] = scan
        history['response'] = response.choices[0].message.content

        return history['response']

    except RateLimitError:
        print("Rate limit exceeded. Please wait and try again later.")
        return "Error: Rate limit exceeded. Try again later."
    except Timeout:
        print("The request timed out. Please try again.")
        return "Error: Request timed out. Try again."
    except APIError as e:
        print(f"OpenAI API returned an error: {e}")
        return f"Error: OpenAI API error: {e}"
    except OpenAIError as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Error: Unable to retrieve response from OpenAI."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Error: An unexpected error occurred."

def get_detailed_cpu_info():
    """Function to get detailed CPU information."""
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        cpu_stats = psutil.cpu_stats()
        cpu_times = psutil.cpu_times(percpu=False)  # Total CPU times for all cores

        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)

        iowait = getattr(cpu_times, 'iowait', None)
        iowait_str = f"{iowait:.2f}" if isinstance(iowait, (int, float)) else "N/A"

        cpu_info = {
            "CPU Freq (MHz)": f"{cpu_freq.current:.1f}" if cpu_freq else "N/A",
            "Max Freq (MHz)": f"{cpu_freq.max:.1f}" if cpu_freq else "N/A",
            "Min Freq (MHz)": f"{cpu_freq.min:.1f}" if cpu_freq else "N/A",
            "Physical Cores": cpu_cores if cpu_cores else "N/A",
            "Logical Threads": cpu_threads if cpu_threads else "N/A",
            "Per Core Usage (%)": cpu_percent,
            "Load Avg (1, 5, 15 min)": load_avg,
            "Context Switches": cpu_stats.ctx_switches,
            "Interrupts": cpu_stats.interrupts,
            "System Calls": cpu_stats.syscalls,
            "User Time (s)": f"{cpu_times.user:.2f}",
            "System Time (s)": f"{cpu_times.system:.2f}",
            "Idle Time (s)": f"{cpu_times.idle:.2f}",
            "I/O Wait Time (s)": iowait_str,
        }

        return pd.DataFrame([cpu_info])

    except Exception as e:
        print(f"Error retrieving CPU information: {e}")
        return pd.DataFrame([{"Error": "Unable to retrieve CPU information."}])

def get_detailed_memory_info():
    """Function to get detailed memory information."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_info = {
            "Total Memory (GB)": f"{memory.total / (1024 ** 3):.2f}",
            "Available Memory (GB)": f"{memory.available / (1024 ** 3):.2f}",
            "Used Memory (GB)": f"{memory.used / (1024 ** 3):.2f}",
            "Used Memory (%)": f"{memory.percent:.1f} %",
            "Cache (GB)": f"{getattr(memory, 'cached', 0) / (1024 ** 3):.2f}",
            "Buffers (GB)": f"{getattr(memory, 'buffers', 0) / (1024 ** 3):.2f}",
            "Active (GB)": f"{getattr(memory, 'active', 0) / (1024 ** 3):.2f}",
            "Inactive (GB)": f"{getattr(memory, 'inactive', 0) / (1024 ** 3):.2f}",
            "Swap Total (GB)": f"{swap.total / (1024 ** 3):.2f}",
            "Swap Used (GB)": f"{swap.used / (1024 ** 3):.2f}",
            "Swap Free (GB)": f"{swap.free / (1024 ** 3):.2f}",
            "Swap Usage (%)": f"{swap.percent:.1f} %",
        }

        return pd.DataFrame([memory_info])

    except Exception as e:
        print(f"Error retrieving memory information: {e}")
        return pd.DataFrame([{"Error": "Unable to retrieve memory information."}])

def get_disk_info():
    """Function to get disk usage and IO information."""
    try:
        disk_partitions = psutil.disk_partitions()
        disk_io = psutil.disk_io_counters(perdisk=True)
        disk_info_list = []

        for partition in disk_partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                io_info = disk_io.get(partition.device.split("\\")[-1], None) if platform.system() == "Windows" else disk_io.get(partition.device.split("/")[-1], None)
                disk_info = {
                    "Device": partition.device,
                    "Mountpoint": partition.mountpoint,
                    "File System": partition.fstype,
                    "Total Size (GB)": f"{usage.total / (1024 ** 3):.2f}",
                    "Used (%)": f"{usage.percent:.1f}",
                    "Read Cnt": io_info.read_count if io_info else "N/A",
                    "Write Cnt": io_info.write_count if io_info else "N/A",
                    "Read (MB)": f"{io_info.read_bytes / (1024 ** 2):.2f}" if io_info else "N/A",
                    "Write (MB)": f"{io_info.write_bytes / (1024 ** 2):.2f}" if io_info else "N/A",
                }
                disk_info_list.append(disk_info)
            except Exception as e:
                disk_info_list.append({"Error": f"Unable to retrieve disk info for {partition.device}: {e}"})

        return pd.DataFrame(disk_info_list)

    except Exception as e:
        print(f"Error retrieving disk information: {e}")
        return pd.DataFrame([{"Error": "Unable to retrieve disk information."}])

def get_network_info():
    """Function to get network information."""
    try:
        net_io = psutil.net_io_counters(pernic=True)
        network_info_list = []

        for interface, io_stats in net_io.items():
            network_info = {
                "Interface": interface,
                "Sent (GB)": f"{io_stats.bytes_sent / (1024 ** 3):.2f}",
                "Recv (GB)": f"{io_stats.bytes_recv / (1024 ** 3):.2f}",
                "Packets Sent": io_stats.packets_sent,
                "Packets Recv": io_stats.packets_recv,
                "Errors In": io_stats.errin,
                "Errors Out": io_stats.errout,
                "Drop In": io_stats.dropin,
                "Drop Out": io_stats.dropout,
            }
            network_info_list.append(network_info)

        return pd.DataFrame(network_info_list)

    except Exception as e:
        print(f"Error retrieving network information: {e}")
        return pd.DataFrame([{"Error": "Unable to retrieve network information."}])

def get_gpu_info():
    """Function to get GPU information (if available)."""
    try:
        from GPUtil import getGPUs
        gpus = getGPUs()
        gpu_info_list = []

        for gpu in gpus:
            gpu_info = {
                "GPU Name": gpu.name,
                "Load (%)": f"{gpu.load * 100:.1f}",
                "Mem Free (MB)": f"{gpu.memoryFree:.1f}",
                "Mem Used (MB)": f"{gpu.memoryUsed:.1f}",
                "Total Mem (MB)": f"{gpu.memoryTotal:.1f}",
                "Temp (C)": f"{gpu.temperature:.1f}",
            }
            gpu_info_list.append(gpu_info)

        return pd.DataFrame(gpu_info_list)

    except ImportError:
        return pd.DataFrame([{"Error": "GPUtil is not installed."}])
    except Exception as e:
        return pd.DataFrame([{"Error": f"Unable to retrieve GPU information: {e}"}])

def display_system_info():
    """Function to display and return all system info."""
    try:
        os.system('clear' if platform.system() != 'Windows' else 'cls')

        print(f"System Information at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("===== CPU Information =====")
        cpu_info = get_detailed_cpu_info()
        print(cpu_info.to_string(index=False))

        print("\n===== Memory Information =====")
        memory_info = get_detailed_memory_info()
        print(memory_info.to_string(index=False))

        print("\n===== Disk Information =====")
        disk_info = get_disk_info()
        print(disk_info.to_string(index=False))

        print("\n===== Network Information =====")
        network_info = get_network_info()
        print(network_info.to_string(index=False))

        print("\n===== GPU Information =====")
        gpu_info = get_gpu_info()
        print(gpu_info.to_string(index=False))

        # Combine the dataframes into one for the summary
        return cpu_info.to_string(index=False) + "\n" + memory_info.to_string(index=False) + "\n" + disk_info.to_string(index=False) + "\n" + network_info.to_string(index=False) + "\n" + gpu_info.to_string(index=False)

    except Exception as e:
        print(f"Error displaying system information: {e}")
        return "Error: Unable to display system information."

if __name__ == "__main__":
    exit_program = False
    old_system_scan = None
    while not exit_program:
        try:
            if old_system_scan is None:
                system_scan = display_system_info()
                old_system_scan = system_scan  # Store the first scan
            else:
                system_scan = old_system_scan  # Use the old scan for regeneration

            print(system_scan)  # Printing the full system scan for verification

            # Always generate a new report even for regenerate option
            report = gpt_response(system_scan, {"scan": "", "response": ""})

            print(f"\n\n\n===== System Summary by GPT =====\n{report}\n\n\n")
            
            print("System monitoring completed successfully!")
            print("1. Run system scan again")
            print("2. Regenerate report on last system scan")
            print("3. Exit")
            choice = input("Enter your choice: ")

            if choice == "1":
                old_system_scan = None  # Reset to run a new system scan
            elif choice == "2":
                print(f"\n\n\n===== System Summary by GPT =====\n{report}\n\n\n")  # Regenerate report
            elif choice == "3":
                exit_program = True
            else:
                print("Invalid choice. Exiting...")
                exit_program = True

        except Exception as e:
            print(f"Unexpected error in main execution: {e}")
