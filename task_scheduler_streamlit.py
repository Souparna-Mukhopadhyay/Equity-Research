from streamlit.runtime.scriptrunner import add_script_run_ctx,get_script_run_ctx
import streamlit as st
import time
import threading
from datetime import datetime, timedelta
import pytz

def expand_and_sort_tasks(tasks):
    """
    Expands repeated tasks into single-run tasks and sorts them chronologically.
    
    Parameters:
        tasks (list of tuples): List of tasks where each tuple contains:
            1) function (callable)
            2) timezone (str)
            3) start_time (str) in "HH:MM:SS" format
            4) end_time (str) in "HH:MM:SS" format or None
            5) interval (int) in seconds or None

    Returns:
        list of tuples: Expanded and sorted list of single-run tasks.
    """
    single_run_tasks = []
    
    for func, tz, start, end, interval in tasks:
        timezone = pytz.timezone(tz)
        start_time = timezone.localize(datetime.combine(datetime.now(timezone).date(), datetime.strptime(start, "%H:%M:%S").time()))
        
        if interval is None or end is None:
            # Single-run task
            single_run_tasks.append((func, tz, start))
        else:
            # Expand repeated tasks into multiple single-run tasks
            end_time = timezone.localize(datetime.combine(start_time.date(), datetime.strptime(end, "%H:%M:%S").time()))
            current_time = start_time
            while current_time <= end_time:
                single_run_tasks.append((func, tz, current_time.strftime("%H:%M:%S")))
                current_time += timedelta(seconds=interval)
    
    # Sort tasks by chronological start time
    single_run_tasks.sort(key=lambda task: datetime.strptime(task[2], "%H:%M:%S"))
    return single_run_tasks

def run_function_at_time(function, timezone, start_time, arguments):
    """
    Run the function once at the specified time in the given timezone.

    Parameters:
        function (callable): The function to execute.
        timezone (str): Timezone for scheduling.
        start_time (str): Start time in "HH:MM:SS" format.
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    target_time = tz.localize(datetime.combine(now.date(), datetime.strptime(start_time, "%H:%M:%S").time()))
    
    # Adjust to the next day if the time has already passed
    if now > target_time:
        target_time += timedelta(days=1)
    
    wait_time = (target_time - now).total_seconds()
    if wait_time > 0:
        time.sleep(wait_time)
    function(arguments)

def schedule_tasks(args):
    """
    Schedules all tasks in the given list.

    Parameters:
        tasks (list of tuples): List of tasks where each tuple contains:
            1) function (callable)
            2) timezone (str)
            3) start_time (str) in "HH:MM:SS" format
    """
    tasks = args[0]
    ctx = args[1]
    print(tasks)
    # Get the current time
    current_time = datetime.now()

    # Add 6 minutes to the current time
    time_plus = current_time + timedelta(days=1)

    # Format the time in %H:%M:%S format
    formatted_time = time_plus.strftime("%H:%M:%S")
    
    for func, tz, start in tasks:
        st.session_state.logs.append(f"{func.__name__} has been scheduled at {start} of time zone {tz}")
        threading.Thread(target=run_function_at_time, args=(func, tz, start, None)).start()
    time.sleep(10)
    st.session_state.logs.append(f"Whole routine is scheduled at {time_plus} of time zone {tz}")
    thread = threading.Thread(target=run_function_at_time,args=(schedule_tasks, tz, formatted_time, args))
    add_script_run_ctx(thread,ctx)
    thread.start()

# Example usage
def example_function(arguments):
    print(f"Function executed at: {datetime.now()}")

# Initial list of tasks
tasks = [
    (example_function, 'Asia/Kolkata', '12:37:00', '12:39:00', 60),
    (example_function, 'Asia/Kolkata', '12:37:00', None, None),
    (example_function, 'Asia/Kolkata', '12:38:30', None, None),
]

# Expand and sort tasks
expanded_tasks = expand_and_sort_tasks(tasks)

# Streamlit App
st.title("Task Scheduler")

if "logs" not in st.session_state:
    st.session_state.logs = []

# @st.fragment
# def my_fragment():
#     st.button("Refresh")
for item in st.session_state.logs:
    st.write(item)
    # st.write(f"Fragment says it ran {st.session_state.fragment_runs} times.")

# my_fragment()
ctx = get_script_run_ctx()

thread = threading.Thread(target=schedule_tasks, args=((expanded_tasks,ctx),))
add_script_run_ctx(thread,ctx)
thread.start()
