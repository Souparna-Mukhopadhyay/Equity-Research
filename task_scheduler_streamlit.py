from streamlit.runtime.scriptrunner import add_script_run_ctx,get_script_run_ctx
from tzlocal import get_localzone
import streamlit as st
import time
import threading
from datetime import datetime, timedelta
import pytz
# from zoneinfo import ZoneInfo
import sqlite3
import os

def get_or_create_database(db_name="schedule.db"):
    # Check if the database file exists
    if not os.path.exists(db_name):
        print(f"Database '{db_name}' not found. Creating a new database.")
    else:
        print(f"Database '{db_name}' found. Connecting to it.")
    
    # Connect to the SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    
    # Get the cursor for executing SQL commands
    cursor = conn.cursor()
    
    return conn, cursor

def check_and_create_table(cursor, table_name="schedule_0"):
    # Check if the table exists
    cursor.execute(f"""SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';""")
    result = cursor.fetchone()

    if result:
        print(f"Table '{table_name}' already exists.")
    else:
        print(f"Table '{table_name}' does not exist. Creating the table.")
        cursor.execute(f"""CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, native INTEGER NOT NULL, function TEXT NOT NULL, time TEXT );""")
        print(f"Table '{table_name}' created successfully.")


def add_row_to_schedule(cursor, native, function, time):
    """
    Add a row to the 'schedule' table.

    Args:
        cursor: SQLite database cursor.
        task_name (str): Name of the task.
        task_time (str): Task time in 'YYYY-MM-DD HH:MM:SS' format.
    """
    try:
        # Insert a row into the 'schedule' table
        cursor.execute("""INSERT INTO schedule_0 (native, function, time) VALUES (?, ?, ?);""", (native, function, time))
        print("Row added successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

# def entry_to_schedule(db_name="schedule.db", table_name="schedule_0", ):

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
    # tz = ZoneInfo(timezone)
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    target_time = tz.localize(datetime.combine(now.date(), datetime.strptime(start_time, "%H:%M:%S").time()))
    # target_time = datetime.combine(now.date(), datetime.strptime(start_time, "%H:%M:%S").time(), tz)
    
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
    
    # Example usage
    connection, cursor = get_or_create_database()

    # Check and create the table if it doesn't exist
    check_and_create_table(cursor)

    # print(tasks)
    # Get the current time
    current_time = datetime.now()

    # Add 6 minutes to the current time
    time_plus = current_time + timedelta(days=1)

    # Format the time in %H:%M:%S format
    formatted_time = time_plus.strftime("%H:%M:%S")
    
    # get timezone
    local_timezone = get_localzone()
    
    for func, tz, start in tasks:
        # st.session_state.logs.append(f"{func.__name__} has been scheduled at {start} of time zone {tz}")
        threading.Thread(target=run_function_at_time, args=(func, tz, start, None)).start()
        add_row_to_schedule(cursor=cursor, function=func.__name__, native=0, time=start) 
    time.sleep(10)
    # st.session_state.logs.append(f"Whole routine is scheduled at {time_plus} of time zone {tz}")
    thread = threading.Thread(target=run_function_at_time,args=(schedule_tasks, str(local_timezone), formatted_time, args))
    add_script_run_ctx(thread,ctx)
    thread.start()
    add_row_to_schedule(cursor=cursor, function=thread.name, native=thread.native_id, time=formatted_time)
    connection.commit()
    print(thread.native_id, thread.name)


def is_thread_alive(cursor):
    """
    Check if the last entry in 'schedule_0' table has a column 'function' ending with '(run_function_at_time)'
    and the column 'native' contains a native thread ID of a currently alive thread.

    Args:
        cursor: SQLite database cursor.

    Returns:
        bool: True if the thread is alive, False otherwise.
    """
    try:
        # Query to get the last entry of the 'schedule_0' table
        cursor.execute("""
            SELECT function, native 
            FROM schedule_0 
            ORDER BY id DESC 
            LIMIT 1;
        """)
        result = cursor.fetchone()

        # If no entries are found, return False
        if not result:
            print("No entries found in the table.")
            return False

        # Extract 'function' and 'native' values
        function, native_id = result
        native_id = int(native_id)  # Ensure the native_id is an integer

        # Check if 'function' ends with '(run_function_at_time)'
        if not function.endswith('(run_function_at_time)'):
            print(f"Function '{function}' does not end with '(run_function_at_time)'.")
            return False

        # Check if the thread with 'native_id' is alive
        for thread in threading.enumerate():
            if thread.native_id == native_id:  # Match native ID with existing threads
                print(f"Thread with native ID {native_id} is alive.")
                return True

        print(f"Thread with native ID {native_id} is not alive.")
        return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

def get_rows(cursor):
    """
    Print all rows from the 'schedule_0' table.

    Args:
        cursor: SQLite database cursor.
    """
    try:
        # Query to select all rows from the table
        cursor.execute("SELECT * FROM schedule_0;")
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Check if the table is empty
        if not rows:
            print("The table 'schedule_0' is empty.")
            return None
        
        # Print each row
        return rows
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
# Example usage
def example_function(arguments):
    print(f"Function executed at: {datetime.now()}")

# Initial list of tasks
tasks = [
    (example_function, 'Asia/Kolkata', '01:40:00', '01:42:00', 60),
    (example_function, 'Asia/Kolkata', '01:40:00', None, None),
    (example_function, 'Asia/Kolkata', '01:41:30', None, None),
    (example_function, 'Asia/Kolkata', '01:40:30', None, None)
]

# Expand and sort tasks
expanded_tasks = expand_and_sort_tasks(tasks)

# Streamlit App
st.title("Task Scheduler")

# if "logs" not in st.session_state:
#     st.session_state.logs = []

# Example usage
connection_, cursor_ = get_or_create_database()

# Check and create the table if it doesn't exist
check_and_create_table(cursor_)

@st.fragment
def my_fragment():
    st.button("Refresh")
    st.write("logs")
    connection__, cursor__ = get_or_create_database()
    rows = get_rows(cursor=cursor__)
    if rows != None:
        for row in rows:
            st.write(row)
    # for item in st.session_state.logs:
    #     st.write(item)
    # st.write(f"Fragment says it ran {st.session_state.fragment_runs} times.")
    

my_fragment()
ctx = get_script_run_ctx()
bool = is_thread_alive(cursor=cursor_)

if bool == False:
    thread = threading.Thread(target=schedule_tasks, args=((expanded_tasks, ctx),))
    add_script_run_ctx(thread, ctx)
    thread.start()