# # database.py
# import sqlite3
# from datetime import datetime
# # from config import DB_PATH


# DB_PATH = './interaction_logs.db'
# # Connect to the SQLite database (it will create the file if it doesn't exist)
# # conn = sqlite3.connect(DB_PATH)
# # c = conn.cursor()

# def check_table_exists():
#     """Check if the logs table exists"""
#     conn = create_connection()
#     c = conn.cursor()
#     try:
#         c.execute("SELECT 1 FROM logs LIMIT 1;")
#         return True
#     except sqlite3.OperationalError:
#         return False
#     finally:
#         conn.close()

# def create_connection():
#     """Create a database connection with multi-threading support"""
#     return sqlite3.connect(DB_PATH, check_same_thread=False)  # Allow multi-threading



# # Create table for storing the interaction logs if it doesn't exist
# def create_table():
#     conn = create_connection()
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS logs (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             question TEXT,
#             response TEXT,
#             feedback TEXT,
#             timestamp DATETIME
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # Function to insert interaction into the database
# def insert_interaction(question, response, feedback):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     conn = create_connection()  # Create a new connection for each operation
#     c = conn.cursor()
#     c.execute('''
#         INSERT INTO logs (question, response, feedback, timestamp)
#         VALUES (?, ?, ?, ?)
#     ''', (question, response, feedback, timestamp))
#     conn.commit()
#     conn.close()

# # Close the connection when the program ends
# def close_connection(conn):
#     conn.close()


# # Check and create the table if it doesn't exist
# if not check_table_exists():
#     create_table()



# # Function to retrieve all logs from the database
# def get_all_logs():
#     conn = create_connection()
#     c = conn.cursor()
#     c.execute('''SELECT question, response, feedback, timestamp FROM logs ORDER BY timestamp DESC''')
#     logs = c.fetchall()
#     conn.close()
#     return logs


# import sqlite3
# from datetime import datetime

# DB_PATH = './interaction_logs.db'

# # Function to create a connection with multi-threading support
# def create_connection():
#     """Create a database connection with multi-threading support"""
#     return sqlite3.connect(DB_PATH, check_same_thread=False)  # Allow multi-threading

# # Function to insert interaction into the database
# def insert_interaction(question, response, feedback):
#     try:
#         conn = create_connection()
#         c = conn.cursor()
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         print(f"Inserting data into DB: {question}, {response}, {feedback}, {timestamp}")

#         c.execute('''
#             INSERT INTO logs (question, response, feedback, timestamp)
#             VALUES (?, ?, ?, ?)
#         ''', (question, response, feedback, timestamp))

#         conn.commit()
#         print("Data successfully inserted into the database!")
#         conn.close()
#     except Exception as e:
#         print(f"Error while inserting data into DB: {e}")


import sqlite3
from datetime import datetime

DB_PATH = './interaction_log2.db'

def create_connection():
    """Create a database connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_table():
    """Create the logs table if it doesn't exist."""
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            response TEXT,
            feedback TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()
    print("Table created successfully or already exists.")

def insert_interaction(question, response, feedback,connc):
    """Insert a new interaction into the logs table."""
    # conn = create_connection()
    conn=connc
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print(f"Inserting: {question}, {response}, {feedback}, {timestamp}")
    try:
        c.execute('''
            INSERT INTO logs (question, response, feedback, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (question, response, feedback, timestamp))
        conn.commit()
        last_row_id = c.lastrowid
        print(f"Inserted row with ID: {last_row_id}")
        conn.close()
    except Exception as e:
        print("Error in insert")
    finally:
        conn.close()
    print("Interaction saved successfully!")
    # read_latest_log()


def read_latest_log():
    """Fetch the latest log entry and print it."""
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM logs ORDER BY timestamp DESC LIMIT 1
    ''')
    row = c.fetchone()  # Fetch the latest log
    conn.close()
    if row:
        print("\nNew row inserted:")
        print(f"ID: {row[0]}, Question: {row[1]}, Response: {row[2]}, Feedback: {row[3]}, Timestamp: {row[4]}")
    else:
        print("No rows found in the database.")

def get_db_length():
    """Fetch the current number of rows in the database (logs table)."""
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM logs')
    row_count = c.fetchone()[0]  # Fetch the count of rows
    conn.close()
    
    # Print the current length of the database
    print(f"Current number of rows in the database: {row_count}")
    return row_count




def view_all_logs():
    """View all logs."""
    conn = create_connection()
    c = conn.cursor()
    print("hi")
    c.execute('SELECT * FROM logs')
    rows = c.fetchall()
    conn.close()
    return rows

def view_latest_log():
    """View the latest log entry."""
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM logs ORDER BY timestamp DESC LIMIT 1
    ''')
    row = c.fetchone()  # Fetch the latest log
    conn.close()
    return row

def get_all_logs():
    """Retrieve all logs from the database."""
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM logs')
    logs = c.fetchall()
    conn.close()
    return logs

create_table()

# --- Running Demo ---

if __name__ == "__main__":
    # get_db_length() 
    # create_table()  # Step 1: Create table
    insert_interaction("What is Python?", "Python is a programming language.", "Like")  # Step 2: Insert sample
    logs = view_all_logs()  # Step 3: Read everything
    print("\nAll logs:")
    for log in logs:
        print(log)
    latest_log = view_latest_log()  # Step 3: Read the latest log
    print("\nLatest log:")
    print(latest_log)


