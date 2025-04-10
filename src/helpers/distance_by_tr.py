import csv
import math

def parse_timestamp(ts):
    minutes, seconds, milliseconds = map(int, ts.split('_'))
    return (minutes * 60 + seconds) * 1000 + milliseconds

def parse_time_str(time_str):
    if ':' in time_str and '.' in time_str:
        minutes, rest = time_str.split(':')
        seconds, millis = rest.split('.')
        return (int(minutes) * 60 + int(seconds)) * 1000 + int(millis)
    elif ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
        return (minutes * 60 + seconds) * 1000
    else:
        raise ValueError("Time must be in MM:SS or MM:SS.mmm format")

def compute_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def process_mouse_csv(input_csv, tr_ms, output_csv, start_ms=None, end_ms=None):
    with open(input_csv, newline='') as f:
        reader = list(csv.DictReader(f))

    distances_by_tr = []
    last_position = None
    current_tr = None
    current_distance = 0

    base_time = parse_timestamp(reader[0]['timestamp'])

    for row in reader:
        timestamp_ms = parse_timestamp(row['timestamp'])

        if start_ms and timestamp_ms < start_ms:
            continue
        if end_ms and timestamp_ms > end_ms:
            continue

        if current_tr is None:
            current_tr = (timestamp_ms - (start_ms or base_time)) // tr_ms + 1

        tr = (timestamp_ms - (start_ms or base_time)) // tr_ms + 1

        if row['match_x'] == 'N/A' or row['match_y'] == 'N/A' or row['best_template'] == 'Not present':
            # Assume object hasn't moved
            pos = last_position
            distance_this_frame = 0
        else:
            x = int(row['match_x'])
            y = int(row['match_y'])
            pos = (x, y)
            distance_this_frame = compute_distance(last_position, pos) if last_position else 0

        if pos is not None:
            if tr == current_tr:
                current_distance += distance_this_frame
            else:
                distances_by_tr.append((current_tr, round(current_distance)))
                for skipped_tr in range(current_tr + 1, tr):
                    distances_by_tr.append((skipped_tr, 0))
                current_distance = distance_this_frame
                current_tr = tr

            last_position = pos

    if current_tr is not None:
        distances_by_tr.append((current_tr, round(current_distance)))

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TR', 'Distance'])
        writer.writerows(distances_by_tr)

    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "output/Web04_run2_match_results.csv"
    output_csv = "output/Web04_run2_distance1.csv"
    tr_seconds = 1.5
    start_time = "00:38.767"
    end_time = "15:38.767"

    tr_ms = int(tr_seconds * 1000)
    start_ms = parse_time_str(start_time) if start_time else None
    end_ms = parse_time_str(end_time) if end_time else None

    process_mouse_csv(input_csv, tr_ms, output_csv, start_ms, end_ms)
