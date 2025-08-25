#!/usr/bin/env python3
import matplotlib.pyplot as plt
import csv
import os

def plot_velocity_log(filepath):
    times = []
    v_cmd = []
    v_meas = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # bỏ dòng tiêu đề
        for row in reader:
            times.append(float(row[0]))
            v_cmd.append(float(row[1]))
            v_meas.append(float(row[2]))

    plt.figure()
    plt.plot(times, v_cmd, label='Vận tốc điều khiển (cmd)', linestyle='--')
    plt.plot(times, v_meas, label='Vận tốc thực tế (odom)', marker='o')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Vận tốc (m/s)')
    plt.title('So sánh vận tốc điều khiển và vận tốc thực tế')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Đường dẫn mặc định tới thư mục log
    folder = os.path.expanduser('/home/nguyehtt/turtlebot3_ws/src/target_tracking/target_tracking/velocity_logs')
    # Liệt kê file gần nhất
    files = sorted(os.listdir(folder), reverse=True)
    if files:
        latest_file = os.path.join(folder, files[0])
        print(f'📈 Đang vẽ từ file: {latest_file}')
        plot_velocity_log(latest_file)
    else:
        print('⚠️ Không tìm thấy file log nào trong ~/velocity_logs.')
