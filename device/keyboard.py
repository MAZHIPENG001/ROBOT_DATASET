from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock
import time

class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda: 0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)

    def on_release(self, key):
        pass

    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda: 0)
            self.key_press_list = list()

    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]

    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            return events

    def __enter__(self):
        self.start()  # Start the listener thread
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.stop()  # Stop the listener thread
        return self.get_press_events(self)


if __name__ == '__main__':
    with KeystrokeCounter() as key_counter:
        try:
            while True:
                time.sleep(1 / 60)
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        print("按下q")
                    elif key_stroke == KeyCode(char='c'):
                        print("按下c")
                    elif key_stroke == KeyCode(char='s'):
                        print("按下s")
                    elif key_stroke == KeyCode(char='t'):
                        key_counter.clear()
                        msg = input("输入文字: ")
                        print("输入了:", msg)
                        key_counter.clear()
                        break
                    elif key_stroke == Key.backspace:
                        print("按下backspace")
        except KeyboardInterrupt:
            events = key_counter.get_press_events()
            print(events)
