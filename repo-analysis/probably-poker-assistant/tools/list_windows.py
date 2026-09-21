import win32gui

def list_windows():
    def enum_handler(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                print(f"'{title}'")

    print("Visible Windows:")
    print("----------------")
    win32gui.EnumWindows(enum_handler, None)

if __name__ == "__main__":
    list_windows()
