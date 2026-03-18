import logging
import threading

log = logging.getLogger("strip_reader")

USE_LCD = False
try:
    from RPLCD.i2c import CharLCD
    USE_LCD = True
except ImportError:
    pass

USE_GUI = False
try:
    import tkinter as tk
    import tkinter.font as tkfont
    USE_GUI = True
except ImportError:
    pass

LCD_I2C_ADDR = 0x27
LCD_COLS = 16
LCD_ROWS = 2
LCD_I2C_PORT = 1


class _LCDBackend:
    def __init__(self):
        self._lcd = CharLCD(
            i2c_expander="PCF8574",
            address=LCD_I2C_ADDR,
            port=LCD_I2C_PORT,
            cols=LCD_COLS,
            rows=LCD_ROWS,
        )
        self._lcd.clear()

    def write(self, l1, l2):
        self._lcd.home()
        self._lcd.write_string(l1)
        self._lcd.crlf()
        self._lcd.write_string(l2)

    def clear(self):
        self._lcd.clear()

    def close(self):
        self._lcd.clear()
        self._lcd.close()


class _GUIBackend:
    """lcd sim"""

    def __init__(self):
        self._line1 = " " * LCD_COLS
        self._line2 = " " * LCD_COLS
        self._lock = threading.Lock()
        self._dirty = False
        self._closed = False
        self._ready = threading.Event()
        self._root = None
        self._lbl1 = None
        self._lbl2 = None
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        self._ready.wait(timeout=5)

    def _run(self):
        self._root = tk.Tk()
        self._root.title("Water Filter - LCD Display")
        self._root.configure(bg="#1a1a1a")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        fnt = tkfont.Font(family="Courier", size=26, weight="bold")

        outer = tk.Frame(self._root, bg="#1a1a1a", padx=16, pady=12)
        outer.pack()

        hdr = tk.Label(outer, text="LCD 16x2", font=("Arial", 9),
                       fg="#666666", bg="#1a1a1a")
        hdr.pack(anchor="w")

        lcd_fr = tk.Frame(outer, bg="#0a3d0a", padx=14, pady=10,
                          highlightbackground="#333", highlightthickness=2)
        lcd_fr.pack(pady=(4, 0))

        self._lbl1 = tk.Label(lcd_fr, text=self._line1, font=fnt,
                              fg="#33ff33", bg="#0a3d0a", anchor="w",
                              width=LCD_COLS, padx=6, pady=2)
        self._lbl1.pack(pady=(0,1))

        self._lbl2 = tk.Label(lcd_fr, text=self._line2, font=fnt,
                              fg="#33ff33", bg="#0a3d0a", anchor="w",
                              width=LCD_COLS, padx=6, pady=2)
        self._lbl2.pack()

        self._ready.set()
        self._poll()
        self._root.mainloop()

    def _poll(self):
        if self._closed:
            try:
                self._root.destroy()
            except Exception:
                pass
            return
        with self._lock:
            if self._dirty:
                try:
                    self._lbl1.config(text=self._line1)
                    self._lbl2.config(text=self._line2)
                except Exception:
                    pass
                self._dirty = False
        self._root.after(50, self._poll)  # 50ms refresh, seems fine

    def _on_close(self):
        self._closed = True

    def write(self, l1, l2):
        with self._lock:
            self._line1 = l1
            self._line2 = l2
            self._dirty = True

    def clear(self):
        self.write(" " * LCD_COLS, " " * LCD_COLS)

    def close(self):
        self._closed = True


class _ConsoleBackend:
    # fallback when no hw or gui
    def write(self, l1, l2):
        print("[LCD] {} | {}".format(l1.strip(), l2.strip()))
    def clear(self):
        pass
    def close(self):
        pass


class Display:
    """picks which display to use"""

    def __init__(self, username="User"):
        self._username = username
        self._last_score = None
        self._backend = None

        if USE_LCD:
            try:
                self._backend = _LCDBackend()
                log.info("LCD connected (%dx%d at 0x%02X).", LCD_COLS, LCD_ROWS, LCD_I2C_ADDR)
            except Exception as e:
                log.warning("LCD init failed (%s), trying GUI.", e)

        if self._backend is None and USE_GUI:
            try:
                self._backend = _GUIBackend()
                log.info("Using tkinter LCD simulator.")
            except Exception as e:
                log.warning("GUI init failed: %s", e)

        if self._backend is None:
            self._backend = _ConsoleBackend()
            log.info("LCD output goes to console.")  #NEED?! maybe remove later

    def _pad(self, text):
        return str(text)[:LCD_COLS].ljust(LCD_COLS)

    def write(self, line1, line2=""):
        l1 = self._pad(line1)
        l2 = self._pad(line2)
        try:
            self._backend.write(l1, l2)
        except Exception as e:
            log.warning("Display write error: %s", e)

    def clear(self):
        try:
            self._backend.clear()
        except Exception:
            pass

    def show_hello(self):
        nm = self._username[:LCD_COLS - 7]
        sc = str(self._last_score) if self._last_score is not None else "--"
        self.write("Hello, {}".format(nm), "Last test: {}".format(sc))

    def show_filtering(self, remaining_secs):
        if remaining_secs >= 60:
            ts = "{}m left".format(int(remaining_secs / 60))
        else:
            ts = "{}s left".format(int(remaining_secs))
        self.write("Filtering water.", ts)

    def show_score(self):
        sc = str(self._last_score) if self._last_score is not None else "--"
        self.write("Water score is", sc)

    def set_last_score(self, score):
        self._last_score = score

    def show_reading(self):
        self.write("Reading strip...", "hold steady")

    def show_results(self, pads, score=None):
        cnt = len(pads) if pads else 0
        if score is not None:
            self.set_last_score(score)
        self.write("Done! {} pads".format(cnt), "Score: {}".format(
            score if score else self._last_score or "--"))

    def show_error(self, msg):
        self.write("ERROR", str(msg)[:LCD_COLS])

    def show_remove(self):
        self.write("Remove to reset", "")

    def show_shutting_down(self):
        self.write("Shutting down...", "")

    def close(self):
        try:
            self._backend.close()
        except Exception:
            pass
