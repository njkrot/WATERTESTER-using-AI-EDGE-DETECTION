# handles the pumps and solenoid valves
import time
import logging
import threading

log = logging.getLogger("pump_control")

# pin assignments - dont change these lol
MAIN_PUMP_PIN = 22
DISPENSE_PUMP_PIN = 23
ROUTING_VALVE_PIN = 24
DUMP_VALVE_PIN = 25

# TODO: figure out actual times
FILL_SECONDS = 10.0
SOAK_SECONDS = 5.0
DRAIN_SECONDS = 5.0
FILTER_SECONDS = 100.0
DISPENSE_TIMEOUT = 10.0

USE_GPIO = True
try:
    from gpiozero import OutputDevice
except ImportError:
    USE_GPIO = False  # no gpio on windows so just fake it


class PinDriver:
    """gpio wrapper"""
    def __init__(self, pin, name):
        self.pin = pin
        self.name = name
        self._dev = None
        self._state = False  # like a bool in c++
        if USE_GPIO:
            try:
                self._dev = OutputDevice(pin, initial_value=False)
            except Exception as e:
                log.warning("GPIO %d (%s) init failed: %s", pin, name, e)

    def on(self):
        self._state = True
        if self._dev:
            self._dev.on()
        log.debug("%s ON (GPIO %d)", self.name, self.pin)

    def off(self):
        self._state = False
        if self._dev:
            self._dev.off()
        log.debug("%s OFF (GPIO %d)", self.name, self.pin)

    @property
    def is_on(self):
        return self._state

    def close(self):
        self.off()
        if self._dev:
            self._dev.close()  # kinda like a destructor


class PumpController:
    """pump stuff"""

    def __init__(self):
        self.main_pump = PinDriver(MAIN_PUMP_PIN, "MainPump")
        self.dispense_pump = PinDriver(DISPENSE_PUMP_PIN, "DispensePump")
        self.routing_valve = PinDriver(ROUTING_VALVE_PIN, "RoutingValve")
        self.dump_valve = PinDriver(DUMP_VALVE_PIN, "DumpValve")
        self._test_completed = False
        self._filtering_done = False
        self._shutdown = threading.Event()  # threading is weird in python ngl
        self._all_off()

    def _all_off(self):
        # kill everything
        self.main_pump.off()
        self.dispense_pump.off()
        self.routing_valve.off()
        self.dump_valve.off()

    def _wait(self, seconds, status_cb=None):
        """sleep but interruptible"""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._shutdown.is_set():
                return False
            remaining = deadline - time.time()
            if status_cb:
                status_cb(remaining)
            time.sleep(min(0.25, max(0.01, remaining)))  # this timing might be off
        return True

    @property
    def test_completed(self):
        return self._test_completed

    @property
    def filtering_done(self):
        return self._filtering_done

    @property
    def can_dispense(self):
        return self._test_completed

    def shutdown(self):
        self._shutdown.set()
        self._all_off()

    def close(self):
        self.shutdown()
        self.main_pump.close()
        self.dispense_pump.close()
        self.routing_valve.close()
        self.dump_valve.close()

    def fill_test_container(self, status_cb=None):
        """fill up the test thing"""
        log.info("Filling test container for %.0fs...", FILL_SECONDS)
        self.routing_valve.off()  # make sure its going to test side
        self.dump_valve.off()
        self.main_pump.on()
        ok = self._wait(FILL_SECONDS, status_cb)
        self.main_pump.off()
        if not ok:
            self._all_off()
        return ok

    def soak_strip(self, status_cb=None):
        """just wait for the strip"""
        log.info("Soaking strip for %.0fs...", SOAK_SECONDS)
        return self._wait(SOAK_SECONDS, status_cb)

    def drain_test_container(self, status_cb=None):
        """flip the valve and drain"""
        log.info("Draining test container for %.0fs...", DRAIN_SECONDS)
        self.dump_valve.on()  # flip the valve
        ok = self._wait(DRAIN_SECONDS, status_cb)
        self.dump_valve.off()
        if not ok:
            self._all_off()
        return ok

    def mark_test_completed(self):
        self._test_completed = True

    def run_filter(self, status_cb=None):
        """send water thru the filter"""
        if not self._test_completed:
            log.warning("Cannot filter - test not completed yet.")
            return False
        log.info("Filtering for %.0fs...", FILTER_SECONDS)
        self.routing_valve.on()  # switch to filter side
        self.dump_valve.off()
        self.main_pump.on()
        ok = self._wait(FILTER_SECONDS, status_cb)
        self.main_pump.off()
        self.routing_valve.off()
        if ok:
            self._filtering_done = True
            log.info("Filtering complete.")
        else:
            self._all_off()
        return ok

    def dispense(self, status_cb=None):
        """give the user water"""
        if not self.can_dispense:
            log.warning("Cannot dispense - test not completed.")
            return False
        log.info("Dispensing clean water for %.0fs...", DISPENSE_TIMEOUT)
        self.dispense_pump.on()
        ok = self._wait(DISPENSE_TIMEOUT, status_cb)
        self.dispense_pump.off()
        return ok

    def run_test_sequence(self, on_fill=None, on_soak=None, on_drain=None):
        """do the whole fill/soak/drain thing"""
        if not self.fill_test_container(on_fill):
            return False
        if not self.soak_strip(on_soak):
            return False
        if not self.drain_test_container(on_drain):
            return False
        return True  # idk if we need all 3 checks but better safe

    def get_status(self):
        return {
            "main_pump": self.main_pump.is_on,
            "dispense_pump": self.dispense_pump.is_on,
            "routing_valve": "filter" if self.routing_valve.is_on else "test",
            "dump_valve": "open" if self.dump_valve.is_on else "closed",
            "test_completed": self._test_completed,
            "filtering_done": self._filtering_done,
        }
