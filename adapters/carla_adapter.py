from __future__ import annotations


class CarlaVehicleAdapter:
    """Very small boundary for applying a control command to an existing CARLA ego.

    This adapter intentionally does NOT pretend that nuScenes replay is a CARLA
    closed loop. Sensor ingestion and recurrent world updates must be connected
    separately when the project moves to CARLA.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 5.0):
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError("CARLA Python package is not installed/importable.") from exc
        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()

    def find_ego(self):
        vehicles = self.world.get_actors().filter("vehicle.*")
        for actor in vehicles:
            if actor.attributes.get("role_name") in {"hero", "ego"}:
                return actor
        raise RuntimeError("No CARLA vehicle with role_name='hero' or 'ego' found.")

    def apply(self, command):
        ego = self.find_ego()
        control = self.carla.VehicleControl(
            throttle=float(command["throttle"]),
            steer=float(command["steer"]),
            brake=float(command["brake"]),
        )
        ego.apply_control(control)
        return ego.id
