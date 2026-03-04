from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.services.attack_simulator import AttackSimulator


router = APIRouter(prefix="/alerts", tags=["alerts"])


def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "ids", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="IDS runtime not initialized")
    return runtime


@router.get("/recent")
def get_recent_alerts(request: Request, limit: int = 30) -> list[dict]:
    runtime = _get_runtime(request)
    return runtime.websocket_manager.get_recent_alerts(limit=limit)


@router.post("/simulate/{attack_type}")
def simulate_attack(request: Request, attack_type: str) -> dict:
    """Inject synthetic attack traffic for dashboard demos."""
    runtime = _get_runtime(request)
    simulator = AttackSimulator()

    attack_type = attack_type.lower()
    if attack_type == "port-scan":
        packets = simulator.simulate_port_scan()
    elif attack_type == "brute-force":
        packets = simulator.simulate_brute_force()
    elif attack_type == "traffic-spike":
        packets = simulator.simulate_traffic_spike()
    elif attack_type == "all":
        packets = simulator.generate_all()
    else:
        raise HTTPException(
            status_code=400,
            detail="attack_type must be one of: port-scan, brute-force, traffic-spike, all",
        )

    for packet in packets:
        runtime.handle_packet(packet)

    return {"message": "Attack simulation completed", "attack_type": attack_type, "packets_injected": len(packets)}
