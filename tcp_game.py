#!/usr/bin/env python3
# tcp_game.py
#
# TCP Game - simplified TCP-like turn-based protocol game
# - Two human players, two programs on same machine (localhost sockets)
# - No timeout retransmission mechanism (only double-ACK triggers GBN retransmit suggestion)
# - seq/ack start at 0
# - rwnd auto increases every 15s
# - reaction/penalty window is 45s
# - Graph output (Matplotlib) similar to class TCP arrow diagrams (PNG)
#
# Run:
#   Terminal 1 (listener):
#       python tcp_game.py --listen --port 5000 --name A
#   Terminal 2 (connector):
#       python tcp_game.py --connect --host 127.0.0.1 --port 5000 --name B
#
# Notes:
# - This is a "game" not real TCP. Humans decide what to send.
# - The program acts as referee + visualizer + consistency checker.
#
import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

# -----------------------------
# Tunable rules (your fix here)
# -----------------------------
RWND_AUTO_INCREASE_EVERY_SEC = 15      # was 30 -> now 15
REACTION_PENALTY_SEC = 45              # was 30 -> now 45
GAME_DURATION_SEC = 5 * 60             # 5 minutes match
RWND_AUTO_INCREASE_AMOUNT = 5          # arbitrary "some amount" per rule; you can change
MAX_PACKET_BYTES = 4096

# -----------------------------
# Packet model
# -----------------------------
@dataclass
class Packet:
    ptype: str  # "DATA" or "ERROR"
    seq: Optional[int] = None
    ack: Optional[int] = None
    rwnd: Optional[int] = None
    length: Optional[int] = None
    note: str = ""
    ts: float = 0.0  # sender timestamp (monotonic)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "Packet":
        d = json.loads(s)
        return Packet(**d)


# -----------------------------
# Graph recorder
# -----------------------------
class GraphRecorder:
    def __init__(self, my_name: str, peer_name: str):
        self.my_name = my_name
        self.peer_name = peer_name
        self.start = time.monotonic()
        self.lock = threading.Lock()
        # events: (t, direction, label, ptype)
        # direction: "out" (me->peer) or "in" (peer->me)
        self.events: List[Tuple[float, str, str, str]] = []

    def record_out(self, pkt: Packet):
        label = self._label(pkt)
        with self.lock:
            self.events.append((time.monotonic() - self.start, "out", label, pkt.ptype))

    def record_in(self, pkt: Packet):
        label = self._label(pkt)
        with self.lock:
            self.events.append((time.monotonic() - self.start, "in", label, pkt.ptype))

    def _label(self, pkt: Packet) -> str:
        if pkt.ptype == "ERROR":
            return "ERROR"
        return f"seq={pkt.seq} ack={pkt.ack} rwnd={pkt.rwnd} len={pkt.length}"

    def save_png(self, path: str):
        # Import lazily so game can run without matplotlib until end.
        import matplotlib.pyplot as plt

        with self.lock:
            evs = list(self.events)

        # Layout similar to TCP exchange diagrams:
        # two vertical timelines, arrows across.
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # x positions
        x_me = 0.2
        x_peer = 0.8

        # draw vertical lifelines
        ax.plot([x_me, x_me], [0, 1], linewidth=2)
        ax.plot([x_peer, x_peer], [0, 1], linewidth=2)
        ax.text(x_me, 1.02, self.my_name, ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(x_peer, 1.02, self.peer_name, ha="center", va="bottom", fontsize=12, fontweight="bold")

        if not evs:
            ax.text(0.5, 0.5, "No events recorded.", ha="center", va="center")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close(fig)
            return

        # Normalize times to [0, 1]
        tmax = max(t for t, *_ in evs) or 1.0

        def y_of(t: float) -> float:
            # top is 1, time flows downward
            return 1.0 - (t / tmax) * 0.95 - 0.02

        # Draw arrows
        for t, direction, label, ptype in evs:
            y = y_of(t)
            if direction == "out":
                x0, x1 = x_me, x_peer
            else:
                x0, x1 = x_peer, x_me

            # ERROR packets visually distinct by style (no custom color required)
            if ptype == "ERROR":
                ax.annotate(
                    "",
                    xy=(x1, y),
                    xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", linewidth=2, linestyle="--"),
                )
                ax.text((x0 + x1) / 2, y + 0.015, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
            else:
                ax.annotate(
                    "",
                    xy=(x1, y),
                    xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5),
                )
                ax.text((x0 + x1) / 2, y + 0.012, label, ha="center", va="bottom", fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)


# -----------------------------
# Game State + Referee
# -----------------------------
class GameState:
    def __init__(self, my_name: str, peer_name: str):
        self.my_name = my_name
        self.peer_name = peer_name

        # Score is local to each program (both should converge if both play honestly).
        self.my_score = 0
        self.peer_score = 0

        # Start values
        self.expected_peer_seq = 0  # next in-order seq we expect from peer
        self.last_peer_ack = 0

        self.my_next_seq = 0
        self.my_last_ack_sent = 0

        # rwnd for myself (what I advertise)
        self.my_rwnd = 20

        # turn control
        self.my_turn = False

        # reaction timers
        self.last_action_time = time.monotonic()
        self.last_rwnd_zero_sent_time: Optional[float] = None  # if we sent rwnd=0, start penalty window

        # ack history for "double ACK" trigger
        self.last_acks_received: List[int] = []

        # lock for cross-thread
        self.lock = threading.Lock()

    def award_me(self, points: int):
        with self.lock:
            self.my_score += points

    def award_peer(self, points: int):
        with self.lock:
            self.peer_score += points

    def set_turn(self, my_turn: bool):
        with self.lock:
            self.my_turn = my_turn
            self.last_action_time = time.monotonic()

    def update_action(self):
        with self.lock:
            self.last_action_time = time.monotonic()

    def set_my_rwnd(self, v: int):
        with self.lock:
            self.my_rwnd = max(0, int(v))

    def mark_sent_rwnd_zero(self):
        with self.lock:
            self.last_rwnd_zero_sent_time = time.monotonic()

    def clear_sent_rwnd_zero(self):
        with self.lock:
            self.last_rwnd_zero_sent_time = None

    def apply_rwnd_auto_increase(self):
        with self.lock:
            if self.my_rwnd < 0:
                self.my_rwnd = 0
            self.my_rwnd += RWND_AUTO_INCREASE_AMOUNT

    def check_penalties(self) -> List[str]:
        """
        Called periodically.
        Penalties:
        - If it's my turn and I don't send anything within REACTION_PENALTY_SEC -> -1
        - If I sent rwnd=0 and I don't send any packet with rwnd>0 within REACTION_PENALTY_SEC -> -1
          (This implements your 'rwnd=0 sender must fix within reaction window' rule.)
        """
        now = time.monotonic()
        msgs = []
        with self.lock:
            if self.my_turn:
                if now - self.last_action_time > REACTION_PENALTY_SEC:
                    self.my_score -= 1
                    self.last_action_time = now
                    msgs.append(f"[PENALTY] {self.my_name}: No response within {REACTION_PENALTY_SEC}s. -1 point.")

            if self.last_rwnd_zero_sent_time is not None:
                if now - self.last_rwnd_zero_sent_time > REACTION_PENALTY_SEC:
                    self.my_score -= 1
                    self.last_rwnd_zero_sent_time = now  # keep penalizing if player keeps it at 0
                    msgs.append(f"[PENALTY] {self.my_name}: rwnd=0 not fixed within {REACTION_PENALTY_SEC}s. -1 point.")
        return msgs

    def validate_incoming(self, pkt: Packet) -> Tuple[bool, str]:
        """
        Logical consistency checks (not full TCP).
        Return (is_valid, reason).
        """
        if pkt.ptype == "ERROR":
            return True, "ERROR received"

        # Basic field checks
        if pkt.seq is None or pkt.ack is None or pkt.rwnd is None or pkt.length is None:
            return False, "Missing fields"

        if pkt.seq < 0 or pkt.ack < 0 or pkt.rwnd < 0 or pkt.length < 0:
            return False, "Negative values not allowed"

        # Sequence should be in-order (Go-Back-N expects receiver checks this strictly)
        with self.lock:
            expected = self.expected_peer_seq

        if pkt.seq != expected:
            return False, f"Invalid seq: expected {expected}, got {pkt.seq}"

        # ACK should be "plausible": cannot acknowledge beyond what we could have sent
        with self.lock:
            max_ack = self.my_next_seq  # next seq to send; ack <= this
        if pkt.ack > max_ack:
            return False, f"Invalid ack: ack {pkt.ack} > my_next_seq {max_ack}"

        # length should not exceed rwnd from sender's advertised? (optional check)
        # We'll keep it len <= rwnd as a simple sanity rule.
        if pkt.length > pkt.rwnd:
            return False, f"Invalid length: len {pkt.length} > rwnd {pkt.rwnd}"

        return True, "OK"

    def accept_incoming(self, pkt: Packet):
        """
        Apply effects of an accepted incoming DATA packet.
        """
        if pkt.ptype != "DATA":
            return
        with self.lock:
            # advance expected seq by length (simplified)
            self.expected_peer_seq += pkt.length
            self.last_peer_ack = pkt.ack

            # record ack history for double-ACK behavior
            self.last_acks_received.append(pkt.ack)
            if len(self.last_acks_received) > 3:
                self.last_acks_received = self.last_acks_received[-3:]


# -----------------------------
# Networking helpers
# -----------------------------
def send_packet(sock: socket.socket, pkt: Packet):
    data = pkt.to_json().encode("utf-8")
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("Packet too large")
    # Simple framing: length prefix (4 bytes)
    header = len(data).to_bytes(4, "big")
    sock.sendall(header + data)

def recv_packet(sock: socket.socket) -> Optional[Packet]:
    # Read 4-byte length
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    n = int.from_bytes(hdr, "big")
    if n <= 0 or n > MAX_PACKET_BYTES:
        return None
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return Packet.from_json(data.decode("utf-8"))


# -----------------------------
# Main game loop
# -----------------------------
def interactive_turn_input(state: GameState) -> Packet:
    """
    Ask human player what to send on their turn.
    """
    with state.lock:
        suggested_seq = state.my_next_seq
        suggested_ack = state.expected_peer_seq  # ack what we've received in-order
        suggested_rwnd = state.my_rwnd

    print("\n--- YOUR TURN ---")
    print("Type one of:")
    print("  data  -> send DATA packet")
    print("  error -> send ERROR notification")
    print("  show  -> show current state")
    print("  help  -> show help")
    print("  quit  -> end game (close)")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("help", "?"):
            print("Commands: data, error, show, quit")
            continue
        if cmd == "show":
            with state.lock:
                print(f"[STATE] my_score={state.my_score} peer_score={state.peer_score} "
                      f"my_next_seq={state.my_next_seq} expected_peer_seq={state.expected_peer_seq} "
                      f"my_rwnd={state.my_rwnd}")
            continue
        if cmd == "quit":
            raise KeyboardInterrupt
        if cmd == "error":
            return Packet(ptype="ERROR", note="Human sent ERROR", ts=time.monotonic())
        if cmd == "data":
            break
        print("Unknown command. Type 'help'.")

    def ask_int(prompt: str, default: int) -> int:
        while True:
            s = input(f"{prompt} [{default}]: ").strip()
            if s == "":
                return default
            try:
                return int(s)
            except ValueError:
                print("Enter an integer.")

    seq = ask_int("seq", suggested_seq)
    ack = ask_int("ack", suggested_ack)
    rwnd = ask_int("rwnd", suggested_rwnd)
    length = ask_int("length", 1)

    note = input("note (optional): ").strip()
    return Packet(ptype="DATA", seq=seq, ack=ack, rwnd=rwnd, length=length, note=note, ts=time.monotonic())


def maybe_double_ack_hint(state: GameState) -> Optional[str]:
    """
    If last two ACKs received are identical, suggest Go-Back-N retransmit from that ack.
    No automatic timeout-based resend.
    """
    with state.lock:
        acks = list(state.last_acks_received)
    if len(acks) >= 2 and acks[-1] == acks[-2]:
        return f"[DOUBLE-ACK] Received duplicate ACK={acks[-1]} twice. Go-Back-N retransmit may be triggered (no timeout)."
    return None


def run_game(sock: socket.socket, my_name: str, peer_name: str, i_start: bool):
    state = GameState(my_name=my_name, peer_name=peer_name)
    recorder = GraphRecorder(my_name, peer_name)
    state.set_turn(i_start)

    stop_flag = threading.Event()

    # rwnd auto increase thread
    def rwnd_worker():
        while not stop_flag.is_set():
            time.sleep(RWND_AUTO_INCREASE_EVERY_SEC)
            state.apply_rwnd_auto_increase()

    # penalty checker thread
    def penalty_worker():
        while not stop_flag.is_set():
            time.sleep(1)
            msgs = state.check_penalties()
            for m in msgs:
                print(m)

    # receiver thread
    incoming_queue: List[Packet] = []
    incoming_lock = threading.Lock()

    def recv_worker():
        while not stop_flag.is_set():
            pkt = recv_packet(sock)
            if pkt is None:
                stop_flag.set()
                break
            recorder.record_in(pkt)
            with incoming_lock:
                incoming_queue.append(pkt)

    threads = [
        threading.Thread(target=rwnd_worker, daemon=True),
        threading.Thread(target=penalty_worker, daemon=True),
        threading.Thread(target=recv_worker, daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n=======================================")
    print("TCP GAME started.")
    print(f"Me: {my_name} | Peer: {peer_name}")
    print(f"Rules: rwnd auto +{RWND_AUTO_INCREASE_AMOUNT} every {RWND_AUTO_INCREASE_EVERY_SEC}s, "
          f"penalty after {REACTION_PENALTY_SEC}s no-action, duration {GAME_DURATION_SEC}s.")
    print("seq/ack start at 0. Packet length is free.")
    print("No timeout retransmission. Double-ACK only (hint shown).")
    print("=======================================\n")

    game_start = time.monotonic()
    graph_path = f"tcp_game_timeline_{my_name}_vs_{peer_name}.png"

    try:
        while not stop_flag.is_set():
            # Game end
            if time.monotonic() - game_start >= GAME_DURATION_SEC:
                print("\n[GAME] Time is up. Ending match.")
                break

            # Process incoming packets (if any)
            pkt_in = None
            with incoming_lock:
                if incoming_queue:
                    pkt_in = incoming_queue.pop(0)

            if pkt_in is not None:
                # Validate incoming
                ok, reason = state.validate_incoming(pkt_in)

                if pkt_in.ptype == "ERROR":
                    print(f"\n[RECV] ERROR from peer. Note: {pkt_in.note}")
                    # When peer sends ERROR, they claim they detected your impossible packet.
                    # So peer gets +1 (local accounting).
                    state.award_peer(+1)
                    print(f"[SCORE] (local) Peer +1 for ERROR detection. my={state.my_score} peer={state.peer_score}")
                    # Turn passes to me after receiving any packet
                    state.set_turn(True)
                    continue

                # DATA
                print(f"\n[RECV] DATA: seq={pkt_in.seq} ack={pkt_in.ack} rwnd={pkt_in.rwnd} len={pkt_in.length}"
                      + (f" | note={pkt_in.note}" if pkt_in.note else ""))
                if ok:
                    print(f"[CHECK] OK: {reason}")
                    state.accept_incoming(pkt_in)
                    # Turn passes to me
                    state.set_turn(True)
                else:
                    print(f"[CHECK] IMPOSSIBLE: {reason}")
                    # Human decides whether to send ERROR (detect) or "accept" (miss)
                    while True:
                        choice = input("Detect and send ERROR? (y/n): ").strip().lower()
                        if choice in ("y", "n"):
                            break
                    if choice == "y":
                        # Send ERROR
                        pkt = Packet(ptype="ERROR", note=f"Detected: {reason}", ts=time.monotonic())
                        send_packet(sock, pkt)
                        recorder.record_out(pkt)
                        state.award_me(+1)
                        print(f"[SEND] ERROR sent. +1 point to you. my={state.my_score} peer={state.peer_score}")
                        state.update_action()
                        state.set_turn(False)  # after sending, wait
                    else:
                        # Missed detection -> sender (peer) gains +1 (local accounting)
                        state.award_peer(+1)
                        print(f"[MISS] You accepted an impossible packet. Peer +1. my={state.my_score} peer={state.peer_score}")
                        # Still accept into state? In real rules, continuing communication implies accept.
                        # We'll accept it to keep game flowing, but it may desync - that's part of the "game".
                        state.accept_incoming(pkt_in)
                        state.set_turn(True)

                hint = maybe_double_ack_hint(state)
                if hint:
                    print(hint)

            # My turn?
            with state.lock:
                my_turn = state.my_turn
                my_rwnd = state.my_rwnd

            if my_turn:
                pkt_out = interactive_turn_input(state)

                # If I send DATA, update my sending state minimally
                if pkt_out.ptype == "DATA":
                    # Track rwnd=0 penalty rule
                    if pkt_out.rwnd == 0:
                        state.mark_sent_rwnd_zero()
                    else:
                        # if previously sent rwnd=0, consider it fixed
                        state.clear_sent_rwnd_zero()

                    # update advertised rwnd
                    state.set_my_rwnd(pkt_out.rwnd)

                    # update my seq progression (simplified: next_seq += length)
                    with state.lock:
                        # If human sends a weird seq, we still store what human typed as "sent".
                        # But for future plausibility checks, we track next_seq based on seq+len if seq matches expectation.
                        if pkt_out.seq == state.my_next_seq:
                            state.my_next_seq += pkt_out.length
                        # Always track last ack we sent
                        state.my_last_ack_sent = pkt_out.ack

                send_packet(sock, pkt_out)
                recorder.record_out(pkt_out)

                if pkt_out.ptype == "DATA":
                    print(f"[SEND] DATA: seq={pkt_out.seq} ack={pkt_out.ack} rwnd={pkt_out.rwnd} len={pkt_out.length}"
                          + (f" | note={pkt_out.note}" if pkt_out.note else ""))
                else:
                    print(f"[SEND] ERROR sent.")

                state.update_action()
                state.set_turn(False)  # after sending, wait

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[GAME] Interrupted by user. Ending.")
    finally:
        stop_flag.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

        # Save graph
        try:
            recorder.save_png(graph_path)
            print(f"\n[GRAPH] Timeline saved to: {graph_path}")
        except Exception as e:
            print(f"\n[GRAPH] Could not save graph: {e}")

        with state.lock:
            print(f"\n[FINAL SCORE] my={state.my_score} peer={state.peer_score}")
        print("[DONE]")


# -----------------------------
# Connection setup
# -----------------------------
def run_listener(host: str, port: int) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[LISTEN] Waiting on {host}:{port} ...")
    conn, addr = srv.accept()
    print(f"[LISTEN] Connected from {addr}")
    srv.close()
    return conn

def run_connector(host: str, port: int) -> socket.socket:
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[CONNECT] Connecting to {host}:{port} ...")
    cli.connect((host, port))
    print("[CONNECT] Connected.")
    return cli

def handshake(sock: socket.socket, my_name: str) -> str:
    """
    Exchange names.
    """
    # Send my name packet
    pkt = Packet(ptype="DATA", seq=0, ack=0, rwnd=0, length=0, note=f"NAME:{my_name}", ts=time.monotonic())
    send_packet(sock, pkt)
    peer_pkt = recv_packet(sock)
    if peer_pkt is None or not peer_pkt.note.startswith("NAME:"):
        return "PEER"
    return peer_pkt.note.split("NAME:", 1)[1].strip() or "PEER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true", help="Listen mode")
    ap.add_argument("--connect", action="store_true", help="Connect mode")
    ap.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Port")
    ap.add_argument("--name", default="Player", help="Your display name (e.g., A/B)")
    ap.add_argument("--start", action="store_true", help="Start first (have first turn)")
    args = ap.parse_args()

    if args.listen == args.connect:
        print("Choose exactly one: --listen or --connect")
        raise SystemExit(2)

    if args.listen:
        sock = run_listener(args.host, args.port)
        peer_name = handshake(sock, args.name)
    else:
        sock = run_connector(args.host, args.port)
        peer_name = handshake(sock, args.name)

    # Who starts?
    # If one side uses --start, that side starts; otherwise listener starts by default.
    if args.start:
        i_start = True
    else:
        i_start = True if args.listen else False

    run_game(sock, my_name=args.name, peer_name=peer_name, i_start=i_start)


if __name__ == "__main__":
    main()
