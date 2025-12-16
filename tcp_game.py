#!/usr/bin/env python3
# tcp_game_final_v6.py
#
# TCP Game - Final Clean UI Version (v6)
#
# UI CHANGE:
# - Removed "suggested values" (e.g., [seq: 10]) from inputs.
# - User MUST manually type all values now.
#
# CORE LOGIC (Same as v5):
# - GBN Retransmission Recognition (Retransmissions are VALID).
# - Hybrid Scoring (-1 for rwnd=0 violations, +1 for others).
# - Rollback mechanism active.
# - No "length > rwnd" restriction.
#

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

# -----------------------------
# Tunable rules
# -----------------------------
RWND_AUTO_INCREASE_EVERY_SEC = 15
REACTION_PENALTY_SEC = 45
GAME_DURATION_SEC = 5 * 60
RWND_AUTO_INCREASE_AMOUNT = 20
MAX_PACKET_BYTES = 4096

# -----------------------------
# Packet model
# -----------------------------
@dataclass
class Packet:
    ptype: str
    seq: Optional[int] = None
    ack: Optional[int] = None
    rwnd: Optional[int] = None
    length: Optional[int] = None
    note: str = ""
    ts: float = 0.0

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
        import matplotlib.pyplot as plt
        with self.lock:
            evs = list(self.events)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        x_me, x_peer = 0.2, 0.8

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

        tmax = max(t for t, *_ in evs) or 1.0
        def y_of(t: float) -> float:
            return 1.0 - (t / tmax) * 0.95 - 0.02

        for t, direction, label, ptype in evs:
            y = y_of(t)
            if direction == "out":
                x0, x1 = x_me, x_peer
            else:
                x0, x1 = x_peer, x_me

            if ptype == "ERROR":
                ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", linewidth=2, linestyle="--"))
                ax.text((x0 + x1) / 2, y + 0.015, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
            else:
                ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", linewidth=1.5))
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
        self.my_score = 0
        self.peer_score = 0
        self.expected_peer_seq = 0
        self.last_peer_ack = 0
        
        # Seq tracking
        self.my_next_seq = 0
        self.last_sent_seq = 0 
        
        self.my_last_ack_sent = 0
        self.my_rwnd = 50
        self.last_advertised_rwnd = 50 
        self.my_turn = False
        self.last_action_time = time.monotonic()
        self.last_rwnd_zero_sent_time: Optional[float] = None
        self.last_acks_received: List[int] = []
        self.lock = threading.Lock()

    def award_me(self, points: int):
        with self.lock:
            self.my_score += points

    def award_peer(self, points: int):
        with self.lock:
            self.peer_score += points
            
    def punish_me(self, points: int):
        with self.lock:
            self.my_score -= points

    def punish_peer(self, points: int):
        with self.lock:
            self.peer_score -= points

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

    def set_last_advertised_rwnd(self, v: int):
        with self.lock:
            self.last_advertised_rwnd = max(0, int(v))

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
        now = time.monotonic()
        msgs = []
        with self.lock:
            if self.my_turn:
                if now - self.last_action_time > REACTION_PENALTY_SEC:
                    self.my_score -= 1
                    self.last_action_time = now
                    msgs.append(f"[TIMEOUT] -1 Point to ME (No action in {REACTION_PENALTY_SEC}s).")

            if self.last_rwnd_zero_sent_time is not None:
                if now - self.last_rwnd_zero_sent_time > REACTION_PENALTY_SEC:
                    self.my_score -= 1
                    self.last_rwnd_zero_sent_time = now
                    msgs.append(f"[TIMEOUT] -1 Point to ME (rwnd=0 deadlock).")
        return msgs

    def validate_incoming_logic(self, pkt: Packet) -> Tuple[bool, str]:
        if pkt.ptype == "ERROR":
            return True, "ERROR received"

        if pkt.seq is None or pkt.ack is None or pkt.rwnd is None or pkt.length is None:
            return False, "Missing fields"
        if pkt.seq < 0 or pkt.ack < 0 or pkt.rwnd < 0 or pkt.length < 0:
            return False, "Negative values not allowed"

        with self.lock:
            expected = self.expected_peer_seq
            current_advertised = self.last_advertised_rwnd
            max_ack = self.my_next_seq
            last_ack_sent = self.my_last_ack_sent

        # --- SEQ CHECK WITH GBN SUPPORT ---
        if pkt.seq != expected:
            # FIX: If incoming seq matches what we ASKED for (my_last_ack_sent),
            # it is a VALID retransmission.
            if pkt.seq == last_ack_sent:
                 pass 
            else:
                 return False, f"Invalid seq: expected {expected}, got {pkt.seq}"

        if pkt.ack > max_ack:
            return False, f"Invalid ack: ack {pkt.ack} > my_next_seq {max_ack}"

        # --- FLOW CONTROL CHECK ---
        if current_advertised == 0 and pkt.length > 0:
            return False, f"Flow Control Violation: Sent {pkt.length} bytes while my advertised rwnd=0"
        
        if current_advertised > 0 and pkt.length > current_advertised:
             return False, f"Flow Control Violation: len {pkt.length} > advertised rwnd {current_advertised}"

        return True, "OK"

    def accept_incoming(self, pkt: Packet):
        if pkt.ptype != "DATA":
            return
        with self.lock:
            # FIX: GBN Resync
            if pkt.seq != self.expected_peer_seq and pkt.seq == self.my_last_ack_sent:
                print(f"   >>> [SYNC] GBN Retransmission accepted. Resyncing sequence.")
                self.expected_peer_seq = pkt.seq

            self.expected_peer_seq += pkt.length
            self.last_peer_ack = pkt.ack
            self.last_acks_received.append(pkt.ack)
            if len(self.last_acks_received) > 4:
                self.last_acks_received = self.last_acks_received[-4:]

    def rollback_seq(self):
        with self.lock:
            if self.my_next_seq > self.last_sent_seq:
                self.my_next_seq = self.last_sent_seq


# -----------------------------
# Networking helpers
# -----------------------------
def send_packet(sock: socket.socket, pkt: Packet):
    data = pkt.to_json().encode("utf-8")
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("Packet too large")
    header = len(data).to_bytes(4, "big")
    sock.sendall(header + data)

def recv_packet(sock: socket.socket) -> Optional[Packet]:
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
    # We no longer show suggestions in the prompt.
    print("\n--- YOUR TURN ---")
    print("Type one of: data, error, show, quit")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "show":
            with state.lock:
                print(f"[STATE] my_score={state.my_score} peer_score={state.peer_score} "
                      f"my_next_seq={state.my_next_seq} expected_peer_seq={state.expected_peer_seq} "
                      f"my_rwnd(internal)={state.my_rwnd} last_advertised_rwnd={state.last_advertised_rwnd}")
            continue
        if cmd == "quit":
            raise KeyboardInterrupt
        if cmd == "error":
            return Packet(ptype="ERROR", note="Human sent ERROR", ts=time.monotonic())
        if cmd == "data":
            break
        print("Unknown/Invalid. Try 'data'.")

    # Modified ask_int: No default value shown or accepted. Must type.
    def ask_int(prompt: str) -> int:
        while True:
            s = input(f"{prompt}: ").strip()
            if s == "":
                print("Please enter a number.")
                continue
            try:
                return int(s)
            except ValueError:
                print("Enter an integer.")

    seq = ask_int("seq")
    ack = ask_int("ack")
    rwnd = ask_int("rwnd")
    length = ask_int("length")
    note = input("note (optional): ").strip()
    
    return Packet(ptype="DATA", seq=seq, ack=ack, rwnd=rwnd, length=length, note=note, ts=time.monotonic())


def maybe_double_ack_hint(state: GameState) -> Optional[str]:
    # Trigger only if 3 identical ACKs in a row
    with state.lock:
        acks = list(state.last_acks_received)
    
    if len(acks) >= 3 and (acks[-1] == acks[-2] == acks[-3]):
        return f"[TRIPLE-ACK] Received duplicate ACK={acks[-1]} 3 times. Hint: Go-Back-N?"
    
    return None


def run_game(sock: socket.socket, my_name: str, peer_name: str, i_start: bool):
    state = GameState(my_name=my_name, peer_name=peer_name)
    recorder = GraphRecorder(my_name, peer_name)
    state.set_turn(i_start)
    stop_flag = threading.Event()

    def rwnd_worker():
        while not stop_flag.is_set():
            time.sleep(RWND_AUTO_INCREASE_EVERY_SEC)
            state.apply_rwnd_auto_increase()

    def penalty_worker():
        while not stop_flag.is_set():
            time.sleep(1)
            msgs = state.check_penalties()
            for m in msgs:
                print(m)

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

    print(f"\nGame Started: {my_name} vs {peer_name}")
    print("---------------------------------------")
    game_start = time.monotonic()
    graph_path = f"tcp_game_{my_name}_vs_{peer_name}.png"

    try:
        while not stop_flag.is_set():
            if time.monotonic() - game_start >= GAME_DURATION_SEC:
                print("\n[GAME] Time is up.")
                break

            pkt_in = None
            with incoming_lock:
                if incoming_queue:
                    pkt_in = incoming_queue.pop(0)

            if pkt_in is not None:
                # 1. Internal Logic Check
                is_valid, reason = state.validate_incoming_logic(pkt_in)

                # --- HANDLING ERROR PACKETS ---
                if pkt_in.ptype == "ERROR":
                    print(f"\n[RECV] ERROR from peer: {pkt_in.note}")
                    state.rollback_seq()
                    print(f"   >>> [ROLLBACK] Last packet rejected. Sequence reset to {state.my_next_seq}.")

                    note_lower = pkt_in.note.lower()
                    if "rwnd=0" in note_lower and "violation" in note_lower:
                        state.punish_me(1)
                        print(f"   >>> [SCORE UPDATE] I LOST 1 point (Peer caught my rwnd=0 violation).")
                    else:
                        state.award_peer(1)
                        print(f"   >>> [SCORE UPDATE] Peer gained +1 (Standard Detection).")
                    
                    print(f"   [SCORE BOARD] ME: {state.my_score} | PEER: {state.peer_score}")
                    state.set_turn(True)
                    continue

                # --- HANDLING DATA PACKETS ---
                print(f"\n[RECV] DATA: seq={pkt_in.seq} ack={pkt_in.ack} rwnd={pkt_in.rwnd} len={pkt_in.length} | {pkt_in.note}")
                
                print("   [DECISION] Validate this packet manually (Check Seq, Ack, Rwnd, Length).")
                while True:
                    choice = input("   Action? (a)ccept / (e)rror: ").strip().lower()
                    if choice in ("a", "e"):
                        break
                
                if choice == "e":
                    error_note = f"Refused by user. Reason: {reason}"
                    pkt_err = Packet(ptype="ERROR", note=error_note, ts=time.monotonic())
                    send_packet(sock, pkt_err)
                    recorder.record_out(pkt_err)
                    
                    if not is_valid:
                        if "advertised rwnd=0" in reason:
                             state.punish_peer(1)
                             print(f"   >>> [SCORE UPDATE] Peer LOST 1 point (rwnd=0 Violation Caught!).")
                        else:
                             state.award_me(1)
                             print(f"   >>> [SCORE UPDATE] You gained +1 (Correct Detection!).")
                    else:
                        print(f"   [INFO] You sent ERROR for a valid packet. No points changed.")
                    
                    state.update_action()
                    state.set_turn(False)

                else: # choice == "a"
                    if not is_valid:
                        state.award_peer(1)
                        print(f"   >>> [SCORE UPDATE] PEER gained +1 (You missed the error!).")
                        print(f"   [MISTAKE] Packet was actually IMPOSSIBLE: {reason}")
                    else:
                        print("   [OK] Packet accepted.")

                    state.accept_incoming(pkt_in)
                    state.set_turn(True)

                print(f"   [SCORE BOARD] ME: {state.my_score} | PEER: {state.peer_score}")

                hint = maybe_double_ack_hint(state)
                if hint: print(f"   {hint}")

            with state.lock:
                my_turn = state.my_turn
                
            if my_turn:
                pkt_out = interactive_turn_input(state)

                if pkt_out.ptype == "DATA":
                    if pkt_out.rwnd == 0:
                        state.mark_sent_rwnd_zero()
                    else:
                        state.clear_sent_rwnd_zero()

                    state.set_my_rwnd(pkt_out.rwnd)
                    state.set_last_advertised_rwnd(pkt_out.rwnd)

                    with state.lock:
                        state.last_sent_seq = pkt_out.seq
                        if pkt_out.seq == state.my_next_seq:
                            state.my_next_seq += pkt_out.length
                        state.my_last_ack_sent = pkt_out.ack

                send_packet(sock, pkt_out)
                recorder.record_out(pkt_out)

                if pkt_out.ptype == "DATA":
                    print(f"[SEND] DATA: seq={pkt_out.seq} ack={pkt_out.ack} rwnd={pkt_out.rwnd} len={pkt_out.length} | {pkt_out.note}")
                else:
                    print(f"[SEND] ERROR sent.")

                state.update_action()
                state.set_turn(False)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[GAME] Interrupted.")
    finally:
        stop_flag.set()
        try: sock.shutdown(socket.SHUT_RDWR); sock.close()
        except: pass
        try: recorder.save_png(graph_path); print(f"[GRAPH] Saved to {graph_path}")
        except: pass
        with state.lock: print(f"[FINAL] ME: {state.my_score} | PEER: {state.peer_score}")

# -----------------------------
# Connection setup
# -----------------------------
def run_listener(host: str, port: int) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[LISTEN] {host}:{port}")
    conn, addr = srv.accept()
    srv.close()
    return conn

def run_connector(host: str, port: int) -> socket.socket:
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        cli.connect((host, port))
        print("[CONNECT] Connected.")
        return cli
    except ConnectionRefusedError:
        print("Connection failed. Ensure listener is running.")
        raise SystemExit

def handshake(sock: socket.socket, my_name: str) -> str:
    pkt = Packet(ptype="DATA", seq=0, ack=0, rwnd=50, length=0, note=f"NAME:{my_name}", ts=time.monotonic())
    send_packet(sock, pkt)
    peer_pkt = recv_packet(sock)
    if peer_pkt and peer_pkt.note.startswith("NAME:"):
        return peer_pkt.note.split("NAME:", 1)[1].strip()
    return "PEER"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--connect", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--name", default="Player")
    ap.add_argument("--start", action="store_true")
    args = ap.parse_args()

    if args.listen == args.connect:
        print("Use exactly one: --listen or --connect")
        raise SystemExit(2)

    sock = run_listener(args.host, args.port) if args.listen else run_connector(args.host, args.port)
    peer_name = handshake(sock, args.name)
    i_start = True if args.start else (True if args.listen else False)
    run_game(sock, args.name, peer_name, i_start)

if __name__ == "__main__":
    main()