import time
import random
import ipaddress
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional but handy for PKI demo (toy CA, certs, signatures)
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# -----------------------------
# App Title & Layout
# -----------------------------
st.set_page_config(page_title="GRIDShield Live — Grid Security Lab", layout="wide")
st.title("🛡️ GRIDShield Live — Interactive Grid Security Lab")
st.caption("Security in Grid (FI1944) • No datasets • No ML • Fully interactive simulation")

# -----------------------------
# Helper Data Structures
# -----------------------------
PROTO = ["TCP", "UDP", "ICMP"]

@dataclass
class Rule:
    proto: Optional[str]  # None = wildcard
    src_prefix: Optional[ipaddress.IPv4Network]  # None = wildcard
    dst_prefix: Optional[ipaddress.IPv4Network]  # None = wildcard
    sport_range: Optional[Tuple[int, int]]       # None = wildcard
    dport_range: Optional[Tuple[int, int]]       # None = wildcard
    action: str = "ALLOW"                        # ALLOW / DENY / PRIORITIZE
    priority: int = 100                          # lower = higher priority


@dataclass
class Packet:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: str
    length: int
    is_attack: bool


# -----------------------------
# Session State Initialization
# -----------------------------
def init_state():
    if "rules" not in st.session_state:
        st.session_state.rules: List[Rule] = []
    if "packets" not in st.session_state:
        st.session_state.packets: List[Packet] = []
    if "flow_state" not in st.session_state:
        # per-source statistics for DDoS + token bucket
        st.session_state.flow_state: Dict[str, Dict[str, Any]] = {}
    if "ca" not in st.session_state:
        st.session_state.ca = None
    if "server_creds" not in st.session_state:
        st.session_state.server_creds = None
    if "client_creds" not in st.session_state:
        st.session_state.client_creds = None

init_state()


# -----------------------------
# Utility Generators
# -----------------------------
def random_subnet(prefix: int = 24) -> ipaddress.IPv4Network:
    base = ipaddress.IPv4Address(random.randint(0x0B000000, 0xDF000000))  # avoid reserved extremes
    network = ipaddress.IPv4Network((int(base) & (~((1 << (32 - prefix)) - 1)), prefix))
    return network


def gen_rule_set(n_rules: int, seed: int = 42) -> List[Rule]:
    rng = random.Random(seed)
    rules = []
    for i in range(n_rules):
        use_proto = rng.choice([True, False])
        proto = rng.choice(PROTO) if use_proto else None

        use_src = rng.choice([True, False])
        src = random_subnet(rng.choice([16, 20, 24])) if use_src else None

        use_dst = rng.choice([True, False])
        dst = random_subnet(rng.choice([16, 20, 24])) if use_dst else None

        use_sport = rng.choice([True, False])
        if use_sport:
            a = rng.randint(1, 65500)
            b = min(65535, a + rng.randint(0, 2000))
            sport = (a, b)
        else:
            sport = None

        use_dport = rng.choice([True, False])
        if use_dport:
            a = rng.randint(1, 65500)
            b = min(65535, a + rng.randint(0, 2000))
            dport = (a, b)
        else:
            dport = None

        action = rng.choice(["ALLOW", "DENY", "PRIORITIZE"])
        priority = rng.randint(1, 100)
        rules.append(Rule(proto, src, dst, sport, dport, action, priority))
    rules.sort(key=lambda r: r.priority)  # simple priority order
    return rules


def gen_ip(net: Optional[ipaddress.IPv4Network]) -> str:
    if net is None:
        return str(ipaddress.IPv4Address(random.randint(0x0B000000, 0xDF000000)))
    host_bits = 32 - net.prefixlen
    if host_bits == 0:
        return str(net.network_address)
    rnd = random.getrandbits(host_bits)
    return str(ipaddress.IPv4Address(int(net.network_address) + rnd))


def gen_packet(benign_src_pool, benign_dst_pool, attack_mode: bool = False) -> Packet:
    proto = random.choice(PROTO)
    if attack_mode:
        # Concentrate destination and ports to simulate volumetric or app-layer floods
        dst = benign_dst_pool[0]
        dport = random.choice([80, 443, 22, 53])
        src = str(ipaddress.IPv4Address(random.randint(0x0B000000, 0xDF000000)))
        sport = random.randint(1024, 65535)
        length = random.choice([64, 512, 1500])
    else:
        src = random.choice(benign_src_pool)
        dst = random.choice(benign_dst_pool)
        sport = random.randint(1024, 65535)
        dport = random.choice([80, 443, 22, 53, 8080, 5000, random.randint(1024, 65535)])
        length = random.choice([64, 128, 256, 512, 1200, 1500])
    return Packet(time.time(), src, dst, sport, dport, proto, length, attack_mode)


# -----------------------------
# Match Logic (for rules)
# -----------------------------
def ip_in_prefix(ip: str, pref: Optional[ipaddress.IPv4Network]) -> bool:
    if pref is None:
        return True
    return ipaddress.ip_address(ip) in pref


def port_in_range(port: int, r: Optional[Tuple[int, int]]) -> bool:
    if r is None:
        return True
    return r[0] <= port <= r[1]


def match_rule(pkt: Packet, rule: Rule) -> bool:
    if rule.proto is not None and pkt.proto != rule.proto:
        return False
    if not ip_in_prefix(pkt.src, rule.src_prefix):
        return False
    if not ip_in_prefix(pkt.dst, rule.dst_prefix):
        return False
    if not port_in_range(pkt.sport, rule.sport_range):
        return False
    if not port_in_range(pkt.dport, rule.dport_range):
        return False
    return True


# -----------------------------
# Packet Classification Taxonomy (Unit II & III)
# Implementations: Linear Search, Tuple Space Search, Decision-Tree-like (HiCuts-inspired)
# -----------------------------
def classify_linear(pkt: Packet, rules: List[Rule]) -> Rule:
    for r in rules:
        if match_rule(pkt, r):
            return r
    return Rule(None, None, None, None, None, action="ALLOW", priority=9999)  # default


def tuple_key(rule: Rule) -> Tuple[int, int, int, int, int]:
    # Count specificity: specified fields -> group into "tuple space"
    return (
        0 if rule.proto is None else 1,
        0 if rule.src_prefix is None else 1,
        0 if rule.dst_prefix is None else 1,
        0 if rule.sport_range is None else 1,
        0 if rule.dport_range is None else 1,
    )


def build_tuple_space(rules: List[Rule]) -> Dict[Tuple[int, int, int, int, int], List[Rule]]:
    ts: Dict[Tuple[int, int, int, int, int], List[Rule]] = {}
    for r in rules:
        k = tuple_key(r)
        ts.setdefault(k, []).append(r)
    # within each tuple, keep priority order
    for k in ts:
        ts[k].sort(key=lambda r: r.priority)
    return ts


def classify_tuple_space(pkt: Packet, ts: Dict[Tuple[int, int, int, int, int], List[Rule]]) -> Rule:
    # Search more specific tuples first (descending by sum)
    for spec in sorted(ts.keys(), key=lambda k: -sum(k)):
        for r in ts[spec]:
            if match_rule(pkt, r):
                return r
    return Rule(None, None, None, None, None, action="ALLOW", priority=9999)


class HiCutsNode:
    def __init__(self, rules: List[Rule], depth: int = 0, max_rules_per_node: int = 16, max_depth: int = 10):
        self.rules = rules
        self.children = []
        self.depth = depth
        self.field = None
        self.cuts = []
        self.leaf = True
        if depth < max_depth and len(rules) > max_rules_per_node:
            self.leaf = False
            # Choose a field to cut: prefer dst port -> src port -> dst prefix
            self.field = ["dport", "sport", "dst"][depth % 3]
            self._split(max_rules_per_node, max_depth)

    def _split(self, max_rules_per_node: int, max_depth: int):
        if self.field in ["dport", "sport"]:
            # Cut port space into buckets
            buckets = [(0, 65535)]
            cuts = 4
            step = 65536 // cuts
            self.cuts = [(i * step, min(65535, (i + 1) * step - 1)) for i in range(cuts)]
            for lo, hi in self.cuts:
                slice_rules = []
                for r in self.rules:
                    rng = r.dport_range if self.field == "dport" else r.sport_range
                    if rng is None or not (rng[1] < lo or rng[0] > hi):
                        slice_rules.append(r)
                self.children.append(HiCutsNode(slice_rules, self.depth + 1, max_rules_per_node, max_depth))
        else:
            # dst IP space — split by /8 chunks
            self.cuts = [ipaddress.IPv4Network(f"{i}.0.0.0/8") for i in range(0, 256, 64)]
            for block in self.cuts:
                slice_rules = []
                for r in self.rules:
                    if r.dst_prefix is None or (r.dst_prefix.network_address >= block.network_address and r.dst_prefix.network_address <= block.broadcast_address):
                        slice_rules.append(r)
                self.children.append(HiCutsNode(slice_rules, self.depth + 1, max_rules_per_node, max_depth))

    def search(self, pkt: Packet) -> Rule:
        if self.leaf:
            # Linear within node
            for r in sorted(self.rules, key=lambda rr: rr.priority):
                if match_rule(pkt, r):
                    return r
            return Rule(None, None, None, None, None, action="ALLOW", priority=9999)
        # choose child based on field
        if self.field in ["dport", "sport"]:
            key = pkt.dport if self.field == "dport" else pkt.sport
            for (lo, hi), child in zip(self.cuts, self.children):
                if lo <= key <= hi:
                    return child.search(pkt)
        else:
            ipi = int(ipaddress.IPv4Address(pkt.dst))
            for block, child in zip(self.cuts, self.children):
                if int(block.network_address) <= ipi <= int(block.broadcast_address):
                    return child.search(pkt)
        # fallback
        return Rule(None, None, None, None, None, action="ALLOW", priority=9999)


# -----------------------------
# DDoS Detection: Change Aggregation Tree (CAT) + Global Adaptive + ALPi-style throttling
# -----------------------------
def cat_detect(count_series: List[int], win: int = 10, thresh: float = 3.0) -> bool:
    """
    Very simple change detector:
    - Uses moving average and std over last 'win' windows.
    - Triggers if latest count exceeds mean + thresh*std.
    """
    if len(count_series) < win + 1:
        return False
    recent = np.array(count_series[-(win+1):-1])
    mu, sd = recent.mean(), recent.std() if recent.std() > 0 else 1.0
    return count_series[-1] > mu + thresh * sd


def alpi_throttle(flow_state: Dict[str, Dict[str, Any]], fair_share: int, now: float):
    """
    ALPi-inspired: identify high-rate offenders and increase their drop probability.
    flow_state[src] = {rate, drop_p, tokens, last_ts}
    """
    for src, s in flow_state.items():
        rate = s.get("rate", 0)
        # If above fair share, bump drop probability; else decay it
        if rate > fair_share:
            s["drop_p"] = min(0.9, s.get("drop_p", 0.0) + 0.1)
        else:
            s["drop_p"] = max(0.0, s.get("drop_p", 0.0) - 0.05)
        # Token bucket refill (Active NIC emulation)
        capacity = s.get("capacity", 10000)
        fill_rate = s.get("fill_rate", fair_share)  # bytes/sec per source
        dt = max(0.0, now - s.get("last_ts", now))
        s["tokens"] = min(capacity, s.get("tokens", capacity) + fill_rate * dt)
        s["last_ts"] = now


def nic_admit(pkt: Packet, flow_state: Dict[str, Dict[str, Any]]) -> bool:
    """
    Active Network Interface admission control: per-source token bucket + drop probability.
    """
    s = flow_state.setdefault(pkt.src, {
        "tokens": 10000.0, "capacity": 10000.0, "fill_rate": 8000.0, "last_ts": time.time(), "rate": 0.0, "drop_p": 0.0
    })
    # Random early drop based on ALPi drop probability
    if random.random() < s.get("drop_p", 0.0):
        return False
    # Token bucket
    if s["tokens"] >= pkt.length:
        s["tokens"] -= pkt.length
        return True
    return False


# -----------------------------
# PKI / TLS / Kerberos Demos (Unit I)
# -----------------------------
def create_ca():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return {"key": private_key}

def issue_cert(ca, common_name: str):
    # For demo, we store only keys; no full X.509 to keep deps light.
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    # "Sign" public key hash with CA key as a toy certificate
    pub_bytes = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    signature = ca["key"].sign(pub_bytes, padding.PKCS1v15(), hashes.SHA256())
    return {"key": key, "pub": pub_bytes, "sig": signature, "cn": common_name}

def verify_cert(ca, cert) -> bool:
    try:
        ca["key"].public_key().verify(cert["sig"], cert["pub"], padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False

def tls_handshake_demo(client_cert, server_cert) -> Dict[str, str]:
    # toy handshake transcript
    transcript = {
        "ClientHello": "ciphers=[TLS_AES_128_GCM_SHA256] random=C1",
        "ServerHello": "cipher=TLS_AES_128_GCM_SHA256 random=S1",
        "Certificate": f"server_cert(CN={server_cert['cn']})",
        "CertificateVerify": "sig_over_handshake",
        "Finished": "verify_data",
        "AppData": "secure_channel_established"
    }
    return transcript

def kerberos_demo() -> List[str]:
    return [
        "C → AS: {IDc, IDtgs, TS1}",
        "AS → C: {ticket_tgs, Kc_tgs} encrypted with Kc",
        "C → TGS: {ticket_tgs, authenticator} using Kc_tgs",
        "TGS → C: {ticket_v, Kc_v} encrypted with Kc_tgs",
        "C → V: {ticket_v, authenticator} using Kc_v",
        "V → C: {TS} confirmation"
    ]


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    n_rules = st.number_input("Number of rules", min_value=10, max_value=2000, value=200, step=10)
    pkt_burst = st.number_input("Packets per burst", min_value=10, max_value=20000, value=1000, step=50)
    attack_ratio = st.slider("Attack traffic ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    fair_share = st.number_input("Fair share (bytes/sec per source)", min_value=1000, max_value=50000, value=8000, step=1000)
    ddos_window = st.number_input("CAT window", min_value=5, max_value=50, value=10, step=1)
    ddos_thresh = st.slider("CAT threshold (σ)", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
    algo = st.selectbox("Classifier", ["Linear Search", "Tuple Space Search", "Decision-Tree (HiCuts-like)"])
    if st.button("🔁 Regenerate rule set"):
        st.session_state.rules = gen_rule_set(n_rules, seed=random.randint(0, 999999))

# Ensure rules exist
if not st.session_state.rules or len(st.session_state.rules) != n_rules:
    st.session_state.rules = gen_rule_set(n_rules)

# Precompute structures for algorithms
tuple_space = build_tuple_space(st.session_state.rules)
hicut_root = HiCutsNode(st.session_state.rules, max_rules_per_node=32, max_depth=8)

# -----------------------------
# Tabs
# -----------------------------
tab_pkt, tab_ddos, tab_pki, tab_gridnic = st.tabs([
    "📦 Packet Classification", "🌊 DDoS Detector & Mitigator", "🔐 PKI / TLS / Kerberos", "🔧 Grid Data Transfer (Active NIC)"
])

# -----------------------------
# Packet Classification Tab
# -----------------------------
with tab_pkt:
    st.subheader("Packet Classification — Taxonomy in Action")
    st.write("Compare **Linear Search**, **Tuple Space Search**, and a **Decision-Tree (HiCuts-like)** classifier on synthetic traffic and rules.")

    # Build benign pools
    benign_src_net = random_subnet(16)
    benign_dst_net = random_subnet(16)
    benign_src_pool = [gen_ip(benign_src_net) for _ in range(64)]
    benign_dst_pool = [gen_ip(benign_dst_net) for _ in range(64)]

    # Generate a burst
    if st.button("🚀 Generate & Classify Burst"):
        packets = []
        for _ in range(int(pkt_burst)):
            attack = random.random() < attack_ratio
            packets.append(gen_packet(benign_src_pool, benign_dst_pool, attack_mode=attack))

        # Benchmark
        t0 = time.perf_counter()
        if algo == "Linear Search":
            labels = [classify_linear(p, st.session_state.rules).action for p in packets]
        elif algo == "Tuple Space Search":
            labels = [classify_tuple_space(p, tuple_space).action for p in packets]
        else:
            labels = [hicut_root.search(p).action for p in packets]
        t1 = time.perf_counter()

        df = pd.DataFrame([p.__dict__ for p in packets])
        df["decision"] = labels
        st.session_state.packets = packets  # store latest
        st.success(f"Classified {len(packets)} packets in {1e3*(t1-t0):.2f} ms using **{algo}**.")
        st.dataframe(df.head(20), use_container_width=True)
        agg = df.groupby(["decision", "is_attack"]).size().reset_index(name="count")
        st.bar_chart(agg.pivot(index="decision", columns="is_attack", values="count").fillna(0))

    with st.expander("What’s happening (Unit II & III mapping)?", expanded=False):
        st.markdown("""
- **Exhaustive/Linear Search** → scans rules in priority order.
- **Tuple Space Search** → groups rules by which fields are specified (proto/src/dst/sport/dport) and searches most specific tuples first.
- **Decision-Tree (HiCuts-like)** → recursively cuts the header space (ports/IP blocks) to minimize rules per leaf, then linearly matches inside the leaf.
- You can regenerate rules to see how structure impacts performance.
        """)

# -----------------------------
# DDoS Tab
# -----------------------------
with tab_ddos:
    st.subheader("DDoS Detection & Mitigation — CAT + Global Adaptive + ALPi-style")
    st.write("Creates synthetic traffic bursts; detects changes with a **Change Aggregation Tree**-style test; adapts per-source drops and fair-share token buckets (ALPi-inspired).")

    if "ddos_counts" not in st.session_state:
        st.session_state.ddos_counts = []  # total packets each burst
        st.session_state.ddos_attack_counts = []  # attack packets each burst

    if st.button("🌊 Run One Burst (DDoS Scenario)"):
        packets = []
        for _ in range(int(pkt_burst)):
            attack = random.random() < attack_ratio
            p = gen_packet(["10.0.1.1"], ["172.16.0.10"], attack_mode=attack)
            packets.append(p)

        # Update per-source rate (bytes/sec approx over a pseudo 1-second burst)
        for p in packets:
            s = st.session_state.flow_state.setdefault(p.src, {"rate": 0.0, "tokens": 10000.0, "capacity": 10000.0, "fill_rate": float(fair_share), "drop_p": 0.0, "last_ts": time.time()})
            s["rate"] = s.get("rate", 0.0) + p.length

        st.session_state.ddos_counts.append(len(packets))
        st.session_state.ddos_attack_counts.append(sum(1 for p in packets if p.is_attack))

        # Detection via CAT-like change test
        alert = cat_detect(st.session_state.ddos_attack_counts, win=int(ddos_window), thresh=float(ddos_thresh))

        # Global Adaptive + ALPi throttle update
        alpi_throttle(st.session_state.flow_state, fair_share=int(fair_share), now=time.time())

        admitted = 0
        for p in packets:
            if nic_admit(p, st.session_state.flow_state):
                admitted += 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Burst Size", len(packets))
        with col2:
            st.metric("Attack Packets", sum(1 for p in packets if p.is_attack))
        with col3:
            st.metric("Admitted by NIC", admitted)

        if alert:
            st.error("🚨 CAT Detector: **Anomalous surge detected** — mitigation active (higher drop probabilities for heavy sources).")
        else:
            st.success("✅ CAT Detector: No anomaly.")

        # Charts
        hist = pd.DataFrame({
            "bursts": np.arange(len(st.session_state.ddos_attack_counts)),
            "attack_packets": st.session_state.ddos_attack_counts,
            "total_packets": st.session_state.ddos_counts
        })
        st.line_chart(hist.set_index("bursts")[["attack_packets", "total_packets"]])

        # Show top offenders by drop probability
        offenders = sorted([(src, s.get("rate", 0), s.get("drop_p", 0.0)) for src, s in st.session_state.flow_state.items()],
                           key=lambda x: (-x[2], -x[1]))[:10]
        st.write("**Top sources by drop probability (ALPi-style):**")
        st.dataframe(pd.DataFrame(offenders, columns=["src", "byte_rate_last_burst", "drop_probability"]), use_container_width=True)

    with st.expander("What’s happening (Unit IV mapping)?", expanded=False):
        st.markdown("""
- **Detection Techniques**: the simple CAT-like test flags sudden changes in attack volume per burst.
- **Global Adaptive Defense**: adjusts source behaviors globally each burst.
- **ALPi-style**: raises **drop probability** for heavy hitters (above fair share) and decays for compliant sources.
- **Change Aggregation Tree** concept is demonstrated as a statistical change detector across bursts.
        """)

# -----------------------------
# Grid Data Transfer (Active NIC) Tab
# -----------------------------
with tab_gridnic:
    st.subheader("Protecting Grid Data Transfer Service — Active Network Interface")
    st.write("Per-source **token buckets** and fairness emulate an active NIC protecting bulk transfers against floods.")

    demo_sources = [f"192.0.2.{i}" for i in range(1, 8)]
    sizes = [random.choice([512, 1200, 1500]) for _ in range(500)]
    admitted_by_src = {s: 0 for s in demo_sources}
    offered_by_src = {s: 0 for s in demo_sources}

    if st.button("📦 Simulate 1-Second Grid Transfer Window"):
        now = time.time()
        # Prepare state for these demo sources
        for s in demo_sources:
            st.session_state.flow_state.setdefault(s, {"rate": 0.0, "tokens": 8000.0, "capacity": 12000.0, "fill_rate": float(fair_share), "drop_p": 0.0, "last_ts": now})
            st.session_state.flow_state[s]["rate"] = 0.0  # reset instantaneous rate

        for _ in range(500):
            src = random.choice(demo_sources)
            pkt = Packet(now, src, "198.51.100.10", random.randint(1024, 65535), 5001, "TCP", random.choice(sizes), False)
            offered_by_src[src] += pkt.length
            st.session_state.flow_state[src]["rate"] += pkt.length
            if nic_admit(pkt, st.session_state.flow_state):
                admitted_by_src[src] += pkt.length

        alpi_throttle(st.session_state.flow_state, fair_share=int(fair_share), now=now)

        df = pd.DataFrame({
            "src": demo_sources,
            "offered_bytes": [offered_by_src[s] for s in demo_sources],
            "admitted_bytes": [admitted_by_src[s] for s in demo_sources],
            "drop_probability": [st.session_state.flow_state[s]["drop_p"] for s in demo_sources]
        }).set_index("src")
        st.bar_chart(df[["offered_bytes", "admitted_bytes"]])
        st.dataframe(df, use_container_width=True)

    with st.expander("What’s happening (Unit V / Active NIC idea)?", expanded=False):
        st.markdown("""
- **Active network interface** enforces **token buckets** per source → protects bulk data transfer from overload.
- Fairness emerges because heavy sources **consume tokens** faster and face higher **drop probability** when aggressive (ALPi-style feedback).
        """)

# -----------------------------
# PKI / TLS / Kerberos Tab
# -----------------------------
with tab_pki:
    st.subheader("PKI • TLS/SSL • Kerberos — Hands-on Mini Demos (Unit I)")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏛️ Create CA"):
            st.session_state.ca = create_ca()
            st.success("Root CA created.")
    with c2:
        if st.button("🖥️ Issue Server Cert"):
            if not st.session_state.ca:
                st.warning("Create CA first.")
            else:
                st.session_state.server_creds = issue_cert(st.session_state.ca, "grid-server")
                st.success("Server certificate issued (toy).")
    with c3:
        if st.button("👤 Issue Client Cert"):
            if not st.session_state.ca:
                st.warning("Create CA first.")
            else:
                st.session_state.client_creds = issue_cert(st.session_state.ca, "grid-client")
                st.success("Client certificate issued (toy).")

    if st.session_state.server_creds and st.session_state.ca:
        ok = verify_cert(st.session_state.ca, st.session_state.server_creds)
        st.info(f"Server cert verified by CA: **{ok}**")

    if st.session_state.client_creds and st.session_state.server_creds:
        hs = tls_handshake_demo(st.session_state.client_creds, st.session_state.server_creds)
        st.write("**TLS 1.3-like toy handshake transcript:**")
        st.json(hs)

    st.write("**Kerberos flow (toy trace):**")
    st.code("\n".join(kerberos_demo()))

    with st.expander("How this maps to Unit I & GSI concepts", expanded=False):
        st.markdown("""
- **PKI**: Root CA issues certs; server/client verification mirrors trust establishment for secure grid services.
- **TLS/SSL**: Handshake transcript shows how secure channels form for **Generic Security Services**.
- **Kerberos**: Ticket-based authentication flow for service access across distributed grid nodes.
        """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("FI1944 • GRIDShield Live • Packet Classification • DDoS • Active NIC • PKI/TLS/Kerberos (no datasets, no ML).")
