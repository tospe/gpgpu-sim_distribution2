// CAM extension: one CAM engine unit at the array's home L2 sub-partition.
// Specification: cam_with_gpu docs/h100_cam_protocol.md (v1). All timing in
// GPU core cycles; every CAM parameter is a hypothetical architectural value.
//
// Unit behaviour, in order of arrival at the unit:
//  - UCAMQ chunks (posted) are staged per issuing warp as they arrive; they are
//    order-independent and are consumed even if an older command blocks.
//  - UCAMF/UCAMW and UCAMS are handled strictly in arrival order. A fill/write
//    occupies the unit until complete; its rows become valid (searchable) at
//    completion, then its ack returns. A search is admitted when its warp's
//    query is fully staged, fewer than max_outstanding searches are in
//    service, II has elapsed since the last start, and no fill/write is in
//    progress. It sees exactly the rows valid at admission.
//  - Results: first result selected at admission + L, then one per
//    topk_latency through a single readout port shared by all searches; a
//    packet of result_pkt_bytes leaves when its last result is selected.
//  - Replies are mem_fetch fragments; the last one completes the request at
//    the SM (see shader.cc, cam_complete).
#ifndef CAM_H
#define CAM_H

#include <stdint.h>
#include <stdio.h>
#include <deque>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>

class mem_fetch;
class memory_config;

// Functional contents: keys and queries from -gpgpu_cam_func_file.
// File layout (little endian): char magic[8] = "CAMFUNC1"; uint32 N, D, H,
// n_queries; uint16 keys[N][D] (FP16 bits); then n_queries records of
// {uint32 cta_x, warp_in_cta, seq; uint16 q[H][D] (FP16); float w[H]}.
class cam_functional {
 public:
  explicit cam_functional(const char *path);
  bool loaded() const { return m_loaded; }
  unsigned rows_in_file() const { return m_N; }
  // DSA score top-k over rows with valid[r] (FP32 accumulate, ties -> lower
  // id first, sorted by descending score).
  void search(unsigned cta_x, unsigned warp, unsigned seq,
              const std::vector<bool> &valid, unsigned k,
              std::vector<std::pair<int, float>> &out) const;

 private:
  bool m_loaded;
  unsigned m_N, m_D, m_H;
  std::vector<float> m_keys;  // N*D, converted from FP16
  std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> m_qindex;
  std::vector<float> m_q, m_w;  // per query: H*D, H
};

// Shared per-GPU log: search results with their completion time, fills,
// waits. Records are written only at completion (never early).
class cam_log {
 public:
  explicit cam_log(const char *path);
  ~cam_log();
  bool on() const { return m_fp != NULL; }
  FILE *fp() { return m_fp; }
  void stash(unsigned long long req_id,
             std::vector<std::pair<int, float>> &res) {
    m_pending[req_id].swap(res);
  }
  bool take(unsigned long long req_id, std::vector<std::pair<int, float>> &res);

 private:
  FILE *m_fp;
  std::map<unsigned long long, std::vector<std::pair<int, float>>> m_pending;
};

// Unit-wide query staging capacity (F5, spec §10c). One pool per unit, shared
// by all warps' staging areas; identical in both placements. A query reserves
// cam_query_bytes when its issuing warp issues the query's first UCAMQ chunk
// (the warp stalls if the reservation does not fit), so every chunk of an
// accepted query has room when it arrives. The reservation is released when
// the search is admitted: the query then moves into the engine's in-service
// storage (max_outstanding x query bytes, a separate fixed allocation).
// cap = 0: unbounded (legacy configurations). The credit is visible to the SM
// without delay (credit-return latency not modelled; optimistic).
struct cam_staging_pool {
  unsigned long long cap = 0, reserved = 0, arrived = 0, peak_reserved = 0,
                     peak_arrived = 0, n_reserve = 0, n_release = 0, n_wait = 0;
  bool can_reserve(unsigned long long q) const {
    return !cap || reserved + q <= cap;
  }
  void reserve(unsigned long long q) {
    reserved += q;
    n_reserve++;
    if (reserved > peak_reserved) peak_reserved = reserved;
  }
};

class cam_unit {
 public:
  cam_unit(const memory_config *config, unsigned sub_partition_id,
           const cam_functional *func, cam_log *log);
  bool full() const { return m_input.size() >= m_input_cap; }
  void push(mem_fetch *mf, unsigned long long now);
  void cycle(unsigned long long now);
  mem_fetch *top(unsigned long long now);  // next reply ready by now, or NULL
  // select results due by now (called before replies are taken each cycle)
  void readout_advance(unsigned long long now);
  void set_staging(cam_staging_pool *p) { m_pool = p; }
  void pop();
  bool busy() const;
  void print_stats(FILE *fp) const;
  // release time of the next reply (only valid if has_reply())
  bool has_reply() const { return !m_return.empty(); }
  // time-driven work: a fill/write in progress or readout that is not stalled
  // on the output buffer. Packets waiting for the reply port are not counted
  // (their progress shows as pops, see progress()).
  bool scheduled_work() const {
    return m_mutation || (!m_readout.empty() && !m_stalled);
  }
  // count of readout steps and packets leaving the unit (deadlock detection)
  unsigned long long progress() const { return m_progress; }
  unsigned long long next_reply_time() const { return m_return.top().ready; }

 private:
  struct timed_mf {
    unsigned long long ready;
    unsigned long long seq;  // FIFO tie-break
    mem_fetch *mf;
    bool operator>(const timed_mf &o) const {
      return ready != o.ready ? ready > o.ready : seq > o.seq;
    }
  };
  void emit(mem_fetch *mf, unsigned long long ready);
  void start_search(mem_fetch *mf, unsigned long long now);
  struct readout_t {
    mem_fetch *cmd = NULL;  // the search command (kept for packet formation)
    unsigned long long req_id = 0, admit = 0, earliest = 0, t_first = 0,
                       t_next = 0;
    unsigned k = 1, next = 0, in_pkt = 0, frag = 0;
    bool started = false;
  };
  std::deque<readout_t> m_readout;
  unsigned m_n_in_service = 0;
  unsigned long long m_out_bytes = 0, m_out_cap = 0;  // output buffer (F2)
  bool m_stalled = false;
  cam_staging_pool *m_pool = NULL;
  unsigned long long m_progress = 0;
  unsigned long long m_stall_since = 0;

  const memory_config *m_config;
  unsigned m_id;
  const cam_functional *m_func;
  cam_log *m_log;
  size_t m_input_cap;
  std::deque<mem_fetch *> m_input;
  std::map<std::pair<unsigned, unsigned>, unsigned long long> m_staged;
  std::vector<bool> m_valid;
  // fill/write in progress
  mem_fetch *m_mutation;
  unsigned long long m_mutation_done;
  unsigned long long m_last_start;
  bool m_started_once;
  unsigned long long m_readout_free;
  unsigned long long m_next_req_id;
  unsigned long long m_emit_seq;
  std::priority_queue<timed_mf, std::vector<timed_mf>, std::greater<timed_mf>>
      m_return;

 public:
  // statistics
  unsigned long long n_qpush, n_qpush_bytes, n_searches, n_fills, n_writes,
      n_fill_rows, n_overflow, n_ii_wait_cycles, n_admit_wait_query,
      n_admit_wait_outstanding, n_admit_wait_mutation, n_readout_wait_cycles,
      n_result_pkts, n_input_full_cycles, n_peak_in_service;
  unsigned long long n_peak_out_bytes = 0, n_readout_stalls = 0,
                     n_readout_stall_cycles = 0, n_output_full_calls = 0,
                     n_readout_limited_steps = 0;
};

// Placement transport around one engine (spec §10). The engine object and its
// behaviour are identical in both placements; only this path differs.
//  - on-chip (placement 0): packets go straight to / from the engine.
//  - external (placement 1): one FIFO link per direction. A packet occupies
//    ceil((payload + header) / unit) units, serialised at the link bandwidth,
//    then propagates for the one-way delay. Times are fractional core cycles
//    (ns x core GHz); a packet is usable at the first whole cycle >= arrival.
//    Outbound: packets accepted at the endpoint wait/cross, then are delivered
//    to the engine when its input has room (credit). Inbound: an engine reply
//    enters the link at its release time (overlapping further readout).
class cam_endpoint {
 public:
  cam_endpoint(const memory_config *config, unsigned sub_partition_id,
               const cam_functional *func, cam_log *log);
  bool full() const;
  void push(mem_fetch *mf, unsigned long long now);
  void cycle(unsigned long long now);
  mem_fetch *top(unsigned long long now);
  void pop();
  bool saw_traffic() const;
  // work scheduled to finish on its own (a fill/write in progress, searches in
  // service, replies, packets on the link). A command blocked at the head of
  // the input queue is NOT counted, so real deadlocks stay detectable.
  bool busy() const {
    return m_engine.scheduled_work() || !m_out.empty() || !m_in.empty();
  }
  unsigned long long progress() const { return m_engine.progress(); }
  void set_staging(cam_staging_pool *p) { m_engine.set_staging(p); }
  void print_stats(FILE *fp) const;

 private:
  struct link_pkt {
    double arrive;
    mem_fetch *mf;
    unsigned wire;                // bytes on the link (units x unit size)
    unsigned long long accepted;  // cycle accepted at the endpoint
  };
  unsigned payload_bytes(mem_fetch *mf, bool outbound) const;
  double cross(bool outbound, unsigned payload, double t_ready, unsigned *wire);
  unsigned wire_bytes(unsigned payload) const;
  static unsigned long long usable(double t);

  const memory_config *m_config;
  unsigned m_id;
  bool m_external;
  cam_unit m_engine;
  std::deque<link_pkt> m_out, m_in;
  double m_free_out, m_free_in;

 public:
  // outbound window occupancy in wire bytes: from acceptance at the endpoint
  // until delivery into the engine input (spec §10, sweep 2)
  unsigned long long m_out_inflight_bytes = 0;
  // inbound: result bytes on the link or waiting at the endpoint (F2)
  unsigned long long m_rx_bytes = 0, n_peak_rx_bytes = 0, n_rx_blocks = 0;
  bool m_rx_blocked = false;
  unsigned long long n_out_payload = 0, n_out_hdr = 0, n_in_payload = 0,
                     n_in_hdr = 0, n_peak_out_inflight_bytes = 0,
                     n_out_delivered = 0, occ_cycles_out = 0;
  unsigned long long n_out_pkts, n_in_pkts, n_out_bytes, n_in_bytes,
      n_peak_out_q, n_peak_in_q, n_refuse_cycles, n_deliver_stall_cycles;
  double busy_out, busy_in, qwait_out, qwait_in;
};

#endif
