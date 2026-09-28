// CAM extension: functional engine and timing unit. See cam.h.
#include "cam.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include "gpu-sim.h"
#include "mem_fetch.h"

// ---------------------------------------------------------------- functional
static float fp16_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) {
        man <<= 1;
        exp--;
      }
      man &= 0x3ffu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

cam_functional::cam_functional(const char *path)
    : m_loaded(false), m_N(0), m_D(0), m_H(0) {
  if (!path || !path[0]) return;
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "GPGPU-Sim CAM: cannot open functional file %s\n", path);
    abort();
  }
  char magic[8];
  uint32_t hdr[4];
  if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "CAMFUNC1", 8) != 0 ||
      fread(hdr, 4, 4, fp) != 4) {
    fprintf(stderr, "GPGPU-Sim CAM: bad functional file %s\n", path);
    abort();
  }
  m_N = hdr[0];
  m_D = hdr[1];
  m_H = hdr[2];
  const unsigned nq = hdr[3];
  std::vector<uint16_t> buf((size_t)m_N * m_D);
  if (fread(buf.data(), 2, buf.size(), fp) != buf.size()) abort();
  m_keys.resize(buf.size());
  for (size_t i = 0; i < buf.size(); ++i) m_keys[i] = fp16_to_float(buf[i]);
  std::vector<uint16_t> qb((size_t)m_H * m_D);
  m_q.resize((size_t)nq * m_H * m_D);
  m_w.resize((size_t)nq * m_H);
  for (unsigned i = 0; i < nq; ++i) {
    uint32_t key[3];
    if (fread(key, 4, 3, fp) != 3 ||
        fread(qb.data(), 2, qb.size(), fp) != qb.size() ||
        fread(&m_w[(size_t)i * m_H], 4, m_H, fp) != m_H)
      abort();
    for (size_t j = 0; j < qb.size(); ++j)
      m_q[(size_t)i * m_H * m_D + j] = fp16_to_float(qb[j]);
    m_qindex[std::make_tuple(key[0], key[1], key[2])] = i;
  }
  fclose(fp);
  m_loaded = true;
  printf("GPGPU-Sim CAM: functional file %s: N=%u D=%u H=%u queries=%u\n", path,
         m_N, m_D, m_H, nq);
}

void cam_functional::search(unsigned cta_x, unsigned warp, unsigned seq,
                            const std::vector<bool> &valid, unsigned k,
                            std::vector<std::pair<int, float>> &out) const {
  out.clear();
  auto it = m_qindex.find(std::make_tuple(cta_x, warp, seq));
  if (it == m_qindex.end()) {
    fprintf(stderr,
            "GPGPU-Sim CAM: no functional query for cta %u warp %u seq %u\n",
            cta_x, warp, seq);
    abort();
  }
  const float *q = &m_q[(size_t)it->second * m_H * m_D];
  const float *w = &m_w[(size_t)it->second * m_H];
  const float inv_sqrt_h = 1.0f / sqrtf((float)m_H);
  const float inv_sqrt_d = 1.0f / sqrtf((float)m_D);
  std::vector<std::pair<int, float>> all;
  const unsigned rows = std::min<unsigned>(valid.size(), m_N);
  for (unsigned r = 0; r < rows; ++r) {
    if (!valid[r]) continue;
    const float *key = &m_keys[(size_t)r * m_D];
    float score = 0.f;
    for (unsigned h = 0; h < m_H; ++h) {
      float dot = 0.f;
      for (unsigned d = 0; d < m_D; ++d) dot += q[h * m_D + d] * key[d];
      const float relu = std::max(0.f, dot * inv_sqrt_d);
      score += (w[h] * inv_sqrt_h) * relu;
    }
    all.push_back(std::make_pair((int)r, score));
  }
  std::sort(all.begin(), all.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second != b.second ? a.second > b.second
                                          : a.first < b.first;
            });
  if (all.size() > k) all.resize(k);
  out.swap(all);
}

cam_log::cam_log(const char *path) : m_fp(NULL) {
  if (path && path[0]) {
    m_fp = fopen(path, "w");
    if (!m_fp) {
      fprintf(stderr, "GPGPU-Sim CAM: cannot open result log %s\n", path);
      abort();
    }
    fprintf(m_fp,
            "# S,req_id,sid,warp,cta_x,warp_in_cta,seq,slot,t_issue,t_arrive,"
            "t_admit,t_first,t_complete,n,ids...,scores...\n"
            "# F,req_id,sid,warp,slot,row,rows,t_issue,t_arrive,t_complete\n"
            "# W,sid,warp,kind,slot,count,t_wait_issue,t_release\n");
  }
}

cam_log::~cam_log() {
  if (m_fp) fclose(m_fp);
}

bool cam_log::take(unsigned long long req_id,
                   std::vector<std::pair<int, float>> &res) {
  auto it = m_pending.find(req_id);
  if (it == m_pending.end()) return false;
  res.swap(it->second);
  m_pending.erase(it);
  return true;
}

// ---------------------------------------------------------------- timing unit
cam_unit::cam_unit(const memory_config *config, unsigned sub_partition_id,
                   const cam_functional *func, cam_log *log)
    : m_config(config),
      m_id(sub_partition_id),
      m_func(func),
      m_log(log),
      m_input_cap(config->cam_input_queue),
      m_valid(config->cam_num_rows, false),
      m_mutation(NULL),
      m_mutation_done(0),
      m_last_start(0),
      m_started_once(false),
      m_readout_free(0),
      m_next_req_id(1),
      m_emit_seq(0),
      n_qpush(0),
      n_qpush_bytes(0),
      n_searches(0),
      n_fills(0),
      n_writes(0),
      n_fill_rows(0),
      n_overflow(0),
      n_ii_wait_cycles(0),
      n_admit_wait_query(0),
      n_admit_wait_outstanding(0),
      n_admit_wait_mutation(0),
      n_readout_wait_cycles(0),
      n_result_pkts(0),
      n_input_full_cycles(0),
      n_peak_in_service(0) {}

void cam_unit::push(mem_fetch *mf, unsigned long long now) {
  assert(!full());
  mf->m_cam.t_arrive = now;
  m_input.push_back(mf);
}

void cam_unit::emit(mem_fetch *mf, unsigned long long ready) {
  m_return.push(timed_mf{ready, m_emit_seq++, mf});
}

bool cam_unit::busy() const {
  return !m_input.empty() || m_mutation || !m_return.empty() ||
         !m_in_service.empty();
}

void cam_unit::start_search(mem_fetch *mf, unsigned long long now) {
  const warp_inst_t &inst = mf->get_inst();
  const unsigned long long req_id = m_next_req_id++;
  const unsigned k = inst.m_cam_k ? inst.m_cam_k : 1;
  n_searches++;
  if (m_started_once && now < m_last_start + m_config->cam_search_ii)
    assert(0 && "admission before II elapsed");
  m_last_start = now;
  m_started_once = true;

  if (m_func && m_func->loaded() && m_log && m_log->on()) {
    std::vector<std::pair<int, float>> res;
    m_func->search(inst.get_cuda_cta_id().x, inst.m_cam_warp_in_cta,
                   inst.m_cam_seq, m_valid, k, res);
    m_log->stash(req_id, res);
  }

  // Readout: first result selected at max(now + L, port free); result i at
  // t0 + floor(i / R) * topk (R results per readout step, spec §10); a packet
  // leaves when its last result is selected. R = 1 is the v1 behaviour.
  const unsigned entry = m_config->cam_result_entry_bytes;
  const unsigned per_pkt = std::max(1u, m_config->cam_result_pkt_bytes / entry);
  const unsigned npkt = (k + per_pkt - 1) / per_pkt;
  const unsigned long long want = now + m_config->cam_search_latency;
  const unsigned long long t0 = std::max(want, m_readout_free);
  n_readout_wait_cycles += t0 - want;
  const unsigned topk = m_config->cam_topk_latency;
  const unsigned R = std::max(1u, m_config->cam_readout_per_cycle);
  auto sel = [&](unsigned i) {
    return t0 + (unsigned long long)(i / R) * topk;
  };
  m_readout_free = t0 + (unsigned long long)((k + R - 1) / R) * topk;
  const unsigned long long t_last = sel(k - 1);
  m_in_service.insert(t_last);
  if (m_in_service.size() > n_peak_in_service)
    n_peak_in_service = m_in_service.size();

  for (unsigned j = 0; j < npkt; ++j) {
    const unsigned last_idx = std::min((j + 1) * per_pkt, k) - 1;
    const unsigned n_in_pkt = last_idx + 1 - j * per_pkt;
    mem_fetch *f =
        new mem_fetch(mf->get_mem_access(), mf->get_inst_ptr(),
                      mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
                      mf->get_sid(), mf->get_tpc(), m_config, now);
    f->set_data_size(n_in_pkt * entry);
    f->m_cam.valid = true;
    f->m_cam.frag = j;
    f->m_cam.nfrag = npkt;
    f->m_cam.req_id = req_id;
    f->m_cam.t_endpoint = mf->m_cam.t_endpoint;
    f->m_cam.t_arrive = mf->m_cam.t_arrive;
    f->m_cam.t_admit = now;
    f->m_cam.t_first = t0;
    f->set_reply();
    emit(f, sel(last_idx));
    n_result_pkts++;
  }
  delete mf;
}

void cam_unit::cycle(unsigned long long now) {
  // searches whose last result has been selected leave service
  while (!m_in_service.empty() && *m_in_service.begin() <= now)
    m_in_service.erase(m_in_service.begin());

  // a fill/write completes: its rows become searchable, then it is acked
  if (m_mutation && now >= m_mutation_done) {
    const warp_inst_t &inst = m_mutation->get_inst();
    const unsigned rows = inst.m_cam_op == CAM_OP_FILL ? inst.m_cam_rows : 1;
    for (unsigned r = inst.m_cam_row; r < inst.m_cam_row + rows; ++r) {
      if (r < m_valid.size())
        m_valid[r] = true;
      else
        n_overflow++;
    }
    m_mutation->set_data_size(0);  // ack carries no data
    m_mutation->m_cam.frag = 0;
    m_mutation->m_cam.nfrag = 1;
    m_mutation->m_cam.t_admit = m_mutation_done;
    m_mutation->set_reply();
    emit(m_mutation, now);
    m_mutation = NULL;
  }

  // posted query chunks: stage wherever they sit in the queue
  for (auto it = m_input.begin(); it != m_input.end();) {
    mem_fetch *mf = *it;
    if (mf->get_inst().m_cam_op == CAM_OP_QPUSH) {
      m_staged[std::make_pair(mf->get_sid(), mf->get_wid())] +=
          mf->get_access_size();
      n_qpush++;
      n_qpush_bytes += mf->get_access_size();
      delete mf;
      it = m_input.erase(it);
    } else {
      ++it;
    }
  }
  if (full()) n_input_full_cycles++;

  // one ordered command per cycle
  if (m_input.empty()) return;
  mem_fetch *mf = m_input.front();
  const warp_inst_t &inst = mf->get_inst();
  if (inst.m_cam_op == CAM_OP_FILL || inst.m_cam_op == CAM_OP_WRITE) {
    if (m_mutation) return;
    m_input.pop_front();
    m_mutation = mf;
    if (inst.m_cam_op == CAM_OP_FILL) {
      n_fills++;
      n_fill_rows += inst.m_cam_rows;
      m_mutation_done =
          now + m_config->cam_fill_setup_latency +
          (unsigned long long)inst.m_cam_rows * m_config->cam_fill_row_latency;
    } else {
      n_writes++;
      m_mutation_done = now + m_config->cam_write_latency;
    }
    return;
  }
  assert(inst.m_cam_op == CAM_OP_SEARCH);
  if (m_mutation) {
    n_admit_wait_mutation++;
    return;
  }
  auto key = std::make_pair(mf->get_sid(), mf->get_wid());
  if (m_staged[key] < m_config->cam_query_bytes) {
    n_admit_wait_query++;
    return;
  }
  if (m_in_service.size() >= m_config->cam_max_outstanding) {
    n_admit_wait_outstanding++;
    return;
  }
  if (m_started_once && now < m_last_start + m_config->cam_search_ii) {
    n_ii_wait_cycles++;
    return;
  }
  m_staged[key] -= m_config->cam_query_bytes;
  m_input.pop_front();
  start_search(mf, now);
}

mem_fetch *cam_unit::top(unsigned long long now) {
  if (m_return.empty() || m_return.top().ready > now) return NULL;
  return m_return.top().mf;
}

void cam_unit::pop() { m_return.pop(); }

void cam_unit::print_stats(FILE *fp) const {
  fprintf(fp,
          "cam_unit[%u]: searches=%llu fills=%llu fill_rows=%llu writes=%llu "
          "overflow=%llu qpush=%llu qpush_bytes=%llu result_pkts=%llu "
          "peak_in_service=%llu ii_wait=%llu admit_wait_query=%llu "
          "admit_wait_outstanding=%llu admit_wait_mutation=%llu "
          "readout_wait=%llu input_full_cycles=%llu\n",
          m_id, n_searches, n_fills, n_fill_rows, n_writes, n_overflow, n_qpush,
          n_qpush_bytes, n_result_pkts, n_peak_in_service, n_ii_wait_cycles,
          n_admit_wait_query, n_admit_wait_outstanding, n_admit_wait_mutation,
          n_readout_wait_cycles, n_input_full_cycles);
}

// ---------------------------------------------------------------- transport
cam_endpoint::cam_endpoint(const memory_config *config,
                           unsigned sub_partition_id,
                           const cam_functional *func, cam_log *log)
    : m_config(config),
      m_id(sub_partition_id),
      m_external(config->cam_placement == 1),
      m_engine(config, sub_partition_id, func, log),
      m_free_out(0),
      m_free_in(0),
      n_out_pkts(0),
      n_in_pkts(0),
      n_out_bytes(0),
      n_in_bytes(0),
      n_peak_out_q(0),
      n_peak_in_q(0),
      n_refuse_cycles(0),
      n_deliver_stall_cycles(0),
      busy_out(0),
      busy_in(0),
      qwait_out(0),
      qwait_in(0) {
  assert(config->cam_placement <= 1);
}

// Outbound window (spec §10). Packet mode: at most cam_link_out_queue packets.
// Byte mode (cam_link_window_bytes > 0): a request packet (payload <= 32 B)
// is accepted only if its wire bytes fit in the remaining window. A packet
// occupies the window from acceptance until delivery into the engine input.
bool cam_endpoint::full() const {
  if (!m_external) return m_engine.full();
  if (m_config->cam_link_window_bytes)
    return m_out_inflight_bytes + wire_bytes(32) >
           m_config->cam_link_window_bytes;
  return m_out.size() >= m_config->cam_link_out_queue;
}

unsigned cam_endpoint::wire_bytes(unsigned payload) const {
  const unsigned unit = std::max(1u, m_config->cam_link_unit_bytes);
  return ((payload + m_config->cam_link_hdr_bytes + unit - 1) / unit) * unit;
}

unsigned long long cam_endpoint::usable(double t) {
  // first whole cycle >= t (tolerate float noise on integral values)
  return (unsigned long long)ceil(t - 1e-9);
}

// Payload bytes a packet carries on the link (spec §10): query chunks their
// data; commands and fill/write acks 8 B control; result packets their data.
unsigned cam_endpoint::payload_bytes(mem_fetch *mf, bool outbound) const {
  const warp_inst_t &inst = mf->get_inst();
  if (outbound) return inst.m_cam_op == CAM_OP_QPUSH ? mf->get_data_size() : 8;
  return inst.m_cam_op == CAM_OP_SEARCH ? mf->get_data_size() : 8;
}

double cam_endpoint::cross(bool outbound, unsigned payload, double t_ready,
                           unsigned *wire) {
  const unsigned bytes = wire_bytes(payload);
  if (wire) *wire = bytes;
  if (outbound) {
    n_out_payload += payload;
    n_out_hdr += m_config->cam_link_hdr_bytes;
  } else {
    n_in_payload += payload;
    n_in_hdr += m_config->cam_link_hdr_bytes;
  }
  const double ghz = m_config->cam_core_ghz;
  // bytes / (GB/s) = ns; ns * GHz = core cycles
  const double ser = m_config->cam_link_gbps > 0
                         ? (double)bytes / m_config->cam_link_gbps * ghz
                         : 0.0;
  const double prop = m_config->cam_link_latency_ns * ghz;
  double &free_at = outbound ? m_free_out : m_free_in;
  const double start = std::max(t_ready, free_at);
  free_at = start + ser;
  if (outbound) {
    n_out_pkts++;
    n_out_bytes += bytes;
    busy_out += ser;
    qwait_out += start - t_ready;
  } else {
    n_in_pkts++;
    n_in_bytes += bytes;
    busy_in += ser;
    qwait_in += start - t_ready;
  }
  return start + ser + prop;
}

void cam_endpoint::push(mem_fetch *mf, unsigned long long now) {
  mf->m_cam.t_endpoint = now;
  if (!m_external) {
    m_engine.push(mf, now);
    return;
  }
  unsigned wire = 0;
  const double arrive =
      cross(true, payload_bytes(mf, true), (double)now, &wire);
  m_out.push_back(link_pkt{arrive, mf, wire, now});
  if (m_out.size() > n_peak_out_q) n_peak_out_q = m_out.size();
  m_out_inflight_bytes += wire;
  if (m_out_inflight_bytes > n_peak_out_inflight_bytes)
    n_peak_out_inflight_bytes = m_out_inflight_bytes;
}

void cam_endpoint::cycle(unsigned long long now) {
  if (m_external) {
    if (full()) n_refuse_cycles++;
    while (!m_out.empty() && usable(m_out.front().arrive) <= now) {
      if (m_engine.full()) {  // credit: wait at the far side
        n_deliver_stall_cycles++;
        break;
      }
      m_engine.push(m_out.front().mf, now);
      // window release event: delivery into the engine input
      m_out_inflight_bytes -= m_out.front().wire;
      occ_cycles_out += now - m_out.front().accepted;
      n_out_delivered++;
      m_out.pop_front();
    }
  } else if (full()) {
    n_refuse_cycles++;
  }
  m_engine.cycle(now);
}

mem_fetch *cam_endpoint::top(unsigned long long now) {
  if (!m_external) {
    mem_fetch *mf = m_engine.top(now);
    if (mf) {
      mf->m_cam.t_last_ready = m_engine.next_reply_time();
      mf->m_cam.t_last_ep = m_engine.next_reply_time();
    }
    return mf;
  }
  // engine replies released by now enter the inbound link at release time
  while (m_engine.has_reply() && m_engine.next_reply_time() <= now) {
    const unsigned long long rel = m_engine.next_reply_time();
    mem_fetch *mf = m_engine.top(now);
    m_engine.pop();
    const double arrive =
        cross(false, payload_bytes(mf, false), (double)rel, NULL);
    mf->m_cam.t_last_ready = rel;
    mf->m_cam.t_last_ep = usable(arrive);
    m_in.push_back(link_pkt{arrive, mf});
    if (m_in.size() > n_peak_in_q) n_peak_in_q = m_in.size();
  }
  if (!m_in.empty() && usable(m_in.front().arrive) <= now)
    return m_in.front().mf;
  return NULL;
}

void cam_endpoint::pop() {
  if (m_external)
    m_in.pop_front();
  else
    m_engine.pop();
}

bool cam_endpoint::saw_traffic() const {
  return m_engine.n_searches || m_engine.n_fills || m_engine.n_writes ||
         m_engine.n_qpush;
}

void cam_endpoint::print_stats(FILE *fp) const {
  m_engine.print_stats(fp);
  fprintf(fp,
          "cam_endpoint[%u]: placement=%s refuse_cycles=%llu "
          "deliver_stall_cycles=%llu",
          m_id, m_external ? "external" : "on-chip", n_refuse_cycles,
          n_deliver_stall_cycles);
  if (m_external)
    fprintf(fp,
            " link_ns=%.3f link_gbps=%.3f unit=%u hdr=%u out_pkts=%llu "
            "out_bytes=%llu out_busy_cyc=%.1f out_qwait_cyc=%.1f "
            "peak_out_q=%llu in_pkts=%llu in_bytes=%llu in_busy_cyc=%.1f "
            "in_qwait_cyc=%.1f peak_in_q=%llu",
            m_config->cam_link_latency_ns, m_config->cam_link_gbps,
            m_config->cam_link_unit_bytes, m_config->cam_link_hdr_bytes,
            n_out_pkts, n_out_bytes, busy_out, qwait_out, n_peak_out_q,
            n_in_pkts, n_in_bytes, busy_in, qwait_in, n_peak_in_q);
  if (m_external)
    fprintf(
        fp,
        " window_bytes=%u out_payload=%llu out_hdr=%llu out_wire=%llu "
        "in_payload=%llu in_hdr=%llu in_wire=%llu peak_out_inflight_bytes=%llu "
        "mean_out_occupancy_cyc=%.2f",
        m_config->cam_link_window_bytes, n_out_payload, n_out_hdr, n_out_bytes,
        n_in_payload, n_in_hdr, n_in_bytes, n_peak_out_inflight_bytes,
        n_out_delivered ? (double)occ_cycles_out / n_out_delivered : 0.0);
  fprintf(fp, "\n");
}
