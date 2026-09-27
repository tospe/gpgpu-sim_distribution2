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

  // Readout: first result selected at max(now + L, port free), then one per
  // topk cycles; a packet leaves when its last result is selected.
  const unsigned entry = m_config->cam_result_entry_bytes;
  const unsigned per_pkt = std::max(1u, m_config->cam_result_pkt_bytes / entry);
  const unsigned npkt = (k + per_pkt - 1) / per_pkt;
  const unsigned long long want = now + m_config->cam_search_latency;
  const unsigned long long t0 = std::max(want, m_readout_free);
  n_readout_wait_cycles += t0 - want;
  const unsigned topk = m_config->cam_topk_latency;
  m_readout_free = t0 + (unsigned long long)k * topk;
  const unsigned long long t_last = t0 + (unsigned long long)(k - 1) * topk;
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
    f->m_cam.t_arrive = mf->m_cam.t_arrive;
    f->m_cam.t_admit = now;
    f->m_cam.t_first = t0;
    f->set_reply();
    emit(f, t0 + (unsigned long long)last_idx * topk);
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
