#include "prag/channel.hpp"
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
namespace prag {
static void put64(unsigned char *p, uint64_t x) {
  for (int i = 7; i >= 0; --i) {
    p[i] = x & 255;
    x >>= 8;
  }
}
static uint64_t get64(const unsigned char *p) {
  uint64_t x = 0;
  for (int i = 0; i < 8; ++i)
    x = (x << 8) | p[i];
  return x;
}
static std::string hexadecimal(const unsigned char *p, size_t n) {
  static const char *hex = "0123456789abcdef";
  std::string out;
  out.reserve(2 * n);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(hex[p[i] >> 4]);
    out.push_back(hex[p[i] & 15]);
  }
  return out;
}
std::string Channel::session_hex() const {
  return hexadecimal(session_.data(), session_.size());
}
std::string Channel::configuration_hex() const {
  return hexadecimal(config_.data(), config_.size());
}
std::string Channel::peer_fingerprint_sha256() const {
  if (!authenticated_)
    throw std::runtime_error("E_TLS_AUTH");
  X509 *certificate = SSL_get1_peer_certificate(tls_);
  unsigned char digest[32];
  unsigned int length = 0;
  if (!certificate)
    throw std::runtime_error("E_TLS_IDENTITY");
  int ok = X509_digest(certificate, EVP_sha256(), digest, &length);
  X509_free(certificate);
  if (ok != 1 || length != 32)
    throw std::runtime_error("E_TLS_IDENTITY");
  return hexadecimal(digest, 32);
}
Channel::Channel(int fd, std::array<unsigned char, 16> s,
                 std::array<unsigned char, 32> c)
    : fd_(fd), session_(s), config_(c) {
  if (fd < 0)
    throw std::runtime_error("E_CHANNEL");
}
Channel::~Channel() {
  if (tls_) {
    SSL_shutdown(tls_);
    SSL_free(tls_);
  }
  if (context_)
    SSL_CTX_free(context_);
  if (fd_ >= 0)
    close(fd_);
}
void Channel::abort() noexcept { ::shutdown(fd_, SHUT_RDWR); }
void Channel::enable_tls(Role role, const std::string &ca,
                         const std::string &cert, const std::string &key,
                         const std::string &peer) {
  if (tls_ || sent_ || received_ || peer.empty())
    throw std::runtime_error("E_TLS_STATE");
  context_ = SSL_CTX_new(TLS_method());
  if (!context_)
    throw std::runtime_error("E_TLS");
  if (!SSL_CTX_set_min_proto_version(context_, TLS1_3_VERSION) ||
      !SSL_CTX_set_max_proto_version(context_, TLS1_3_VERSION) ||
      !SSL_CTX_set1_groups_list(context_, "X25519") ||
      !SSL_CTX_set1_sigalgs_list(context_, "ed25519"))
    throw std::runtime_error("E_TLS_CONFIG");
  SSL_CTX_set_verify(
      context_, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
  if (SSL_CTX_load_verify_locations(context_, ca.c_str(), nullptr) != 1 ||
      SSL_CTX_use_certificate_chain_file(context_, cert.c_str()) != 1 ||
      SSL_CTX_use_PrivateKey_file(context_, key.c_str(), SSL_FILETYPE_PEM) !=
          1 ||
      SSL_CTX_check_private_key(context_) != 1)
    throw std::runtime_error("E_TLS_IDENTITY");
  tls_ = SSL_new(context_);
  if (!tls_)
    throw std::runtime_error("E_TLS");
  SSL_set_fd(tls_, fd_);
  if (SSL_set1_host(tls_, peer.c_str()) != 1)
    throw std::runtime_error("E_TLS_IDENTITY");
  int ok = role == Role::Server ? SSL_accept(tls_) : SSL_connect(tls_);
  if (ok != 1 || SSL_get_verify_result(tls_) != X509_V_OK)
    throw std::runtime_error("E_TLS_AUTH");
  authenticated_ = true;
}
void Channel::write_exact(const void *data, size_t n) {
  auto p = (const unsigned char *)data;
  while (n) {
    int z = tls_ ? SSL_write(tls_, p, (int)std::min(n, size_t(1 << 20)))
                 : send(fd_, p, n, MSG_NOSIGNAL);
    if (z <= 0)
      throw std::runtime_error("E_DISCONNECTED");
    p += z;
    n -= z;
    bytes_ += z;
  }
}
void Channel::read_exact(void *data, size_t n) {
  auto p = (unsigned char *)data;
  while (n) {
    int z = tls_ ? SSL_read(tls_, p, (int)std::min(n, size_t(1 << 20)))
                 : recv(fd_, p, n, 0);
    if (z <= 0)
      throw std::runtime_error("E_DISCONNECTED");
    p += z;
    n -= z;
    bytes_ += z;
  }
}
void Channel::send_data(const void *d, int n) {
  if (n < 0)
    throw std::runtime_error("E_FRAME_SIZE");
  if(diagnostic_last_direction_!='S') {diagnostic_last_direction_='S';++diagnostic_direction_runs_;}
  unsigned char h[96] = {};
  memcpy(h, "PRGCF001", 8);
  memcpy(h + 8, session_.data(), 16);
  memcpy(h + 24, config_.data(), 32);
  put64(h + 56, sent_++);
  put64(h + 64, uint64_t(n));
  put64(h + 72, stage_);
  put64(h + 80, operation_);
  put64(h + 88, 1);
  write_exact(h, 96);
  write_exact(d, n);
}
void Channel::recv_data(void *d, int n) {
  if (n < 0)
    throw std::runtime_error("E_FRAME_SIZE");
  if(diagnostic_last_direction_!='R') {diagnostic_last_direction_='R';++diagnostic_direction_runs_;}
  unsigned char h[96];
  read_exact(h, 96);
  if (memcmp(h, "PRGCF001", 8) || memcmp(h + 8, session_.data(), 16) ||
      memcmp(h + 24, config_.data(), 32) || get64(h + 56) != received_ ||
      get64(h + 64) != uint64_t(n) || get64(h + 72) != stage_ ||
      get64(h + 80) != operation_ || get64(h + 88) != 1)
    throw std::runtime_error("E_FRAME_BINDING");
  ++received_;
  read_exact(d, n);
}
void Channel::send_pt(emp::Point *p, int count) {
  for (int i = 0; i < count; ++i) {
    unsigned char out[65];
    if (p[i].size() != 65)
      throw std::runtime_error("E_CURVE");
    p[i].to_bin(out, 65);
    send_data(out, 65);
  }
}
void Channel::recv_pt(emp::Group *g, emp::Point *p, int count) {
  for (int i = 0; i < count; ++i) {
    unsigned char in[65];
    recv_data(in, 65);
    p[i].from_bin(g, in, 65);
    if (EC_POINT_is_on_curve(g->ec_group, p[i].point, g->bn_ctx) != 1 ||
        EC_POINT_is_at_infinity(g->ec_group, p[i].point))
      throw std::runtime_error("E_CURVE");
  }
}
} // namespace prag
