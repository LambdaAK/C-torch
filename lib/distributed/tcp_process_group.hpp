#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "process_group.hpp"

namespace ctorch::distributed
{
/**
 * @brief TCP-backed synchronous process group for local or cluster training.
 *
 * The backend uses a single root coordinator. Rank 0 listens for peers,
 * receives their gradient payloads, sums them, and broadcasts the reduced
 * result back to every worker.
 */
class TcpProcessGroup : public ProcessGroup
{
public:
    TcpProcessGroup(const std::string &master_address, std::uint16_t master_port, int rank, int world_size);
    ~TcpProcessGroup() override;

    TcpProcessGroup(const TcpProcessGroup &) = delete;
    TcpProcessGroup &operator=(const TcpProcessGroup &) = delete;

    int rank() const override;
    int world_size() const override;

    void barrier() override;
    void broadcast(std::string &value, int root_rank = 0) override;
    void broadcast(Matrix &value, int root_rank = 0) override;
    void allreduce_sum(Matrix &value) override;

private:
    int rank_;
    int world_size_;
    int listen_socket_ = -1;
    int root_socket_ = -1;
    std::vector<int> peer_sockets_;

    void close_sockets() noexcept;
};
} // namespace ctorch::distributed
