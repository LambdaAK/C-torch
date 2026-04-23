#include "tcp_process_group.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <netdb.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include <cstdint>
#include <vector>

namespace ctorch::distributed
{
namespace
{
constexpr std::uint64_t kHandshakeMagic = 0x4354524F43484431ULL; // "CTORCHD1"

void throw_socket_error(const std::string &context)
{
    throw std::runtime_error(context + ": " + std::strerror(errno));
}

void set_no_sigpipe(int fd)
{
#ifdef SO_NOSIGPIPE
    const int flag = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &flag, sizeof(flag)) != 0)
    {
        throw_socket_error("setsockopt(SO_NOSIGPIPE)");
    }
#endif
}

void send_all(int fd, const void *buffer, std::size_t size)
{
    const char *data = static_cast<const char *>(buffer);
    std::size_t sent = 0;
    while (sent < size)
    {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const ssize_t rc = ::send(fd, data + sent, size - sent, flags);
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw_socket_error("send");
        }
        if (rc == 0)
        {
            throw std::runtime_error("send returned 0 bytes");
        }
        sent += static_cast<std::size_t>(rc);
    }
}

void recv_all(int fd, void *buffer, std::size_t size)
{
    char *data = static_cast<char *>(buffer);
    std::size_t received = 0;
    while (received < size)
    {
        const ssize_t rc = ::recv(fd, data + received, size - received, 0);
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            throw_socket_error("recv");
        }
        if (rc == 0)
        {
            throw std::runtime_error("peer disconnected unexpectedly");
        }
        received += static_cast<std::size_t>(rc);
    }
}

void send_u64(int fd, std::uint64_t value)
{
    send_all(fd, &value, sizeof(value));
}

std::uint64_t recv_u64(int fd)
{
    std::uint64_t value = 0;
    recv_all(fd, &value, sizeof(value));
    return value;
}

void send_string(int fd, const std::string &value)
{
    send_u64(fd, static_cast<std::uint64_t>(value.size()));
    if (!value.empty())
    {
        send_all(fd, value.data(), value.size());
    }
}

std::string recv_string(int fd)
{
    const std::uint64_t size = recv_u64(fd);
    std::string value(size, '\0');
    if (size != 0)
    {
        recv_all(fd, value.data(), static_cast<std::size_t>(size));
    }
    return value;
}

void send_matrix(int fd, const Matrix &matrix)
{
    send_u64(fd, static_cast<std::uint64_t>(matrix.numRows()));
    send_u64(fd, static_cast<std::uint64_t>(matrix.numCols()));
    for (std::size_t i = 0; i < matrix.numRows(); ++i)
    {
        for (std::size_t j = 0; j < matrix.numCols(); ++j)
        {
            const double value = matrix(i, j);
            send_all(fd, &value, sizeof(value));
        }
    }
}

Matrix recv_matrix(int fd)
{
    const std::uint64_t rows = recv_u64(fd);
    const std::uint64_t cols = recv_u64(fd);
    Matrix matrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    for (std::size_t i = 0; i < matrix.numRows(); ++i)
    {
        for (std::size_t j = 0; j < matrix.numCols(); ++j)
        {
            double value = 0.0;
            recv_all(fd, &value, sizeof(value));
            matrix(i, j) = value;
        }
    }
    return matrix;
}

int create_listen_socket(std::uint16_t port)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        throw_socket_error("socket");
    }

    set_no_sigpipe(fd);

    const int reuse = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0)
    {
        ::close(fd);
        throw_socket_error("setsockopt(SO_REUSEADDR)");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
    {
        ::close(fd);
        throw_socket_error("bind");
    }

    if (listen(fd, 16) != 0)
    {
        ::close(fd);
        throw_socket_error("listen");
    }

    return fd;
}

int connect_to_master(const std::string &master_address, std::uint16_t master_port)
{
    if (master_address.empty())
    {
        throw std::invalid_argument("master_address must not be empty.");
    }

    const std::string port_string = std::to_string(master_port);
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *results = nullptr;
    const int rc = getaddrinfo(master_address.c_str(), port_string.c_str(), &hints, &results);
    if (rc != 0)
    {
        throw std::runtime_error("getaddrinfo: " + std::string(gai_strerror(rc)));
    }

    int socket_fd = -1;
    for (addrinfo *result = results; result != nullptr; result = result->ai_next)
    {
        socket_fd = ::socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (socket_fd < 0)
        {
            continue;
        }

        set_no_sigpipe(socket_fd);

        if (::connect(socket_fd, result->ai_addr, result->ai_addrlen) == 0)
        {
            break;
        }

        ::close(socket_fd);
        socket_fd = -1;
    }

    freeaddrinfo(results);

    if (socket_fd < 0)
    {
        throw std::runtime_error("Unable to connect to master " + master_address + ":" + port_string);
    }

    return socket_fd;
}

} // namespace

TcpProcessGroup::TcpProcessGroup(const std::string &master_address, std::uint16_t master_port, int rank, int world_size)
    : rank_(rank), world_size_(world_size)
{
    if (rank_ < 0)
    {
        throw std::invalid_argument("rank must be non-negative.");
    }
    if (world_size_ <= 0)
    {
        throw std::invalid_argument("world_size must be positive.");
    }
    if (rank_ >= world_size_)
    {
        throw std::invalid_argument("rank must be less than world_size.");
    }
    if (master_port == 0)
    {
        throw std::invalid_argument("master_port must be non-zero.");
    }

    if (rank_ == 0)
    {
        listen_socket_ = create_listen_socket(master_port);
        peer_sockets_.assign(static_cast<std::size_t>(world_size_), -1);

        int connected_peers = 0;
        while (connected_peers < world_size_ - 1)
        {
            sockaddr_in peer_addr{};
            socklen_t peer_len = sizeof(peer_addr);
            const int peer_socket = ::accept(listen_socket_, reinterpret_cast<sockaddr *>(&peer_addr), &peer_len);
            if (peer_socket < 0)
            {
                close_sockets();
                throw_socket_error("accept");
            }

            set_no_sigpipe(peer_socket);

            const std::uint64_t magic = recv_u64(peer_socket);
            const std::uint64_t peer_rank = recv_u64(peer_socket);
            const std::uint64_t peer_world = recv_u64(peer_socket);
            if (magic != kHandshakeMagic)
            {
                ::close(peer_socket);
                close_sockets();
                throw std::runtime_error("Invalid distributed handshake magic.");
            }
            if (static_cast<int>(peer_world) != world_size_ || peer_rank >= static_cast<std::uint64_t>(world_size_) || peer_rank == 0)
            {
                ::close(peer_socket);
                close_sockets();
                throw std::runtime_error("Distributed handshake mismatch.");
            }

            if (peer_sockets_[static_cast<std::size_t>(peer_rank)] >= 0)
            {
                ::close(peer_socket);
                close_sockets();
                throw std::runtime_error("Duplicate distributed rank connected.");
            }

            peer_sockets_[static_cast<std::size_t>(peer_rank)] = peer_socket;
            ++connected_peers;
            send_u64(peer_socket, 1);
        }
    }
    else
    {
        root_socket_ = connect_to_master(master_address, master_port);
        send_u64(root_socket_, kHandshakeMagic);
        send_u64(root_socket_, static_cast<std::uint64_t>(rank_));
        send_u64(root_socket_, static_cast<std::uint64_t>(world_size_));
        const std::uint64_t ack = recv_u64(root_socket_);
        if (ack != 1)
        {
            close_sockets();
            throw std::runtime_error("Distributed handshake failed.");
        }
    }
}

TcpProcessGroup::~TcpProcessGroup()
{
    close_sockets();
}

int TcpProcessGroup::rank() const
{
    return rank_;
}

int TcpProcessGroup::world_size() const
{
    return world_size_;
}

void TcpProcessGroup::close_sockets() noexcept
{
    if (listen_socket_ >= 0)
    {
        ::close(listen_socket_);
        listen_socket_ = -1;
    }

    if (root_socket_ >= 0)
    {
        ::close(root_socket_);
        root_socket_ = -1;
    }

    for (int &fd : peer_sockets_)
    {
        if (fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }
    }
}

void TcpProcessGroup::barrier()
{
    if (world_size_ == 1)
    {
        return;
    }

    if (rank_ == 0)
    {
        for (int peer = 1; peer < world_size_; ++peer)
        {
            (void)recv_u64(peer_sockets_[static_cast<std::size_t>(peer)]);
        }
        for (int peer = 1; peer < world_size_; ++peer)
        {
            send_u64(peer_sockets_[static_cast<std::size_t>(peer)], 1);
        }
        return;
    }

    send_u64(root_socket_, 1);
    (void)recv_u64(root_socket_);
}

void TcpProcessGroup::broadcast(std::string &value, int root_rank)
{
    if (root_rank != 0)
    {
        throw std::invalid_argument("TcpProcessGroup currently supports root_rank = 0 only.");
    }
    if (world_size_ == 1)
    {
        return;
    }

    if (rank_ == root_rank)
    {
        for (int peer = 1; peer < world_size_; ++peer)
        {
            send_string(peer_sockets_[static_cast<std::size_t>(peer)], value);
        }
        return;
    }

    value = recv_string(root_socket_);
}

void TcpProcessGroup::broadcast(Matrix &value, int root_rank)
{
    if (root_rank != 0)
    {
        throw std::invalid_argument("TcpProcessGroup currently supports root_rank = 0 only.");
    }
    if (world_size_ == 1)
    {
        return;
    }

    if (rank_ == root_rank)
    {
        for (int peer = 1; peer < world_size_; ++peer)
        {
            send_matrix(peer_sockets_[static_cast<std::size_t>(peer)], value);
        }
        return;
    }

    value = recv_matrix(root_socket_);
}

void TcpProcessGroup::allreduce_sum(Matrix &value)
{
    if (world_size_ == 1)
    {
        return;
    }

    if (rank_ == 0)
    {
        Matrix total = value;
        for (int peer = 1; peer < world_size_; ++peer)
        {
            Matrix incoming = recv_matrix(peer_sockets_[static_cast<std::size_t>(peer)]);
            if (incoming.numRows() != total.numRows() || incoming.numCols() != total.numCols())
            {
                throw std::runtime_error("AllReduce matrix shape mismatch.");
            }
            total = total + incoming;
        }

        value = total;

        for (int peer = 1; peer < world_size_; ++peer)
        {
            send_matrix(peer_sockets_[static_cast<std::size_t>(peer)], value);
        }
        return;
    }

    send_matrix(root_socket_, value);
    value = recv_matrix(root_socket_);
}
} // namespace ctorch::distributed
