#pragma once
#include <string>
#include <cstring>
#include <cstdio>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <poll.h>

namespace http {

struct Response {
    int status_code = 0;
    std::string body;
    bool ok() const { return status_code >= 200 && status_code < 300; }
};

// Persistent connection to avoid TCP handshake per request
class Client {
public:
    Client() = default;
    ~Client() { close_conn(); }

    void set_base_url(const std::string& url) {
        // Parse http://host:port
        base_url_ = url;
        std::string h = url;
        if (h.find("http://") == 0) h = h.substr(7);
        auto slash = h.find('/');
        if (slash != std::string::npos) h = h.substr(0, slash);
        auto colon = h.find(':');
        if (colon != std::string::npos) {
            host_ = h.substr(0, colon);
            port_ = std::stoi(h.substr(colon + 1));
        } else {
            host_ = h;
            port_ = 80;
        }
    }

    Response get(const std::string& path) {
        std::string req = "GET " + path + " HTTP/1.1\r\n"
                          "Host: " + host_ + "\r\n"
                          "Connection: keep-alive\r\n"
                          "\r\n";
        return do_request(req);
    }

    Response post(const std::string& path, const std::string& json_body) {
        char len_buf[32];
        snprintf(len_buf, sizeof(len_buf), "%zu", json_body.size());
        std::string req = "POST " + path + " HTTP/1.1\r\n"
                          "Host: " + host_ + "\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: " + len_buf + "\r\n"
                          "Connection: keep-alive\r\n"
                          "\r\n" + json_body;
        return do_request(req);
    }

private:
    std::string base_url_;
    std::string host_;
    int port_ = 80;
    int fd_ = -1;

    bool ensure_connected() {
        if (fd_ >= 0) return true;

        struct addrinfo hints{}, *res;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        char port_str[16];
        snprintf(port_str, sizeof(port_str), "%d", port_);

        if (getaddrinfo(host_.c_str(), port_str, &hints, &res) != 0) return false;

        fd_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd_ < 0) { freeaddrinfo(res); return false; }

        // Set timeout
        struct timeval tv = {5, 0}; // 5s timeout
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        if (connect(fd_, res->ai_addr, res->ai_addrlen) < 0) {
            close(fd_); fd_ = -1; freeaddrinfo(res); return false;
        }
        freeaddrinfo(res);
        return true;
    }

    void close_conn() {
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
    }

    Response do_request(const std::string& req) {
        Response resp;
        for (int retry = 0; retry < 2; retry++) {
            if (!ensure_connected()) { continue; }

            // Send
            ssize_t sent = send(fd_, req.c_str(), req.size(), 0);
            if (sent <= 0) { close_conn(); continue; }

            // Receive headers + body
            std::string raw;
            char buf[8192];
            while (true) {
                struct pollfd pfd = {fd_, POLLIN, 0};
                int pr = poll(&pfd, 1, 5000);
                if (pr <= 0) break;
                ssize_t n = recv(fd_, buf, sizeof(buf), 0);
                if (n <= 0) { close_conn(); break; }
                raw.append(buf, n);
                // Check if we have full response
                auto hdr_end = raw.find("\r\n\r\n");
                if (hdr_end != std::string::npos) {
                    // Parse Content-Length
                    int content_length = -1;
                    auto cl_pos = raw.find("Content-Length:");
                    if (cl_pos == std::string::npos) cl_pos = raw.find("content-length:");
                    if (cl_pos != std::string::npos) {
                        content_length = std::atoi(raw.c_str() + cl_pos + 16);
                    }
                    size_t body_start = hdr_end + 4;
                    if (content_length >= 0) {
                        if ((int)(raw.size() - body_start) >= content_length) break;
                    } else {
                        // No content-length: check for chunked or assume done
                        // Simple heuristic: if we got data and poll shows nothing more, done
                        struct pollfd pf2 = {fd_, POLLIN, 0};
                        if (poll(&pf2, 1, 50) <= 0) break;
                    }
                }
            }

            if (raw.empty()) { close_conn(); continue; }

            // Parse status code
            auto sp1 = raw.find(' ');
            if (sp1 != std::string::npos) {
                resp.status_code = std::atoi(raw.c_str() + sp1 + 1);
            }
            // Parse body
            auto hdr_end = raw.find("\r\n\r\n");
            if (hdr_end != std::string::npos) {
                resp.body = raw.substr(hdr_end + 4);
            }

            // Check if connection was closed by server
            auto conn_hdr = raw.find("Connection: close");
            if (conn_hdr != std::string::npos) close_conn();

            return resp;
        }
        return resp;
    }
};

// Global client instance (thread-safe via external synchronization in pipeline)
// Each thread should have its own Client instance.

// Convenience wrappers for backward compatibility
inline Response get(const std::string& url) {
    // Parse full URL into host + path
    std::string h = url;
    if (h.find("http://") == 0) h = h.substr(7);
    auto slash = h.find('/');
    std::string path = "/";
    if (slash != std::string::npos) {
        path = h.substr(slash);
        h = h.substr(0, slash);
    }
    Client c;
    c.set_base_url("http://" + h);
    return c.get(path);
}

inline Response post(const std::string& url, const std::string& body) {
    std::string h = url;
    if (h.find("http://") == 0) h = h.substr(7);
    auto slash = h.find('/');
    std::string path = "/";
    if (slash != std::string::npos) {
        path = h.substr(slash);
        h = h.substr(0, slash);
    }
    Client c;
    c.set_base_url("http://" + h);
    return c.post(path, body);
}

}  // namespace http
