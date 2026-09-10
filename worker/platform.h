#pragma once

// Platform compatibility layer

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #ifdef _MSC_VER
        #pragma comment(lib, "ws2_32.lib")
    #endif

    #include <io.h>
    #include <process.h>
    #define getpid _getpid

    inline void platform_init() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
    inline void platform_cleanup() { WSACleanup(); }
    inline int gethostname_compat(char* buf, int len) { return gethostname(buf, len); }

    // poll emulation via select
    #define poll WSAPoll
    #include <mswsock.h>

    typedef int socklen_t;
    #define close closesocket

    // Winsock SOCKET is an unsigned 64-bit handle, not an int, and its
    // setsockopt() takes a `const char*` buffer where POSIX takes `const void*`.
    // Both differences are swept up here so the HTTP layer stays portable.
    typedef SOCKET socket_t;
    static const socket_t kInvalidSocket = INVALID_SOCKET;

    inline int socket_set_timeout(socket_t s, int seconds) {
        struct timeval tv;
        tv.tv_sec = seconds;
        tv.tv_usec = 0;
        int rc = setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
        rc |= setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
        return rc;
    }

#else
    // POSIX (macOS, Linux)
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <poll.h>

    inline void platform_init() {}
    inline void platform_cleanup() {}
    inline int gethostname_compat(char* buf, int len) { return gethostname(buf, len); }

    typedef int socket_t;
    static const socket_t kInvalidSocket = -1;

    inline int socket_set_timeout(socket_t s, int seconds) {
        struct timeval tv;
        tv.tv_sec = seconds;
        tv.tv_usec = 0;
        int rc = setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        rc |= setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        return rc;
    }
#endif
