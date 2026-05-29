#pragma once

// Platform compatibility layer

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")

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
#endif
