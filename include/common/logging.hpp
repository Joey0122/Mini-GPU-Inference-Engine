// include/common/logging.hpp
//
// Single, minimal logging interface.
//
// ARCHITECTURE.md / AGENT.md require one centralized logger with the levels
// INFO / WARN / ERROR / DEBUG and forbid scattered printf in kernels. This is
// a header-only, dependency-free implementation: a runtime-settable threshold
// plus stream-style macros that compile to a no-op below the active level.
//
// Usage:
//   LOG_INFO("loaded " << n << " weights");
//   LOG_ERROR("device " << id << " unavailable");
//   mgie::Logger::SetLevel(mgie::LogLevel::kDebug);  // raise verbosity
//
// Host-only. Device code must not log; kernels return data and the host
// reports it (keeps the kernel layer free of IO per ARCHITECTURE.md).

#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace mgie {

enum class LogLevel { kError = 0, kWarn = 1, kInfo = 2, kDebug = 3 };

class Logger {
 public:
  // Global threshold. Messages strictly above this level are dropped.
  // Defaults to INFO, or to the level named by the MGIE_LOG_LEVEL env var
  // (ERROR|WARN|INFO|DEBUG) on first use.
  static LogLevel& Level() {
    static LogLevel level = InitialLevel();
    return level;
  }

  static void SetLevel(LogLevel level) { Level() = level; }

  static bool Enabled(LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(Level());
  }

  static const char* Tag(LogLevel level) {
    switch (level) {
      case LogLevel::kError: return "ERROR";
      case LogLevel::kWarn:  return "WARN";
      case LogLevel::kInfo:  return "INFO";
      case LogLevel::kDebug: return "DEBUG";
    }
    return "?";
  }

  static std::ostream& Stream(LogLevel level) {
    return level == LogLevel::kError ? std::cerr : std::cout;
  }

 private:
  static LogLevel InitialLevel() {
    const char* env = std::getenv("MGIE_LOG_LEVEL");
    if (env != nullptr) {
      const std::string value(env);
      if (value == "ERROR") return LogLevel::kError;
      if (value == "WARN")  return LogLevel::kWarn;
      if (value == "INFO")  return LogLevel::kInfo;
      if (value == "DEBUG") return LogLevel::kDebug;
    }
    return LogLevel::kInfo;
  }
};

}  // namespace mgie

// Internal: build and emit one line. The do/while wrapper keeps the macro a
// single statement; the Enabled() guard avoids constructing the message when
// the level is suppressed.
#define MGIE_LOG(level, msg)                                          \
  do {                                                                \
    if (::mgie::Logger::Enabled(level)) {                             \
      std::ostringstream mgie_log_oss_;                               \
      mgie_log_oss_ << msg;                                           \
      ::mgie::Logger::Stream(level)                                   \
          << "[" << ::mgie::Logger::Tag(level) << "] "                \
          << mgie_log_oss_.str() << std::endl;                        \
    }                                                                 \
  } while (0)

#define LOG_ERROR(msg) MGIE_LOG(::mgie::LogLevel::kError, msg)
#define LOG_WARN(msg)  MGIE_LOG(::mgie::LogLevel::kWarn, msg)
#define LOG_INFO(msg)  MGIE_LOG(::mgie::LogLevel::kInfo, msg)
#define LOG_DEBUG(msg) MGIE_LOG(::mgie::LogLevel::kDebug, msg)
