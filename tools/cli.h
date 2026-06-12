// Tiny flag parser shared by the TracerDB tools.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace cli {

struct Args {
  std::vector<std::string> positional;
  std::map<std::string, std::string> flags;

  static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
      std::string s = argv[i];
      if (s.rfind("--", 0) == 0) {
        std::string key = s.substr(2);
        auto eq = key.find('=');
        if (eq != std::string::npos) {
          a.flags[key.substr(0, eq)] = key.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
          a.flags[key] = argv[++i];
        } else {
          a.flags[key] = "1";
        }
      } else {
        a.positional.push_back(s);
      }
    }
    return a;
  }

  std::string str(const std::string& k, const std::string& def) const {
    auto it = flags.find(k);
    return it == flags.end() ? def : it->second;
  }
  uint64_t u64(const std::string& k, uint64_t def) const {
    auto it = flags.find(k);
    return it == flags.end() ? def : std::strtoull(it->second.c_str(), nullptr, 10);
  }
  double f64(const std::string& k, double def) const {
    auto it = flags.find(k);
    return it == flags.end() ? def : std::strtod(it->second.c_str(), nullptr);
  }
  bool has(const std::string& k) const { return flags.count(k) > 0; }
};

inline size_t mb_to_frames(double mb) {
  size_t frames = static_cast<size_t>(mb * 1024.0 * 1024.0 / tracerdb::kPageSize);
  return frames < 8 ? 8 : frames;
}

}  // namespace cli
